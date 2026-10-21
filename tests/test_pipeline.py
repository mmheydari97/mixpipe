"""Tests for the parts where a bug would be silent.

Run with:  PYTHONPATH=src python -m pytest tests/ -q
           PYTHONPATH=src python tests/test_pipeline.py   (no pytest needed)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mixpipe import SAMPLE_RATE
from mixpipe.analysis import (
    analyze_all, band_energy_db, crest_factor_db, peak_db, rms_db,
    spectral_centroid_hz,
)
from mixpipe.cleanup import fade_edges, noise_gate, remove_dc
from mixpipe.config import load_config, treatment_for
from mixpipe.dynamics import apply_gain_db, compress, limit, normalize_peak
from mixpipe.eq import apply_eq_chain, highpass, peaking
from mixpipe.io_utils import Stem, infer_role, pad_to_common_length, read_audio, write_audio
from mixpipe.mixdown import mixdown, sum_bus
from mixpipe.panning import pan_mono_to_stereo, width
from mixpipe.plan import validate_plan
from mixpipe.advisor import rules_advisor
from mixpipe.master import master


def sine(freq=440.0, dur=1.0, amp=0.5, sr=SAMPLE_RATE):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(dur=1.0, amp=0.1, sr=SAMPLE_RATE, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, amp, int(dur * sr))).astype(np.float32)


def fake_stems():
    return [
        Stem("acoustic_gtr", sine(330, 2.0, 0.3), SAMPLE_RATE, "acoustic"),
        Stem("electric_lead", sine(660, 2.0, 0.6), SAMPLE_RATE, "electric_lead"),
        Stem("harmonica", sine(880, 2.0, 0.4), SAMPLE_RATE, "harmonica"),
        Stem("seaboard_pad", sine(110, 2.0, 0.25), SAMPLE_RATE, "keys"),
    ]


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def test_peak_and_rms_of_sine():
    s = sine(amp=0.5)
    assert abs(peak_db(s) - (-6.02)) < 0.2
    # RMS of a sine is peak/sqrt(2), i.e. 3 dB below peak.
    assert abs(rms_db(s) - (-9.03)) < 0.2


def test_crest_factor_sine_vs_noise():
    # A sine has ~3 dB crest; gaussian noise has much more.
    assert abs(crest_factor_db(sine()) - 3.0) < 0.5
    assert crest_factor_db(noise()) > 8.0


def test_spectral_centroid_tracks_frequency():
    low = spectral_centroid_hz(sine(200, 2.0))
    high = spectral_centroid_hz(sine(4000, 2.0))
    assert low < high
    assert 100 < low < 400
    assert 3000 < high < 5000


def test_band_energy_is_normalized_and_finds_the_band():
    bands = band_energy_db(sine(100, 2.0))
    assert max(bands.values()) == 0.0          # loudest band is the reference
    assert bands["low"] == 0.0                 # 100 Hz sits in 60-200
    assert bands["high"] < -20.0


# --------------------------------------------------------------------------- #
# gain and dynamics
# --------------------------------------------------------------------------- #

def test_gain_is_exact():
    s = sine(amp=0.25)
    assert abs(peak_db(apply_gain_db(s, 6.0)) - (peak_db(s) + 6.0)) < 0.01


def test_normalize_peak_hits_target():
    for target in (-1.0, -6.0, -20.0):
        out = normalize_peak(sine(amp=0.9), target_db=target)
        assert abs(peak_db(out) - target) < 0.05


def test_compressor_reduces_dynamic_range():
    # Quiet passage then loud passage; compression should narrow the gap.
    quiet, loud = sine(440, 1.0, 0.05), sine(440, 1.0, 0.9)
    sig = np.concatenate([quiet, loud])
    out = compress(sig, threshold_db=-20.0, ratio=8.0, attack_ms=5.0,
                   release_ms=100.0, makeup_db=0.0)
    n = quiet.size
    before = rms_db(sig[n:]) - rms_db(sig[:n])
    after = rms_db(out[n:]) - rms_db(out[:n])
    assert after < before - 3.0


def test_compressor_ratio_one_is_transparent():
    s = sine(amp=0.5)
    out = compress(s, ratio=1.0, makeup_db=0.0)
    assert np.allclose(out, s, atol=1e-6)


def test_limiter_respects_ceiling():
    for ceiling in (-0.5, -1.0, -3.0):
        hot = np.stack([sine(amp=2.5), sine(amp=2.5)], axis=1)
        out = limit(hot, ceiling_db=ceiling)
        assert peak_db(out) <= ceiling + 0.05


def test_limiter_does_not_touch_quiet_audio():
    quiet = np.stack([sine(amp=0.1)] * 2, axis=1)
    out = limit(quiet, ceiling_db=-1.0)
    # Allow for the lookahead delay by comparing peak, not sample-wise.
    assert abs(peak_db(out) - peak_db(quiet)) < 0.1


# --------------------------------------------------------------------------- #
# EQ
# --------------------------------------------------------------------------- #

def test_highpass_attenuates_below_cutoff():
    low, high = sine(50, 1.0, 0.5), sine(2000, 1.0, 0.5)
    assert rms_db(highpass(low, 500.0)) < rms_db(low) - 20.0
    assert abs(rms_db(highpass(high, 500.0)) - rms_db(high)) < 1.0


def test_peaking_boost_and_cut():
    s = sine(1000, 2.0, 0.3)
    boosted = peaking(s, 1000.0, 6.0, q=1.0)
    cut = peaking(s, 1000.0, -6.0, q=1.0)
    assert rms_db(boosted) > rms_db(s) + 4.0
    assert rms_db(cut) < rms_db(s) - 4.0


def test_peaking_leaves_distant_frequency_alone():
    s = sine(100, 2.0, 0.3)
    out = peaking(s, 8000.0, 9.0, q=2.0)
    assert abs(rms_db(out) - rms_db(s)) < 1.0


def test_eq_chain_runs_with_empty_bands():
    s = sine()
    assert apply_eq_chain(s, 0, 0, []).shape == s.shape


# --------------------------------------------------------------------------- #
# cleanup
# --------------------------------------------------------------------------- #

def test_remove_dc():
    s = sine() + 0.2
    assert abs(np.mean(remove_dc(s))) < 1e-6


def test_gate_silences_below_threshold():
    # Loud burst surrounded by near-silence.
    sig = np.concatenate([noise(0.5, 1e-4), sine(440, 0.5, 0.8), noise(0.5, 1e-4)])
    out = noise_gate(sig, threshold_db=-40.0, hold_ms=10.0, release_ms=20.0)
    n = int(0.5 * SAMPLE_RATE)
    assert rms_db(out[:n]) < rms_db(sig[:n]) - 10.0    # silence got quieter
    assert rms_db(out[n:2 * n]) > rms_db(sig[n:2 * n]) - 1.0  # signal survived


def test_fade_edges_starts_and_ends_at_zero():
    out = fade_edges(sine(amp=0.9), fade_ms=10.0)
    assert abs(out[0]) < 1e-6
    assert abs(out[-1]) < 1e-6


# --------------------------------------------------------------------------- #
# panning
# --------------------------------------------------------------------------- #

def test_pan_centre_is_equal_and_constant_power():
    out = pan_mono_to_stereo(sine(amp=0.5), 0.0)
    assert out.shape[1] == 2
    assert np.allclose(out[:, 0], out[:, 1])
    # Constant power: centre is ~-3 dB per side.
    assert abs(peak_db(out[:, 0]) - (peak_db(sine(amp=0.5)) - 3.0)) < 0.2


def test_pan_hard_left_silences_right():
    out = pan_mono_to_stereo(sine(amp=0.5), -1.0)
    assert peak_db(out[:, 1]) < -60.0
    assert peak_db(out[:, 0]) > -7.0


def test_width_zero_collapses_to_mono():
    stereo = np.stack([sine(440, 1.0, 0.5), sine(880, 1.0, 0.5)], axis=1)
    out = width(stereo, 0.0)
    assert np.allclose(out[:, 0], out[:, 1], atol=1e-6)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def test_write_then_read_roundtrip():
    original = sine(440, 0.5, 0.5)
    with tempfile.TemporaryDirectory() as d:
        p = write_audio(Path(d) / "t.wav", original)
        back, sr = read_audio(p)
        assert sr == SAMPLE_RATE
        assert back.shape[0] == original.shape[0]
        assert np.max(np.abs(back - original)) < 1e-3   # 24-bit quantization


def test_infer_role_prefers_longest_match():
    assert infer_role("electric_lead_take3.wav") == "electric_lead"
    assert infer_role("electric_rhythm.wav") == "electric"
    assert infer_role("my_harmonica.wav") == "harmonica"
    assert infer_role("seaboard_bass.wav") == "bass"
    assert infer_role("seaboard_pad.wav") == "keys"
    assert infer_role("something_weird.wav") == "default"


def test_pad_to_common_length():
    stems = [
        Stem("a", sine(dur=1.0), SAMPLE_RATE),
        Stem("b", sine(dur=2.0), SAMPLE_RATE),
    ]
    padded = pad_to_common_length(stems)
    assert len({s.n_samples for s in padded}) == 1


# --------------------------------------------------------------------------- #
# plan validation - the safety-critical part
# --------------------------------------------------------------------------- #

def test_plan_clamps_absurd_llm_output():
    stems = fake_stems()
    evil = {
        "stems": {
            "acoustic_gtr": {
                "gain_db": 400.0,
                "pan": -99.0,
                "hpf_hz": 999999.0,
                "eq": [{"freq": 1e9, "gain_db": 60.0, "q": -5.0}],
                "comp": {"ratio": 1000.0, "attack_ms": -3.0, "threshold_db": 50.0},
            }
        },
        "master": {"target_lufs": 20.0, "ceiling_db": 12.0},
    }
    plan = validate_plan(evil, stems)
    s = plan["stems"]["acoustic_gtr"]
    assert s["gain_db"] <= 12.0
    assert -1.0 <= s["pan"] <= 1.0
    assert s["hpf_hz"] <= 1000.0
    assert s["eq"][0]["freq"] <= 20000.0
    assert s["eq"][0]["gain_db"] <= 9.0
    assert s["eq"][0]["q"] >= 0.2
    assert s["comp"]["ratio"] <= 20.0
    assert s["comp"]["attack_ms"] >= 0.1
    assert plan["master"]["target_lufs"] <= -6.0
    assert plan["master"]["ceiling_db"] <= -0.1


def test_plan_handles_nan_and_garbage():
    stems = fake_stems()
    plan = validate_plan(
        {"stems": {"harmonica": {"gain_db": float("nan"), "pan": "left",
                                 "eq": "not a list", "comp": None}}},
        stems,
    )
    s = plan["stems"]["harmonica"]
    assert s["gain_db"] == s["gain_db"]        # not NaN
    assert isinstance(s["eq"], list)
    assert s["comp"]["ratio"] >= 1.0


def test_plan_fills_missing_stems_from_role_defaults():
    stems = fake_stems()
    plan = validate_plan({"stems": {}}, stems)
    assert set(plan["stems"]) == {s.name for s in stems}
    assert plan["stems"]["harmonica"]["hpf_hz"] == treatment_for("harmonica")["hpf_hz"]


def test_plan_matches_stem_names_case_insensitively():
    stems = fake_stems()
    plan = validate_plan({"stems": {"HARMONICA": {"gain_db": -4.0}}}, stems)
    assert plan["stems"]["harmonica"]["gain_db"] == -4.0


def test_plan_caps_eq_band_count():
    stems = fake_stems()
    many = [{"freq": 100 * i, "gain_db": 1.0, "q": 1.0} for i in range(1, 30)]
    plan = validate_plan({"stems": {"harmonica": {"eq": many}}}, stems)
    assert len(plan["stems"]["harmonica"]["eq"]) <= 6


# --------------------------------------------------------------------------- #
# advisor and full pipeline
# --------------------------------------------------------------------------- #

def test_rules_advisor_covers_every_stem_and_explains_itself():
    stems = fake_stems()
    plan = rules_advisor(stems, analyze_all(stems), load_config())
    assert set(plan["stems"]) == {s.name for s in stems}
    assert all(plan["stems"][n]["reason"] for n in plan["stems"])


def test_rules_advisor_pulls_down_the_loud_stem():
    stems = fake_stems()   # electric_lead is the loudest at amp 0.6
    plan = rules_advisor(stems, analyze_all(stems), load_config())
    assert plan["stems"]["electric_lead"]["gain_db"] < plan["stems"]["acoustic_gtr"]["gain_db"]


def test_sum_bus_is_stereo_and_longest_length():
    stems = [
        Stem("a", pan_mono_to_stereo(sine(dur=1.0), -0.5), SAMPLE_RATE),
        Stem("b", pan_mono_to_stereo(sine(dur=2.0), 0.5), SAMPLE_RATE),
    ]
    bus = sum_bus(stems)
    assert bus.ndim == 2 and bus.shape[1] == 2
    assert bus.shape[0] == 2 * SAMPLE_RATE


def test_mixdown_then_master_produces_valid_audio():
    cfg = load_config()
    stems = fake_stems()
    plan = rules_advisor(stems, analyze_all(stems), cfg)
    bus, processed, report = mixdown(stems, plan, cfg, do_cleanup=False, verbose=False)
    final, m_report = master(bus, plan, cfg, verbose=False)

    assert final.ndim == 2 and final.shape[1] == 2
    assert np.all(np.isfinite(final))
    assert peak_db(final) <= plan["master"]["ceiling_db"] + 0.05
    assert len(processed) == len(stems)
    assert m_report["output_lufs"] > m_report["input_lufs"]   # got louder


def test_muted_stem_is_excluded():
    cfg = load_config()
    stems = fake_stems()
    plan = rules_advisor(stems, analyze_all(stems), cfg)
    plan["stems"]["harmonica"]["mute"] = True
    _, processed, report = mixdown(stems, plan, cfg, do_cleanup=False, verbose=False)
    assert "harmonica" not in [p.name for p in processed]
    assert any("harmonica" in w for w in report["warnings"])


def test_former_stubs_are_safe_to_call():
    """Smoke test: every function that used to be a # TODO stub runs and returns audio."""
    from mixpipe import pitch, timing, reverb
    from mixpipe.eq import deesser, dynamic_eq
    from mixpipe.dynamics import multiband_compress, transient_shaper
    from mixpipe.cleanup import declick, dehum

    s = sine()
    assert pitch.correct_pitch(s) is not None
    assert timing.align_stems(fake_stems()) is not None
    assert timing.estimate_offset_samples(s, s) == 0
    assert reverb.algorithmic_reverb(s) is not None
    assert deesser(s) is not None
    assert dynamic_eq(s, 1000.0, -20.0) is not None
    assert multiband_compress(s) is not None
    assert transient_shaper(s) is not None
    assert declick(s) is not None
    assert dehum(s) is not None


def test_detect_key():
    """No longer a stub: Krumhansl-Schmuckler over an STFT chroma."""
    from mixpipe import pitch

    assert pitch.detect_key(np.zeros(SAMPLE_RATE, dtype=np.float32)) is None
    # A minor triad + A bass, sustained
    a_minor = sine(220.0, 2.0, 0.3) + sine(261.63, 2.0, 0.2) + sine(329.63, 2.0, 0.2)
    assert pitch.detect_key(a_minor) in ("A:min", "C:maj")  # relative keys share a profile


if __name__ == "__main__":
    # Minimal runner so the suite works without pytest installed.
    fns = {k: v for k, v in sorted(globals().items()) if k.startswith("test_")}
    failed = []
    for name, fn in fns.items():
        try:
            fn()
            print(f"  pass  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
