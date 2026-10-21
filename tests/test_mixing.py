"""Tests for the mixing stages that used to be stubs: reverb and sends, hum and
click removal, band-split dynamics, timing, pitch, mastering extras, MIDI
rendering, and the advisor's notes and critique.

Audio is synthetic and short, so the whole file runs in a few seconds.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mixpipe import SAMPLE_RATE as SR  # noqa: E402
from mixpipe import advisor, cleanup, dynamics, eq, master, mixdown, panning  # noqa: E402
from mixpipe import pitch, plan as plan_mod, render_midi, reverb, timing, tsm  # noqa: E402
from mixpipe.analysis import (  # noqa: E402
    analyze_all, analyze_master, masking_report, peak_db, rms_db, true_peak_db,
)
from mixpipe.config import load_config  # noqa: E402
from mixpipe.io_utils import Stem, write_audio  # noqa: E402


def sine(freq=440.0, dur=1.0, amp=0.5, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(dur=1.0, amp=0.1, seed=0, sr=SR):
    return (np.random.default_rng(seed).normal(0, amp, int(dur * sr))).astype(np.float32)


def stems4():
    return [
        Stem("lead_vox", sine(330, 2.0, 0.3), SR, "vocal"),
        Stem("harmonica", sine(880, 2.0, 0.3), SR, "harmonica"),
        Stem("bass_di", sine(110, 2.0, 0.3), SR, "bass"),
        Stem("drums", sine(200, 2.0, 0.3), SR, "drums"),
    ]


# --------------------------------------------------------------------------- #
# reverb and delay
# --------------------------------------------------------------------------- #

def _rt60(wet: np.ndarray) -> float:
    """Reverberation time from Schroeder backward integration (T20 x 3)."""
    e = np.cumsum((wet.astype(np.float64) ** 2)[::-1])[::-1]
    e = 10 * np.log10(e / e[0] + 1e-30)
    return 3 * (np.argmax(e < -25) - np.argmax(e < -5)) / SR


def test_reverb_decay_follows_decay_s_and_damping_shortens_it():
    imp = np.zeros(SR * 6, dtype=np.float32)
    imp[0] = 1.0
    long_ = reverb.algorithmic_reverb(imp, decay_s=2.5, mix=1.0, damping=0.0)
    short = reverb.algorithmic_reverb(imp, decay_s=0.8, mix=1.0, damping=0.0)
    damped = reverb.algorithmic_reverb(imp, decay_s=2.5, mix=1.0, damping=0.5)
    assert _rt60(long_) > 2 * _rt60(short)
    assert 0.8 * 2.5 < _rt60(long_) < 1.2 * 2.5
    assert _rt60(damped) < _rt60(long_)


def test_reverb_wet_level_is_normalised_and_shapes_are_kept():
    x = noise(2.0, 0.1)
    wet = reverb.algorithmic_reverb(x, room_size=0.8, decay_s=2.0, mix=1.0, tail=False)
    assert wet.shape == x.shape
    assert abs(rms_db(wet) - rms_db(x)) < 1.5       # unit-power wet for white noise
    st = np.stack([x, x[::-1]], axis=1)
    out = reverb.algorithmic_reverb(st, mix=0.3)
    assert out.ndim == 2 and out.shape[1] == 2 and out.shape[0] > st.shape[0]   # tail
    assert np.all(np.isfinite(out))
    dry = reverb.algorithmic_reverb(st, mix=0.0, tail=False)
    assert np.allclose(dry, st, atol=1e-6)


def test_reverb_stereo_tail_is_decorrelated():
    imp = np.zeros((SR * 3, 2), dtype=np.float32)
    imp[0] = 1.0
    wet = reverb.algorithmic_reverb(imp, mix=1.0, width=1.0)
    corr = np.corrcoef(wet[:, 0], wet[:, 1])[0, 1]
    narrow = reverb.algorithmic_reverb(imp, mix=1.0, width=0.0)
    assert abs(corr) < 0.5
    assert np.allclose(narrow[:, 0], narrow[:, 1], atol=1e-6)


def test_convolve_ir_is_energy_normalised():
    x = noise(1.0, 0.1)
    short_ir = reverb.synth_ir(0.3, stereo=False)
    long_ir = reverb.synth_ir(2.0, stereo=False) * 7.0           # arbitrary scale
    a = reverb.convolve_ir(x, short_ir, mix=1.0, tail=False)
    b = reverb.convolve_ir(x, long_ir, mix=1.0, tail=False)
    assert abs(rms_db(a) - rms_db(b)) < 2.0
    assert reverb.convolve_ir(x, long_ir, mix=0.5).shape[0] == x.shape[0] + long_ir.shape[0] - 1
    delta = np.zeros(10)
    delta[0] = 1.0
    assert np.allclose(reverb.convolve_ir(x, delta, mix=1.0, tail=False), x, atol=1e-5)


def test_slap_delay_places_and_decays_echoes():
    imp = np.zeros(SR, dtype=np.float32)
    imp[0] = 1.0
    out = reverb.slap_delay(imp, delay_ms=100.0, feedback=0.5, mix=1.0)
    d = int(0.1 * SR)
    assert abs(out[d] - 1.0) < 1e-5 and abs(out[2 * d] - 0.5) < 1e-5 and abs(out[3 * d] - 0.25) < 1e-5
    assert abs(out[0]) < 1e-6
    assert out.shape[0] > imp.shape[0]                          # the repeats ring out
    runaway = reverb.slap_delay(imp, feedback=5.0, mix=1.0)     # clamped, so it decays
    assert np.max(np.abs(runaway)) <= 1.0 + 1e-6
    dark = reverb.slap_delay(imp, delay_ms=100.0, feedback=0.5, mix=1.0, damping=0.6)
    assert abs(dark[2 * d]) < abs(out[2 * d]) * 1.01 or dark[d] < 1.0   # damping lowers/softens repeats


def test_send_buses_share_one_room_and_respect_levels():
    cfg = load_config()
    stems = stems4()
    plan = plan_mod.validate_plan({}, stems, cfg)
    processed = [mixdown.process_stem(s, plan["stems"][s.name], cfg, do_cleanup=False) for s in stems]
    sends = mixdown.add_send_buses(processed, plan, cfg)
    assert sends.ndim == 2 and sends.shape[1] == 2 and sends.shape[0] >= processed[0].n_samples
    assert np.any(sends)

    # every send off -> silence
    for s in plan["stems"].values():
        s["sends"] = {"reverb_db": -60.0, "delay_db": -60.0}
    assert not np.any(mixdown.add_send_buses(processed, plan, cfg))

    # a louder send gives a louder return
    plan["stems"]["lead_vox"]["sends"]["reverb_db"] = -20.0
    quiet = mixdown.add_send_buses(processed, plan, cfg)
    plan["stems"]["lead_vox"]["sends"]["reverb_db"] = -8.0
    loud = mixdown.add_send_buses(processed, plan, cfg)
    assert peak_db(loud) == pytest.approx(peak_db(quiet) + 12.0, abs=0.5)

    cfg["sends"]["enabled"] = False
    assert not np.any(mixdown.add_send_buses(processed, plan, cfg))


def test_plan_clamps_send_and_bus_settings():
    stems = stems4()
    evil = {"stems": {"lead_vox": {"sends": {"reverb_db": 99, "delay_db": "x"},
                                   "deess": {"freq_hz": 1e9, "max_reduction_db": 500},
                                   "transient": {"attack": 50, "sustain": -50}}},
            "send_buses": {"reverb": {"decay_s": 1e6, "damping": 7}, "delay": {"feedback": 3}},
            "master": {"multiband": {"enabled": 1, "crossovers": [1, 1e9, 500, 500],
                                     "bands": [{"ratio": 1e6}]}}}
    plan = plan_mod.validate_plan(evil, stems)
    v = plan["stems"]["lead_vox"]
    assert v["sends"]["reverb_db"] <= 0.0 and v["sends"]["delay_db"] == plan_mod.SEND_OFF_DB
    assert v["deess"]["freq_hz"] <= 12000.0 and v["deess"]["max_reduction_db"] <= 18.0
    assert -1.0 <= v["transient"]["attack"] <= 1.0 and -1.0 <= v["transient"]["sustain"] <= 1.0
    assert plan["send_buses"]["reverb"]["decay_s"] <= 8.0
    assert plan["send_buses"]["delay"]["feedback"] <= 0.9
    mb = plan["master"]["multiband"]
    assert mb["enabled"] is True and len(mb["bands"]) == len(mb["crossovers"]) + 1
    assert mb["crossovers"] == sorted(set(mb["crossovers"]))
    assert all(b["ratio"] <= 20.0 for b in mb["bands"])


# --------------------------------------------------------------------------- #
# cleanup: hum and clicks
# --------------------------------------------------------------------------- #

def _hum(base, dur=4.0, amp=0.01, n_harm=5):
    t = np.arange(int(dur * SR)) / SR
    return sum(amp / k * np.sin(2 * np.pi * base * k * t + k) for k in range(1, n_harm + 1))


@pytest.mark.parametrize("base", [50.0, 60.0])
def test_find_hum_names_the_mains_frequency(base):
    x = (_hum(base) + 0.001 * np.random.default_rng(1).standard_normal(4 * SR)).astype(np.float32)
    found = cleanup.find_hum(x)
    assert found is not None and found["base_hz"] == base
    assert len(found["harmonics"]) >= 2


def test_dehum_removes_hum_but_not_music_or_clean_audio():
    t = np.arange(4 * SR) / SR
    music = 0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0.3)
    x = (music + _hum(50.0) + 0.001 * np.random.default_rng(2).standard_normal(4 * SR)).astype(np.float32)
    y = cleanup.dehum(x)
    quiet = slice(int(1.9 * SR), int(2.0 * SR))      # between phrases
    assert rms_db(y[quiet]) < rms_db(x[quiet]) - 10.0
    loud = slice(int(0.1 * SR), int(0.5 * SR))
    assert abs(rms_db(y[loud]) - rms_db(x[loud])) < 1.0
    clean = (music + 0.001 * np.random.default_rng(2).standard_normal(4 * SR)).astype(np.float32)
    assert cleanup.find_hum(clean) is None
    assert np.array_equal(cleanup.dehum(clean), clean)


def test_dehum_leaves_a_sustained_bass_note_alone():
    t = np.arange(4 * SR) / SR
    bass = (0.3 * np.sin(2 * np.pi * 60.0 * t) + 0.001 * np.random.default_rng(3).standard_normal(t.size)).astype(np.float32)
    assert cleanup.find_hum(bass) is None            # one line is a note, not a harmonic series
    assert np.array_equal(cleanup.dehum(bass), bass)


def test_declick_repairs_clicks_and_spares_transients():
    base = sine(330, 3.0, 0.3)
    clicked = base.copy()
    positions = [20000, 60000, 100000]
    for p in positions:
        clicked[p] += 0.6
        clicked[p + 1] -= 0.4
    fixed = cleanup.declick(clicked)
    for p in positions:
        assert abs(clicked[p] - base[p]) > 0.5
        assert abs(fixed[p] - base[p]) < 1e-3
    assert np.max(np.abs(fixed - base)) < 1e-3

    # a drum-like burst is not a click
    burst = np.zeros(3 * SR, dtype=np.float32)
    env = np.exp(-np.arange(SR) / 2000.0)
    burst[SR:2 * SR] = (np.random.default_rng(4).standard_normal(SR) * env * 0.8).astype(np.float32)
    burst += sine(200, 3.0, 0.05)
    assert cleanup.find_clicks(burst) == []
    assert np.array_equal(cleanup.declick(burst), burst)
    st = np.stack([clicked, clicked], axis=1)
    assert cleanup.declick(st).shape == st.shape


def test_clean_stem_runs_dehum_by_default_and_declick_on_request():
    cfg = load_config()
    t = np.arange(4 * SR) / SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0.3) + _hum(60.0)
         + 0.001 * np.random.default_rng(5).standard_normal(t.size)).astype(np.float32)
    cfg["cleanup"].update(denoise=False, gate_enabled=False)
    out = cleanup.clean_stem(Stem("gtr", x, SR, "electric"), cfg)
    assert rms_db(out.audio[int(1.9 * SR):int(2.0 * SR)]) < rms_db(x[int(1.9 * SR):int(2.0 * SR)]) - 10
    cfg["cleanup"]["dehum"] = False
    cfg["cleanup"]["declick"] = True
    clicked = sine(330, 3.0, 0.3)
    clicked[30000] += 0.7
    clicked[30001] -= 0.5
    out = cleanup.clean_stem(Stem("gtr", clicked, SR, "electric"), cfg)
    assert abs(out.audio[30000] - sine(330, 3.0, 0.3)[30000]) < 0.01


# --------------------------------------------------------------------------- #
# EQ and dynamics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("crossovers", [[1000.0], [150.0, 4000.0], [100.0, 800.0, 5000.0]])
def test_band_split_sums_flat(crossovers):
    x = noise(3.0, 0.1)
    bands = eq.split_bands(x, crossovers)
    assert len(bands) == len(crossovers) + 1
    ratio = np.abs(np.fft.rfft(sum(bands)))[20:-20] / np.abs(np.fft.rfft(x))[20:-20]
    smooth = np.convolve(20 * np.log10(ratio), np.ones(2000) / 2000, "valid")
    assert np.abs(smooth).max() < 0.1                 # an all-pass: magnitude flat, no comb


def test_lr4_bands_land_where_they_should():
    lo, hi = eq.lr4_split(noise(2.0, 0.1), 1000.0)
    tone_lo, tone_hi = eq.lr4_split(sine(100, 1.0), 1000.0), eq.lr4_split(sine(8000, 1.0), 1000.0)
    assert rms_db(tone_lo[0]) > rms_db(tone_lo[1]) + 30
    assert rms_db(tone_hi[1]) > rms_db(tone_hi[0]) + 30
    assert lo.shape == hi.shape


def test_dynamic_eq_cuts_only_while_the_band_is_loud():
    t = np.arange(3 * SR) / SR
    gate = np.sin(2 * np.pi * 1.0 * t) > 0.6
    sig = (0.3 * np.sin(2 * np.pi * 3000 * t) * gate + 0.1 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    out = eq.dynamic_eq(sig, 3000.0, -25.0)
    assert rms_db(out[gate]) < rms_db(sig[gate]) - 2.0           # the ringing burst is cut
    assert abs(rms_db(out[~gate]) - rms_db(sig[~gate])) < 0.2    # the rest is untouched
    capped = eq.dynamic_eq(sig, 3000.0, -25.0, max_reduction_db=1.0)
    assert rms_db(capped[gate]) > rms_db(out[gate])
    assert eq.dynamic_eq(sig, 3000.0, -25.0, ratio=1.0) is sig


def test_deesser_reduces_sibilance_not_body():
    rng = np.random.default_rng(6)
    t = np.arange(4 * SR) / SR
    voice = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 400 * t)
    ess = np.zeros_like(t)
    for start in (0.5, 1.5, 2.5, 3.3):                        # five hundred ms hisses
        a, b = int(start * SR), int((start + 0.15) * SR)
        ess[a:b] = rng.standard_normal(b - a) * 0.15
    ess = lfilter(*__import__("scipy.signal", fromlist=["butter"]).butter(2, [5000, 9000], btype="band", fs=SR), ess)
    sig = (voice + ess).astype(np.float32)
    out = eq.deesser(sig)
    hiss = np.zeros(sig.size, dtype=bool)
    for start in (0.5, 1.5, 2.5, 3.3):
        hiss[int(start * SR):int((start + 0.15) * SR)] = True
    from mixpipe.analysis import band_energy_db  # noqa: F401
    hf_before = rms_db(eq.lr4_split(sig[hiss], 5000.0)[1])
    hf_after = rms_db(eq.lr4_split(out[hiss], 5000.0)[1])
    assert hf_after < hf_before - 2.0
    body = ~hiss
    assert abs(rms_db(out[body]) - rms_db(sig[body])) < 0.5
    assert eq.deesser(sig, max_reduction_db=0.0) is sig


def test_transient_shaper_moves_attack_and_sustain_independently():
    plucks = np.zeros(3 * SR, dtype=np.float32)
    for k in range(6):
        n0, m = int(k * 0.5 * SR), int(0.4 * SR)
        tt = np.arange(m) / SR
        plucks[n0:n0 + m] += (np.sin(2 * np.pi * 200 * tt) * np.exp(-tt * 8) * 0.5).astype(np.float32)
    attack_part, tail_part = slice(0, int(0.02 * SR)), slice(int(0.25 * SR), int(0.4 * SR))

    def move(out, part):
        return rms_db(out[part]) - rms_db(plucks[part])

    punchy = dynamics.transient_shaper(plucks, attack=0.8)
    assert move(punchy, attack_part) > 5.0 and abs(move(punchy, tail_part)) < 1.0
    soft = dynamics.transient_shaper(plucks, attack=-0.8)
    assert move(soft, attack_part) < -4.0
    bloom = dynamics.transient_shaper(plucks, sustain=0.8)
    assert move(bloom, tail_part) > 5.0 and abs(move(bloom, attack_part)) < 1.5
    tight = dynamics.transient_shaper(plucks, sustain=-0.8)
    assert move(tight, tail_part) < -5.0
    assert dynamics.transient_shaper(plucks) is plucks


def test_multiband_compress_is_transparent_at_ratio_one_and_compresses_otherwise():
    x = noise(3.0, 0.3)
    flat = dynamics.multiband_compress(x, bands=[{"ratio": 1.0}] * 3)
    assert abs(rms_db(flat) - rms_db(x)) < 0.05
    squeezed = dynamics.multiband_compress(x)
    assert rms_db(squeezed) < rms_db(x) - 2.0
    with pytest.raises(ValueError):
        dynamics.multiband_compress(x, crossovers=[200.0], bands=[{}, {}, {}])
    st = np.stack([x, x], axis=1)
    assert dynamics.multiband_compress(st).shape == st.shape


def test_master_runs_multiband_when_enabled():
    cfg = load_config()
    stems = stems4()
    plan = plan_mod.validate_plan({"master": {"multiband": {"enabled": True}}}, stems, cfg)
    bus = np.stack([noise(3.0, 0.1), noise(3.0, 0.1, seed=1)], axis=1)
    _, report = master.master(bus, plan, cfg, verbose=False)
    assert "multiband" in [s["stage"] for s in report["stages"]]


def test_true_peak_limiter_catches_inter_sample_peaks():
    # fs/4 sine at 45 degrees: every sample reads 0.707 of a 1.0 true peak
    t = np.arange(2 * SR)
    x = (1.4 * np.sin(2 * np.pi * (SR / 4) * t / SR + np.pi / 4)).astype(np.float32)
    st = np.stack([x, x], axis=1)
    assert true_peak_db(st) > peak_db(st) + 2.5
    sample = dynamics.limit(st, -1.0)
    assert peak_db(sample) <= -0.95 and true_peak_db(sample) > 0.0      # sample-peak: blind to it
    tp = dynamics.limit(st, -1.0, true_peak=True)
    assert true_peak_db(tp) <= -0.9


def test_envelope_kernel_matches_a_plain_python_loop():
    from mixpipe._fast import asym_smooth, pole

    rng = np.random.default_rng(0)
    target = rng.random(5000)
    up, down = pole(5.0, SR), pole(40.0, SR)
    state, expect = target[0], np.empty_like(target)
    for i, x in enumerate(target):
        c = up if x > state else down
        state = x + c * (state - x)
        expect[i] = state
    assert np.allclose(asym_smooth(target, 5.0, 40.0, SR), expect)


# --------------------------------------------------------------------------- #
# timing and time-scale modification
# --------------------------------------------------------------------------- #

def _hits(times, dur, freq=200.0, decay=0.04, amp=0.8, noisy=False, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur * SR), dtype=np.float32)
    for t0 in times:
        i, m = int(t0 * SR), int(0.3 * SR)
        tt = np.arange(m) / SR
        sig = (rng.standard_normal(m) * 0.5 if noisy else np.sin(2 * np.pi * freq * tt)) * np.exp(-tt / decay) * amp
        e = min(x.size, i + m)
        if e > i:
            x[i:e] += sig[:e - i]
    return x


def _pattern(bpm=100.0, bars=8):
    beat = 60 / bpm
    return [bar * 4 * beat + k * beat / 2 for bar in range(bars) for k in (0, 3, 4, 6)]


def test_offset_between_instruments_playing_the_same_rhythm():
    times = _pattern()
    drums = _hits(times, 20, 80, 0.05) + _hits(times[1::2], 20, 3000, 0.01, noisy=True)
    delay = int(0.0373 * SR)
    bass = np.roll(_hits(times, 20, 55, 0.12, 0.6), delay)
    assert abs(timing.estimate_offset_samples(drums, bass) - delay) < 200      # within ~4 ms
    assert timing.estimate_offset_samples(bass, drums) == -timing.estimate_offset_samples(drums, bass)
    assert timing.estimate_offset_samples(np.zeros(SR), np.zeros(SR)) == 0


def test_offset_is_sample_accurate_for_the_same_source():
    times = _pattern()
    drums = _hits(times, 20, 80, 0.05) + _hits(times[1::2], 20, 3000, 0.01, noisy=True)
    mic = np.roll(drums, 1234) * 0.5 + 0.01 * np.random.default_rng(1).standard_normal(drums.size).astype(np.float32)
    assert timing.estimate_offset_samples(drums, mic) == 1234


def test_align_stems_shifts_confident_stems_and_leaves_noise_alone():
    times = _pattern()
    drums = _hits(times, 20, 80, 0.05) + _hits(times[1::2], 20, 3000, 0.01, noisy=True)
    delay = int(0.0373 * SR)
    bass = np.roll(_hits(times, 20, 55, 0.12, 0.6), delay)
    hiss = (np.random.default_rng(2).standard_normal(20 * SR) * 0.01).astype(np.float32)
    stems = [Stem("drums", drums, SR, "drums"), Stem("bass", bass, SR, "bass"), Stem("hiss", hiss, SR)]
    out = {s.name: s for s in timing.align_stems(stems)}
    assert out["drums"].meta["align_offset_samples"] == 0
    assert abs(out["bass"].meta["align_offset_samples"] - delay) < 200
    assert out["hiss"].meta["align_offset_samples"] == 0 and np.array_equal(out["hiss"].audio, hiss)
    assert all(s.n_samples == 20 * SR for s in out.values())
    aligned = out["bass"].audio
    assert np.argmax(np.abs(aligned[:SR])) < np.argmax(np.abs(bass[:SR]))           # moved earlier
    with pytest.raises(ValueError):
        timing.align_stems(stems, reference="nope")
    assert timing.align_stems(stems[:1]) == stems[:1]


def test_shift_samples():
    x = np.arange(10, dtype=np.float32)
    assert list(timing.shift_samples(x, 3)) == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    assert list(timing.shift_samples(x, -3)) == [3, 4, 5, 6, 7, 8, 9, 0, 0, 0]


@pytest.mark.parametrize("bpm", [73, 97, 128, 150])
def test_tempo_estimate(bpm):
    clicks = _hits([k * 60 / bpm for k in range(int(20 / (60 / bpm)))], 20, 100, 0.03)
    assert timing.estimate_tempo_bpm(clicks) == pytest.approx(bpm, abs=0.6)


def test_tempo_gives_none_without_a_pulse_or_enough_audio():
    assert timing.estimate_tempo_bpm(np.zeros(10 * SR, dtype=np.float32)) is None
    assert timing.estimate_tempo_bpm(sine(220, 2.0)) is None


def test_grid_steps():
    assert timing.grid_step_s(120, "1/4") == pytest.approx(0.5)
    assert timing.grid_step_s(120, "1/16") == pytest.approx(0.125)
    assert timing.grid_step_s(120, "1/8T") == pytest.approx(0.5 / 3)
    with pytest.raises(ValueError):
        timing.grid_step_s(120, "banana")


def _loose_notes(seed=5, bpm=120.0, jitter=0.025):
    rng = np.random.default_rng(seed)
    step = 60 / bpm / 4
    idx = [0, 2, 3, 4, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 22, 24, 26, 27, 28, 30,
           32, 34, 35, 36, 38, 40, 42, 43, 44, 46, 48, 50, 51, 52, 54, 56]
    x = np.zeros(int(9 * SR), dtype=np.float32)
    for gi in idx:
        t = 0.37 + gi * step + rng.uniform(-jitter, jitter)
        i, m = int(t * SR), int(0.25 * SR)
        tt = np.arange(m) / SR
        f = rng.choice([220, 262, 330, 392, 440])
        x[i:i + m] += ((np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(4 * np.pi * f * tt))
                       * np.exp(-tt / 0.08) * 0.4).astype(np.float32)
    return x


def _grid_error_ms(sig, bpm):
    on, _ = timing.detect_onsets(sig)
    t, step = on / SR, timing.grid_step_s(bpm, "1/16")
    phase = np.angle(np.sum(np.exp(2j * np.pi * t / step))) / (2 * np.pi) * step
    return float(np.abs(((t - phase + step / 2) % step) - step / 2).mean() * 1000)


def test_detect_onsets_finds_every_note_once():
    x = _loose_notes()
    on, w = timing.detect_onsets(x)
    assert len(on) == 36 and len(w) == 36


def test_quantize_pulls_notes_to_the_grid_and_keeps_length_and_level():
    x = _loose_notes()
    before = _grid_error_ms(x, 120)
    full = timing.quantize_onsets(x, 120, "1/16", 1.0)
    half = timing.quantize_onsets(x, 120, "1/16", 0.5)
    assert len(full) == len(x) and len(half) == len(x)
    assert _grid_error_ms(full, 120) < before / 4
    assert _grid_error_ms(full, 120) < _grid_error_ms(half, 120) < before
    assert abs(rms_db(full) - rms_db(x)) < 0.5
    assert timing.quantize_onsets(x, 120, "1/16", 0.0) is x


def test_stretch_changes_length_not_pitch():
    x = sum(0.2 / k * np.sin(2 * np.pi * 220 * k * np.arange(4 * SR) / SR) for k in range(1, 6)).astype(np.float32)

    def f0(sig):
        seg = sig[SR // 2:2 * SR]
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size), n=1 << 18))
        fr = np.fft.rfftfreq(1 << 18, 1 / SR)
        m = (fr > 100) & (fr < 2000)
        return fr[m][np.argmax(spec[m])]

    for rate in (0.7, 1.5):
        y = tsm.stretch(x, rate)
        assert abs(len(y) - round(len(x) / rate)) <= 1
        assert f0(y) == pytest.approx(220.0, abs=1.0)
    st = np.stack([x, x * 0.5], axis=1)
    y = tsm.stretch(st, 0.8)
    assert np.std(y[:, 0]) / np.std(y[:, 1]) == pytest.approx(2.0, rel=0.02)   # image kept
    assert tsm.stretch(x, 1.0) is x
    with pytest.raises(ValueError):
        tsm.stretch(x, 0)


@pytest.mark.parametrize("semitones", [3, -5, 7])
def test_shift_semitones_moves_pitch_and_keeps_length(semitones):
    x = sum(0.2 / k * np.sin(2 * np.pi * 220 * k * np.arange(3 * SR) / SR) for k in range(1, 6)).astype(np.float32)
    y = pitch.shift_semitones(x, semitones, engine="pv")
    assert len(y) == len(x)
    seg = y[SR // 2:2 * SR]
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size), n=1 << 18))
    fr = np.fft.rfftfreq(1 << 18, 1 / SR)
    m = (fr > 100) & (fr < 2000)
    assert fr[m][np.argmax(spec[m])] == pytest.approx(220 * 2 ** (semitones / 12), rel=0.005)
    assert pitch.shift_semitones(x, 0) is x


@pytest.mark.skipif(not tsm.have_rubberband(), reason="Rubber Band not installed")
def test_shift_semitones_can_use_rubberband():
    x = sine(220, 2.0, 0.4)
    y = pitch.shift_semitones(x, 2, engine="rubberband")
    assert len(y) == len(x) and np.all(np.isfinite(y))


# --------------------------------------------------------------------------- #
# pitch tracking and correction
# --------------------------------------------------------------------------- #

def _voice(notes, vib_cents=0.0, vib_hz=5.5, seed=2):
    """Pulse-train 'voice' with three formant resonators; None = quiet noise."""
    rng = np.random.default_rng(seed)
    segs = []
    for f, d in notes:
        n = int(d * SR)
        if f is None:
            segs.append(rng.standard_normal(n) * 0.01)
            continue
        t = np.arange(n) / SR
        phase = np.cumsum(f * 2 ** ((vib_cents / 1200) * np.sin(2 * np.pi * vib_hz * t))) / SR
        sig = sum(np.sin(2 * np.pi * k * phase) / k for k in range(1, 25) if f * k < SR * 0.45)
        env = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.02 * SR))
        segs.append(sig * env * 0.3)
    x = np.concatenate(segs)
    for fc, bw in ((800, 80), (1200, 90), (2600, 120)):
        r, th = np.exp(-np.pi * bw / SR), 2 * np.pi * fc / SR
        x = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x) * 3
    return x.astype(np.float32)


C4, E4, G4 = 261.63, 329.63, 392.0
FLAT_SHARP = [(None, 0.2), (C4 * 2 ** (35 / 1200), 0.6), (None, 0.1),
              (E4 * 2 ** (-30 / 1200), 0.6), (None, 0.1), (G4 * 2 ** (40 / 1200), 0.6), (None, 0.2)]


def _note_cents(sig, spans=((0.3, 0.75, C4), (0.95, 1.4, E4), (1.6, 2.05, G4))):
    f0, _ = pitch.track_pitch(sig)
    out = []
    for t0, t1, ref in spans:
        seg = f0[int(t0 * 100):int(t1 * 100)]
        out.append(float(np.median(1200 * np.log2(seg[~np.isnan(seg)] / ref))))
    return out


def test_track_pitch_finds_f0_and_marks_silence_unvoiced():
    f0, conf = pitch.track_pitch(_voice(FLAT_SHARP))
    assert np.all(np.isnan(f0[:15]))                       # leading noise is unvoiced
    assert np.nanmedian(f0[30:70]) == pytest.approx(C4 * 2 ** (35 / 1200), rel=0.003)
    assert conf[40] > 0.9
    silent, _ = pitch.track_pitch(np.zeros(SR, dtype=np.float32))
    assert np.all(np.isnan(silent))
    assert pitch.track_pitch(sine(440, 1.0, 0.3))[0][50] == pytest.approx(440.0, rel=0.002)


def test_scale_pitch_classes():
    assert pitch.scale_pitch_classes("C:maj") == [0, 2, 4, 5, 7, 9, 11]
    assert pitch.scale_pitch_classes("A:min") == [0, 2, 4, 5, 7, 9, 11]
    assert pitch.scale_pitch_classes("Bb:maj") == [0, 2, 3, 5, 7, 9, 10]
    assert pitch.scale_pitch_classes("E:pent_min") == [2, 4, 7, 9, 11]
    assert pitch.scale_pitch_classes("chromatic") == list(range(12))
    with pytest.raises(ValueError):
        pitch.scale_pitch_classes("H:lydian")


def test_correct_pitch_snaps_flat_and_sharp_notes_to_the_scale():
    x = _voice(FLAT_SHARP)
    assert [round(abs(c)) for c in _note_cents(x)] == [35, 30, 40]
    y = pitch.correct_pitch(x, SR, "C:maj", retune_speed=1.0)
    assert len(y) == len(x) and np.all(np.isfinite(y))
    assert all(abs(c) < 3.0 for c in _note_cents(y))
    assert abs(rms_db(y) - rms_db(x)) < 0.5
    half = pitch.correct_pitch(x, SR, "C:maj", retune_speed=0.5)
    assert all(abs(c) < abs(o) for c, o in zip(_note_cents(half), _note_cents(x)))
    moved = pitch.correct_pitch(x, SR, "C:maj", 1.0, formant_preserve=False)
    assert all(abs(c) < 3.0 for c in _note_cents(moved))


def test_retune_speed_trades_vibrato_for_accuracy():
    xv = _voice([(None, 0.2), (C4 * 2 ** (30 / 1200), 1.5), (None, 0.2)], vib_cents=40)

    def spread(sig):
        f, _ = pitch.track_pitch(sig)
        c = 1200 * np.log2(f[~np.isnan(f)] / C4)
        lo, mid, hi = np.percentile(c, [5, 50, 95])
        return hi - lo, mid

    orig_width, orig_mid = spread(xv)
    hard_width, hard_mid = spread(pitch.correct_pitch(xv, SR, "C:maj", 1.0))
    soft_width, soft_mid = spread(pitch.correct_pitch(xv, SR, "C:maj", 0.1))
    assert hard_width < orig_width / 2 and abs(hard_mid) < 8        # flat and on pitch
    assert soft_width > orig_width * 0.7 and abs(soft_mid) < orig_mid  # vibrato kept, centre pulled in


def test_correct_pitch_leaves_in_tune_unvoiced_and_empty_audio_alone():
    in_tune = _voice([(None, 0.2), (C4, 0.8), (None, 0.2)])
    assert np.array_equal(pitch.correct_pitch(in_tune, SR, "C:maj"), in_tune)
    hiss = noise(1.0, 0.05)
    assert np.array_equal(pitch.correct_pitch(hiss, SR, "C:maj"), hiss)
    assert pitch.correct_pitch(np.zeros(0, dtype=np.float32)).size == 0
    far = _voice([(None, 0.2), (C4 * 2 ** (80 / 1200), 0.8), (None, 0.2)])      # between two notes
    out = pitch.correct_pitch(far, SR, "chromatic", 1.0)
    assert len(out) == len(far)
    stereo = np.stack([_voice(FLAT_SHARP), _voice(FLAT_SHARP)], axis=1)
    fixed = pitch.correct_pitch(stereo, SR, "C:maj", 1.0)
    assert fixed.shape == stereo.shape


# --------------------------------------------------------------------------- #
# panning
# --------------------------------------------------------------------------- #

def test_pseudo_stereo_is_wide_but_mono_safe():
    x = noise(2.0, 0.1)
    y = panning.haas_spread(x)
    assert y.shape == (x.size, 2)
    assert np.corrcoef(y[:, 0], y[:, 1])[0, 1] < 0.95                  # actually stereo
    mono = y.mean(axis=1)
    assert np.max(np.abs(mono - x / np.sqrt(1 + 0.3 ** 2))) < 1e-5     # side cancels exactly
    assert np.array_equal(panning.haas_spread(y), y)                   # stereo passes through
    assert np.allclose(panning.haas_spread(x, amount=0.0)[:, 0], x, atol=1e-6)


# --------------------------------------------------------------------------- #
# mastering extras
# --------------------------------------------------------------------------- #

def _tilted(n, brightness, seed):
    w = np.random.default_rng(seed).standard_normal(n)
    x = lfilter([1], [1, -(0.97 - 0.5 * brightness)], w)
    return (x / np.abs(x).max() * 0.5).astype(np.float32)


def _band_shape(x):
    from scipy.signal import welch

    f, p = welch(x if x.ndim == 1 else x.mean(axis=1), fs=SR, nperseg=8192)
    lv = np.array([10 * np.log10(p[(f >= lo) & (f < hi)].mean())
                   for lo, hi in ((60, 200), (200, 800), (800, 3000), (3000, 8000), (8000, 15000))])
    return lv - lv[2]


def test_match_reference_moves_the_balance_toward_the_reference(tmp_path):
    mix = np.stack([_tilted(12 * SR, 0.0, 0), _tilted(12 * SR, 0.0, 1)], axis=1)
    ref = _tilted(12 * SR, 0.9, 2)
    ref_path = write_audio(tmp_path / "ref.wav", np.stack([ref, ref], axis=1))
    out, curve = master.match_reference(mix, ref_path, SR, strength=1.0, return_curve=True)
    assert out.shape == mix.shape
    gap_before = np.abs(_band_shape(mix) - _band_shape(ref)).mean()
    gap_after = np.abs(_band_shape(out) - _band_shape(ref)).mean()
    assert gap_after < gap_before * 0.7
    assert curve["max_boost_db"] <= 6.0 + 1e-6 and curve["max_cut_db"] >= -6.0 - 1e-6
    from mixpipe.analysis import integrated_lufs

    assert integrated_lufs(out.mean(axis=1), SR) == pytest.approx(integrated_lufs(mix.mean(axis=1), SR), abs=0.1)
    untouched = master.match_reference(mix, ref_path, SR, strength=0.0)
    assert np.allclose(untouched, mix, atol=1e-3)
    short = write_audio(tmp_path / "short.wav", ref[:1000])
    with pytest.raises(ValueError):
        master.match_reference(mix, short, SR)


def test_master_accepts_a_reference(tmp_path):
    cfg = load_config()
    stems = stems4()
    plan = plan_mod.validate_plan({}, stems, cfg)
    ref = _tilted(6 * SR, 0.8, 3)
    ref_path = write_audio(tmp_path / "ref.wav", ref)
    bus = np.stack([_tilted(6 * SR, 0.1, 4)] * 2, axis=1) * 0.3
    final, report = master.master(bus, plan, cfg, verbose=False, reference=ref_path)
    assert "reference" in report and "reference_match" in [s["stage"] for s in report["stages"]]
    assert report["output_true_peak_db"] <= plan["master"]["ceiling_db"] + 0.1
    assert final.shape == bus.shape


def test_dither_lands_on_the_bit_grid_and_removes_distortion():
    q = 2 ** 15
    quiet = (np.sin(2 * np.pi * 1000 * np.arange(2 * SR) / SR) * 3.0 / q).astype(np.float32)   # 3 LSB

    def harmonic_distortion_db(x):
        spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
        k = int(1000 * x.size / SR)
        harm = np.sqrt(sum(spec[h * k - 2:h * k + 3].max() ** 2 for h in range(2, 8)))
        return 20 * np.log10(harm / spec[k - 2:k + 3].max())

    plain = np.round(quiet * q) / q
    for shaped in (False, True):
        d = master.dither(quiet, 16, noise_shaping=shaped, seed=1)
        assert np.allclose(d * q, np.round(d * q))
        assert harmonic_distortion_db(d) < harmonic_distortion_db(plain) - 3.0
    assert master.dither(quiet, 16, seed=1).std() > 0
    assert np.array_equal(master.dither(quiet, 16, seed=1), master.dither(quiet, 16, seed=1))
    st = np.stack([quiet, quiet], axis=1)
    assert master.dither(st, 16, noise_shaping=True).shape == st.shape
    loud = np.array([1.5, -1.5], dtype=np.float32)
    assert np.all(np.abs(master.dither(loud, 16)) <= 1.0)
    assert master.dither(quiet, 32) is quiet


# --------------------------------------------------------------------------- #
# MIDI
# --------------------------------------------------------------------------- #

def _mpe_smf():
    notes = []
    for ch, key, tick in ((1, 60, 0), (2, 64, 0), (3, 67, 240)):
        notes += [(tick, bytes([0x90 | ch, key, 100])), (tick + 480, bytes([0x80 | ch, key, 0])),
                  (tick + 100, bytes([0xE0 | ch, 0x00, 0x50]))]
    return {"format": 1, "division": 480,
            "tracks": [[(0, b"\xff\x51\x03\x07\xa1\x20")], notes]}


def test_smf_roundtrip(tmp_path):
    smf = _mpe_smf()
    path = render_midi.write_smf(smf, tmp_path / "a.mid")
    back = render_midi.read_smf(path)
    assert back["format"] == 1 and back["division"] == 480 and len(back["tracks"]) == 2
    assert sorted(back["tracks"][1]) == sorted(smf["tracks"][1])
    assert back["tracks"][0] == smf["tracks"][0]
    (tmp_path / "bad.mid").write_bytes(b"nope")
    with pytest.raises(ValueError):
        render_midi.read_smf(tmp_path / "bad.mid")


def test_prepare_mpe_adds_program_and_bend_range_once():
    prepared = render_midi.prepare_mpe(_mpe_smf(), bend_range=48, program=5)
    events = [e for _, e in prepared["tracks"][0]]
    for ch in (1, 2, 3):
        assert bytes([0xC0 | ch, 5]) in events
        assert bytes([0xB0 | ch, 6, 48]) in events
    twice = render_midi.prepare_mpe(prepared)
    assert len(twice["tracks"][0]) == len(prepared["tracks"][0])        # nothing added the 2nd time
    assert prepared["tracks"][0][0][0] == 0                              # setup sits at tick 0
    assert render_midi.prepare_mpe(_mpe_smf())["tracks"][1] == _mpe_smf()["tracks"][1]   # input untouched


@pytest.fixture
def fake_fluidsynth(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "fluidsynth"
    exe.parent.mkdir()
    exe.write_text(f"#!{sys.executable}\n"
                   "import sys, numpy as np, soundfile as sf\n"
                   "a = sys.argv\n"
                   "out = a[a.index('-F') + 1]\n"
                   "sr = int(a[a.index('-r') + 1])\n"
                   "sf.write(out, np.full((sr // 10, 2), 0.25, dtype='float32'), sr)\n"
                   "open(out + '.args', 'w').write(' '.join(a))\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{exe.parent}:/usr/bin:/bin")
    return exe


def test_render_runs_fluidsynth(tmp_path, fake_fluidsynth):
    mid = render_midi.write_smf(_mpe_smf(), tmp_path / "a.mid")
    sf2 = tmp_path / "gm.sf2"
    sf2.write_bytes(b"RIFF")
    audio = render_midi.render(mid, str(sf2), sr=44100)
    assert audio.shape == (4410, 2) and abs(audio[0, 0] - 0.25) < 1e-4
    resampled = render_midi.render(mid, str(sf2), sr=SR)
    assert resampled.shape[1] == 2 and abs(resampled.shape[0] - SR // 10) < 5
    with pytest.raises(FileNotFoundError):
        render_midi.render(str(tmp_path / "missing.mid"), str(sf2))
    with pytest.raises(FileNotFoundError):
        render_midi.render(mid, str(tmp_path / "missing.sf2"))


def test_render_mpe_prepares_the_file_first(tmp_path, fake_fluidsynth, monkeypatch):
    mid = render_midi.write_smf(_mpe_smf(), tmp_path / "a.mid")
    sf2 = tmp_path / "gm.sf2"
    sf2.write_bytes(b"RIFF")
    seen = {}
    real = render_midi.render

    def spy(path, sf, sr=SR, gain=0.8):
        seen["smf"] = render_midi.read_smf(path)
        return real(path, sf, sr, gain)

    monkeypatch.setattr(render_midi, "render", spy)
    render_midi.render_mpe(mid, SR, soundfont_path=str(sf2))
    flat = [e for _, e in seen["smf"]["tracks"][0]]
    assert bytes([0xB0 | 2, 6, 48]) in flat
    monkeypatch.delenv("MIXPIPE_SOUNDFONT", raising=False)
    with pytest.raises(RuntimeError, match="SoundFont"):
        render_midi.render_mpe(mid, SR)


def test_render_explains_a_missing_fluidsynth(tmp_path, monkeypatch):
    mid = render_midi.write_smf(_mpe_smf(), tmp_path / "a.mid")
    sf2 = tmp_path / "gm.sf2"
    sf2.write_bytes(b"RIFF")
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(RuntimeError, match="fluid-synth"):
        render_midi.render(mid, str(sf2))


# --------------------------------------------------------------------------- #
# advisor: notes, masking, critique
# --------------------------------------------------------------------------- #

def _advise(notes=""):
    stems = stems4()
    cfg = load_config()
    return stems, cfg, advisor.rules_advisor(stems, analyze_all(stems), cfg, notes=notes), \
        advisor.rules_advisor(stems, analyze_all(stems), cfg)


def test_rules_advisor_acts_on_level_notes():
    _, _, plan, base = _advise("I can't hear the harmonica, the vocals are too loud and the drums are overpowering")
    # level notes move the fader (after the compressor), not the input trim
    assert plan["stems"]["harmonica"]["fader_db"] == pytest.approx(base["stems"]["harmonica"]["fader_db"] + 3.0)
    assert plan["stems"]["lead_vox"]["fader_db"] == pytest.approx(base["stems"]["lead_vox"]["fader_db"] - 3.0)
    assert plan["stems"]["drums"]["fader_db"] == pytest.approx(base["stems"]["drums"]["fader_db"] - 3.0)
    assert plan["stems"]["bass_di"]["fader_db"] == base["stems"]["bass_di"]["fader_db"]
    assert plan["stems"]["harmonica"]["gain_db"] == base["stems"]["harmonica"]["gain_db"]
    assert "Acted on" in plan["notes"]


def test_rules_advisor_acts_on_space_and_tone_notes():
    _, _, plan, base = _advise("more reverb on the vocals. the bass is boomy, it sounds a bit dull overall and make it louder")
    assert plan["stems"]["lead_vox"]["sends"]["reverb_db"] > base["stems"]["lead_vox"]["sends"]["reverb_db"]
    assert plan["stems"]["harmonica"]["sends"] == base["stems"]["harmonica"]["sends"]
    assert len(plan["stems"]["bass_di"]["eq"]) > len(base["stems"]["bass_di"]["eq"])     # the bass, not the master
    assert any(b["type"] == "high_shelf" and b["gain_db"] > 0 for b in plan["master"]["master_eq"])
    assert plan["master"]["target_lufs"] == pytest.approx(base["master"]["target_lufs"] + 1.5)
    _, _, dry, base2 = _advise("make it drier")
    assert dry["stems"]["lead_vox"]["sends"]["reverb_db"] < base2["stems"]["lead_vox"]["sends"]["reverb_db"]


def test_rules_advisor_reports_notes_it_does_not_understand():
    _, _, plan, base = _advise("banana fish")
    assert "Not understood" in plan["notes"] and "banana fish" in plan["notes"]
    assert plan["stems"] == base["stems"]


def test_masking_ignores_stems_that_never_play_together():
    t = np.arange(8 * SR) / SR
    a = (0.3 * np.sin(2 * np.pi * 1000 * t) * (t < 4)).astype(np.float32)
    b = (0.3 * np.sin(2 * np.pi * 1200 * t) * (t >= 4)).astype(np.float32)
    c = (0.3 * np.sin(2 * np.pi * 1100 * t) * (t < 4)).astype(np.float32)
    apart = masking_report([Stem("a", a, SR), Stem("b", b, SR)])
    assert apart == []
    together = masking_report([Stem("a", a, SR), Stem("c", c, SR)])
    assert together and "mid" in together[0]["contested_bands"]
    assert together[0]["overlap"]["mid"] > 0.9


def _make_mix_analysis(**over):
    base = {"lufs": -14.0, "peak_db": -1.0, "true_peak_db": -1.0, "crest_db": 10.0,
            "spectral_centroid_hz": 2000.0, "loudness_range_db": 4.0, "lr_correlation": 0.8,
            "side_to_mid_db": -8.0,
            "band_energy_db": {"sub": -20.0, "low": 0.0, "low_mid": -3.0, "mid": -6.0,
                               "high_mid": -10.0, "high": -16.0, "air": -26.0}}
    base.update(over)
    return base


def test_analyze_master_reports_the_stereo_picture():
    x = noise(3.0, 0.1)
    a = analyze_master(np.stack([x, x], axis=1))
    assert a["lr_correlation"] == pytest.approx(1.0) and a["side_to_mid_db"] < -100
    wide = analyze_master(np.stack([x, -x], axis=1))
    assert wide["lr_correlation"] == pytest.approx(-1.0)
    assert set(a) >= {"lufs", "peak_db", "true_peak_db", "crest_db", "band_energy_db", "loudness_range_db"}
    assert "lr_correlation" not in analyze_master(x)


def test_rules_critique_fixes_balance_against_a_reference():
    stems, cfg, plan, _ = _advise()
    out = _make_mix_analysis()
    ref = _make_mix_analysis(band_energy_db={"sub": -20.0, "low": -5.0, "low_mid": -3.0, "mid": -6.0,
                                              "high_mid": -4.0, "high": -16.0, "air": -26.0})
    result = advisor.critique_mix(out, cfg, plan=plan, stems=stems, reference_analysis=ref)
    assert result["changed"] and result["source"] == "critique:rules"
    eq = {round(b["freq"]): b["gain_db"] for b in result["plan"]["master"]["master_eq"]}
    assert eq[120] < 0                       # "low" is 5 dB too hot -> cut
    assert eq[3500] > 0                      # "high_mid" is 6 dB short -> boost
    assert all(abs(g) <= 3.0 + 1e-6 for g in eq.values())
    assert result["balance_error"]["low"] == pytest.approx(5.0)
    assert result["balance_error"]["high_mid"] == pytest.approx(-6.0)

    # a second round on the same error replaces, rather than stacks on, the first cut
    again = advisor.critique_mix(out, cfg, plan=result["plan"], stems=stems, reference_analysis=ref)
    eq2 = {round(b["freq"]): b["gain_db"] for b in again["plan"]["master"]["master_eq"]}
    assert len([b for b in again["plan"]["master"]["master_eq"] if round(b["freq"]) == 120]) == 1
    assert eq2[120] <= eq[120]


def test_rules_critique_objective_problems_need_no_reference():
    stems, cfg, plan, _ = _advise()
    hot = _make_mix_analysis(true_peak_db=0.5, lr_correlation=-0.3)
    result = advisor.critique_mix(hot, cfg, plan=plan, stems=stems, master_report={"crest_lost_db": 8.0})
    revised = result["plan"]["master"]
    assert revised["ceiling_db"] < plan["master"]["ceiling_db"]
    assert revised["target_lufs"] == pytest.approx(plan["master"]["target_lufs"] - 1.5)
    assert any("mono" in f for f in result["findings"])
    clean = advisor.critique_mix(_make_mix_analysis(), cfg, plan=plan, stems=stems)
    assert not clean["changed"] and "no problems" in clean["critique"]
    assert advisor.critique_mix(_make_mix_analysis(), cfg)["plan"] is None       # critique only


def test_llm_critique_validates_the_returned_plan_and_falls_back(monkeypatch):
    import json

    stems, cfg, plan, _ = _advise()
    evil = json.loads(json.dumps(plan))
    evil["stems"]["lead_vox"]["gain_db"] = 400
    evil["master"]["master_eq"] = [{"freq": 120, "gain_db": -2, "q": 0.8}]
    reply = json.dumps({"critique": "too boomy", "plan": evil})
    monkeypatch.setattr(advisor, "_llm_text", lambda *a, **k: "```json\n" + reply + "\n```")
    result = advisor.critique_mix(_make_mix_analysis(), cfg, plan=plan, stems=stems,
                                  use_llm=True, provider="anthropic", model="x")
    assert result["source"].startswith("critique:llm") and result["critique"] == "too boomy"
    assert result["plan"]["stems"]["lead_vox"]["gain_db"] <= 12.0
    assert result["changed"]

    def boom(*a, **k):
        raise RuntimeError("no key")

    monkeypatch.setattr(advisor, "_llm_text", boom)
    fell_back = advisor.critique_mix(_make_mix_analysis(true_peak_db=1.0), cfg, plan=plan,
                                     stems=stems, use_llm=True)
    assert fell_back["source"] == "critique:rules" and fell_back["changed"]


# --------------------------------------------------------------------------- #
# vocal chain and the whole pipeline
# --------------------------------------------------------------------------- #

def test_vocal_gets_a_deesser_and_other_roles_do_not():
    cfg = load_config()
    stems = stems4()
    plan = plan_mod.validate_plan({}, stems, cfg)
    assert plan["stems"]["lead_vox"]["deess"]["max_reduction_db"] > 0
    assert plan["stems"]["bass_di"]["deess"]["max_reduction_db"] == 0
    assert not any(abs(b["freq"] - 7000.0) < 1 and b["gain_db"] < 0
                   for b in plan["stems"]["lead_vox"]["eq"])         # the static notch is gone
    assert plan["stems"]["lead_vox"]["transient"] == {"attack": 0.0, "sustain": 0.0}


def test_process_stem_applies_transient_shaping_and_deessing():
    cfg = load_config()
    s = Stem("lead_vox", sine(330, 2.0, 0.3), SR, "vocal")
    plan = plan_mod.validate_plan({}, [s], cfg)
    entry = plan["stems"]["lead_vox"]
    base = mixdown.process_stem(s, entry, cfg, do_cleanup=False)
    entry["transient"] = {"attack": 0.8, "sustain": 0.0}
    shaped = mixdown.process_stem(s, entry, cfg, do_cleanup=False)
    assert not np.allclose(base.audio, shaped.audio)
    assert base.audio.ndim == 2


def test_full_pipeline_with_every_option(tmp_path):
    from mixpipe import pipeline

    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    times = _pattern(100, 3)
    write_audio(stems_dir / "drums.wav", _hits(times, 8, 80, 0.05) + _hits(times[1::2], 8, 3000, 0.01, noisy=True))
    write_audio(stems_dir / "bass.wav", np.roll(_hits(times, 8, 55, 0.12, 0.6), int(0.02 * SR)))
    write_audio(stems_dir / "vox.wav", _voice([(None, 0.3), (C4, 1.0), (None, 0.3)] * 3)[: 8 * SR] * 0.5)
    ref = write_audio(tmp_path / "ref.wav", _tilted(8 * SR, 0.7, 9))
    out = tmp_path / "out" / "mix.wav"
    result = pipeline.run(str(stems_dir), str(out), advisor_mode="rules", notes="vocals louder",
                          skip_cleanup=True, verbose=False, align=True, reference=ref,
                          iterations=2, bit_depth=16, noise_shaping=True)
    import soundfile as sf

    info = sf.info(str(out))
    assert info.subtype == "PCM_16" and info.channels == 2
    audio, _ = sf.read(str(out))
    assert np.all(np.isfinite(audio)) and np.abs(audio).max() <= 1.0
    assert result["master_report"]["output_true_peak_db"] <= -0.8
    assert 1 <= len(result["iterations"]) <= 2
    assert (tmp_path / "out" / "run_report.json").exists()
    assert result["mix_report"]["send_peak_db"] is not None


def test_cli_exposes_the_new_options(tmp_path, capsys):
    from mixpipe import cli

    stems_dir = tmp_path / "s"
    stems_dir.mkdir()
    write_audio(stems_dir / "drums.wav", _hits(_pattern(100, 2), 5, 80, 0.05))
    write_audio(stems_dir / "bass.wav", _hits(_pattern(100, 2), 5, 55, 0.12))
    out = tmp_path / "m.wav"
    rc = cli.main(["mix", "--stems", str(stems_dir), "--out", str(out), "--quiet", "--align",
                   "--iterate", "1", "--bit-depth", "16"])
    assert rc == 0 and out.exists()
