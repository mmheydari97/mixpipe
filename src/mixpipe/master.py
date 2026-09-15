"""Mastering chain.

Deliberately short. Mastering a mix that isn't good yet just makes a loud bad
mix. The chain here is: clean up the extremes, glue it gently, hit a loudness
target, and stop peaks from clipping.
"""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE
from .analysis import crest_factor_db, integrated_lufs, peak_db
from .config import DEFAULT_CONFIG
from .dynamics import compress_with_settings, limit, normalize_loudness
from .eq import apply_eq_chain


def master(
    bus: np.ndarray,
    plan: dict,
    cfg: dict | None = None,
    sr: int = SAMPLE_RATE,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Run the master chain. Returns (audio, report)."""
    cfg = cfg or DEFAULT_CONFIG
    m = plan.get("master", cfg["master"])

    report: dict = {
        "input_peak_db": round(peak_db(bus), 2),
        "input_lufs": round(integrated_lufs(bus.mean(axis=1) if bus.ndim == 2 else bus, sr), 2),
        "input_crest_db": round(crest_factor_db(bus), 2),
        "stages": [],
    }

    audio = bus

    # 1. Clear out subsonic energy that eats limiter headroom inaudibly.
    audio = apply_eq_chain(
        audio,
        hpf_hz=m.get("master_hpf_hz", 25.0),
        lpf_hz=0.0,
        bands=m.get("master_eq", []),
        sr=sr,
    )
    report["stages"].append({"stage": "master_eq", "peak_db": round(peak_db(audio), 2)})

    # 2. Glue compression: low ratio, slow attack, just to bind the parts.
    if cfg["mixdown"].get("bus_comp_enabled", True):
        audio = compress_with_settings(audio, m.get("glue_comp", {}), sr)
        report["stages"].append(
            {"stage": "glue_comp", "peak_db": round(peak_db(audio), 2)}
        )

    # 3. Loudness target. Pure gain, so this can push past 0 dBFS.
    audio, applied = normalize_loudness(audio, m.get("target_lufs", -14.0), sr)
    report["loudness_gain_db"] = round(applied, 2)
    report["stages"].append(
        {"stage": "loudness_norm", "peak_db": round(peak_db(audio), 2)}
    )

    # 4. Limiter catches what step 3 pushed over.
    audio = limit(
        audio,
        ceiling_db=m.get("ceiling_db", -1.0),
        lookahead_ms=cfg["master"]["limiter_lookahead_ms"],
        release_ms=cfg["master"]["limiter_release_ms"],
        sr=sr,
    )
    report["stages"].append({"stage": "limiter", "peak_db": round(peak_db(audio), 2)})

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    report["output_peak_db"] = round(peak_db(audio), 2)
    report["output_lufs"] = round(integrated_lufs(mono, sr), 2)
    report["output_crest_db"] = round(crest_factor_db(audio), 2)
    report["crest_lost_db"] = round(
        report["input_crest_db"] - report["output_crest_db"], 2
    )

    # A crest loss over ~6 dB means the limiter is doing structural work that
    # the mix should have done. Worth telling the user rather than hiding.
    if report["crest_lost_db"] > 6.0:
        report["warning"] = (
            f"limiter squashed {report['crest_lost_db']:.1f} dB of dynamic range; "
            "the mix is probably too hot or too peaky before the master"
        )

    if verbose:
        print(
            f"  [mstr] {report['input_lufs']:.1f} -> {report['output_lufs']:.1f} LUFS, "
            f"peak {report['output_peak_db']:+.2f} dBFS, "
            f"crest {report['input_crest_db']:.1f} -> {report['output_crest_db']:.1f} dB"
        )
        if "warning" in report:
            print(f"  [warn] {report['warning']}")

    return audio, report


def match_reference(
    audio: np.ndarray, reference_path: str, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Match the tonal balance of a reference track.

    This is the single highest-value thing that could be added. It's what
    Matchering does: take the average spectrum of a commercial track you like,
    compute the difference against your mix, and apply the corrective curve.

    # TODO: implement as
    #   1. long-window average magnitude spectrum of both, in log-spaced bands
    #   2. smooth the difference curve (octave-band smoothing, cap at +/-6 dB)
    #   3. apply as a linear-phase FIR, or as a bank of peaking filters
    #   4. match loudness afterwards
    # Or just shell out to `matchering` (pip install matchering), which does all
    # of this well already.
    """
    pass  # TODO
    return audio


def dither(audio: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """TPDF dither for bit-depth reduction.

    # TODO: only needed when exporting to 16-bit. Add triangular-PDF noise at
    # 1 LSB amplitude before truncation, optionally with noise shaping.
    """
    pass  # TODO
    return audio


def export_lufs_report(report: dict) -> str:
    """Format the master report for printing."""
    lines = ["master report"]
    for key in (
        "input_lufs", "output_lufs", "loudness_gain_db",
        "input_peak_db", "output_peak_db",
        "input_crest_db", "output_crest_db", "crest_lost_db",
    ):
        if key in report:
            lines.append(f"  {key:<20} {report[key]}")
    for stage in report.get("stages", []):
        lines.append(f"  after {stage['stage']:<16} peak {stage['peak_db']:+.2f} dBFS")
    if "warning" in report:
        lines.append(f"  WARNING: {report['warning']}")
    return "\n".join(lines)
