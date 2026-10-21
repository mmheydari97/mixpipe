#!/usr/bin/env python3
"""Run the whole mixer on a folder of stems and check what came out.

    python scripts/verify_mix.py stems_demo/                       # the synthetic session
    python scripts/verify_mix.py out/splitter/<id>/stems --start 60 --duration 45
    python scripts/verify_mix.py stems_demo/ --advisor llm         # OpenRouter plan + critique

Every feature is switched on (alignment, declick, multiband, sends, reference
matching, critique rounds, 16-bit shaped dither). The checks that hold for any
material always run; those that need to know where the flaws are (hum, clicks,
sibilance, the late bass) run when the folder has a `session_meta.json`, which
`make_demo_session.py` writes. A markdown report goes next to the mix.
Exit status 1 when a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mixpipe.analysis import integrated_lufs, true_peak_db  # noqa: E402
from mixpipe.cleanup import clean_stem  # noqa: E402
from mixpipe.config import load_config  # noqa: E402
from mixpipe.io_utils import load_stems, read_audio, stems_mono, write_audio  # noqa: E402
from mixpipe.mixdown import process_stem  # noqa: E402
from mixpipe.pipeline import run  # noqa: E402
from mixpipe.timing import align_stems  # noqa: E402


def feature_config() -> dict:
    cfg = load_config(None)
    cfg["cleanup"]["declick"] = True
    cfg["master"].setdefault("multiband", {})["enabled"] = True
    return cfg


def _line_db(x: np.ndarray, sr: int, f0: float) -> float:
    """Level of the spectral line at f0 over the median of its neighbourhood, in dB."""
    f, p = welch(x, fs=sr, nperseg=1 << 16)
    line = (f > f0 - 0.6) & (f < f0 + 0.6)
    near = (np.abs(f - f0) > 3) & (np.abs(f - f0) < 25)
    return float(10 * np.log10(max(p[line].max(), 1e-30) / max(np.median(p[near]), 1e-30)))


def _click_height(x: np.ndarray, sr: int, times: list[float]) -> float:
    """Median over the click positions of (peak at the click) / (peak of the music
    in the 5 ms either side). About 1 once a click is gone."""
    vals = []
    w, gap = int(0.005 * sr), int(0.0005 * sr)
    for t in times:
        i = int(t * sr)
        around = np.concatenate([x[max(0, i - gap - w): max(0, i - gap)], x[i + 8 + gap: i + 8 + gap + w]])
        vals.append(np.abs(x[max(0, i - 4): i + 8]).max() / (np.abs(around).max() + 1e-9))
    return float(np.median(vals))


def _band_rms(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return float(np.sqrt(np.mean(sosfiltfilt(sos, x) ** 2)) + 1e-12)


def _windows(x: np.ndarray, sr: int, spans: list[list[float]]) -> np.ndarray:
    return np.concatenate([x[int(a * sr): int(b * sr)] for a, b in spans])


def check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def synthetic_checks(stems_dir: str, cfg: dict, plan: dict, meta: dict) -> list[dict]:
    """Per-feature checks that need to know where the flaws were put."""
    stems = {s.name: s for s in load_stems(stems_dir, force_mono=stems_mono(cfg), role_table=cfg["roles"])}
    sr = next(iter(stems.values())).sr
    out = []

    if "electric_lead" in stems:                                # dehum
        s = stems["electric_lead"]
        before = _line_db(s.audio, sr, meta["hum_hz"])
        after = _line_db(clean_stem(s, cfg).audio, sr, meta["hum_hz"])
        out.append(check("dehum", before - after >= 10.0,
                         f"{meta['hum_hz']:.0f} Hz line {before:.1f} -> {after:.1f} dB over its surroundings"))

    if "acoustic_gtr" in stems and meta["clicks"]:              # declick
        s = stems["acoustic_gtr"]
        before = _click_height(s.audio, sr, meta["clicks"])
        after = _click_height(clean_stem(s, cfg).audio, sr, meta["clicks"])
        out.append(check("declick", before > 1.5 and after < 1.2,
                         f"click peak over the music around it {before:.1f}x -> {after:.2f}x"))

    if "bass" in stems and "drums" in stems:                    # alignment
        al = {s.name: s for s in align_stems(list(stems.values()))}
        shift = al["bass"].meta.get("align_offset_samples", 0) / sr * 1000.0
        out.append(check("alignment", abs(shift - meta["bass_delay_ms"]) <= 5.0,
                         f"bass found {shift:+.1f} ms late, put in {meta['bass_delay_ms']:.0f} ms"))

    if "vocal_lead" in stems and meta["sibilance"]:             # de-esser
        s = stems["vocal_lead"]
        settings = dict(plan["stems"]["vocal_lead"])
        off = {**settings, "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0}}
        on = {**settings, "deess": {"freq_hz": 6500.0, "max_reduction_db": 8.0}}
        cleaned = clean_stem(s, cfg)
        a = process_stem(s, off, cfg, cleaned=cleaned).audio[:, 0]
        b = process_stem(s, on, cfg, cleaned=cleaned).audio[:, 0]
        hiss_a = _band_rms(_windows(a, sr, meta["sibilance"]), sr, 5500, 11000)
        hiss_b = _band_rms(_windows(b, sr, meta["sibilance"]), sr, 5500, 11000)
        body_a, body_b = _band_rms(a, sr, 300, 3000), _band_rms(b, sr, 300, 3000)
        drop = 20 * np.log10(hiss_a / hiss_b)
        body = 20 * np.log10(body_b / body_a)
        out.append(check("de-esser", drop >= 3.0 and abs(body) <= 1.0,
                         f"hiss bursts {-drop:+.1f} dB, voiced body {body:+.2f} dB"))
    return out


def generic_checks(result: dict, cfg: dict, duration_s: float, n_stems: int, bit_depth: int) -> list[dict]:
    audio, sr = read_audio(result["out_path"])
    audio = audio if audio.ndim == 2 else np.stack([audio, audio], axis=1)
    master = result["master_report"]
    ceiling = result["plan"]["master"]["ceiling_db"]
    target = result["plan"]["master"]["target_lufs"]
    tp = true_peak_db(audio)
    lufs = integrated_lufs(audio.mean(axis=1), sr)
    mid, side = audio.sum(axis=1), audio[:, 0] - audio[:, 1]
    corr = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
    out = [
        check("true peak under the ceiling", tp <= ceiling + 0.05, f"{tp:+.2f} dBTP vs ceiling {ceiling:+.1f}"),
        check("loudness on target", abs(lufs - target) <= 0.5, f"{lufs:.2f} LUFS vs target {target:.1f}"),
        check("no clipping", float(np.abs(audio).max()) < 1.0, f"sample peak {np.abs(audio).max():.4f}"),
        check("mono-safe", corr > 0.0 and float(np.mean(side ** 2)) < float(np.mean(mid ** 2)),
              f"L/R correlation {corr:+.2f}"),
        check("crest loss reported", "crest_lost_db" in master,
              f"crest lost {master.get('crest_lost_db')} dB"),
    ]
    if bit_depth == 16:
        grid = np.abs(audio * 32768.0 - np.round(audio * 32768.0)).max()
        out.append(check("16-bit grid", grid < 1e-3, f"max distance from the 16-bit grid {grid:.5f} LSB"))
    tail = audio.shape[0] / sr - duration_s
    out.append(check("reverb tail", tail > 0.05 or not any(
        e["sends"]["reverb_db"] > -60 for e in result["plan"]["stems"].values()),
        f"output is {tail:+.2f} s longer than the stems"))
    hist = [h["balance_score_db"] for h in result["iterations"] if h.get("balance_score_db") is not None]
    if len(hist) >= 2:
        out.append(check("critique does not worsen", hist[-1] <= hist[0] + 0.05,
                         "balance error per round: " + " -> ".join(f"{h:.2f}" for h in hist) + " dB"))
    out.append(check("render time", True,
                     f"{result['elapsed_s']:.1f} s for {duration_s:.0f} s x {n_stems} stems "
                     f"({result['elapsed_s'] / max(duration_s / 60.0, 1e-9) / max(n_stems, 1):.1f} s per stem-minute)"))
    return out


def reference_from_stems(stems_dir: str, out_path: Path, start: float, duration: float | None) -> str:
    """The sum of the stems is the original mix: a fair target for tonal balance."""
    stems = load_stems(stems_dir, force_mono=False, start_s=start, duration_s=duration)
    n = max(s.n_samples for s in stems)
    mix = np.zeros((n, 2))
    for s in stems:
        a = s.audio if s.audio.ndim == 2 else np.stack([s.audio] * 2, axis=1)
        mix[: a.shape[0]] += a
    mix /= max(np.abs(mix).max(), 1e-9) / 0.7
    write_audio(str(out_path), mix.astype(np.float32), stems[0].sr)
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems")
    ap.add_argument("--out", default=None, help="output folder (default out/verify/<name>)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--advisor", choices=["rules", "llm"], default="rules")
    ap.add_argument("--provider", choices=["anthropic", "openrouter"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--iterate", type=int, default=2)
    ap.add_argument("--notes", default="")
    ap.add_argument("--bit-depth", type=int, choices=[16, 24], default=16)
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args()

    stems_dir = Path(args.stems)
    name = stems_dir.parent.name if stems_dir.name == "stems" else stems_dir.name
    out_dir = Path(args.out or f"out/verify/{name}-{args.advisor}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = feature_config()
    cfg_path = out_dir / "verify_config.json"
    cfg_path.write_text(json.dumps(cfg, default=str))
    ref = None if args.no_reference else reference_from_stems(
        str(stems_dir), out_dir / "reference.wav", args.start, args.duration)

    t0 = time.time()
    result = run(
        str(stems_dir), out_path=str(out_dir / "mix.wav"), config_path=str(cfg_path),
        advisor_mode=args.advisor, notes=args.notes, provider=args.provider, model=args.model, align=True, reference=ref,
        iterations=args.iterate, bit_depth=args.bit_depth, noise_shaping=True,
        start_s=args.start, duration_s=args.duration, verbose=False,
    )
    wall = time.time() - t0
    n_stems = len(result["plan"]["stems"])
    dur = max(a["duration_s"] for a in result["analysis"]["stems"])

    checks = generic_checks(result, cfg, dur, n_stems, args.bit_depth)
    meta_path = stems_dir / "session_meta.json"
    if meta_path.exists() and not args.start and args.duration is None:
        checks += synthetic_checks(str(stems_dir), cfg, result["plan"], json.loads(meta_path.read_text()))

    lines = [f"# Mix verification: {name} ({args.advisor} advisor)", "",
             f"{n_stems} stems, {dur:.0f} s, wall time {wall:.1f} s. Plan source: `{result['plan'].get('source')}`", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {c['name']} | {'pass' if c['ok'] else '**FAIL**'} | {c['detail']} |" for c in checks]
    lines += ["", "## Critique rounds", ""]
    lines += [f"- round {h['round']}: {h['critique']}" for h in result["iterations"]] or ["- none"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport: {out_dir / 'report.md'}   mix: {out_dir / 'mix.wav'}")
    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
