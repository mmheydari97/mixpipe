#!/usr/bin/env python3
"""Generate synthetic stems so the pipeline can be run before you record anything.

Four fake takes matching your actual setup, all in A minor at 96 BPM, each with
a deliberate flaw for the advisor to react to:

  acoustic_gtr  - plucked-string arpeggios, quiet, hissy (low SNR)
  electric_lead - saturated sustained lead, too loud
  harmonica     - reedy sustained chords with a nasty 1.8 kHz honk
  seaboard_pad  - warm sub-heavy pad, muddy low end

    python scripts/make_test_stems.py [--out stems/] [--seconds 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mixpipe.io_utils import write_audio  # noqa: E402

SR = 48_000
BPM = 96.0
A_MINOR = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00]


def adsr(n: int, a: float, d: float, s: float, r: float, sustain: float = 0.6):
    """Simple ADSR envelope of length n samples, times in seconds."""
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    s_n = max(0, n - a_n - d_n - r_n)
    return np.concatenate([
        np.linspace(0, 1, a_n, endpoint=False),
        np.linspace(1, sustain, d_n, endpoint=False),
        np.full(s_n, sustain),
        np.linspace(sustain, 0, r_n),
    ])[:n]


def note(freq: float, dur: float, harmonics: list[float], env: tuple, detune=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for i, amp in enumerate(harmonics, start=1):
        f = freq * i * (1.0 + detune * (i - 1))
        sig += amp * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    return sig * adsr(n, *env)


def place(buf: np.ndarray, sig: np.ndarray, at_s: float, gain: float = 1.0):
    start = int(at_s * SR)
    end = min(buf.size, start + sig.size)
    if start < buf.size:
        buf[start:end] += sig[: end - start] * gain


def make_acoustic(seconds: float) -> np.ndarray:
    """Arpeggios. Bright, plucky, with tape-ish hiss and a bit of rumble."""
    buf = np.zeros(int(seconds * SR))
    beat = 60.0 / BPM
    pattern = [0, 2, 4, 7, 4, 2]
    i = 0
    t = 0.5  # leading silence for the noise profile
    while t < seconds - 1.0:
        f = A_MINOR[pattern[i % len(pattern)] % len(A_MINOR)]
        place(buf, note(f, 1.2, [1.0, 0.5, 0.3, 0.18, 0.1, 0.06],
                        (0.004, 0.08, 0.5, 0.9, 0.35)), t, 0.5)
        t += beat / 2
        i += 1
    buf += np.random.normal(0, 0.004, buf.size)              # mic hiss
    rumble_t = np.arange(buf.size) / SR
    buf += 0.01 * np.sin(2 * np.pi * 42 * rumble_t)          # floor rumble
    return (buf * 0.35).astype(np.float32)


def make_electric_lead(seconds: float) -> np.ndarray:
    """Sustained lead line, soft-clipped, recorded way too hot."""
    buf = np.zeros(int(seconds * SR))
    melody = [(0.5, 4, 1.4), (2.2, 5, 1.0), (3.4, 7, 1.8), (5.6, 4, 1.2),
              (7.2, 2, 1.6), (9.0, 4, 2.2)]
    for at, deg, dur in melody:
        if at >= seconds:
            break
        f = A_MINOR[deg % len(A_MINOR)]
        place(buf, note(f, dur, [1.0, 0.7, 0.5, 0.35, 0.22, 0.15, 0.1],
                        (0.02, 0.15, 0.75, 0.3), ), at, 0.8)
    buf = np.tanh(buf * 3.0) / 1.2          # amp saturation, kills the crest factor
    buf += np.random.normal(0, 0.002, buf.size)
    return (buf * 0.85).astype(np.float32)   # deliberately too loud


def make_harmonica(seconds: float) -> np.ndarray:
    """Reedy sustained double-stops with a resonant honk around 1.8 kHz."""
    buf = np.zeros(int(seconds * SR))
    chords = [(1.0, [2, 4], 2.0), (3.5, [4, 7], 2.2), (6.2, [0, 4], 2.4),
              (9.0, [2, 5], 2.0)]
    for at, degs, dur in chords:
        if at >= seconds:
            break
        for deg in degs:
            f = A_MINOR[deg % len(A_MINOR)]
            place(buf, note(f, dur, [1.0, 0.85, 0.7, 0.55, 0.45, 0.3, 0.25, 0.2],
                            (0.05, 0.2, 0.8, 0.25), detune=0.002), at, 0.45)
    # The honk: a resonant peak where the lead guitar also lives.
    t = np.arange(buf.size) / SR
    honk = 0.12 * np.sin(2 * np.pi * 1800 * t) * (np.abs(buf) > 0.02)
    buf += honk
    buf += np.random.normal(0, 0.006, buf.size)   # breathy, noisy take
    return (buf * 0.6).astype(np.float32)


def make_seaboard_pad(seconds: float) -> np.ndarray:
    """Warm sub-heavy pad with slow MPE-ish pressure swells. Muddy on purpose."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    buf = np.zeros(n)
    for deg, amp in [(0, 0.5), (2, 0.35), (4, 0.3)]:
        f = A_MINOR[deg] / 2.0     # an octave down, lots of low energy
        lfo = 1.0 + 0.4 * np.sin(2 * np.pi * 0.15 * t)   # pressure swell
        buf += amp * lfo * (
            np.sin(2 * np.pi * f * t)
            + 0.5 * np.sin(2 * np.pi * f * 2 * t)
            + 0.2 * np.sin(2 * np.pi * f * 3 * t)
        )
    buf *= adsr(n, 1.5, 0.5, 0.0, 2.0, sustain=0.9)
    return (buf * 0.3).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stems")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stems = {
        "acoustic_gtr": make_acoustic(args.seconds),
        "electric_lead": make_electric_lead(args.seconds),
        "harmonica": make_harmonica(args.seconds),
        "seaboard_pad": make_seaboard_pad(args.seconds),
    }
    for name, audio in stems.items():
        path = write_audio(out / f"{name}.wav", audio, SR)
        peak = 20 * np.log10(max(np.max(np.abs(audio)), 1e-9))
        print(f"wrote {path}  ({audio.size / SR:.1f}s, peak {peak:+.1f} dBFS)")

    print(f"\n{len(stems)} test stems in {out}/")
    print("now run: python -m mixpipe.cli mix --stems stems/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
