#!/usr/bin/env python3
"""Synthetic multitrack session for exercising every mixer feature at once.

Eight stems, A minor, 100 BPM, each with a deliberate flaw so a feature has
something to fix. `session_meta.json` records where the flaws are, so the
verification can measure them before and after:

  vocal_lead      sibilant hiss bursts (de-esser)
  vocal_harmony   a third above the lead, quieter (same-role panning, masking)
  acoustic_gtr    a dozen 1-3 sample clicks (declick)
  electric_lead   saturated, with 50 Hz hum + harmonics (dehum)
  bass            recorded 37 ms late against the drums (time alignment)
  drums           kick / snare / hat
  keys_pad        warm pad, muddy low end
  harmonica       reedy chords with a 1.8 kHz honk

    python scripts/make_demo_session.py [--out stems_demo/] [--seconds 40] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mixpipe.io_utils import write_audio  # noqa: E402

SR = 48_000
BPM = 100.0
BEAT = 60.0 / BPM
BASS_DELAY_MS = 37.0
HUM_HZ = 50.0
# A minor progression, one chord per bar: Am F C G (root MIDI, third and fifth in semitones)
PROG = [(45, (0, 3, 7)), (41, (0, 4, 7)), (48, (0, 4, 7)), (43, (0, 4, 7))]


def hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _env(n: int, a: float, r: float, sustain: float = 1.0) -> np.ndarray:
    a_n, r_n = min(n, int(a * SR)), min(n, int(r * SR))
    e = np.full(n, sustain)
    e[:a_n] = np.linspace(0, sustain, a_n, endpoint=False)
    if r_n:
        e[n - r_n:] *= np.linspace(1, 0, r_n)
    return e


def _tone(freq: float, dur: float, partials, rng, env=(0.01, 0.05)) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for i, amp in enumerate(partials, start=1):
        if freq * i < SR / 2.2:
            y += amp * np.sin(2 * np.pi * freq * i * t + rng.uniform(0, 6.28))
    return y * _env(n, *env)


def _place(buf: np.ndarray, sig: np.ndarray, at: float, gain: float = 1.0) -> None:
    a = int(at * SR)
    b = min(buf.size, a + sig.size)
    if 0 <= a < buf.size:
        buf[a:b] += sig[: b - a] * gain


def _highpass_noise(n: int, rng, lo: float, hi: float) -> np.ndarray:
    """White noise band-limited by FFT masking."""
    x = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    x[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(x, n)
    return y / (np.abs(y).max() + 1e-9)


def _vowel(freq: float, dur: float, rng, formants=((700, 1.0), (1200, 0.5), (2600, 0.2))) -> np.ndarray:
    """Harmonic source shaped by a few formant humps: enough to read as a voice."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t)
    y = np.zeros(n)
    for k in range(1, int(5000 / freq)):
        f = freq * k
        gain = sum(a * np.exp(-0.5 * ((f - c) / (0.18 * c + 80)) ** 2) for c, a in formants)
        y += gain / k ** 0.3 * np.sin(2 * np.pi * np.cumsum(f * vib) / SR + rng.uniform(0, 6.28))
    return y * _env(n, 0.03, 0.08)


def build(seconds: float = 40.0, seed: int = 7) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    bars = int(seconds / (4 * BEAT))
    out = {k: np.zeros(n) for k in (
        "vocal_lead", "vocal_harmony", "acoustic_gtr", "electric_lead",
        "bass", "drums", "keys_pad", "harmonica")}
    meta: dict = {"sr": SR, "bpm": BPM, "seconds": seconds, "hum_hz": HUM_HZ,
                  "bass_delay_ms": BASS_DELAY_MS, "sibilance": [], "clicks": []}

    # drums ---------------------------------------------------------------
    kick = _tone(55, 0.25, [1.0], rng, env=(0.001, 0.2)) * np.exp(-np.arange(int(0.25 * SR)) / (0.06 * SR))
    kick += 0.5 * np.sin(2 * np.pi * (140 * np.exp(-np.arange(int(0.25 * SR)) / (0.02 * SR))
                                      * np.arange(int(0.25 * SR)) / SR)) * np.exp(-np.arange(int(0.25 * SR)) / (0.03 * SR))
    snare_n = int(0.2 * SR)
    snare = (_highpass_noise(snare_n, rng, 1500, 9000) * np.exp(-np.arange(snare_n) / (0.05 * SR))
             + 0.5 * np.sin(2 * np.pi * 190 * np.arange(snare_n) / SR) * np.exp(-np.arange(snare_n) / (0.03 * SR)))
    hat_n = int(0.05 * SR)
    hat = _highpass_noise(hat_n, rng, 6000, 16000) * np.exp(-np.arange(hat_n) / (0.012 * SR))
    for beat in range(int(seconds / BEAT)):
        t = beat * BEAT
        if beat % 4 in (0, 2):
            _place(out["drums"], kick, t, 0.9)
        if beat % 4 in (1, 3):
            _place(out["drums"], snare, t, 0.6)
        _place(out["drums"], hat, t, 0.25)
        _place(out["drums"], hat, t + BEAT / 2, 0.15)

    # bass (late) ---------------------------------------------------------
    delay = BASS_DELAY_MS / 1000.0
    for bar in range(bars):
        root = PROG[bar % 4][0] - 12
        for step, off in enumerate((0, 1.5, 2, 3)):         # rhythmic figure
            _place(out["bass"], _tone(hz(root), BEAT * (1.0 if step != 1 else 0.5),
                                      [1.0, 0.45, 0.2, 0.1], rng, env=(0.005, 0.06)),
                   bar * 4 * BEAT + off * BEAT + delay, 0.6)

    # keys pad (muddy: a lot of 150-300 Hz) -------------------------------
    for bar in range(bars):
        root, ints = PROG[bar % 4]
        for i in ints:
            _place(out["keys_pad"], _tone(hz(root + 12 + i), 4 * BEAT, [1.0, 0.4, 0.5, 0.2], rng,
                                          env=(0.35, 0.5)), bar * 4 * BEAT, 0.16)
        _place(out["keys_pad"], _tone(hz(root), 4 * BEAT, [1.0, 1.1, 0.6], rng, env=(0.3, 0.4)),
               bar * 4 * BEAT, 0.12)

    # acoustic: plucked arpeggio + clicks ---------------------------------
    for bar in range(bars):
        root, ints = PROG[bar % 4]
        for k in range(8):
            midi = root + 12 + ints[k % 3] + (12 if k % 4 == 3 else 0)
            _place(out["acoustic_gtr"], _tone(hz(midi), 0.7, [1.0, 0.5, 0.3, 0.15, 0.08], rng,
                                              env=(0.002, 0.25)) * np.exp(-np.arange(int(0.7 * SR)) / (0.35 * SR)),
                   bar * 4 * BEAT + k * BEAT / 2, 0.35)
    for at in sorted(rng.uniform(1.0, seconds - 1.0, 12)):
        i = int(at * SR)
        width = int(rng.integers(1, 4))
        out["acoustic_gtr"][i:i + width] += rng.choice([-1, 1]) * 0.85
        meta["clicks"].append(round(float(at), 4))

    # electric lead: saturated + hum -------------------------------------
    melody = [0, 3, 7, 5, 3, 0, -2, 0]
    for bar in range(bars):
        root = PROG[bar % 4][0]
        for k in range(4):
            semis = melody[(bar * 4 + k) % len(melody)]
            _place(out["electric_lead"], np.tanh(2.5 * _tone(hz(root + 24 + semis), BEAT * 0.95,
                                                            [1.0, 0.6, 0.4, 0.2], rng, env=(0.01, 0.08))),
                   bar * 4 * BEAT + k * BEAT, 0.45)
    t = np.arange(n) / SR
    for h, a in ((1, 0.07), (2, 0.045), (3, 0.03)):
        out["electric_lead"] += a * np.sin(2 * np.pi * HUM_HZ * h * t)

    # harmonica: reedy chords + honk --------------------------------------
    for bar in range(0, bars, 2):
        root, ints = PROG[bar % 4]
        for i in ints:
            _place(out["harmonica"], _tone(hz(root + 24 + i), 2 * BEAT, [1.0, 0.6, 0.55, 0.4, 0.3], rng,
                                           env=(0.05, 0.2)), (bar * 4 + 2) * BEAT, 0.12)
    out["harmonica"] += 0.04 * np.sin(2 * np.pi * 1800 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t))

    # vocals ---------------------------------------------------------------
    phrase = [0, 3, 5, 7, 5, 3, 0, 3]
    for bar in range(bars):
        root = PROG[bar % 4][0]
        for k in range(4):
            at = bar * 4 * BEAT + k * BEAT
            if bar % 2 == 1:        # leave air between phrases
                continue
            semis = phrase[(bar * 2 + k) % len(phrase)]
            dur = BEAT * 0.9
            _place(out["vocal_lead"], _vowel(hz(root + 24 + semis), dur, rng), at, 0.5)
            _place(out["vocal_harmony"], _vowel(hz(root + 24 + semis + 4), dur, rng,
                                                formants=((500, 1.0), (1500, 0.4), (2500, 0.2))),
                   at, 0.22)
            if k in (0, 2):             # an "s" at the start of the word
                hiss = _highpass_noise(int(0.09 * SR), rng, 5500, 11000)
                hiss *= _env(hiss.size, 0.005, 0.03)
                _place(out["vocal_lead"], hiss, at, 0.28)
                meta["sibilance"].append([round(at, 3), round(at + 0.09, 3)])

    for k, v in out.items():
        out[k] = np.clip(v, -0.98, 0.98).astype(np.float32)
    # the bass is what is "recorded late": its content starts later, the length is the same
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="stems_demo/")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    stems, meta = build(args.seconds, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, audio in stems.items():
        write_audio(str(out / f"{name}.wav"), audio, SR)
    (out / "session_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(stems)} stems ({args.seconds:.0f} s) to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
