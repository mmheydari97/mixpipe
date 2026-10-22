"""Effects the mixer and the splitter share, written once.

The mixer keeps them in its plan (the chat and the keyword rules change them;
mixdown.process_stem and add_send_buses apply them). The splitter's knobs send
the same settings to splitter/effects.py, which renders them with these same
functions, so "drive 0.3" sounds the same in both.

  tone_bands(bass_db, treble_db)          shelf bands for eq.apply_eq_chain
  drive(audio, sr, amount, tone)          saturation: warm at 0.2, fuzzy at 1
  chorus(audio, sr, mix, rate_hz, depth_ms)
  color(audio, sr, settings)              drive, then chorus: the order both use
  send_returns(sources, buses, sr, ...)   the shared room (reverb) and echo
  delay_ms_for(sync, bpm)                 an echo time from a note value
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .reverb import algorithmic_reverb, slap_delay

OFF_DB = -60.0          # a send at or below this is off
BASS_HZ, TREBLE_HZ = 120.0, 6000.0
# echo times as fractions of a beat ("d" = dotted, "t" = triplet)
SYNC = {"1/4": 1.0, "1/4d": 1.5, "1/4t": 2 / 3, "1/8": 0.5, "1/8d": 0.75, "1/8t": 1 / 3, "1/16": 0.25}


def as_stereo(audio: np.ndarray) -> np.ndarray:
    return np.stack([audio, audio], axis=1) if audio.ndim == 1 else audio


def tone_bands(bass_db: float = 0.0, treble_db: float = 0.0) -> list[dict]:
    """Bass and treble as EQ bands: a low shelf at 120 Hz, a high shelf at 6 kHz."""
    bands = []
    if abs(bass_db) > 0.05:
        bands.append({"type": "low_shelf", "freq": BASS_HZ, "gain_db": float(bass_db), "q": 0.7})
    if abs(treble_db) > 0.05:
        bands.append({"type": "high_shelf", "freq": TREBLE_HZ, "gain_db": float(treble_db), "q": 0.7})
    return bands


def drive(audio: np.ndarray, sr: int, amount: float, tone: float = 0.5) -> np.ndarray:
    """Saturation. `amount` 0..1 (0 = off, 0.2 warm, 1 fuzzy) pushes up to 30 dB
    into a tanh curve, at twice the sample rate so the new harmonics don't fold
    back as aliasing, and blends that much of it with the dry sound. `tone` 0..1
    is a low-pass on the driven part, from dark (2 kHz) to open (16 kHz). The
    level is matched to the input: more drive is more grit, not more volume."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-3 or audio.size == 0:
        return audio
    x = np.asarray(audio, dtype=np.float64)
    hot = 10 ** (amount * 30.0 / 20.0)
    wet = resample_poly(np.tanh(hot * resample_poly(x, 2, 1, axis=0)), 1, 2, axis=0)[: x.shape[0]]
    cutoff = 2000.0 * 2 ** (3.0 * float(np.clip(tone, 0.0, 1.0)))
    if cutoff < 0.45 * sr:
        wet = sosfilt(butter(2, cutoff, btype="low", fs=sr, output="sos"), wet, axis=0)
    y = (1.0 - amount) * x + amount * wet
    rms_in, rms_out = np.sqrt(np.mean(x * x)), np.sqrt(np.mean(y * y))
    if rms_out > 1e-12:
        y *= rms_in / rms_out
    return y.astype(np.float32)


def chorus(audio: np.ndarray, sr: int, mix: float, rate_hz: float = 0.8,
           depth_ms: float = 3.0, delay_ms: float = 12.0) -> np.ndarray:
    """Two copies, each delayed by a slowly swinging delay_ms ± depth_ms (a
    quarter-cycle apart), mixed back in: the thickening of a doubled part.
    `mix` 0..1 is how much (1 = equal parts dry and chorus). On a stereo source
    the right side swings a quarter-cycle later than the left, which widens it."""
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 1e-3 or audio.size == 0:
        return audio
    x = np.asarray(audio, dtype=np.float64)
    chans = x[:, None] if x.ndim == 1 else x
    t = np.arange(chans.shape[0], dtype=np.float64)
    swing = 2 * np.pi * rate_hz * t / sr
    base, depth = delay_ms * sr / 1000.0, depth_ms * sr / 1000.0
    out = np.empty_like(chans)
    for c in range(chans.shape[1]):
        taps = [np.interp(t - (base + depth * np.sin(swing + np.pi / 2 * (k + c))), t,
                          chans[:, c], left=0.0) for k in (0, 1)]
        out[:, c] = (1.0 - 0.5 * mix) * chans[:, c] + 0.25 * mix * (taps[0] + taps[1])
    return (out[:, 0] if x.ndim == 1 else out).astype(np.float32)


def color(audio: np.ndarray, sr: int, settings: dict) -> np.ndarray:
    """A stem's insert effects in the order both apps use: drive, then chorus.
    `settings` is a plan stem entry (or the splitter's): {"drive": {"amount",
    "tone"}, "chorus": {"mix", "rate_hz", "depth_ms"}}; missing means off."""
    d = settings.get("drive") or {}
    c = settings.get("chorus") or {}
    audio = drive(audio, sr, d.get("amount", 0.0), d.get("tone", 0.5))
    return chorus(audio, sr, c.get("mix", 0.0), c.get("rate_hz", 0.8), c.get("depth_ms", 3.0))


def delay_ms_for(sync: str | None, bpm: float | None) -> float | None:
    """An echo time locked to the tempo ("1/8" at 120 BPM = 250 ms); None
    without a note value or a tempo."""
    if not sync or sync not in SYNC or not bpm:
        return None
    return 60000.0 / float(bpm) * SYNC[sync]


def send_returns(sources: list[tuple[np.ndarray, float, float]], buses: dict, sr: int,
                 bpm: float | None = None, tail: bool = True) -> np.ndarray:
    """The shared reverb and echo for `sources`, to sum in parallel with the dry
    mix. Each source is (audio, reverb_db, delay_db): it feeds each effect at its
    send level (dB, OFF_DB = off), and each effect runs once on the summed feed,
    so every instrument sits in the *same* room: that is what stops a mix of
    close-miked dry sources sounding like separate recordings. `buses` holds
    the plan's send_buses settings; a delay with a "sync" note value follows
    `bpm`. Returns a stereo array as long as the longest source, plus the tails
    when `tail` is on; all zeros when nothing is sent."""
    n = max((len(a) for a, _, _ in sources), default=0)
    feeds = {"reverb": np.zeros((n, 2)), "delay": np.zeros((n, 2))}
    used = {"reverb": False, "delay": False}
    for audio, reverb_db, delay_db in sources:
        a = as_stereo(audio)
        for bus, level in (("reverb", reverb_db), ("delay", delay_db)):
            if float(level) > OFF_DB:
                feeds[bus][: a.shape[0]] += a * 10.0 ** (float(level) / 20.0)
                used[bus] = True

    returns = []
    if used["reverb"]:
        p = buses["reverb"]
        wet = algorithmic_reverb(
            feeds["reverb"].astype(np.float32), room_size=p["room_size"],
            decay_s=p["decay_s"], mix=1.0, sr=sr, damping=p["damping"],
            pre_delay_ms=p["pre_delay_ms"], width=p["width"], tail=tail,
        )
        returns.append(wet * 10.0 ** (p["return_db"] / 20.0))
    if used["delay"]:
        p = buses["delay"]
        wet = slap_delay(
            feeds["delay"].astype(np.float32),
            delay_ms=delay_ms_for(p.get("sync"), bpm) or p["delay_ms"],
            feedback=p["feedback"], mix=1.0, sr=sr, damping=p["damping"], tail=tail,
        )
        returns.append(wet * 10.0 ** (p["return_db"] / 20.0))

    out = np.zeros((max([n] + [r.shape[0] for r in returns]), 2), dtype=np.float32)
    for r in returns:
        out[: r.shape[0]] += r
    return out
