"""Dynamics: gain staging, compression, limiting.

The compressor is a feed-forward design with a smoothed gain-reduction envelope
computed vectorized where possible. It is not a model of any specific hardware.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import SAMPLE_RATE
from ._fast import asym_smooth
from .analysis import from_db, db, peak_db, integrated_lufs, true_peak_db, true_peak_detector

EPS = 1e-12


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 1e-6:
        return audio
    return (audio * from_db(gain_db)).astype(np.float32)


def normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Scale so the highest peak lands at target_db."""
    p = peak_db(audio)
    if p <= -200.0:
        return audio
    return apply_gain_db(audio, target_db - p)


def normalize_loudness(
    audio: np.ndarray, target_lufs: float = -14.0, sr: int = SAMPLE_RATE
) -> tuple[np.ndarray, float]:
    """Scale to a loudness target. Returns (audio, applied_gain_db).

    Pure gain, no limiting, so this can push peaks over 0 dBFS. Always follow
    with the limiter.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    current = integrated_lufs(mono, sr)
    if current <= -70.0:
        return audio, 0.0
    gain = float(target_lufs - current)
    return apply_gain_db(audio, gain), gain


def _smooth_gain(
    target_gain: np.ndarray, attack_ms: float, release_ms: float, sr: int
) -> np.ndarray:
    """One-pole smoothing of a *gain* envelope (linear, <= 1.0).

    Direction matters and is easy to get backwards: gain moving DOWN is the
    compressor clamping down, which is the attack; gain moving UP is it letting
    go, which is the release. Swapping these makes the compressor ignore
    sustained loud passages, because the gain snaps back to unity on every
    waveform zero-crossing.
    """
    return asym_smooth(target_gain, up_ms=release_ms, down_ms=attack_ms, sr=sr)


def compress(
    audio: np.ndarray,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 10.0,
    release_ms: float = 150.0,
    knee_db: float = 6.0,
    makeup_db: float = 0.0,
    sr: int = SAMPLE_RATE,
    return_reduction: bool = False,
):
    """Soft-knee feed-forward compressor.

    Detector is the peak of the (linked) channels, so stereo images don't wander.
    """
    if ratio <= 1.0:
        out = apply_gain_db(audio, makeup_db)
        return (out, np.zeros(audio.shape[0], dtype=np.float32)) if return_reduction else out

    detector = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=1)
    level_db = db(np.maximum(detector, EPS))

    # Static gain-computer curve with a quadratic soft knee.
    over = level_db - threshold_db
    gain_db = np.zeros_like(over)
    half_knee = knee_db / 2.0

    in_knee = (over > -half_knee) & (over <= half_knee)
    above = over > half_knee
    if knee_db > 0:
        gain_db[in_knee] = (
            (1.0 / ratio - 1.0) * (over[in_knee] + half_knee) ** 2 / (2.0 * knee_db)
        )
    gain_db[above] = (1.0 / ratio - 1.0) * over[above]

    # Smooth in the linear domain so attack/release behave like an analog VCA.
    # The detector is a peak follower, so hold the minimum gain over a short
    # window first: without this, the gain recovers on every zero-crossing and
    # sustained material barely gets compressed.
    target_gain = from_db(gain_db).astype(np.float64)
    hold = max(1, int(0.002 * sr))
    if hold > 1:
        padded = np.pad(target_gain, (hold - 1, 0), mode="edge")
        target_gain = np.lib.stride_tricks.sliding_window_view(padded, hold).min(axis=1)
    smoothed = _smooth_gain(target_gain, attack_ms, release_ms, sr)

    out = audio * (smoothed if audio.ndim == 1 else smoothed[:, None])
    out = apply_gain_db(out.astype(np.float32), makeup_db)

    if return_reduction:
        return out, db(np.maximum(smoothed, EPS)).astype(np.float32)
    return out


def compress_with_settings(audio: np.ndarray, settings: dict, sr: int = SAMPLE_RATE):
    """Convenience wrapper taking a comp settings dict from the config/plan."""
    return compress(
        audio,
        threshold_db=float(settings.get("threshold_db", -18.0)),
        ratio=float(settings.get("ratio", 3.0)),
        attack_ms=float(settings.get("attack_ms", 10.0)),
        release_ms=float(settings.get("release_ms", 150.0)),
        knee_db=float(settings.get("knee_db", 6.0)),
        makeup_db=float(settings.get("makeup_db", 0.0)),
        sr=sr,
    )


TP_MARGIN_DB = 0.1


def limit(
    audio: np.ndarray,
    ceiling_db: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 60.0,
    sr: int = SAMPLE_RATE,
    true_peak: bool = False,
) -> np.ndarray:
    """Lookahead peak limiter.

    Delays the signal by the lookahead time and computes a gain envelope from the
    future peak, so attacks are caught rather than clipped.

    By default this watches *sample* peaks, so inter-sample peaks can still
    exceed the ceiling after a DAC or lossy encode. With `true_peak=True` the
    detector sees the 4x-oversampled signal instead, and the result is checked
    against it: if the true peak still lands over the ceiling (the gain moves
    between samples) the limiter runs again with the ceiling lowered by the
    overshoot, up to three times.
    """
    ceiling = float(from_db(ceiling_db))
    look = max(1, int(lookahead_ms / 1000.0 * sr))

    if true_peak:
        detector = true_peak_detector(audio)
    else:
        detector = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=1)

    # Maximum over the next `look` samples (inclusive), in O(n).
    padded = np.pad(detector, (0, look), mode="edge")
    future_peak = maximum_filter1d(padded, size=look + 1, origin=-(look // 2) - (look % 2))[: detector.size]

    # the gain moves between samples, so aim a hair under the ceiling when watching
    # true peaks; that usually saves the corrective re-run below
    aim = ceiling * (from_db(-TP_MARGIN_DB) if true_peak else 1.0)
    required = np.minimum(1.0, aim / np.maximum(future_peak, EPS))

    # Release-only smoothing: gain must drop instantly, recover slowly.
    gain = asym_smooth(required, up_ms=release_ms, down_ms=0.0, sr=sr, init=1.0)

    # Delay the audio to align with the gain envelope it was computed from.
    if audio.ndim == 1:
        delayed = np.concatenate([np.zeros(look, dtype=audio.dtype), audio])[: audio.size]
        out = delayed * gain
    else:
        delayed = np.concatenate(
            [np.zeros((look, audio.shape[1]), dtype=audio.dtype), audio]
        )[: audio.shape[0]]
        out = delayed * gain[:, None]

    out = np.clip(out, -ceiling, ceiling).astype(np.float32)
    if true_peak and ceiling_db > -40.0:
        over = true_peak_db(out) - ceiling_db
        if over > 0.01:
            for _ in range(3):
                out = limit(out, ceiling_db - over - 0.02, lookahead_ms, release_ms, sr, true_peak=True)
                over = true_peak_db(out) - ceiling_db
                if over <= 0.01:
                    break
    return out


def gain_staging_report(audio: np.ndarray, label: str = "") -> dict:
    """Small helper so the CLI can print where headroom is going."""
    return {"label": label, "peak_db": round(peak_db(audio), 2)}


def _envelope_db(
    detector: np.ndarray, attack_ms: float, release_ms: float, sr: int
) -> np.ndarray:
    """Peak envelope follower, in dB."""
    env = asym_smooth(detector, up_ms=attack_ms, down_ms=release_ms, sr=sr, init=0.0)
    return db(np.maximum(env, EPS))


def transient_shaper(
    audio: np.ndarray, attack: float = 0.0, sustain: float = 0.0,
    sr: int = SAMPLE_RATE, max_gain_db: float = 12.0,
) -> np.ndarray:
    """Attack/sustain shaping that follows the signal, independent of level.

    Two pairs of envelope followers; the difference of each pair isolates one
    part of a note:

      attack   a fast and a slow follower with the same release. On a note onset
               the fast one jumps first, so (fast - slow) is positive only
               during the attack.
      sustain  two followers with the same attack but a fast and a slow release.
               After the peak the slow-release one holds up, so (slow - fast) is
               positive only during the decay.

    Each difference, in dB, scaled by `attack` / `sustain` (-1 to +1), becomes a
    gain move: positive adds punch / lets the tail bloom, negative softens the
    pick or tightens a boomy room. Because it works on the difference between
    envelopes, a loud passage and a quiet one are shaped alike. The total move
    is capped at +/-`max_gain_db`.
    """
    attack = float(np.clip(attack, -1.0, 1.0))
    sustain = float(np.clip(sustain, -1.0, 1.0))
    if audio.size == 0 or (abs(attack) < 1e-3 and abs(sustain) < 1e-3):
        return audio

    detector = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=1)
    gain_db = np.zeros(detector.size)
    if abs(attack) >= 1e-3:
        fast = _envelope_db(detector, 1.0, 60.0, sr)
        slow = _envelope_db(detector, 25.0, 60.0, sr)
        gain_db += attack * np.maximum(fast - slow, 0.0)
    if abs(sustain) >= 1e-3:
        quick = _envelope_db(detector, 1.0, 8.0, sr)
        held = _envelope_db(detector, 1.0, 150.0, sr)
        gain_db += sustain * np.maximum(held - quick, 0.0)
    gain_db = np.clip(gain_db, -max_gain_db, max_gain_db)

    # Smooth the gain move so it never zippers (a few ms is inaudible as a lag).
    gain = asym_smooth(from_db(gain_db), up_ms=3.0, down_ms=3.0, sr=sr, init=1.0)
    out = audio * (gain if audio.ndim == 1 else gain[:, None])
    return out.astype(np.float32)


# A gentle starting point for a mastering bus: wide, slow, low ratios.
DEFAULT_MULTIBAND = {
    "crossovers": [150.0, 4000.0],
    "bands": [
        {"threshold_db": -18.0, "ratio": 2.0, "attack_ms": 30.0, "release_ms": 180.0,
         "makeup_db": 0.0},
        {"threshold_db": -16.0, "ratio": 1.8, "attack_ms": 15.0, "release_ms": 120.0,
         "makeup_db": 0.0},
        {"threshold_db": -20.0, "ratio": 1.8, "attack_ms": 5.0, "release_ms": 80.0,
         "makeup_db": 0.0},
    ],
}


def multiband_compress(
    audio: np.ndarray, sr: int = SAMPLE_RATE, crossovers: list[float] | None = None,
    bands: list[dict] | None = None,
) -> np.ndarray:
    """Per-band compression for the master bus.

    Splits the signal with phase-coherent Linkwitz-Riley (LR4) crossovers, runs
    an independent compressor on each band, and sums. With every band's ratio at
    1 the sum is the input again (an LR4 crossover sums flat in magnitude; only
    the phase rotates). `crossovers` has one entry fewer than `bands`.
    """
    from .eq import split_bands

    cfg = DEFAULT_MULTIBAND
    crossovers = list(crossovers if crossovers is not None else cfg["crossovers"])
    bands = list(bands if bands is not None else cfg["bands"])
    if len(bands) != len(crossovers) + 1:
        raise ValueError("multiband_compress needs one more band than crossovers")
    if audio.size == 0:
        return audio

    parts = split_bands(audio, crossovers, sr)
    out = np.zeros(audio.shape, dtype=np.float64)
    for part, settings in zip(parts, bands):
        out += compress_with_settings(part, settings, sr)
    return out.astype(np.float32)
