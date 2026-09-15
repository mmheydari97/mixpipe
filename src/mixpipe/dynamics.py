"""Dynamics: gain staging, compression, limiting.

The compressor is a feed-forward design with a smoothed gain-reduction envelope
computed vectorized where possible. It is not a model of any specific hardware.
"""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE
from .analysis import from_db, db, peak_db, integrated_lufs

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
    a = np.exp(-1.0 / max(1.0, attack_ms / 1000.0 * sr))
    r = np.exp(-1.0 / max(1.0, release_ms / 1000.0 * sr))
    out = np.empty_like(target_gain)
    state = target_gain[0] if target_gain.size else 1.0
    for i in range(target_gain.size):
        x = target_gain[i]
        coef = a if x < state else r   # reducing gain -> attack
        state = x + coef * (state - x)
        out[i] = state
    return out


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


def limit(
    audio: np.ndarray,
    ceiling_db: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 60.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Lookahead peak limiter.

    Delays the signal by the lookahead time and computes a gain envelope from the
    future peak, so attacks are caught rather than clipped. This is a *sample*
    peak limiter: it does not oversample, so inter-sample peaks can still exceed
    the ceiling after conversion to lossy formats. Set the ceiling to -1 dB and
    don't treat this as release-grade.
    """
    ceiling = float(from_db(ceiling_db))
    look = max(1, int(lookahead_ms / 1000.0 * sr))

    detector = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=1)

    # Sliding maximum over the lookahead window, via a padded strided max.
    padded = np.pad(detector, (0, look), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, look + 1)
    future_peak = windows.max(axis=1)[: detector.size]

    required = np.minimum(1.0, ceiling / np.maximum(future_peak, EPS))

    # Release-only smoothing: gain must drop instantly, recover slowly.
    r = np.exp(-1.0 / max(1.0, release_ms / 1000.0 * sr))
    gain = np.empty_like(required)
    state = 1.0
    for i in range(required.size):
        target = required[i]
        state = target if target < state else target + r * (state - target)
        gain[i] = state

    # Delay the audio to align with the gain envelope it was computed from.
    if audio.ndim == 1:
        delayed = np.concatenate([np.zeros(look, dtype=audio.dtype), audio])[: audio.size]
        out = delayed * gain
    else:
        delayed = np.concatenate(
            [np.zeros((look, audio.shape[1]), dtype=audio.dtype), audio]
        )[: audio.shape[0]]
        out = delayed * gain[:, None]

    return np.clip(out, -ceiling, ceiling).astype(np.float32)


def gain_staging_report(audio: np.ndarray, label: str = "") -> dict:
    """Small helper so the CLI can print where headroom is going."""
    return {"label": label, "peak_db": round(peak_db(audio), 2)}


def transient_shaper(
    audio: np.ndarray, attack: float = 0.0, sustain: float = 0.0, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Attack/sustain shaping, independent of level.

    # TODO: implement with two envelope followers at different time constants;
    # their difference isolates transients. Useful for acoustic guitar pick
    # attack and for taming a boomy room on the sustain.
    """
    pass  # TODO
    return audio


def multiband_compress(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Per-band compression for the master bus.

    # TODO: 3-4 band Linkwitz-Riley crossover, independent compressors, sum.
    # The single-band glue comp in master.py is the interim stand-in.
    """
    pass  # TODO
    return audio
