"""Stereo placement."""

from __future__ import annotations

import numpy as np


def pan_mono_to_stereo(audio: np.ndarray, pan: float = 0.0, law: str = "constant_power"):
    """Place a mono signal in the stereo field.

    pan: -1.0 hard left, 0.0 centre, +1.0 hard right.

    Constant-power (sin/cos) law keeps perceived loudness steady across the
    field, which is what you want for a mix. Linear law is provided because it
    sums to the same level in mono, which matters if the track will be folded
    down.
    """
    if audio.ndim == 2:
        return audio  # already stereo, use `width()` instead

    p = float(np.clip(pan, -1.0, 1.0))
    if law == "linear":
        left, right = (1.0 - p) / 2.0, (1.0 + p) / 2.0
    else:
        angle = (p + 1.0) * np.pi / 4.0  # 0..pi/2
        left, right = np.cos(angle), np.sin(angle)

    return np.stack([audio * left, audio * right], axis=1).astype(np.float32)


def width(audio: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Mid/side width control. 0 = mono, 1 = unchanged, >1 = wider."""
    if audio.ndim == 1:
        return audio
    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * 0.5 * float(amount)
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def haas_spread(audio: np.ndarray, delay_ms: float = 12.0, sr: int = 48_000):
    """Pseudo-stereo by delaying one side.

    # TODO: works but comb-filters in mono. Should be replaced with an
    # all-pass-based decorrelator, or a short stereo ambience send in reverb.py.
    """
    pass  # TODO
    return audio
