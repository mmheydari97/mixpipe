"""Stereo placement."""

from __future__ import annotations

import numpy as np


def pan_mono_to_stereo(audio: np.ndarray, pan: float = 0.0, law: str = "constant_power"):
    """Place a mono signal in the stereo field.

    pan: -1.0 hard left, 0.0 centre, +1.0 hard right. A stereo input keeps its
    image and `pan` works as a balance control instead.

    Constant-power (sin/cos) law keeps perceived loudness steady across the
    field, which is what you want for a mix. Linear law is provided because it
    sums to the same level in mono, which matters if the track will be folded
    down.
    """
    if audio.ndim == 2:
        # already stereo: pan is a balance control, the far side turned down
        p = float(np.clip(pan, -1.0, 1.0))
        if abs(p) < 1e-6:
            return audio
        gains = np.array([min(1.0, 1.0 - p), min(1.0, 1.0 + p)], dtype=np.float32)
        return (audio * gains).astype(np.float32)

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


_DECORR_DELAYS_MS = (0.31, 0.47, 0.67, 0.91, 1.13, 1.37)   # mutually "prime" ratios


def haas_spread(
    audio: np.ndarray, delay_ms: float = 12.0, sr: int = 48_000, amount: float = 0.6,
    low_cut_hz: float = 200.0,
) -> np.ndarray:
    """Pseudo-stereo from a mono source, without the comb filter of a Haas delay.

    A plain delay on one side sounds wide but collapses to a comb filter when
    the mix is folded to mono. Here a cascade of Schroeder all-pass filters turns
    the signal into a decorrelated copy `d` (same magnitude spectrum, scrambled
    phase) and the sides are L = x + a*d, R = x - a*d. In mono L + R = 2x: the
    decorrelated part cancels exactly, so nothing combs or thins out.

    `delay_ms` scales the all-pass delays (about 0.3-1.4x of it in total), i.e.
    how smeared the copy is; `amount` (0-1) is the width; `low_cut_hz` keeps the
    low end of the copy out, so bass stays centred. Stereo input is returned as is.
    """
    from scipy.signal import butter, sosfilt

    from .reverb import _allpass

    if audio.ndim == 2 or audio.size == 0:
        return audio
    amount = float(np.clip(amount, 0.0, 1.0))
    x = audio.astype(np.float64)
    d = x
    for base in _DECORR_DELAYS_MS:
        d = _allpass(d, max(2, int(round(base * delay_ms / 12.0 * sr / 1000.0))), 0.6)
    if low_cut_hz > 0:
        d = sosfilt(butter(2, low_cut_hz, btype="highpass", fs=sr, output="sos"), d)
    a = 0.5 * amount
    norm = 1.0 / np.sqrt(1.0 + a * a)
    return (np.stack([x + a * d, x - a * d], axis=1) * norm).astype(np.float32)
