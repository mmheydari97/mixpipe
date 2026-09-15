"""Biquad EQ. RBJ cookbook coefficients, applied with scipy's zero-phase-free
sosfilt (causal, like a real analog-modelled EQ would be).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt, butter

from . import SAMPLE_RATE


def _biquad_peaking(freq: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """Peaking (bell) filter as a single second-order section."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    alpha = np.sin(w0) / (2.0 * max(q, 1e-3))
    cos_w0 = np.cos(w0)

    b = [1 + alpha * A, -2 * cos_w0, 1 - alpha * A]
    a = [1 + alpha / A, -2 * cos_w0, 1 - alpha / A]
    return np.array([[*(np.array(b) / a[0]), *(np.array(a) / a[0])]])


def _biquad_shelf(
    freq: float, gain_db: float, q: float, sr: int, kind: str = "low"
) -> np.ndarray:
    """Low or high shelf."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1 / A) * (1 / max(q, 1e-3) - 1) + 2)
    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha

    if kind == "low":
        b = [
            A * ((A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha),
            2 * A * ((A - 1) - (A + 1) * cos_w0),
            A * ((A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha),
        ]
        a = [
            (A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha,
            -2 * ((A - 1) + (A + 1) * cos_w0),
            (A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha,
        ]
    else:
        b = [
            A * ((A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha),
            -2 * A * ((A - 1) + (A + 1) * cos_w0),
            A * ((A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha),
        ]
        a = [
            (A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha,
            2 * ((A - 1) - (A + 1) * cos_w0),
            (A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha,
        ]
    return np.array([[*(np.array(b) / a[0]), *(np.array(a) / a[0])]])


def _apply_sos(audio: np.ndarray, sos: np.ndarray) -> np.ndarray:
    axis = 0
    return sosfilt(sos, audio, axis=axis).astype(np.float32)


def highpass(audio: np.ndarray, freq: float, sr: int = SAMPLE_RATE, order: int = 4):
    """Butterworth high-pass. The single most useful move in a home mix."""
    if not freq or freq <= 0:
        return audio
    freq = min(freq, sr / 2.0 * 0.95)
    return _apply_sos(audio, butter(order, freq, btype="highpass", fs=sr, output="sos"))


def lowpass(audio: np.ndarray, freq: float, sr: int = SAMPLE_RATE, order: int = 4):
    if not freq or freq <= 0:
        return audio
    freq = min(freq, sr / 2.0 * 0.95)
    return _apply_sos(audio, butter(order, freq, btype="lowpass", fs=sr, output="sos"))


def peaking(
    audio: np.ndarray, freq: float, gain_db: float, q: float = 1.0, sr: int = SAMPLE_RATE
) -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return audio
    freq = min(max(freq, 10.0), sr / 2.0 * 0.95)
    return _apply_sos(audio, _biquad_peaking(freq, gain_db, q, sr))


def low_shelf(
    audio: np.ndarray, freq: float, gain_db: float, q: float = 0.7, sr: int = SAMPLE_RATE
) -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return audio
    return _apply_sos(audio, _biquad_shelf(freq, gain_db, q, sr, "low"))


def high_shelf(
    audio: np.ndarray, freq: float, gain_db: float, q: float = 0.7, sr: int = SAMPLE_RATE
) -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return audio
    return _apply_sos(audio, _biquad_shelf(freq, gain_db, q, sr, "high"))


def apply_eq_chain(
    audio: np.ndarray,
    hpf_hz: float = 0.0,
    lpf_hz: float = 0.0,
    bands: list[dict] | None = None,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Run the full EQ chain for one stem: HPF -> LPF -> peaking/shelf bands.

    Each band dict: {"freq": Hz, "gain_db": dB, "q": Q, "type": peaking|low_shelf|high_shelf}
    """
    out = audio
    if hpf_hz:
        out = highpass(out, hpf_hz, sr)
    if lpf_hz:
        out = lowpass(out, lpf_hz, sr)

    for band in bands or []:
        kind = band.get("type", "peaking")
        f = float(band.get("freq", 1000.0))
        g = float(band.get("gain_db", 0.0))
        q = float(band.get("q", 1.0))
        if kind == "low_shelf":
            out = low_shelf(out, f, g, q, sr)
        elif kind == "high_shelf":
            out = high_shelf(out, f, g, q, sr)
        else:
            out = peaking(out, f, g, q, sr)
    return out


def dynamic_eq(
    audio: np.ndarray, freq: float, threshold_db: float, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Frequency-selective compression (e.g. tame a resonance only when it rings).

    # TODO: split the band with a Linkwitz-Riley crossover, run the band through
    # the compressor in dynamics.py, recombine. Needs phase-coherent crossover
    # design or the recombination will comb-filter.
    """
    pass  # TODO
    return audio


def deesser(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Sibilance control for vocals.

    # TODO: implement properly as a split-band dynamic EQ around 5-9 kHz with a
    # sidechain detector. The static -2 dB notch in the vocal role treatment is
    # a placeholder, not a de-esser.
    """
    pass  # TODO
    return audio
