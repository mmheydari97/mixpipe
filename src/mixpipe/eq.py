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


# --------------------------------------------------------------------------- #
# crossovers and band-split dynamics
# --------------------------------------------------------------------------- #


def lr4_split(
    audio: np.ndarray, freq: float, sr: int = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray]:
    """Split into (low, high) with a Linkwitz-Riley 4th-order crossover.

    Each side is a 2nd-order Butterworth run twice, so both are -6 dB at `freq`
    and in phase there; low + high sums to a flat magnitude (an all-pass), which
    is what makes recombining after separate processing free of comb filtering.
    """
    freq = float(np.clip(freq, 10.0, sr / 2.0 * 0.95))
    lo = butter(2, freq, btype="lowpass", fs=sr, output="sos")
    hi = butter(2, freq, btype="highpass", fs=sr, output="sos")
    x = np.asarray(audio, dtype=np.float64)
    low = sosfilt(lo, sosfilt(lo, x, axis=0), axis=0)
    high = sosfilt(hi, sosfilt(hi, x, axis=0), axis=0)
    return low, high


def split_bands(
    audio: np.ndarray, crossovers: list[float], sr: int = SAMPLE_RATE
) -> list[np.ndarray]:
    """Split into len(crossovers) + 1 bands, lowest first, that sum back flat.

    The bands are split off one crossover at a time. Each earlier band is also
    passed through the all-pass the *later* crossovers apply to the remainder
    (its own low + high), so every band has been rotated by the same total phase
    and their sum is an all-pass of the input, not a comb.
    """
    bands: list[np.ndarray] = []
    rest = np.asarray(audio, dtype=np.float64)
    for f in sorted(float(c) for c in crossovers):
        low, high = lr4_split(rest, f, sr)
        bands = [sum(lr4_split(b, f, sr)) for b in bands]
        bands.append(low)
        rest = high
    bands.append(rest)
    return bands


def _band_edges(freq: float, q: float) -> tuple[float, float]:
    """Lower and upper edge of a band of centre `freq` and quality `q`.

    The edges are the -3 dB points of a band-pass with that Q: geometric mean
    `freq`, width freq/q.
    """
    half = 1.0 / (2.0 * max(q, 0.1))
    root = np.sqrt(1.0 + half * half)
    return freq * (root - half), freq * (root + half)


def _split_mid(audio: np.ndarray, freq: float, q: float, sr: int):
    """(below, band, above) around `freq`; the three sum back flat."""
    lo_f, hi_f = _band_edges(freq, q)
    below, band, above = split_bands(audio, [lo_f, hi_f], sr)
    return below, band, above


def dynamic_eq(
    audio: np.ndarray, freq: float, threshold_db: float, sr: int = SAMPLE_RATE,
    ratio: float = 4.0, q: float = 1.5, attack_ms: float = 3.0,
    release_ms: float = 60.0, knee_db: float = 6.0, max_reduction_db: float = 12.0,
) -> np.ndarray:
    """Frequency-selective compression: tame a resonance only when it rings.

    The band around `freq` (width freq/q) is split off with a phase-coherent
    crossover and compressed on its own level, while the rest of the spectrum
    passes untouched. Outside the moments the band crosses `threshold_db` (a
    level in dBFS of the band signal itself) nothing changes, which is the
    difference from a static cut. `max_reduction_db` caps the cut.
    """
    from .dynamics import compress

    if audio.size == 0 or ratio <= 1.0:
        return audio
    below, band, above = _split_mid(audio, freq, q, sr)
    _, reduction_db = compress(
        band.astype(np.float32), threshold_db=threshold_db, ratio=ratio,
        attack_ms=attack_ms, release_ms=release_ms, knee_db=knee_db, sr=sr,
        return_reduction=True,
    )
    gain = 10.0 ** (np.maximum(reduction_db, -abs(max_reduction_db)) / 20.0)
    band = band * (gain if band.ndim == 1 else gain[:, None])
    return (below + band + above).astype(np.float32)


def _auto_threshold_db(band: np.ndarray, full: np.ndarray, sr: int, pct: float) -> float:
    """Level that the loudest `100 - pct` % of the band's active frames exceed.

    "Active" means the full-band signal is above -50 dBFS, so silence between
    phrases doesn't drag the percentile down.
    """
    mono_b = band if band.ndim == 1 else band.mean(axis=1)
    mono_f = full if full.ndim == 1 else full.mean(axis=1)
    hop = max(1, int(0.01 * sr))
    n = mono_b.size // hop
    if n < 4:
        return -30.0
    rms = lambda x: np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-20)  # noqa: E731
    band_db = 20.0 * np.log10(rms(mono_b))
    full_db = 20.0 * np.log10(rms(mono_f))
    active = full_db > -50.0
    if active.sum() < 4:
        return float(np.percentile(band_db, pct))
    return float(np.percentile(band_db[active], pct))


def deesser(
    audio: np.ndarray, sr: int = SAMPLE_RATE, freq: float = 6500.0,
    threshold_db: float | None = None, ratio: float = 5.0,
    max_reduction_db: float = 8.0, q: float = 1.3, auto_percentile: float = 90.0,
) -> np.ndarray:
    """Sibilance control for vocals.

    A split-band compressor around `freq` (default 6.5 kHz, roughly 5-9 kHz at
    Q 1.3) with its detector on that same band, so it ducks only the "s" and "t"
    energy and leaves the rest of the voice alone. Pass `threshold_db` in dBFS to
    fix it. With the default None the threshold is measured from the take: the
    level the loudest 10 % of active frames reach in the band, so a take with
    little sibilance isn't squashed and a hissy one is. `max_reduction_db` caps
    the cut.
    """
    from .dynamics import compress

    if audio.size == 0 or max_reduction_db <= 0:
        return audio
    below, band, above = _split_mid(audio, freq, q, sr)
    if threshold_db is None:
        threshold_db = _auto_threshold_db(band, audio, sr, auto_percentile)
    _, reduction_db = compress(
        band.astype(np.float32), threshold_db=threshold_db, ratio=ratio,
        attack_ms=1.0, release_ms=40.0, knee_db=4.0, sr=sr, return_reduction=True,
    )
    gain = 10.0 ** (np.maximum(reduction_db, -abs(max_reduction_db)) / 20.0)
    band = band * (gain if band.ndim == 1 else gain[:, None])
    return (below + band + above).astype(np.float32)
