"""Measurement. Produces the numbers the advisor reasons about.

Nothing here modifies audio. Every function is a pure measurement so the output
can be serialized to JSON and handed to an LLM, or diffed between runs.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.signal import stft

from . import SAMPLE_RATE
from .io_utils import Stem

try:
    import pyloudnorm

    HAVE_PYLOUDNORM = True
except ImportError:  # pragma: no cover
    pyloudnorm = None
    HAVE_PYLOUDNORM = False


EPS = 1e-12

# Bands used for the spectral balance report, in Hz.
BANDS: dict[str, tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "low": (60.0, 200.0),
    "low_mid": (200.0, 600.0),
    "mid": (600.0, 2000.0),
    "high_mid": (2000.0, 6000.0),
    "high": (6000.0, 12000.0),
    "air": (12000.0, 20000.0),
}


def db(x: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), EPS))


def from_db(x: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(x, dtype=np.float64) / 20.0)


def peak_db(audio: np.ndarray) -> float:
    return float(db(np.max(np.abs(audio)) if audio.size else EPS))


@functools.lru_cache(maxsize=4)
def _tp_filter(oversample: int) -> np.ndarray:
    """Interpolation filter for true-peak metering: 12 taps per phase, as in
    ITU-R BS.1770 (scipy's default for 4x is twice as long, for no gain here)."""
    from scipy.signal import firwin

    return firwin(12 * oversample + 1, 1.0 / oversample, window=("kaiser", 5.0)).astype(np.float32)


def true_peak_detector(audio: np.ndarray, oversample: int = 4) -> np.ndarray:
    """Per-sample peak that includes the inter-sample peaks after reconstruction.

    The signal is upsampled `oversample`x (the way a DAC or a lossy encoder's
    filter would see it) and, for each original sample, the largest value among
    the oversampled points from that sample up to the next, across channels, is
    returned. Done in chunks so a long track doesn't need 4x its size in memory.
    """
    from scipy.signal import resample_poly

    a = audio[:, None] if audio.ndim == 1 else audio
    n = a.shape[0]
    out = np.empty(n)
    chunk, pad = 1 << 18, 64
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        lo, hi = max(0, s - pad), min(n, e + pad)
        up = np.abs(resample_poly(a[lo:hi].astype(np.float32), oversample, 1, axis=0,
                                  window=_tp_filter(oversample)))
        # elementwise maxima: numpy's reduce over a 2- or 4-wide axis is very slow
        up = np.maximum.reduce([up[:, c] for c in range(up.shape[1])]) if up.shape[1] > 1 else up[:, 0]
        seg = up[(s - lo) * oversample:(e - lo) * oversample]
        out[s:e] = np.maximum.reduce([seg[k::oversample] for k in range(oversample)])
    return out


def true_peak_db(audio: np.ndarray, oversample: int = 4) -> float:
    """True peak in dBTP (4x oversampled), at least the sample peak."""
    if audio.size == 0:
        return float(db(EPS))
    return max(peak_db(audio), float(db(true_peak_detector(audio, oversample).max())))


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float(db(EPS))
    return float(db(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))


def crest_factor_db(audio: np.ndarray) -> float:
    """Peak-to-RMS. High = spiky/dynamic, low = squashed or sustained."""
    return peak_db(audio) - rms_db(audio)


def integrated_lufs(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """ITU-R BS.1770 integrated loudness if pyloudnorm is available.

    The fallback is *not* LUFS. It's K-weighting-free RMS with a fixed offset
    that lands in roughly the right neighbourhood for typical music. Good
    enough for relative balancing, not good enough to claim a LUFS number.
    """
    if audio.size == 0:
        return -70.0
    if HAVE_PYLOUDNORM:
        meter = pyloudnorm.Meter(sr)
        a = audio if audio.ndim == 1 else audio
        try:
            return float(meter.integrated_loudness(a))
        except ValueError:
            # pyloudnorm raises on clips shorter than its block size
            return rms_db(audio) - 3.0
    return rms_db(audio) - 3.0


def short_term_lufs(
    audio: np.ndarray, sr: int = SAMPLE_RATE, window_s: float = 3.0
) -> list[float]:
    """Loudness over time, so the advisor can see arrangement dynamics."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    hop = int(window_s * sr)
    if hop <= 0 or mono.size < hop:
        return [integrated_lufs(mono, sr)]
    return [
        integrated_lufs(mono[i : i + hop], sr)
        for i in range(0, mono.size - hop + 1, hop)
    ]


def band_energy_db(
    audio: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float]:
    """Energy per frequency band, normalized so the loudest band reads 0 dB."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if mono.size < 1024:
        return {k: -70.0 for k in BANDS}

    nper = min(4096, 1 << int(np.floor(np.log2(mono.size))))
    freqs, _, Z = stft(mono, fs=sr, nperseg=nper, noverlap=nper // 2)
    psd = np.mean(np.abs(Z) ** 2, axis=1)

    out: dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = float(db(np.sqrt(psd[sel].sum())) if sel.any() else -70.0)

    top = max(out.values())
    return {k: round(v - top, 2) for k, v in out.items()}


def spectral_centroid_hz(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Brightness. Low = dark/muddy, high = thin/bright."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if mono.size < 1024:
        return 0.0
    nper = min(4096, 1 << int(np.floor(np.log2(mono.size))))
    freqs, _, Z = stft(mono, fs=sr, nperseg=nper, noverlap=nper // 2)
    mag = np.abs(Z).mean(axis=1)
    total = mag.sum()
    return float((freqs * mag).sum() / total) if total > EPS else 0.0


def dc_offset(audio: np.ndarray) -> float:
    return float(np.mean(audio)) if audio.size else 0.0


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at or beyond full scale. Non-zero = already clipped."""
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


def estimate_noise_floor_db(
    audio: np.ndarray, sr: int = SAMPLE_RATE, percentile: float = 10.0
) -> float:
    """Noise floor from the quietest frames. Tells you if a take needs denoise."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    win = max(256, int(0.02 * sr))
    n_frames = mono.size // win
    if n_frames < 2:
        return rms_db(mono)
    frames = mono[: n_frames * win].reshape(n_frames, win)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return float(db(np.percentile(frame_rms, percentile)))


def silence_lead_ms(
    audio: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = -50.0
) -> float:
    """Leading silence, used both for trimming and as a noise profile."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    thresh = from_db(threshold_db)
    above = np.nonzero(np.abs(mono) > thresh)[0]
    if above.size == 0:
        return mono.size / sr * 1000.0
    return float(above[0] / sr * 1000.0)


def analyze_stem(stem: Stem) -> dict:
    """Full measurement set for one stem."""
    a = stem.audio
    mono = stem.mono()
    return {
        "name": stem.name,
        "role": stem.role,
        "duration_s": round(stem.duration, 3),
        "channels": stem.channels,
        "source_sr": stem.meta.get("source_sr", stem.sr),
        "peak_db": round(peak_db(a), 2),
        "rms_db": round(rms_db(a), 2),
        "lufs": round(integrated_lufs(mono, stem.sr), 2),
        "crest_db": round(crest_factor_db(a), 2),
        "dc_offset": round(dc_offset(a), 6),
        "clipping_ratio": round(clipping_ratio(a), 6),
        "noise_floor_db": round(estimate_noise_floor_db(mono, stem.sr), 2),
        "snr_db": round(
            rms_db(mono) - estimate_noise_floor_db(mono, stem.sr), 2
        ),
        "lead_silence_ms": round(silence_lead_ms(mono, stem.sr), 1),
        "spectral_centroid_hz": round(spectral_centroid_hz(mono, stem.sr), 1),
        "band_energy_db": band_energy_db(mono, stem.sr),
        "lufs_estimated": not HAVE_PYLOUDNORM,
    }


SILENT_DB = -60.0   # a stem quieter than this (LUFS, or RMS dBFS) holds no music, only bleed or noise


def is_silent(stem_analysis: dict) -> bool:
    """A stem that is effectively empty: a separated bass stem of a song without a
    bass, a track muted at the source. Level matching would boost its noise."""
    lufs, rms = stem_analysis.get("lufs", 0.0), stem_analysis.get("rms_db", 0.0)
    return not np.isfinite(lufs) or lufs < SILENT_DB or rms < SILENT_DB


def analyze_all(stems: list[Stem]) -> dict:
    """Per-stem analysis plus a cross-stem summary."""
    per_stem = [analyze_stem(s) for s in stems]
    for s in per_stem:
        s["silent"] = is_silent(s)
    audible = [s for s in per_stem if not s["silent"]] or per_stem
    lufs = [s["lufs"] for s in audible if np.isfinite(s["lufs"])]
    return {
        "sample_rate": SAMPLE_RATE,
        "n_stems": len(stems),
        "stems": per_stem,
        "summary": {
            "longest_s": max((s["duration_s"] for s in per_stem), default=0.0),
            "loudest_stem": (
                max(audible, key=lambda s: s["lufs"])["name"] if audible else None
            ),
            "quietest_stem": (
                min(audible, key=lambda s: s["lufs"])["name"] if audible else None
            ),
            "lufs_spread_db": round(max(lufs) - min(lufs), 2) if lufs else 0.0,
            "any_clipped": any(s["clipping_ratio"] > 0 for s in per_stem),
            "noisy_stems": [s["name"] for s in per_stem if s["snr_db"] < 20.0 and not s["silent"]],
            "silent_stems": [s["name"] for s in per_stem if s["silent"]],
        },
    }


def analyze_master(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Measurements of a finished stereo (or mono) mix, for the critique loop.

    Loudness (integrated and its range over 3 s windows), peak and true peak,
    crest factor, spectral balance (each band relative to the loudest), spectral
    centroid, and the stereo picture: L/R correlation (+1 mono, 0 uncorrelated,
    negative = phase trouble in mono) and side-to-mid energy in dB.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    out = {
        "duration_s": round(audio.shape[0] / sr, 2),
        "lufs": round(integrated_lufs(mono, sr), 2),
        "peak_db": round(peak_db(audio), 2),
        "true_peak_db": round(true_peak_db(audio), 2),
        "crest_db": round(crest_factor_db(audio), 2),
        "spectral_centroid_hz": round(spectral_centroid_hz(mono, sr), 1),
        "band_energy_db": band_energy_db(mono, sr),
    }
    short = short_term_lufs(audio, sr)
    valid = [v for v in short if v > -70.0]
    out["loudness_range_db"] = round(float(np.percentile(valid, 95) - np.percentile(valid, 10)), 2) \
        if len(valid) >= 3 else 0.0
    if audio.ndim == 2 and audio.shape[1] == 2:
        left, right = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
        denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
        out["lr_correlation"] = round(float(np.sum(left * right) / denom), 3) if denom > 0 else 1.0
        mid, side = (left + right) / 2, (left - right) / 2
        out["side_to_mid_db"] = round(float(db(np.sqrt(np.mean(side**2)) / max(np.sqrt(np.mean(mid**2)), EPS))), 2)
    return out


def compare_balance(mix: dict, reference: dict) -> dict[str, float]:
    """Per-band difference (mix - reference, dB) of two `band_energy_db` profiles,
    after removing their overall offset so it reads as a tonal-balance error."""
    diff = {b: mix["band_energy_db"][b] - reference["band_energy_db"][b] for b in BANDS}
    offset = float(np.median(list(diff.values())))
    return {b: round(v - offset, 2) for b, v in diff.items()}


def _band_activity(mono: np.ndarray, sr: int, frame_s: float = 0.1) -> dict[str, np.ndarray]:
    """For each band, a boolean per frame: is the stem playing in this band now?

    Active means within 20 dB of the band's own loudest frame and above -70 dB.
    """
    nper = int(frame_s * sr)
    if mono.size < 2 * nper:
        return {b: np.ones(1, dtype=bool) for b in BANDS}
    freqs, _, Z = stft(mono, fs=sr, nperseg=nper, noverlap=0)
    power = np.abs(Z) ** 2
    out = {}
    for name, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        level = 10.0 * np.log10(power[sel].sum(axis=0) + 1e-20) if sel.any() else np.full(Z.shape[1], -200.0)
        out[name] = (level > level.max() - 20.0) & (level > -70.0)
    return out


def masking_report(stems: list[Stem], min_overlap: float = 0.3) -> list[dict]:
    """Pairs of stems fighting for the same frequency band *at the same time*.

    A band is contested when both stems carry it (within 6 dB of their own
    loudest band) and, frame by frame, they are both actually playing in it for
    at least `min_overlap` of the time the quieter-in-time stem is. A guitar that
    only plays in the verse and a harmonica that only plays in the solo share a
    band on paper but never meet, and are no longer flagged. Each entry has
    `overlap`, the per-band share of simultaneous activity, so the advisor can see
    how serious a clash is.
    """
    profiles = {s.name: band_energy_db(s.mono(), s.sr) for s in stems}
    activity = {s.name: _band_activity(s.mono(), s.sr) for s in stems}
    names = list(profiles)
    out: list[dict] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared, overlap = [], {}
            for band in BANDS:
                if not (profiles[a][band] > -6.0 and profiles[b][band] > -6.0):
                    continue
                xa, xb = activity[a][band], activity[b][band]
                n = min(xa.size, xb.size)
                xa, xb = xa[:n], xb[:n]
                denom = min(int(xa.sum()), int(xb.sum()))
                share = float((xa & xb).sum() / denom) if denom else 0.0
                if share >= min_overlap:
                    shared.append(band)
                    overlap[band] = round(share, 2)
            if shared:
                out.append({"stems": [a, b], "contested_bands": shared, "overlap": overlap})
    return out
