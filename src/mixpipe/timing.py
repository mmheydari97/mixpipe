"""Timing alignment, tempo and quantization.

In order of how much they improve a home recording:

  1. Latency compensation - stems recorded in separate passes through an
     interface with different buffer settings can sit tens of ms apart.
     `estimate_offset_samples` cross-correlates onset envelopes (so two
     instruments playing different material still line up on their rhythm) and
     `align_stems` applies the integer-sample shifts.
  2. Tempo estimation - `estimate_tempo_bpm`, to build a grid.
  3. Onset quantization - `quantize_onsets` moves note onsets toward that grid
     with a pitch-preserving time warp.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, correlate, find_peaks, sosfilt

from . import SAMPLE_RATE
from .io_utils import Stem

# flux frame i spans [i * hop, i * hop + 1024). The flux peaks when the window's
# centre-weighted part reaches the onset, which measured 0.78 of a window after the
# frame start on sine and noise bursts alike.
_N_FFT = 1024
_ONSET_BIAS_FRAC = 0.78
_NEAR_PEAK_FRAC = 0.8       # a peak this close to the tallest one competes on distance from 0


def _mono(audio: np.ndarray) -> np.ndarray:
    return (audio if audio.ndim == 1 else audio.mean(axis=1)).astype(np.float64)


def onset_envelope(
    audio: np.ndarray, sr: int = SAMPLE_RATE, rate_hz: float = 375.0
) -> tuple[np.ndarray, int]:
    """Onset strength as log-magnitude spectral flux.

    Returns (envelope, hop), one value per `hop` samples (about sr / rate_hz).
    Each frame is a 1024-point Hann-windowed FFT; magnitudes are log-compressed
    (a quiet note counts as much as a loud one) and the flux is the summed
    positive change from the previous frame, bins limited to 40 Hz - 16 kHz.
    Unlike band-energy envelopes, a steady tone produces no flux once it is
    sounding, so filter build-up and waveform ripple don't masquerade as onsets.
    The value at index i describes the change into the frame starting at
    i * hop, i.e. the audio from (i * hop) to (i * hop + n_fft).
    """
    n_fft = _N_FFT
    hop = max(16, int(sr / rate_hz))
    x = _mono(audio)
    if x.size < n_fft + 8 * hop:
        return np.zeros(1), hop
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    keep = (freqs >= 40.0) & (freqs <= min(16000.0, sr * 0.45))
    win = np.hanning(n_fft)
    n_frames = (x.size - n_fft) // hop + 1
    scale = float(np.max(np.abs(x))) + 1e-12
    flux = np.zeros(n_frames)
    prev = None
    chunk = 2048
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        idx = (np.arange(c0, c1)[:, None] * hop) + np.arange(n_fft)[None, :]
        mag = np.abs(np.fft.rfft(x[idx] * win, axis=1))[:, keep] / (n_fft / 4 * scale)
        logmag = np.log1p(1000.0 * mag)
        before = np.vstack([prev if prev is not None else logmag[:1], logmag[:-1]])
        flux[c0:c1] = np.maximum(logmag - before, 0.0).sum(axis=1)
        prev = logmag[-1:]
    return flux, hop


# --------------------------------------------------------------------------- #
# latency / offset
# --------------------------------------------------------------------------- #


def _offset_and_confidence(
    reference: np.ndarray, target: np.ndarray, max_lag_ms: float, sr: int
) -> tuple[int, float]:
    ref_env, hop = onset_envelope(reference, sr)
    tgt_env, _ = onset_envelope(target, sr)
    if ref_env.std() < 1e-9 or tgt_env.std() < 1e-9:
        return 0, 0.0
    a, b = ref_env - ref_env.mean(), tgt_env - tgt_env.mean()
    corr = correlate(b, a, mode="full", method="fft")      # peak at +L: target lags by L
    lags = np.arange(-(a.size - 1), b.size)
    max_lag = int(max_lag_ms / 1000.0 * sr / hop)
    window = np.abs(lags) <= max_lag
    c, l = corr[window], lags[window]
    k = int(np.argmax(c))
    spread = c.std()
    med = float(np.median(c))
    confidence = float((c[k] - med) / spread) if spread > 0 else 0.0
    # Rhythmic material repeats every beat and bar, so the tallest peak can be a whole
    # beat away from the true latency. Among peaks nearly as tall, take the one
    # closest to zero: a recording latency is far more likely small than a beat.
    peaks, _ = find_peaks(c, height=med + _NEAR_PEAK_FRAC * (c[k] - med))
    if peaks.size:
        k = int(peaks[np.argmin(np.abs(l[peaks]))])
    offset = int(l[k]) * hop
    return _refine_offset(reference, target, offset, hop, sr), confidence


def _refine_offset(
    reference: np.ndarray, target: np.ndarray, coarse: int, hop: int, sr: int
) -> int:
    """Sample-accurate refinement for stems that carry the same source.

    The onset envelope only knows the lag to within a hop. If the two stems are
    correlated at the waveform level (a DI and a mic on the same guitar, two
    takes of the same riff) the raw cross-correlation peak next to the coarse
    lag gives the exact sample; otherwise the coarse value stands.
    """
    ref, tgt = _mono(reference), _mono(target)
    sos = butter(2, [100.0, min(4000.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    ref, tgt = sosfilt(sos, ref), sosfilt(sos, tgt)
    span = 2 * hop
    seg_len = min(ref.size, 12 * sr)
    if seg_len < sr // 4:
        return coarse
    # the loudest stretch of the reference
    block = max(1, sr // 10)
    frames = ref[: ref.size // block * block].reshape(-1, block)
    power = np.convolve((frames ** 2).sum(axis=1), np.ones(max(1, seg_len // block)),
                        mode="valid")
    start = int(np.argmax(power)) * block
    r = ref[start:start + seg_len]
    t0 = start + coarse - span
    if t0 < 0 or t0 + seg_len + 2 * span > tgt.size:
        return coarse
    t = tgt[t0:t0 + seg_len + 2 * span]
    c = correlate(t, r, mode="valid", method="fft")        # lags -span .. +span
    k = int(np.argmax(c))
    denom = np.sqrt(np.sum(r * r) * np.sum(t[k:k + r.size] ** 2))
    if denom > 0 and c[k] / denom > 0.3:
        return coarse + (k - span)
    return coarse


def estimate_offset_samples(
    reference: np.ndarray, target: np.ndarray, max_lag_ms: float = 500.0,
    sr: int = SAMPLE_RATE,
) -> int:
    """How many samples `target` lags `reference` (negative: it leads).

    Cross-correlates onset-strength envelopes, not raw samples: raw correlation
    locks onto the wrong peak when the two instruments play different material,
    while their rhythm still agrees. Refined to the sample when the waveforms
    themselves correlate. Returns 0 when neither stem has any onsets.
    """
    return _offset_and_confidence(reference, target, max_lag_ms, sr)[0]


def shift_samples(audio: np.ndarray, samples: int) -> np.ndarray:
    """Delay by `samples` (negative advances), keeping the length."""
    n = audio.shape[0]
    if samples == 0 or n == 0:
        return audio
    out = np.zeros_like(audio)
    if samples > 0:
        out[samples:] = audio[: max(0, n - samples)]
    else:
        out[: n + samples] = audio[-samples:]
    return out


def pick_reference(stems: list[Stem]) -> Stem:
    """The stem to align the others to: drums if there are any (the clearest
    onsets), else the first one."""
    for s in stems:
        if s.role == "drums":
            return s
    return stems[0]


def align_stems(
    stems: list[Stem], reference: str | None = None, max_lag_ms: float = 500.0,
    min_confidence: float = 6.0,
) -> list[Stem]:
    """Shift every stem so they share a timeline.

    Each stem is correlated with the reference (a stem name, default: the drums,
    else the first stem) and moved by the integer number of samples that lines
    its onsets up. A stem whose correlation peak isn't clearly above the noise
    (`min_confidence` standard deviations) is left where it is, so unrelated or
    featureless material is never shifted on a guess. Each returned stem records
    `meta["align_offset_samples"]` and `meta["align_confidence"]`. Stems are not
    trimmed, so all keep their length; a stem moved earlier loses its first
    samples, which for a latency offset is silence.
    """
    if len(stems) < 2:
        return stems
    ref = next((s for s in stems if s.name == reference), None) if reference else None
    if reference and ref is None:
        raise ValueError(f"reference stem not found: {reference}")
    ref = ref or pick_reference(stems)

    out = []
    for s in stems:
        if s is ref:
            kept = s.copy_with(s.audio)
            kept.meta.update({"align_offset_samples": 0, "align_confidence": None})
            out.append(kept)
            continue
        offset, conf = _offset_and_confidence(ref.audio, s.audio, max_lag_ms, s.sr)
        if conf >= min_confidence and offset != 0:
            moved = s.copy_with(shift_samples(s.audio, -offset))
        else:
            moved = s.copy_with(s.audio)
            offset = 0
        moved.meta.update({"align_offset_samples": int(offset),
                           "align_confidence": round(conf, 1)})
        out.append(moved)
    return out


# --------------------------------------------------------------------------- #
# onsets and tempo
# --------------------------------------------------------------------------- #


def detect_onsets(
    audio: np.ndarray, sr: int = SAMPLE_RATE, min_gap_ms: float = 50.0,
    sensitivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Note onsets as (sample positions, strengths).

    Peaks of the onset envelope above an adaptive threshold (local median plus
    a multiple of the spread), at least `min_gap_ms` apart. Higher `sensitivity`
    finds more, weaker onsets.
    """
    env, hop = onset_envelope(audio, sr)
    if env.size < 8 or env.max() <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    k = max(3, int(0.4 * sr / hop)) | 1
    kernel = np.ones(k) / k
    local = np.convolve(env, kernel, mode="same")
    height = local + (2.0 / max(sensitivity, 0.1)) * env.std()
    peaks, props = find_peaks(env, height=height, distance=max(1, int(min_gap_ms / 1000 * sr / hop)))
    # the envelope is a difference, so its peak sits a frame after the rise begins
    return peaks.astype(np.int64) * hop + int(_ONSET_BIAS_FRAC * _N_FFT), props["peak_heights"]


def estimate_tempo_bpm(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    """Tempo in BPM from the autocorrelation of the onset envelope, or None.

    The autocorrelation is weighted toward 120 BPM (a log-normal prior a couple
    of octaves wide), which picks between a tempo and its double or half. The
    result is then refined with the multiples of the beat period, whose peaks
    pin the value down much more tightly than the first one. None for material
    shorter than 4 s or with no steady pulse.
    """
    x = _mono(audio)
    if x.size < 4 * sr:
        return None
    env, hop = onset_envelope(x, sr, rate_hz=200.0)
    rate = sr / hop
    if env.std() < 1e-9:
        return None
    e = env - env.mean()
    n = 1 << int(np.ceil(np.log2(e.size * 2)))
    spec = np.fft.rfft(e, n)
    acf = np.fft.irfft(np.abs(spec) ** 2, n)[: e.size]
    acf /= acf[0] + 1e-12
    # tempo on the median-smoothed curve of lags covering 50-220 BPM
    lo, hi = int(rate * 60.0 / 220.0), int(rate * 60.0 / 50.0)
    if hi >= acf.size - 2 or lo < 2:
        return None
    lags = np.arange(lo, hi)
    bpm = 60.0 * rate / lags
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 1.0) ** 2)
    score = acf[lo:hi] * prior
    best = int(np.argmax(score))
    lag = float(lags[best])
    if acf[int(lag)] < 0.05:
        return None
    # parabolic interpolation of the first peak
    i = int(lag)
    y0, y1, y2 = acf[i - 1], acf[i], acf[i + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) > 1e-12:
        lag = i + 0.5 * (y0 - y2) / denom
    # refine with the highest multiple of the period that still has a clear peak
    base = lag
    for mult in range(2, 9):
        centre = base * mult
        if centre * 1.03 >= acf.size - 2:
            break
        a, b = int(centre * 0.97), int(centre * 1.03) + 1
        j = a + int(np.argmax(acf[a:b]))
        if acf[j] < 0.3 * acf[int(round(base))] or j <= 1:
            break
        y0, y1, y2 = acf[j - 1], acf[j], acf[j + 1]
        d = y0 - 2 * y1 + y2
        peak = j + (0.5 * (y0 - y2) / d if abs(d) > 1e-12 else 0.0)
        lag = peak / mult
    return float(60.0 * rate / lag)


# --------------------------------------------------------------------------- #
# quantization
# --------------------------------------------------------------------------- #


def grid_step_s(bpm: float, grid: str = "1/16") -> float:
    """Length of one grid cell in seconds. "1/4" is a beat; a trailing "T" makes
    it a triplet ("1/8T" = three per beat)."""
    g = grid.strip().upper()
    triplet = g.endswith("T")
    g = g.rstrip("T")
    try:
        num, den = g.split("/")
        note = float(num) / float(den)              # 1/4 = a quarter note
    except ValueError as exc:
        raise ValueError(f"grid must look like 1/16 or 1/8T, got {grid!r}") from exc
    if note <= 0:
        raise ValueError(f"grid must be positive, got {grid!r}")
    step = (60.0 / bpm) * (note / 0.25)             # in quarter-note beats
    return step * (2.0 / 3.0 if triplet else 1.0)


def quantize_onsets(
    audio: np.ndarray, bpm: float, grid: str = "1/16", strength: float = 0.5,
    sr: int = SAMPLE_RATE, sensitivity: float = 1.0,
) -> np.ndarray:
    """Move note onsets toward a rhythmic grid, keeping pitch and length.

    Onsets are detected (`detect_onsets`), the grid's phase is fitted to them (the
    take doesn't have to start on the grid), and each onset is moved
    `strength` of the way (0 = leave, 1 = snap) to its nearest grid line. The
    audio between onsets is stretched or squeezed to follow with a pitch-
    preserving time warp (`tsm.time_warp`), restarting the phase vocoder at each
    onset so attacks stay sharp. Local speed changes are held between 0.5x and 2x,
    and the output has the same length as the input.
    """
    from . import tsm

    if bpm is None or bpm <= 0:
        raise ValueError("bpm must be positive")
    strength = float(np.clip(strength, 0.0, 1.0))
    n = audio.shape[0]
    onsets, weights = detect_onsets(audio, sr, sensitivity=sensitivity)
    if strength == 0.0 or onsets.size < 2 or n < 4096:
        return audio

    step = grid_step_s(bpm, grid) * sr
    # grid phase: weighted circular mean of where onsets fall inside a cell
    angle = 2.0 * np.pi * (onsets % step) / step
    phase = np.angle(np.sum(weights * np.exp(1j * angle))) / (2.0 * np.pi) * step
    snapped = phase + np.round((onsets - phase) / step) * step
    target = onsets + strength * (snapped - onsets)

    # keep the map monotone with bounded local speed (0.5x .. 2x)
    dst = np.empty_like(target, dtype=np.float64)
    dst[0] = target[0]
    for i in range(1, onsets.size):
        gap = onsets[i] - onsets[i - 1]
        dst[i] = min(max(target[i], dst[i - 1] + 0.5 * gap), dst[i - 1] + 2.0 * gap)

    src_anchors = np.r_[0.0, onsets.astype(np.float64), float(n)]
    dst_anchors = np.r_[0.0, dst, float(n)]
    if dst_anchors[-2] >= n:                       # an onset pushed past the end
        return audio

    hop = tsm.HOP
    m = int(np.ceil(n / hop)) + 1
    positions = np.interp(np.arange(m) * hop, dst_anchors, src_anchors)
    return tsm.time_warp(audio, positions, n_out=n, reset_at=onsets)
