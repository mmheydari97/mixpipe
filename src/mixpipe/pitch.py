"""Pitch tracking, pitch correction and key detection.

  track_pitch      YIN f0 tracker (numpy only) with voicing confidence
  correct_pitch    snap a monophonic part to a scale: scale-note decision with
                   hysteresis, a retune-speed smoother, TD-PSOLA resynthesis
  shift_semitones  fixed shift of any material, length preserved (phase vocoder,
                   or Rubber Band when installed)
  detect_key       Krumhansl-Schmuckler key estimate

Correction works on *monophonic* material (a voice, a harmonica line, a lead
guitar). TD-PSOLA moves pitch by re-spacing pitch-synchronous grains, which
leaves the formants where they were, so a voice keeps its character. It cannot
retune chords; for polyphonic material use `shift_semitones`, which moves
everything by the same amount.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import SAMPLE_RATE

_YIN_FRAME = 2048          # samples analysed per frame (integration window = half)
_YIN_THRESHOLD = 0.15      # cumulative-mean-normalised difference below this = periodic


def track_pitch(
    audio: np.ndarray, sr: int = SAMPLE_RATE, hop_ms: float = 10.0,
    fmin: float = 70.0, fmax: float = 1000.0, voicing_threshold: float = 0.2,
    min_level_db: float = -50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (f0_hz, confidence) with f0 = NaN where unvoiced.

    YIN (de Cheveigne & Kawahara 2002): the difference function between the
    signal and a delayed copy is normalised by its running mean, the first dip
    below a threshold is taken as the period (so a sub-harmonic dip later on
    can't win), and the dip is refined by parabolic interpolation. Frame i is
    centred on sample i * hop. `confidence` is 1 minus the normalised difference
    at the chosen period; a frame is voiced when that difference is below
    `voicing_threshold` and the frame is louder than `min_level_db`. Isolated
    octave jumps are folded back using the median of the neighbouring frames.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    mono = mono.astype(np.float64)
    hop = max(1, int(round(hop_ms / 1000.0 * sr)))
    n_frames = mono.size // hop + 1
    w = _YIN_FRAME // 2                          # integration window = max lag
    tau_min = max(2, int(sr / fmax))
    tau_max = min(w - 2, int(sr / fmin))
    if tau_max <= tau_min + 2 or mono.size < w:
        return np.full(n_frames, np.nan), np.zeros(n_frames)

    padded = np.pad(mono, (_YIN_FRAME // 2, _YIN_FRAME))
    f0 = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    floor = 10.0 ** (min_level_db / 20.0)
    batch = 512
    for b0 in range(0, n_frames, batch):
        b1 = min(n_frames, b0 + batch)
        starts = np.arange(b0, b1) * hop                   # frame centre at start + N/2
        frames = np.stack([padded[st:st + _YIN_FRAME] for st in starts])
        x = frames[:, :w]
        # d(tau) = sum_j (x[j] - x[j + tau])^2 over the window, via FFT autocorrelation
        nfft = 1 << int(np.ceil(np.log2(_YIN_FRAME + w)))
        fx = np.fft.rfft(frames, nfft, axis=1)
        fw = np.fft.rfft(x, nfft, axis=1)
        r = np.fft.irfft(fx * np.conj(fw), nfft, axis=1)[:, :w]     # r[tau] = sum x[j] f[j+tau]
        sq = np.cumsum(frames ** 2, axis=1)
        e0 = sq[:, w - 1]                                            # energy of x[0:w]
        csum = np.concatenate([np.zeros((frames.shape[0], 1)), sq], axis=1)
        taus = np.arange(w)
        e_shift = csum[:, taus + w] - csum[:, taus]                  # energy of x[tau:tau+w]
        d = e0[:, None] + e_shift - 2.0 * r
        d[:, 0] = 0.0
        run = np.cumsum(d[:, 1:], axis=1)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * np.arange(1, w)[None, :] / np.maximum(run, 1e-12)
        rms = np.sqrt(e0 / w)

        for i in range(frames.shape[0]):
            c = cmnd[i]
            seg = c[tau_min:tau_max]
            below = np.nonzero(seg < _YIN_THRESHOLD)[0]
            if below.size:
                t = tau_min + int(below[0])
                while t + 1 < tau_max and c[t + 1] < c[t]:    # walk to the bottom of the dip
                    t += 1
            else:
                t = tau_min + int(np.argmin(seg))
            if c[t] >= voicing_threshold or rms[i] < floor:
                conf[b0 + i] = max(0.0, 1.0 - c[t]) if rms[i] >= floor else 0.0
                continue
            y0, y1, y2 = c[t - 1], c[t], c[t + 1]
            denom = y0 - 2.0 * y1 + y2
            shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            f0[b0 + i] = sr / (t + float(np.clip(shift, -1.0, 1.0)))
            conf[b0 + i] = 1.0 - c[t]
    return _fold_octave_jumps(f0), conf


def _fold_octave_jumps(f0: np.ndarray, radius: int = 5) -> np.ndarray:
    """Move a frame an octave (or two) if that puts it next to its neighbours."""
    out = f0.copy()
    voiced = np.nonzero(~np.isnan(f0))[0]
    for i in voiced:
        lo, hi = max(0, i - radius), min(f0.size, i + radius + 1)
        neigh = np.concatenate([f0[lo:i], f0[i + 1:hi]])
        neigh = neigh[~np.isnan(neigh)]
        if neigh.size < 3:
            continue
        ref = np.median(neigh)
        octaves = np.log2(f0[i] / ref)
        k = round(octaves)
        if k != 0 and abs(octaves - k) < 0.2:
            out[i] = f0[i] / (2.0 ** k)
    return out


# --------------------------------------------------------------------------- #
# scales
# --------------------------------------------------------------------------- #

_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6,
       "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
_SCALES = {
    "maj": (0, 2, 4, 5, 7, 9, 11), "major": (0, 2, 4, 5, 7, 9, 11),
    "min": (0, 2, 3, 5, 7, 8, 10), "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10), "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "pent_maj": (0, 2, 4, 7, 9), "pent_min": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10), "chromatic": tuple(range(12)),
}


def scale_pitch_classes(scale: str) -> list[int]:
    """Pitch classes (0-11) of a scale given as "A:min", "C:maj", "Bb:dorian",
    "E:pent_min", or "chromatic" (every semitone)."""
    name = scale.strip()
    if name.lower() == "chromatic":
        return list(range(12))
    root, _, mode = name.partition(":")
    if root.upper() not in _PC or mode.lower() not in _SCALES:
        raise ValueError(
            f"unknown scale {scale!r}: use e.g. 'C:maj', 'A:min', 'E:pent_min' "
            f"(modes: {', '.join(sorted(_SCALES))})"
        )
    r = _PC[root.upper()]
    return sorted({(r + step) % 12 for step in _SCALES[mode.lower()]})


def _nearest_scale_note(midi: float, pcs: list[int]) -> float:
    base = int(np.floor(midi))
    cands = [n for n in range(base - 3, base + 5) if n % 12 in pcs]
    return float(min(cands, key=lambda n: abs(n - midi)))


def _decide_targets(
    midi: np.ndarray, pcs: list[int], hold_frames: int, hysteresis: float = 0.65
) -> np.ndarray:
    """Target scale note per frame (NaN where unvoiced).

    A held note changes only when the smoothed pitch has moved more than
    `hysteresis` semitones from it, so vibrato around a note, or a bend that
    hasn't arrived yet, doesn't flip the target between neighbours.
    """
    out = np.full(midi.size, np.nan)
    voiced = ~np.isnan(midi)
    smooth = np.full(midi.size, np.nan)
    idx = np.nonzero(voiced)[0]
    for i in idx:
        lo, hi = max(0, i - hold_frames // 2), min(midi.size, i + hold_frames // 2 + 1)
        win = midi[lo:hi]
        smooth[i] = np.median(win[~np.isnan(win)])
    held = None
    prev = -10
    for i in idx:
        if held is None or i - prev > 3 or abs(smooth[i] - held) > hysteresis:
            held = _nearest_scale_note(smooth[i], pcs)
        out[i] = held
        prev = i
    return out


# --------------------------------------------------------------------------- #
# TD-PSOLA
# --------------------------------------------------------------------------- #


def _pitch_marks(mono: np.ndarray, f0_at, sr: int, start: int, stop: int) -> list[int]:
    """Pitch marks (one per period) across the voiced span [start, stop).

    The first sits on the strongest peak of the first period. Each next mark is
    predicted one period on and moved by up to +/-15 % of a period to wherever
    the waveform best matches the previous cycle, so successive grains stay
    cycle-aligned, which is what PSOLA needs and a peak-picker can't guarantee
    on a complex waveform.
    """
    f = f0_at(start)
    period = int(round(sr / f))
    first = start + int(np.argmax(np.abs(mono[start:start + period])))
    marks = [first]
    while True:
        prev = marks[-1]
        period = int(round(sr / f0_at(prev)))
        pred = prev + period
        if pred >= stop - period // 2:
            break
        reach = max(1, int(0.15 * period))
        half = max(2, period // 2)
        if prev - half < 0 or pred + reach + half >= mono.size:
            break
        ref = mono[prev - half:prev + half]
        cand = sliding_window_view(mono[pred - reach - half:pred + reach + half], 2 * half)
        scores = cand @ ref
        marks.append(pred - reach + int(np.argmax(scores)))
    return marks


def _psola(
    audio: np.ndarray, sr: int, f0: np.ndarray, hop: int, ratio_at, formant_preserve: bool,
) -> np.ndarray:
    """Resynthesise `audio` with local pitch ratio `ratio_at(sample)` (>1 = up)."""
    x = audio if audio.ndim == 2 else audio[:, None]
    n, n_ch = x.shape
    mono = x.mean(axis=1)
    voiced = ~np.isnan(f0)

    def f0_at(sample: int) -> float:
        i = int(np.clip(round(sample / hop), 0, f0.size - 1))
        j = i
        while j < f0.size and np.isnan(f0[j]):
            j += 1
        if j >= f0.size:
            j = i
            while j > 0 and np.isnan(f0[j]):
                j -= 1
        return float(f0[j]) if not np.isnan(f0[j]) else 200.0

    out = np.zeros((n + 4096, n_ch))
    wsum = np.zeros(n + 4096)

    def add_grain(center_src: int, half_src: int, center_dst: int, length_dst_half: int):
        """Window of 2*half_src around center_src, placed (optionally rescaled) at center_dst."""
        a, b = center_src - half_src, center_src + half_src
        if a < 0 or b > n:
            return
        grain = x[a:b]                                         # (2h, ch)
        if length_dst_half != half_src:
            m = 2 * length_dst_half
            pos = np.linspace(0, 2 * half_src - 1, m)
            grain = np.stack([np.interp(pos, np.arange(2 * half_src), grain[:, c])
                              for c in range(n_ch)], axis=1)
        m = grain.shape[0]
        win = np.hanning(m + 2)[1:-1]
        d0 = center_dst - m // 2
        if d0 < 0 or d0 + m > out.shape[0]:
            return
        out[d0:d0 + m] += grain * win[:, None]
        wsum[d0:d0 + m] += win

    # split into voiced runs and the gaps between them
    runs = []
    i = 0
    while i < voiced.size:
        if voiced[i]:
            j = i
            while j < voiced.size and voiced[j]:
                j += 1
            runs.append((i * hop - hop // 2, min(n, j * hop + hop // 2)))
            i = j
        else:
            i += 1

    cursor = 0
    grain_u = max(8, int(0.005 * sr))                          # unvoiced: 5 ms hop, 10 ms window
    for a, b in runs:
        a, b = max(a, 0), min(b, n)
        if b - a < 3 * sr / 1000.0 * 5:
            continue
        # unvoiced stretch before this run: plain overlap-add, identity
        s = cursor
        while s < a:
            add_grain(s, grain_u, s, grain_u)
            s += grain_u
        marks = _pitch_marks(mono, f0_at, sr, a, b)
        if len(marks) < 3:
            cursor = a
            continue
        marks_arr = np.array(marks)
        periods = np.gradient(marks_arr.astype(np.float64))
        s = float(marks[0])
        end = marks[-1]
        while s <= end:
            j = int(np.argmin(np.abs(marks_arr - s)))
            T = max(8, int(round(periods[j])))
            beta = float(ratio_at(int(s)))
            dst_half = T if formant_preserve else max(2, int(round(T / beta)))
            add_grain(int(marks_arr[j]), T, int(round(s)), dst_half)
            s += T / beta
        cursor = int(min(max(s, end), n))
    s = cursor
    while s < n:
        add_grain(min(s, n - 1), grain_u, s, grain_u)
        s += grain_u

    out = out[:n] / np.maximum(wsum[:n], 1e-3)[:, None]
    # outside any grain (before the first / after the last) keep the original
    empty = wsum[:n] < 1e-3
    out[empty] = x[empty]
    out = out.astype(np.float32)
    return out[:, 0] if audio.ndim == 1 else out


def correct_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    scale: str = "C:maj",
    retune_speed: float = 0.5,
    formant_preserve: bool = True,
    max_shift_semitones: float = 1.5,
) -> np.ndarray:
    """Snap the pitch of a monophonic part to the nearest note of a scale.

    Steps: track f0 (`track_pitch`), decide a target note for each stretch of
    voiced frames with hysteresis (`scale` like "A:min" or "E:pent_min"), work
    out how far each frame must move, smooth that correction with a time
    constant set by `retune_speed`, and resynthesise with TD-PSOLA.

    retune_speed  0 - 1. At 1 the pitch snaps instantly to the note (vibrato is
                  flattened, the "hard-tuned" sound). Lower values glide in over
                  up to ~300 ms, so a held note is pulled to pitch while vibrato
                  and bends, which are faster than that, survive.
    formant_preserve  True keeps the voice character (grains are copied, only
                  re-spaced). False resamples each grain like a tape speed change,
                  so formants move with the pitch.
    max_shift_semitones  corrections larger than this are left alone: a singer
                  that far from every scale note is probably using a note outside
                  the scale on purpose.

    Unvoiced stretches (breaths, consonants) and silence pass through untouched.
    """
    pcs = scale_pitch_classes(scale)
    if audio.size == 0:
        return audio
    hop_ms = 10.0
    hop = max(1, int(round(hop_ms / 1000.0 * sr)))
    f0, _ = track_pitch(audio, sr, hop_ms)
    if np.all(np.isnan(f0)):
        return audio

    midi = 69.0 + 12.0 * np.log2(f0 / 440.0)
    targets = _decide_targets(midi, pcs, hold_frames=max(3, int(round(80.0 / hop_ms))))
    needed = targets - midi
    needed = np.where(np.abs(needed) > max_shift_semitones, 0.0, needed)

    speed = float(np.clip(retune_speed, 0.0, 1.0))
    tau_ms = (1.0 - speed) * 300.0
    alpha = 1.0 if tau_ms < 1e-3 else 1.0 - np.exp(-hop_ms / tau_ms)
    applied = np.zeros(f0.size)
    cur = 0.0
    for i in range(f0.size):
        goal = 0.0 if np.isnan(needed[i]) else needed[i]
        cur += alpha * (goal - cur)
        applied[i] = cur
    # between frames, interpolate the ratio linearly
    frame_t = np.arange(f0.size) * hop
    ratio_frames = 2.0 ** (applied / 12.0)

    def ratio_at(sample: int) -> float:
        return float(np.interp(sample, frame_t, ratio_frames))

    if np.max(np.abs(applied)) < 0.02:
        return audio
    return _psola(audio.astype(np.float64), sr, f0, hop, ratio_at, formant_preserve)


def shift_semitones(
    audio: np.ndarray, semitones: float, sr: int = SAMPLE_RATE,
    engine: str = "auto", formant: bool = False,
) -> np.ndarray:
    """Fixed pitch shift with the length preserved, so stems stay in sync.

    Works on any material, chords included. See `tsm.pitch_shift`: the built-in
    phase vocoder, or the Rubber Band binary when it is installed (`engine`
    "auto"), which is cleaner and can keep a voice's formants with `formant=True`.
    """
    from . import tsm

    return tsm.pitch_shift(audio, semitones, sr, engine=engine, formant=formant)


NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler probe-tone profiles, tonic first.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chroma_stft(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    fmin: float = 55.0,
    fmax: float = 5000.0,
) -> np.ndarray:
    """Whole-signal 12-bin pitch-class energy profile from an STFT.

    Coarse (bins are folded onto the nearest semitone) but dependency-free, and
    plenty for key estimation, which only needs the long-term average.
    """
    from scipy.signal import stft

    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    nperseg = 8192 if sr >= 32000 else 4096
    freqs, _, z = stft(mono, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    power = (np.abs(z) ** 2).mean(axis=1)
    keep = (freqs >= fmin) & (freqs <= fmax)
    midi = 69.0 + 12.0 * np.log2(freqs[keep] / 440.0)
    pcs = np.mod(np.round(midi).astype(int), 12)
    chroma = np.bincount(pcs, weights=np.sqrt(power[keep]), minlength=12)
    total = chroma.sum()
    return chroma / total if total > 0 else chroma


def key_from_chroma(chroma: np.ndarray) -> tuple[int, str, float]:
    """Krumhansl-Schmuckler: correlate a 12-bin profile against all 24 keys.

    Returns (tonic pitch class 0-11, "maj" | "min", correlation).
    """
    chroma = np.asarray(chroma, dtype=np.float64)
    if not np.any(chroma > 0):
        return 0, "maj", 0.0
    best = (0, "maj", -np.inf)
    for mode, profile in (("maj", KK_MAJOR), ("min", KK_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            if r > best[2]:
                best = (tonic, mode, r)
    return best


def detect_key(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str | None:
    """Estimate the musical key, e.g. "A:min", in the same format `correct_pitch`
    takes for `scale`. None for silence.
    """
    chroma = chroma_stft(audio, sr)
    if not np.any(chroma > 0):
        return None
    tonic, mode, _ = key_from_chroma(chroma)
    return f"{NOTE_NAMES[tonic]}:{mode}"
