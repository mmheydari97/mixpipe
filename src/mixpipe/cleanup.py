"""Per-stem cleanup: DC offset, mains hum, noise reduction, clicks, gating, edge fades.

Runs before EQ and compression, because compressing a noisy take just makes the
noise louder in the gaps.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import iirnotch, sosfilt, stft, istft, tf2sos, welch

from . import SAMPLE_RATE
from ._fast import asym_smooth
from .analysis import from_db
from .io_utils import Stem

try:
    import noisereduce

    HAVE_NOISEREDUCE = True
except ImportError:  # pragma: no cover
    noisereduce = None
    HAVE_NOISEREDUCE = False


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """Subtract the mean. A DC-offset stem wastes headroom for no sound."""
    return audio - np.mean(audio, axis=0, keepdims=audio.ndim == 2)


def _spectral_gate(
    mono: np.ndarray,
    sr: int,
    noise_profile: np.ndarray,
    strength: float = 0.6,
    n_fft: int = 2048,
) -> np.ndarray:
    """Built-in spectral subtraction denoise (used when noisereduce is absent).

    Estimates a per-bin noise magnitude from a silent region, then attenuates
    bins that sit near that floor. Musical-noise artifacts are possible at high
    strength; that's why `noisereduce` is preferred when installed.
    """
    if noise_profile.size < n_fft:
        return mono

    noverlap = n_fft // 2
    _, _, Zn = stft(noise_profile, fs=sr, nperseg=n_fft, noverlap=noverlap)
    noise_mag = np.mean(np.abs(Zn), axis=1, keepdims=True)

    f, t, Z = stft(mono, fs=sr, nperseg=n_fft, noverlap=noverlap)
    mag, phase = np.abs(Z), np.angle(Z)

    # Wiener-ish soft mask. strength scales how far above the noise floor a bin
    # must sit to survive untouched.
    threshold = noise_mag * (1.0 + 4.0 * float(strength))
    snr = mag / np.maximum(threshold, 1e-12)
    mask = snr**2 / (1.0 + snr**2)
    mask = np.clip(mask, 1.0 - float(strength), 1.0)

    _, cleaned = istft(
        mask * mag * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=noverlap
    )
    return cleaned[: mono.size].astype(np.float32)


NOISE_PROFILE_MARGIN_DB = 20.0   # a noise sample must be this far under the music


def denoise(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    strength: float = 0.6,
    noise_profile_ms: float = 400.0,
) -> np.ndarray:
    """Reduce steady-state noise (room hum, amp hiss, phone-mic floor).

    The noise profile is the quietest `noise_profile_ms` of the take (a pause, or
    the silence before the first note). It is only trusted when it sits at least
    NOISE_PROFILE_MARGIN_DB under the take's loud level: a stem with no pause at all
    (a mid-song excerpt, a separated stem that never stops) has no noise sample,
    and subtracting the music from itself would take several dB off everything.
    Such a take is returned unchanged.
    """
    if strength <= 0.0:
        return audio

    def _one(ch: np.ndarray) -> np.ndarray:
        n_profile = int(noise_profile_ms / 1000.0 * sr)
        if ch.size <= n_profile * 2:
            return ch
        profile = _quietest_segment(ch, sr, noise_profile_ms / 1000.0)
        hop = max(1, int(0.05 * sr))
        frames = ch[: ch.size // hop * hop].astype(np.float64).reshape(-1, hop)
        loud = np.percentile(np.sqrt(np.mean(frames ** 2, axis=1)), 95)
        quiet = np.sqrt(np.mean(profile.astype(np.float64) ** 2))
        if quiet <= 1e-7 or loud <= 0 or 20 * np.log10(loud / quiet) < NOISE_PROFILE_MARGIN_DB:
            return ch                     # digital silence (nothing to remove) or no pause

        if HAVE_NOISEREDUCE:
            return noisereduce.reduce_noise(
                y=ch, sr=sr, prop_decrease=float(strength), y_noise=profile
            ).astype(np.float32)
        return _spectral_gate(ch, sr, profile, strength)

    if audio.ndim == 1:
        return _one(audio)
    return np.stack([_one(audio[:, c]) for c in range(audio.shape[1])], axis=1)


def noise_gate(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    threshold_db: float = -48.0,
    hold_ms: float = 80.0,
    release_ms: float = 120.0,
    attack_ms: float = 2.0,
) -> np.ndarray:
    """Envelope-follower gate with hold, so decaying notes don't get chopped."""
    mono = audio if audio.ndim == 1 else np.abs(audio).mean(axis=1)
    thresh = float(from_db(threshold_db))

    # Smooth the detector a little to avoid chattering on transients.
    win = max(1, int(0.005 * sr))
    env = np.convolve(np.abs(mono), np.ones(win) / win, mode="same")

    open_mask = env > thresh
    hold = int(hold_ms / 1000.0 * sr)
    if hold > 0:
        # Extend every open region forward by the hold time.
        kernel = np.ones(hold)
        open_mask = np.convolve(open_mask.astype(np.float32), kernel, mode="full")[
            : open_mask.size
        ] > 0

    smoothed = asym_smooth(
        open_mask.astype(np.float64), up_ms=attack_ms, down_ms=release_ms, sr=sr, init=0.0
    ).astype(np.float32)

    if audio.ndim == 1:
        return audio * smoothed
    return audio * smoothed[:, None]


def fade_edges(
    audio: np.ndarray, sr: int = SAMPLE_RATE, fade_ms: float = 5.0
) -> np.ndarray:
    """Short fades in/out to kill start and end clicks."""
    n = int(fade_ms / 1000.0 * sr)
    if n <= 0 or audio.shape[0] < 2 * n:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = audio.copy()
    if audio.ndim == 1:
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    else:
        out[:n] *= ramp[:, None]
        out[-n:] *= ramp[::-1, None]
    return out


def trim_lead_silence(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    threshold_db: float = -50.0,
    keep_ms: float = 50.0,
) -> tuple[np.ndarray, float]:
    """Trim leading silence, keeping a little pre-roll.

    Returns (audio, trimmed_ms). Note: trimming stems by different amounts
    destroys their relative timing. Only use this on a single stem, or after
    timing.align_stems() exists.
    """
    mono = audio if audio.ndim == 1 else np.abs(audio).mean(axis=1)
    above = np.nonzero(np.abs(mono) > float(from_db(threshold_db)))[0]
    if above.size == 0:
        return audio, 0.0
    start = max(0, above[0] - int(keep_ms / 1000.0 * sr))
    return audio[start:], start / sr * 1000.0


def clean_stem(stem: Stem, cfg: dict) -> Stem:
    """Apply the configured cleanup chain to one stem."""
    c = cfg["cleanup"]
    audio = stem.audio

    if c.get("remove_dc", True):
        audio = remove_dc(audio)
    if c.get("dehum", "auto"):
        audio = dehum(audio, stem.sr, base_hz=c.get("dehum", "auto"))
    if c.get("declick", False):
        audio = declick(
            audio, stem.sr,
            threshold=c.get("declick_threshold", 8.0),
            max_click_ms=c.get("declick_max_ms", 2.0),
        )
    if c.get("denoise", True):
        audio = denoise(
            audio,
            stem.sr,
            strength=c.get("denoise_strength", 0.6),
            noise_profile_ms=c.get("noise_profile_ms", 400),
        )
    if c.get("gate_enabled", True):
        audio = noise_gate(
            audio,
            stem.sr,
            threshold_db=c.get("gate_threshold_db", -48.0),
            hold_ms=c.get("gate_hold_ms", 80.0),
            release_ms=c.get("gate_release_ms", 120.0),
        )
    audio = fade_edges(audio, stem.sr, c.get("fade_ms", 5.0))

    out = stem.copy_with(audio.astype(np.float32))
    out.meta["cleaned"] = True
    return out


# --------------------------------------------------------------------------- #
# click removal
# --------------------------------------------------------------------------- #


def _robust_scale(d: np.ndarray, block: int) -> np.ndarray:
    """Per-sample robust scale of `d`: the largest block median of |d| nearby.

    A block median ignores a click (a few outliers). Taking the max over the
    neighbouring blocks as well keeps an attack that starts at a block edge, with
    a loud block next to a quiet one, from looking like an outlier in the quiet one.
    """
    n_blocks = int(np.ceil(d.size / block))
    padded = np.pad(np.abs(d), (0, n_blocks * block - d.size), mode="edge")
    med = np.median(padded.reshape(n_blocks, block), axis=1) * 1.4826
    near = np.maximum(med, np.maximum(np.r_[med[1:], med[-1:]], np.r_[med[:1], med[:-1]]))
    return np.repeat(near, block)[: d.size]


def find_clicks(
    mono: np.ndarray, sr: int = SAMPLE_RATE, threshold: float = 8.0,
    max_click_ms: float = 2.0, min_level_db: float = -55.0,
) -> list[tuple[int, int]]:
    """Locate impulsive clicks as (start, stop) sample runs, stop exclusive.

    The detector works on the second difference, which turns a click (a short
    jump away from and back to the waveform) into a large isolated spike while
    cancelling most of the music. A sample is an outlier when it exceeds
    `threshold` x the robust local scale and `min_level_db`. Runs longer than
    `max_click_ms` are dropped: a pick attack or drum hit keeps the second
    difference large for much longer than a click does, and the neighbouring
    blocks' scale rises with it.
    """
    x = np.asarray(mono, dtype=np.float64)
    if x.size < 16:
        return []
    d = np.diff(x, n=2)                     # d[i] is centred on x[i + 1]
    scale = _robust_scale(d, max(32, int(0.005 * sr)))
    floor = float(from_db(min_level_db))
    flagged = (np.abs(d) > threshold * np.maximum(scale, 1e-9)) & (np.abs(d) > floor)
    if not flagged.any():
        return []

    # Group flagged samples into runs, merging gaps of up to 2 samples, then pad
    # by 2 on each side because the second difference spreads a click over three.
    idx = np.nonzero(flagged)[0] + 1        # back to x coordinates
    breaks = np.nonzero(np.diff(idx) > 3)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]] + 1
    max_len = max(4, int(max_click_ms / 1000.0 * sr))
    runs = []
    for a, b in zip(starts, stops):
        a, b = max(1, a - 2), min(x.size - 1, b + 2)
        if b - a <= max_len:
            runs.append((int(a), int(b)))
    return runs


def _repair(x: np.ndarray, a: int, b: int, context: int = 6) -> None:
    """Replace x[a:b] in place with a cubic spline through the samples around it."""
    lo, hi = max(0, a - context), min(x.size, b + context)
    xs = np.r_[np.arange(lo, a), np.arange(b, hi)]
    if xs.size < 4:
        return
    x[a:b] = CubicSpline(xs, x[xs], bc_type="natural")(np.arange(a, b))


def declick(
    audio: np.ndarray, sr: int = SAMPLE_RATE, threshold: float = 8.0,
    max_click_ms: float = 2.0, min_level_db: float = -55.0,
) -> np.ndarray:
    """Detect and interpolate over impulsive clicks and pops.

    Median-based outlier detection on the second difference (`find_clicks`),
    then cubic interpolation across each flagged run. Each channel is treated
    on its own. Conservative by design: it only touches runs shorter than
    `max_click_ms`, so pick attacks and drum transients (which are wider and lift
    the local scale) are left alone. Lower `threshold` to catch quieter clicks
    at the cost of more false positives.
    """
    if audio.size == 0:
        return audio

    def _one(ch: np.ndarray) -> np.ndarray:
        runs = find_clicks(ch, sr, threshold, max_click_ms, min_level_db)
        if not runs:
            return ch
        out = ch.astype(np.float64)
        for a, b in runs:
            _repair(out, a, b)
        return out.astype(np.float32)

    if audio.ndim == 1:
        return _one(audio)
    return np.stack([_one(audio[:, c]) for c in range(audio.shape[1])], axis=1)


# --------------------------------------------------------------------------- #
# mains hum
# --------------------------------------------------------------------------- #

_MAINS = (50.0, 60.0)
_HUM_MAX_HARMONICS = 10
_HUM_PROMINENCE_DB = 10.0     # a line must stand this far above its surroundings
_HUM_MIN_LINES = 2            # ... and at least this many harmonics (two of them in 1-4)


def _quietest_segment(mono: np.ndarray, sr: int, seconds: float = 2.0) -> np.ndarray:
    """The quietest `seconds` of the signal, where hum is not masked by music."""
    n = int(seconds * sr)
    if mono.size <= n:
        return mono
    hop = int(0.1 * sr)
    n_frames = mono.size // hop
    power = np.mean(mono[: n_frames * hop].astype(np.float64).reshape(n_frames, hop) ** 2, axis=1)
    k = max(1, int(round(seconds / 0.1)))
    window = np.convolve(power, np.ones(k), mode="valid")
    start = int(np.argmin(window)) * hop
    return mono[start : start + n]


def find_hum(
    audio: np.ndarray, sr: int = SAMPLE_RATE, max_search_hz: float = 1000.0
) -> dict | None:
    """Detect mains hum and say which family it is.

    Looks at the quietest couple of seconds, and for each of 50 Hz and 60 Hz
    measures how far every harmonic line stands above the surrounding spectrum.
    Returns None unless at least two harmonics of one family clearly stand out,
    otherwise {"base_hz", "harmonics": [k, ...], "prominence_db": [...]}.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    seg = _quietest_segment(mono.astype(np.float64), sr)
    if seg.size < sr // 2 or float(np.max(np.abs(seg))) < 1e-7:
        return None

    nper = min(seg.size, 1 << 16)
    freqs, psd = welch(seg, fs=sr, nperseg=nper, noverlap=nper // 2)
    df = freqs[1] - freqs[0]
    best, best_score = None, 0.0
    for base in _MAINS:
        harmonics, prom = [], []
        for k in range(1, _HUM_MAX_HARMONICS + 1):
            f0 = base * k
            if f0 > max_search_hz or f0 > sr * 0.45:
                break
            line = (freqs > f0 - 0.6) & (freqs < f0 + 0.6)
            around = (np.abs(freqs - f0) > 3.0) & (np.abs(freqs - f0) < 25.0)
            if not line.any() or around.sum() < 4:
                continue
            ratio_db = 10.0 * np.log10(max(psd[line].max(), 1e-30) / max(np.median(psd[around]), 1e-30))
            if ratio_db >= _HUM_PROMINENCE_DB:
                harmonics.append(k)
                prom.append(round(float(ratio_db), 1))
        # mains hum is strongest in its low harmonics; a lone note plus a chance
        # spike far up the series is not hum, so two of the lines must be in 1-4
        low = sum(1 for k in harmonics if k <= 4)
        if len(harmonics) >= _HUM_MIN_LINES and low >= _HUM_MIN_LINES and sum(prom) > best_score:
            best_score = sum(prom)
            best = {"base_hz": base, "harmonics": harmonics, "prominence_db": prom}
    return best


def dehum(
    audio: np.ndarray, sr: int = SAMPLE_RATE, base_hz: float | str | None = "auto",
    harmonics: list[int] | None = None, q: float = 40.0,
) -> np.ndarray:
    """Notch out mains hum and its harmonics.

    `base_hz="auto"` (default) measures the hum (`find_hum`): it picks 50 or 60
    Hz from the spectrum of the quietest part of the take, notches that
    fundamental and its harmonics up to the 6th (or the highest it saw), and
    returns the audio untouched when there is no hum. Pass 50.0 or 60.0 to skip detection and notch `harmonics` (default 1-8)
    unconditionally. `q` sets the notch width: 40 is about 1.5 Hz wide at 60 Hz,
    narrow enough to leave bass notes alone.
    """
    if audio.size == 0 or base_hz is None or base_hz is False:
        return audio
    if base_hz == "auto":
        found = find_hum(audio, sr)
        if found is None:
            return audio
        # Complete the series: a harmonic can be masked by music in the analysed
        # stretch yet still be in the take, and a notch ~1.5 Hz wide costs nothing.
        base_hz = found["base_hz"]
        harmonics = list(range(1, max(6, max(found["harmonics"])) + 1))
    elif harmonics is None:
        harmonics = list(range(1, 9))

    sos = []
    for k in harmonics:
        f0 = float(base_hz) * k
        if f0 >= sr * 0.45:
            continue
        b, a = iirnotch(f0, max(q, 1.0), fs=sr)
        sos.append(tf2sos(b, a))
    if not sos:
        return audio
    return sosfilt(np.vstack(sos), audio, axis=0).astype(np.float32)
