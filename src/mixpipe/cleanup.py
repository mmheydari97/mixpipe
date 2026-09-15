"""Per-stem cleanup: DC offset, noise reduction, gating, edge fades.

Runs before EQ and compression, because compressing a noisy take just makes the
noise louder in the gaps.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft

from . import SAMPLE_RATE
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


def denoise(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    strength: float = 0.6,
    noise_profile_ms: float = 400.0,
) -> np.ndarray:
    """Reduce steady-state noise (room hum, amp hiss, phone-mic floor).

    The noise profile is taken from the first `noise_profile_ms`, so leave a
    little silence at the top of every take. If there is no usable silence, this
    falls back to noisereduce's non-stationary mode, or returns audio unchanged.
    """
    if strength <= 0.0:
        return audio

    def _one(ch: np.ndarray) -> np.ndarray:
        n_profile = int(noise_profile_ms / 1000.0 * sr)
        profile = ch[:n_profile] if ch.size > n_profile * 2 else None

        if HAVE_NOISEREDUCE:
            kwargs = {"y": ch, "sr": sr, "prop_decrease": float(strength)}
            if profile is not None and np.max(np.abs(profile)) > 1e-6:
                kwargs["y_noise"] = profile
            else:
                kwargs["stationary"] = False
            return noisereduce.reduce_noise(**kwargs).astype(np.float32)

        if profile is None:
            return ch
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

    gain = open_mask.astype(np.float32)
    a_coef = np.exp(-1.0 / max(1.0, attack_ms / 1000.0 * sr))
    r_coef = np.exp(-1.0 / max(1.0, release_ms / 1000.0 * sr))

    smoothed = np.empty_like(gain)
    g = 0.0
    for i, target in enumerate(gain):
        coef = a_coef if target > g else r_coef
        g = target + coef * (g - target)
        smoothed[i] = g

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


def declick(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Detect and interpolate over impulsive clicks and pops.

    # TODO: needs median-filter outlier detection on the sample-difference
    # signal, then cubic interpolation across flagged runs. Non-trivial to tune
    # without false-positive-ing on legitimate pick attacks and drum transients.
    """
    pass  # TODO
    return audio


def dehum(audio: np.ndarray, sr: int = SAMPLE_RATE, base_hz: float = 60.0) -> np.ndarray:
    """Notch out mains hum and its harmonics.

    # TODO: implement as a cascade of narrow notch filters at base_hz * n,
    # with the fundamental frequency estimated from the noise profile rather
    # than assumed (50 Hz vs 60 Hz depends on where you are).
    """
    pass  # TODO
    return audio
