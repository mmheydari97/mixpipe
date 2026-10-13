"""Pitch correction.

Every function here is a stub. Pitch correction that sounds musical needs:

  1. A monophonic pitch tracker (YIN / pYIN / CREPE) producing f0 + confidence
     per frame at ~10 ms resolution.
  2. A target-note decision stage: snap to a scale, with a retune speed so
     bends and vibrato survive instead of being flattened.
  3. A pitch shifter that preserves formants and transients - PSOLA for
     monophonic material, or WORLD / phase-vocoder with transient detection.

Doing this badly is worse than not doing it, which is why it isn't hacked in
with a naive resampler. Recommended external options if you need this soon:

  - `librosa.effects.pitch_shift` for fixed-interval shifts (not correction)
  - `pyworld` (WORLD vocoder) for f0 manipulation with formant preservation
  - `pyrubberband` (Rubber Band CLI wrapper) for good quality time/pitch
  - Melodyne / Autotune for anything you actually plan to release
"""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE


def track_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE, hop_ms: float = 10.0):
    """Return per-frame (f0_hz, confidence).

    # TODO: implement pYIN, or wrap librosa.pyin when librosa is available.
    """
    pass  # TODO


def correct_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    scale: str = "C:maj",
    retune_speed: float = 0.5,
    formant_preserve: bool = True,
) -> np.ndarray:
    """Snap pitch to the nearest scale degree.

    # TODO: depends on track_pitch() plus a PSOLA/WORLD shifter. Returns input
    # unchanged for now so the pipeline can run end to end.
    """
    pass  # TODO
    return audio


def shift_semitones(audio: np.ndarray, semitones: float, sr: int = SAMPLE_RATE):
    """Fixed pitch shift with length preserved.

    # TODO: phase vocoder. A naive resample changes duration, which breaks
    # alignment with the other stems.
    """
    pass  # TODO
    return audio


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
