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


def detect_key(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str | None:
    """Estimate the musical key, for scale-aware correction and harmony.

    # TODO: chroma features + Krumhansl-Schmuckler profile correlation.
    """
    pass  # TODO
    return None
