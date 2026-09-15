"""Reverb and delay send buses.

Stubs. A mix of four dry close-miked sources will sound pasted-together without
some shared space, so this is the highest-value missing module after the mix
itself works.
"""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE


def convolve_ir(audio: np.ndarray, ir: np.ndarray, mix: float = 0.2) -> np.ndarray:
    """Convolution reverb from an impulse response.

    # TODO: use scipy.signal.fftconvolve per channel, normalize the IR energy so
    # `mix` means the same thing across different IRs, and handle the tail
    # length (output is longer than input).
    """
    pass  # TODO
    return audio


def algorithmic_reverb(
    audio: np.ndarray, room_size: float = 0.5, decay_s: float = 1.5,
    mix: float = 0.2, sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Freeverb-style reverb, no IR file needed.

    # TODO: 8 parallel comb filters into 4 series all-pass filters (Schroeder /
    # Freeverb topology). Straightforward to implement, just fiddly to tune.
    """
    pass  # TODO
    return audio


def slap_delay(
    audio: np.ndarray, delay_ms: float = 120.0, feedback: float = 0.25,
    mix: float = 0.15, sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """# TODO: simple feedback delay line. Easiest win in this module."""
    pass  # TODO
    return audio
