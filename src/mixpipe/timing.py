"""Timing alignment and quantization.

Stubs. The useful cases, in order of how much they'd improve a home recording:

  1. Latency compensation - if stems were recorded in separate passes through an
     interface with different buffer settings, they can sit tens of ms apart.
     Fixable by cross-correlating each stem against a reference (click or the
     first-recorded stem) and applying an integer-sample shift.
  2. Onset quantization - move note onsets toward a grid. Needs onset detection
     plus time-warping between anchors, not a global shift.
  3. Tempo estimation - to build the grid in the first place.
"""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE
from .io_utils import Stem


def estimate_offset_samples(
    reference: np.ndarray, target: np.ndarray, max_lag_ms: float = 500.0,
    sr: int = SAMPLE_RATE,
) -> int:
    """Cross-correlation lag between two stems.

    # TODO: implement with scipy.signal.correlate on onset-strength envelopes
    # rather than raw samples - raw correlation locks onto the wrong peak when
    # the two instruments play different material.
    """
    pass  # TODO
    return 0


def align_stems(stems: list[Stem], reference: str | None = None) -> list[Stem]:
    """Shift every stem so they share a timeline.

    # TODO: depends on estimate_offset_samples(). Currently a pass-through, so
    # record your takes against a click or to a common backing track.
    """
    pass  # TODO
    return stems


def estimate_tempo_bpm(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    """# TODO: onset autocorrelation or a tempogram."""
    pass  # TODO
    return None


def quantize_onsets(
    audio: np.ndarray, bpm: float, grid: str = "1/16", strength: float = 0.5,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """# TODO: onset detection + piecewise time-warp between anchor points."""
    pass  # TODO
    return audio
