"""Source separation, for when the input is an already-mixed recording.

Stub. If someone hands this pipeline a single stereo voice memo of guitar +
harmonica played together, there are no stems to mix. Demucs can split that into
parts first.
"""

from __future__ import annotations

import numpy as np

from .io_utils import Stem


def separate(audio: np.ndarray, sr: int, model: str = "htdemucs") -> list[Stem]:
    """Split a mixed recording into stems.

    # TODO: wrap Demucs (`pip install demucs`, then `demucs.api.Separator`).
    # Note Demucs is trained on drums/bass/vocals/other, so a guitar+harmonica
    # duo mostly lands in "other" - useful for removing a backing track, less
    # useful for splitting two similar acoustic instruments.
    """
    pass  # TODO
    return []
