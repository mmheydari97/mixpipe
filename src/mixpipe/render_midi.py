"""MIDI rendering, for Seaboard captures saved as MIDI rather than audio.

Stub. The Seaboard's whole point is per-note pressure/glide/slide (MPE), so a
naive General MIDI render throws away most of the performance.
"""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE


def render(midi_path: str, soundfont_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render a MIDI file to audio via a soundfont.

    # TODO: wrap FluidSynth (`pip install pyfluidsynth` + a .sf2 file), or shell
    # out to `fluidsynth -F out.wav soundfont.sf2 in.mid`.
    """
    pass  # TODO


def render_mpe(midi_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render MPE expression (per-note pitch bend, pressure, slide).

    # TODO: needs an MPE-aware synth. Practical advice in the meantime: bounce
    # audio out of the Seaboard's own plugin (Equator) and drop the WAV in
    # stems/ instead of routing MIDI through here.
    """
    pass  # TODO
