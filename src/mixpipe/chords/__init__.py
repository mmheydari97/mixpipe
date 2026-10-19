"""Chord finder: upload a song, get lyrics with chord candidates on the words.

Heavy dependencies (librosa, demucs, mlx-whisper, fastapi) live in
requirements-chords.txt and are imported lazily, so the mixing pipeline never
needs them. This package may only import io_utils, separation, pitch,
openrouter, fetch and songstore from the rest of mixpipe, which keeps it easy
to split out later.
"""

import warnings

# librosa's numba ufuncs (abs2, phasor) warn when a frame holds NaN/inf, e.g.
# pYIN on silent stretches. The NaNs are handled downstream, so the message
# ("RuntimeWarning: invalid value encountered in cast") is only noise. Only
# warnings raised from numba's ufunc wrapper are hidden.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numba\.np\.ufunc\.dufunc")
