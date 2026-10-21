"""Sample-by-sample kernels for the dynamics code.

Compressor gain smoothing, the limiter's release, the gate and the envelope
followers are all the same recursion: a one-pole filter whose coefficient
depends on whether the input is above or below the current state. That is not a
linear filter, so scipy can't do it, and in plain Python it costs ~1 s per
minute of audio per call.

If numba is installed (it is whenever the chord finder's requirements are) the
kernel is compiled and runs at C speed; without it the same code runs as plain
Python, so nothing depends on numba being present.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*_args, **_kwargs):
        def wrap(fn):
            return fn

        return wrap


@njit(cache=True)
def _asym_smooth(target, up_coef, down_coef, init):
    out = np.empty_like(target)
    state = init
    for i in range(target.shape[0]):
        x = target[i]
        coef = up_coef if x > state else down_coef
        state = x + coef * (state - x)
        out[i] = state
    return out


def pole(time_ms: float, sr: int) -> float:
    """One-pole coefficient for a time constant: 0 = instant, ->1 = very slow."""
    return float(np.exp(-1.0 / max(1.0, time_ms / 1000.0 * sr)))


def asym_smooth(
    target: np.ndarray, up_ms: float, down_ms: float, sr: int, init: float | None = None
) -> np.ndarray:
    """Smooth `target` with separate time constants for rising and falling.

    s[i] = t[i] + c * (s[i-1] - t[i]), with c set by `up_ms` when the target is
    above the current state and `down_ms` when it is below (0 ms = follow
    instantly). Used three ways:

      envelope follower   up = attack, down = release
      gain reduction      up = release, down = attack (gain falling = clamping)
      limiter release     up = release, down = 0
    """
    t = np.ascontiguousarray(target, dtype=np.float64)
    if t.size == 0:
        return t
    up = 0.0 if up_ms <= 0 else pole(up_ms, sr)
    down = 0.0 if down_ms <= 0 else pole(down_ms, sr)
    return _asym_smooth(t, up, down, float(t[0] if init is None else init))
