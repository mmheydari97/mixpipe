"""Time-scale modification: stretch, time-warp and pitch-shift without a binary.

The core is a phase vocoder with identity phase locking (Laroche & Dolson):
each frame's phase is advanced by the frequency measured between frames, and
every bin around a spectral peak is locked to that peak's phase, which keeps
partials coherent and avoids the "phasey" smear of a plain vocoder. Two extras
matter for music:

  - channels are processed together: the phase is worked out on the mid signal
    and the same rotation is applied to every channel, so the stereo image
    survives;
  - `reset_at` lists source positions of note onsets. The frame that holds one
    restarts from the analysis phase, so a drum hit or pick attack keeps its
    transient instead of being smeared across a frame.

`time_warp` takes any monotone map from output time to source time, which is
what onset quantization needs; `stretch` and `pitch_shift` are uniform special
cases. The Rubber Band binary is used for `pitch_shift` when it is installed
and `engine="auto"`, because it sounds better (through stretch.py, the one place
MixPipe calls it); the vocoder is the fallback and the dependency-free default
for everything else.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from . import SAMPLE_RATE
from .stretch import has_rubberband as have_rubberband
from .stretch import stretch as rubberband_stretch

N_FFT = 2048
HOP = 512          # synthesis hop: 75 % overlap


def _princarg(x: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi)."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _peak_regions(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(peak bin indices, nearest peak for every bin) from a magnitude frame."""
    n = mag.size
    padded = np.pad(mag, 2, mode="constant")
    is_peak = (
        (mag > padded[1:-3]) & (mag > padded[0:-4])
        & (mag >= padded[3:-1]) & (mag >= padded[4:])
    )
    peaks = np.nonzero(is_peak)[0]
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(mag))])
    # region boundaries halfway between neighbouring peaks
    mids = (peaks[:-1] + peaks[1:]) // 2 + 1
    owner = peaks[np.searchsorted(mids, np.arange(n), side="right")]
    return peaks, owner


def _as_2d(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    a = np.asarray(audio, dtype=np.float64)
    return (a[:, None], True) if a.ndim == 1 else (a, False)


def time_warp(
    audio: np.ndarray, positions: np.ndarray, n_out: int | None = None,
    reset_at: np.ndarray | None = None, n_fft: int = N_FFT, hop: int = HOP,
) -> np.ndarray:
    """Render output frame m from source position `positions[m]` (in samples).

    The output sample at `m * hop` is the audio at `positions[m]`, so a map that
    advances more slowly than `hop` per frame stretches the sound, faster
    compresses it, and pitch is untouched either way. `reset_at` are source
    sample positions of onsets to keep sharp. Returns `n_out` samples (default
    `len(positions) * hop`).
    """
    x, was_mono = _as_2d(audio)
    n, n_ch = x.shape
    positions = np.asarray(positions, dtype=np.float64)
    n_frames = positions.size
    if n_out is None:
        n_out = n_frames * hop
    half = n_fft // 2

    pad = n_fft
    xp = np.pad(x, ((pad, pad), (0, 0)))
    win = np.hanning(n_fft + 1)[:-1]                   # periodic Hann
    omega = 2.0 * np.pi * np.arange(n_fft // 2 + 1) / n_fft

    pos_i = np.round(positions).astype(np.int64)
    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros((out_len, n_ch))
    norm = np.zeros(out_len)

    # frames that hold an onset: the first frame whose window reaches it
    reset_frames: set[int] = set()
    if reset_at is not None and len(reset_at):
        for t in np.asarray(reset_at, dtype=np.float64):
            m = int(np.searchsorted(pos_i, t - half // 2))
            if 0 <= m < n_frames:
                reset_frames.add(m)

    prev_phase = np.zeros(omega.size)
    syn_phase = np.zeros(omega.size)
    prev_pos = None
    for m in range(n_frames):
        start = pos_i[m] - half + pad
        seg = xp[start:start + n_fft] * win[:, None] if 0 <= start <= xp.shape[0] - n_fft \
            else np.zeros((n_fft, n_ch))
        spec = np.fft.rfft(seg, axis=0)                 # (bins, channels)
        mix = spec.sum(axis=1)
        mag, phase = np.abs(mix), np.angle(mix)

        if prev_pos is None or m in reset_frames:
            syn_phase = phase.copy()                    # start / transient: keep analysis phase
        else:
            ha = pos_i[m] - prev_pos
            if ha > 0:
                dphi = _princarg(phase - prev_phase - omega * ha)
                inst = omega + dphi / ha
            else:
                inst = omega
            advanced = syn_phase + inst * hop
            peaks, owner = _peak_regions(mag)
            # identity phase locking: every bin follows its peak's phase offset
            syn_phase = advanced[owner] + (phase - phase[owner])
        prev_phase, prev_pos = phase, pos_i[m]

        rot = np.exp(1j * (syn_phase - phase))
        frame = np.fft.irfft(spec * rot[:, None], n=n_fft, axis=0) * win[:, None]
        o = m * hop
        out[o:o + n_fft] += frame
        norm[o:o + n_fft] += win * win

    out /= np.maximum(norm, 1e-8)[:, None]
    out = out[half:half + n_out]                        # frame m is centred on m * hop
    if out.shape[0] < n_out:
        out = np.pad(out, ((0, n_out - out.shape[0]), (0, 0)))
    out = out.astype(np.float32)
    return out[:, 0] if was_mono else out


def stretch(audio: np.ndarray, rate: float, **kw) -> np.ndarray:
    """Play `audio` at `rate` (2.0 = twice as fast, half as long) at the same pitch."""
    rate = float(rate)
    if rate <= 0:
        raise ValueError("rate must be positive")
    n = np.asarray(audio).shape[0]
    if abs(rate - 1.0) < 1e-6 or n == 0:
        return audio
    n_out = int(round(n / rate))
    hop = kw.get("hop", HOP)
    positions = np.arange(int(np.ceil(n_out / hop)) + 1) * hop * rate
    return time_warp(audio, positions, n_out=n_out, **kw)


def pitch_shift(
    audio: np.ndarray, semitones: float, sr: int = SAMPLE_RATE,
    engine: str = "auto", formant: bool = False,
) -> np.ndarray:
    """Shift pitch by `semitones`, keeping the length (and so the sync) intact.

    engine "pv" is the built-in phase vocoder: stretch by the pitch ratio, then
    resample back to the original length. "rubberband" shells out to the Rubber
    Band binary (cleaner, and the only one that can keep formants with
    `formant=True`); "auto" picks it when it is installed.
    """
    if abs(semitones) < 1e-6 or np.asarray(audio).size == 0:
        return audio
    if engine not in ("auto", "pv", "rubberband"):
        raise ValueError("engine must be auto, pv or rubberband")
    if engine == "rubberband" or (engine == "auto" and have_rubberband()):
        if not have_rubberband():
            raise RuntimeError("engine 'rubberband' needs: brew install rubberband")
        a = np.asarray(audio, dtype=np.float32)
        out = rubberband_stretch(a if a.ndim == 2 else a[:, None], sr, 1.0, semitones, "fine",
                                 formant=formant)
        return out[:, 0] if a.ndim == 1 else out

    ratio = 2.0 ** (semitones / 12.0)
    n = np.asarray(audio).shape[0]
    stretched = stretch(audio, 1.0 / ratio)          # ratio times longer, same pitch
    frac = Fraction(ratio).limit_denominator(1000)   # play it back `ratio` times faster
    out = resample_poly(stretched, frac.denominator, frac.numerator, axis=0)
    if out.shape[0] < n:
        out = np.pad(out, [(0, n - out.shape[0])] + [(0, 0)] * (out.ndim - 1))
    return out[:n].astype(np.float32)
