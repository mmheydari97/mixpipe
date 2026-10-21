"""Reverb and delay send buses.

A mix of four dry close-miked sources sounds pasted together without some shared
space. Each function blends dry and wet by `mix` (0 = dry, 1 = wet only), so it
works as an insert at mix 0.2 or as a send at mix 1.0 summed in parallel with
the dry bus (see `mixdown.add_send_buses`).

All three accept mono (n,) or stereo (n, 2) audio and return the same layout.
By default the output is longer than the input by the tail (`tail=True`); pass
`tail=False` to keep the input length.

The comb, all-pass and delay lines are recursions with a delay of D samples, so
they are evaluated a block of D samples at a time: every output sample in a
block depends only on the previous block. That is O(n) with a handful of numpy
calls per block, where an IIR filter with a D-tap coefficient vector would be
O(n * D).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, lfilter

from . import SAMPLE_RATE

# Freeverb's tunings, in samples at 44.1 kHz, scaled to the working sample rate
# so the room sounds the same at 48 kHz. The right channel's delays are offset
# by _STEREO_SPREAD samples for decorrelation.
_COMB_TUNING = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASS_TUNING = (556, 441, 341, 225)
_STEREO_SPREAD = 23
_TUNING_SR = 44_100
_ALLPASS_GAIN = 0.5
_MAX_TAIL_S = 8.0


def _channels(audio: np.ndarray) -> list[np.ndarray]:
    if audio.ndim == 1:
        return [audio.astype(np.float64)]
    return [audio[:, c].astype(np.float64) for c in range(audio.shape[1])]


def _stack(chans: list[np.ndarray], like: np.ndarray) -> np.ndarray:
    if like.ndim == 1:
        return chans[0].astype(np.float32)
    return np.stack(chans, axis=1).astype(np.float32)


def _blend(
    dry: list[np.ndarray], wet: list[np.ndarray], mix: float, tail: bool
) -> list[np.ndarray]:
    """Dry at (1 - mix) plus wet at mix, on a common length."""
    mix = float(np.clip(mix, 0.0, 1.0))
    n_in = dry[0].size
    n_out = max(n_in, max(w.size for w in wet)) if tail else n_in
    out = []
    for d, w in zip(dry, wet):
        o = np.zeros(n_out)
        o[:n_in] += d * (1.0 - mix)
        w = w[:n_out]
        o[: w.size] += w * mix
        out.append(o)
    return out


def _one_pole(damp: float):
    """Coefficients of the damping lowpass (1 - d) / (1 - d z^-1), unity DC gain."""
    return [1.0 - damp], [1.0, -damp]


# --------------------------------------------------------------------------- #
# convolution
# --------------------------------------------------------------------------- #


def convolve_ir(
    audio: np.ndarray, ir: np.ndarray, mix: float = 0.2, tail: bool = True
) -> np.ndarray:
    """Convolution reverb from an impulse response.

    The IR is normalised to unit energy (sum of squares = 1), so `mix` means
    the same wet level for any IR, whatever its length or how it was recorded.
    A mono IR is used on every channel; a stereo IR maps its channels onto the
    matching input channels (a mono input takes the left one). The output is
    longer than the input by the IR tail unless `tail=False`.
    """
    if audio.size == 0 or np.size(ir) == 0:
        return audio
    ir = np.asarray(ir, dtype=np.float64)
    ir_chans = [ir] if ir.ndim == 1 else [ir[:, c] for c in range(ir.shape[1])]
    dry = _channels(audio)

    wet = []
    for i, ch in enumerate(dry):
        h = ir_chans[min(i, len(ir_chans) - 1)]
        energy = np.sqrt(np.sum(h * h))
        if energy > 0:
            h = h / energy
        wet.append(fftconvolve(ch, h))
    return _stack(_blend(dry, wet, mix, tail), audio)


def synth_ir(
    decay_s: float = 1.5, sr: int = SAMPLE_RATE, pre_delay_ms: float = 10.0,
    stereo: bool = True, seed: int = 7,
) -> np.ndarray:
    """Exponentially decaying noise IR, for when no recorded IR is at hand.

    The envelope falls 60 dB at `decay_s`. The stereo channels use independent
    noise, so the tail decorrelates between left and right and sounds wide.
    """
    decay_s = max(0.05, float(decay_s))
    n = int(decay_s * sr * 1.5)
    t = np.arange(n) / sr
    envelope = 10.0 ** (-3.0 * t / decay_s)
    rng = np.random.default_rng(seed)
    pre = int(pre_delay_ms / 1000.0 * sr)
    chans = [
        np.concatenate([np.zeros(pre), rng.standard_normal(n) * envelope])
        for _ in range(2 if stereo else 1)
    ]
    return np.stack(chans, axis=1) if stereo else chans[0]


# --------------------------------------------------------------------------- #
# block-wise delay-line recursions
# --------------------------------------------------------------------------- #


def _comb(x: np.ndarray, delay: int, feedback: float, damp: float) -> np.ndarray:
    """Damped feedback comb: y[n] = x[n] + feedback * LP(y)[n - delay].

    The lowpass sits inside the loop, so highs die faster than lows, like air
    absorption in a real room. Stable for feedback < 1 (the lowpass has unity
    gain at DC and less elsewhere).
    """
    b, a = _one_pole(damp)
    y = np.empty_like(x)
    prev = np.zeros(delay)
    zi = np.zeros(1)
    for s in range(0, x.size, delay):
        e = min(s + delay, x.size)
        blk = x[s:e] + feedback * prev[: e - s]
        y[s:e] = blk
        prev, zi = lfilter(b, a, blk, zi=zi)
    return y


def _allpass(x: np.ndarray, delay: int, gain: float = _ALLPASS_GAIN) -> np.ndarray:
    """Schroeder all-pass: y[n] = -g x[n] + x[n - D] + g y[n - D].

    Flat magnitude response; it only smears phase, which thickens the echo
    density the combs produce.
    """
    y = np.empty_like(x)
    xp = np.zeros(delay)
    yp = np.zeros(delay)
    for s in range(0, x.size, delay):
        e = min(s + delay, x.size)
        blk = x[s:e]
        out = -gain * blk + xp[: e - s] + gain * yp[: e - s]
        y[s:e] = out
        xp, yp = blk, out
    return y


def _echo(x: np.ndarray, delay: int, feedback: float, damp: float) -> np.ndarray:
    """Feedback echo: w[n] = x[n - D] + feedback * LP(w)[n - D]."""
    b, a = _one_pole(damp)
    w = np.zeros_like(x)
    xp = np.zeros(delay)
    fp = np.zeros(delay)
    zi = np.zeros(1)
    for s in range(0, x.size, delay):
        e = min(s + delay, x.size)
        blk = xp[: e - s] + feedback * fp[: e - s]
        w[s:e] = blk
        fp, zi = lfilter(b, a, blk, zi=zi)
        xp = x[s:e]
    return w


# --------------------------------------------------------------------------- #
# Freeverb
# --------------------------------------------------------------------------- #


def _freeverb_core(
    feed: np.ndarray, n_out_ch: int, scale: float, decay_s: float, damp: float,
    sr: int, width: float,
) -> list[np.ndarray]:
    """One mono feed through 8 combs -> 4 all-passes, per output channel."""
    wets = []
    for ch in range(n_out_ch):
        spread = _STEREO_SPREAD * (ch % 2)
        acc = np.zeros_like(feed)
        for base in _COMB_TUNING:
            d = max(2, int(round((base + spread) * scale)))
            # RT60: each trip round a loop of d samples loses 60 d / (sr decay) dB.
            g = 10.0 ** (-3.0 * (d / sr) / decay_s)
            acc += _comb(feed, d, g, damp)
        out = acc / len(_COMB_TUNING)
        for base in _ALLPASS_TUNING:
            out = _allpass(out, max(2, int(round((base + spread) * scale))))
        wets.append(out)
    if n_out_ch >= 2:
        # width 1 keeps the two tails independent; 0 collapses them to the mid.
        mid = 0.5 * (wets[0] + wets[1])
        wets[0] = mid + width * (wets[0] - mid)
        wets[1] = mid + width * (wets[1] - mid)
    return wets


def algorithmic_reverb(
    audio: np.ndarray, room_size: float = 0.5, decay_s: float = 1.5,
    mix: float = 0.2, sr: int = SAMPLE_RATE, damping: float = 0.4,
    pre_delay_ms: float = 0.0, width: float = 1.0, tail: bool = True,
) -> np.ndarray:
    """Freeverb-style reverb, no IR file needed.

    Eight parallel damped comb filters into four series all-pass filters
    (Schroeder / Freeverb topology).

      room_size   0-1, scales the delay lengths from 0.5x to 1.5x
      decay_s     RT60, the time for the tail to fall 60 dB (at low frequencies)
      damping     0-1, how much the highs die faster than the lows
      width       0-1, how far apart the left and right tails are
      pre_delay_ms  gap before the reverb starts

    The wet signal is level-normalised from the measured impulse response (unit
    power gain for white noise), so `mix` is comparable across room sizes and
    decay times. A mono input gives a mono output.
    """
    if audio.size == 0:
        return audio
    scale = sr / _TUNING_SR * (0.5 + float(np.clip(room_size, 0.0, 1.0)))
    damp = float(np.clip(damping, 0.0, 0.95))
    width = float(np.clip(width, 0.0, 1.0))
    decay_s = float(np.clip(decay_s, 0.05, _MAX_TAIL_S))
    pre = int(max(0.0, pre_delay_ms) / 1000.0 * sr)

    dry = _channels(audio)
    n_ch = len(dry)
    n_out_ch = 1 if n_ch == 1 else 2
    tail_n = int(decay_s * sr * 1.2) if tail else 0

    # The room is excited by the channel sum, so mono and stereo sources behave alike.
    feed = np.concatenate([np.zeros(pre), np.mean(dry, axis=0), np.zeros(tail_n)])

    # Unit-power normalisation: run an impulse through the same network.
    probe = np.zeros(int(decay_s * sr * 1.5) + 1)
    probe[0] = 1.0
    probe_wets = _freeverb_core(probe, n_out_ch, scale, decay_s, damp, sr, width)
    power = float(np.mean([np.sum(w * w) for w in probe_wets]))
    norm = 1.0 / np.sqrt(power) if power > 0 else 1.0

    wets = [w * norm for w in _freeverb_core(feed, n_out_ch, scale, decay_s, damp, sr, width)]
    if n_ch > n_out_ch:                      # >2 channels never happens, but be safe
        wets += [wets[-1]] * (n_ch - n_out_ch)
    return _stack(_blend(dry, wets, mix, tail), audio)


# --------------------------------------------------------------------------- #
# delay
# --------------------------------------------------------------------------- #


def slap_delay(
    audio: np.ndarray, delay_ms: float = 120.0, feedback: float = 0.25,
    mix: float = 0.15, sr: int = SAMPLE_RATE, tail: bool = True,
    damping: float = 0.0,
) -> np.ndarray:
    """Feedback delay line: one slap, or a few repeats with feedback.

    `feedback` is clamped below 1 so it can't run away. `damping` (0-1) puts a
    lowpass in the feedback path so each repeat is darker than the last, which
    sits better behind a vocal than clean repeats. With `tail` on, the output is
    extended until the repeats have fallen 60 dB (at most 5 s).
    """
    if audio.size == 0:
        return audio
    d = max(2, int(delay_ms / 1000.0 * sr))
    fb = float(np.clip(feedback, 0.0, 0.95))
    damp = float(np.clip(damping, 0.0, 0.95))

    extra = 0
    if tail:
        repeats = 1 if fb <= 1e-3 else min(60, int(np.ceil(np.log(1e-3) / np.log(fb))))
        extra = min(d * (repeats + 1), int(5.0 * sr))

    dry = _channels(audio)
    wet = [_echo(np.concatenate([ch, np.zeros(extra)]), d, fb, damp) for ch in dry]
    return _stack(_blend(dry, wet, mix, tail), audio)
