"""Mastering chain.

Deliberately short. Mastering a mix that isn't good yet just makes a loud bad
mix. The chain here is: clean up the extremes, glue it gently, hit a loudness
target, and stop peaks from clipping.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, welch

from . import SAMPLE_RATE
from ._fast import njit
from .analysis import crest_factor_db, integrated_lufs, peak_db, true_peak_db
from .config import DEFAULT_CONFIG
from .dynamics import (
    compress_with_settings, limit, multiband_compress, normalize_loudness,
)
from .eq import apply_eq_chain


def master(
    bus: np.ndarray,
    plan: dict,
    cfg: dict | None = None,
    sr: int = SAMPLE_RATE,
    verbose: bool = True,
    reference: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Run the master chain. Returns (audio, report).

    `reference` is a path to a commercial track whose tonal balance the mix is
    nudged toward (`match_reference`) right after the master EQ.
    """
    cfg = cfg or DEFAULT_CONFIG
    m = plan.get("master", cfg["master"])

    report: dict = {
        "input_peak_db": round(peak_db(bus), 2),
        "input_lufs": round(integrated_lufs(bus.mean(axis=1) if bus.ndim == 2 else bus, sr), 2),
        "input_crest_db": round(crest_factor_db(bus), 2),
        "stages": [],
    }

    audio = bus

    # 1. Clear out subsonic energy that eats limiter headroom inaudibly.
    audio = apply_eq_chain(
        audio,
        hpf_hz=m.get("master_hpf_hz", 25.0),
        lpf_hz=0.0,
        bands=m.get("master_eq", []),
        sr=sr,
    )
    report["stages"].append({"stage": "master_eq", "peak_db": round(peak_db(audio), 2)})

    if reference:
        audio, curve = match_reference(
            audio, reference, sr, strength=m.get("reference_strength", 0.8),
            return_curve=True,
        )
        report["reference"] = {"path": reference, "max_boost_db": curve["max_boost_db"],
                               "max_cut_db": curve["max_cut_db"]}
        report["stages"].append(
            {"stage": "reference_match", "peak_db": round(peak_db(audio), 2)}
        )

    # 2a. Optional three-band compression, so a pumping low end or a harsh top
    # is controlled on its own band rather than ducking the whole bus.
    mb = m.get("multiband") or {}
    if mb.get("enabled"):
        audio = multiband_compress(
            audio, sr, crossovers=mb.get("crossovers"), bands=mb.get("bands")
        )
        report["stages"].append(
            {"stage": "multiband", "peak_db": round(peak_db(audio), 2)}
        )

    # 2b. Glue compression: low ratio, slow attack, just to bind the parts.
    if cfg["mixdown"].get("bus_comp_enabled", True):
        audio = compress_with_settings(audio, m.get("glue_comp", {}), sr)
        report["stages"].append(
            {"stage": "glue_comp", "peak_db": round(peak_db(audio), 2)}
        )

    # 3. Loudness target. Pure gain, so this can push past 0 dBFS.
    pre_gain = audio
    target = m.get("target_lufs", -14.0)
    audio, applied = normalize_loudness(pre_gain, target, sr)
    report["loudness_gain_db"] = round(applied, 2)
    report["stages"].append(
        {"stage": "loudness_norm", "peak_db": round(peak_db(audio), 2)}
    )

    # 4. Limiter catches what step 3 pushed over. It also takes some loudness with
    # it, so measure what is left and push the gain by the shortfall (up to two
    # times; peaky material may never reach the target under the ceiling).
    def limited(x: np.ndarray) -> np.ndarray:
        return limit(
            x,
            ceiling_db=m.get("ceiling_db", -1.0),
            lookahead_ms=cfg["master"]["limiter_lookahead_ms"],
            release_ms=cfg["master"]["limiter_release_ms"],
            sr=sr,
            true_peak=cfg["master"].get("true_peak", True),
        )

    def lufs_of(x: np.ndarray) -> float:
        return integrated_lufs(x.mean(axis=1) if x.ndim == 2 else x, sr)

    audio = limited(audio)
    extra = 0.0
    for _ in range(2):
        short = target - lufs_of(audio)
        if abs(short) <= 0.2 or short < -3.0:
            break
        extra += short
        audio = limited(pre_gain * float(10 ** ((applied + extra) / 20.0)))
    if extra:
        report["loudness_gain_db"] = round(applied + extra, 2)
    report["stages"].append({"stage": "limiter", "peak_db": round(peak_db(audio), 2)})

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    report["output_peak_db"] = round(peak_db(audio), 2)
    report["output_true_peak_db"] = round(true_peak_db(audio), 2)
    report["output_lufs"] = round(integrated_lufs(mono, sr), 2)
    report["output_crest_db"] = round(crest_factor_db(audio), 2)
    report["crest_lost_db"] = round(
        report["input_crest_db"] - report["output_crest_db"], 2
    )

    # A crest loss over ~6 dB means the limiter is doing structural work that
    # the mix should have done. Worth telling the user rather than hiding.
    if report["crest_lost_db"] > 6.0:
        report["warning"] = (
            f"limiter squashed {report['crest_lost_db']:.1f} dB of dynamic range; "
            "the mix is probably too hot or too peaky before the master"
        )

    if verbose:
        print(
            f"  [mstr] {report['input_lufs']:.1f} -> {report['output_lufs']:.1f} LUFS, "
            f"peak {report['output_peak_db']:+.2f} dBFS, "
            f"crest {report['input_crest_db']:.1f} -> {report['output_crest_db']:.1f} dB"
        )
        if "warning" in report:
            print(f"  [warn] {report['warning']}")

    return audio, report


def _smoothed_spectrum_db(mono: np.ndarray, sr: int, centres: np.ndarray, width_oct: float):
    """Long-term average power spectrum, averaged over `width_oct` around each centre."""
    nper = 1 << 14
    freqs, psd = welch(mono, fs=sr, nperseg=min(nper, max(256, mono.size // 2)),
                       noverlap=None, window="hann")
    csum = np.concatenate([[0.0], np.cumsum(psd)])
    lo = np.searchsorted(freqs, centres * 2.0 ** (-width_oct / 2.0))
    hi = np.maximum(np.searchsorted(freqs, centres * 2.0 ** (width_oct / 2.0)), lo + 1)
    hi = np.minimum(hi, freqs.size)
    lo = np.minimum(lo, hi - 1)
    mean_power = (csum[hi] - csum[lo]) / (hi - lo)
    return 10.0 * np.log10(mean_power + 1e-20)


def match_reference(
    audio: np.ndarray, reference_path: str, sr: int = SAMPLE_RATE,
    strength: float = 1.0, max_db: float = 6.0, smoothing_octaves: float = 1.0 / 3.0,
    match_loudness: bool = True, return_curve: bool = False,
):
    """Match the tonal balance of a reference track.

    What Matchering does, in four steps:

      1. long-term average spectra of the mix and the reference (Welch, mid
         channel), averaged over `smoothing_octaves` on a log-frequency grid;
      2. their difference, with the overall level offset removed so only the
         *shape* is compared, limited to +/-`max_db` (a soft limit, so a mix
         that is 12 dB off gets pulled the full amount, not clipped flat) and
         scaled by `strength`; boosts stop at 16 kHz, so a reference that came
         from a lossy file with a lowpass can't make the mix fizz;
      3. that curve applied as a zero-delay linear-phase FIR (8191 taps), the
         same filter on both channels so the stereo image doesn't move;
      4. loudness put back to what it was (`match_loudness`), since the EQ moves
         the level and loudness is the master chain's job.

    `reference_path` is any file `io_utils.read_audio` reads. With
    `return_curve=True` returns (audio, {"freqs", "gain_db", "max_boost_db",
    "max_cut_db"}).
    """
    from .io_utils import normalize_shape, read_audio, to_sample_rate

    ref, ref_sr = read_audio(reference_path)
    ref = to_sample_rate(normalize_shape(ref), ref_sr, sr)
    ref_mono = ref if ref.ndim == 1 else ref.mean(axis=1)
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if ref_mono.size < sr or mono.size < sr:
        raise ValueError("match_reference needs at least a second of audio on both sides")

    centres = 2.0 ** np.linspace(np.log2(30.0), np.log2(min(18000.0, sr * 0.45)), 200)
    diff = _smoothed_spectrum_db(ref_mono, sr, centres, smoothing_octaves) \
        - _smoothed_spectrum_db(mono, sr, centres, smoothing_octaves)
    band = (centres >= 100.0) & (centres <= 10000.0)
    diff = diff - np.median(diff[band])                       # shape only, not level
    gain = max_db * np.tanh(diff / max_db) * float(np.clip(strength, 0.0, 1.0))
    gain[(centres > 16000.0) & (gain > 0)] = 0.0

    # zero-phase magnitude on a dense grid -> windowed symmetric FIR
    nfft, taps = 1 << 16, 8191
    grid = np.fft.rfftfreq(nfft, 1.0 / sr)
    g_db = np.interp(np.log2(np.maximum(grid, 1.0)), np.log2(centres), gain)
    ir = np.fft.irfft(10.0 ** (g_db / 20.0), nfft)
    ir = np.roll(ir, taps // 2)[:taps] * np.blackman(taps)
    if audio.ndim == 1:
        out = fftconvolve(audio.astype(np.float64), ir, mode="same")
    else:
        out = np.stack([fftconvolve(audio[:, c].astype(np.float64), ir, mode="same")
                        for c in range(audio.shape[1])], axis=1)
    out = out.astype(np.float32)

    if match_loudness:
        before = integrated_lufs(mono, sr)
        after = integrated_lufs(out if out.ndim == 1 else out.mean(axis=1), sr)
        if before > -70.0 and after > -70.0:
            out = (out * 10.0 ** ((before - after) / 20.0)).astype(np.float32)
    if return_curve:
        return out, {"freqs": centres, "gain_db": gain,
                     "max_boost_db": round(float(gain.max()), 2),
                     "max_cut_db": round(float(gain.min()), 2)}
    return out


@njit(cache=True)
def _quantize_shaped(x, dither, q, c):
    out = np.empty_like(x)
    err = 0.0
    top = (q - 1.0) / q
    for i in range(x.shape[0]):
        w = x[i] + dither[i] - c * err
        y = np.round(w * q) / q
        y = min(max(y, -1.0), top)
        err = y - w
        out[i] = y
    return out


def dither(
    audio: np.ndarray, bit_depth: int = 16, noise_shaping: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Quantize to `bit_depth` bits with TPDF dither.

    Triangular-PDF noise of +/-1 LSB (the sum of two uniform variables) is added
    before rounding, which makes the quantization error independent of the signal:
    no distortion on quiet passages and fade tails, only a steady noise floor at
    about -98 dBFS for 16 bits. The result is float32 on the exact integer grid
    of that bit depth, so writing it as PCM loses nothing.

    `noise_shaping` feeds the quantization error back (first order, NTF
    1 - 0.7 z^-1) so the noise moves from the midrange up toward 20 kHz, where
    hearing is least sensitive, for roughly 3 dB lower noise at 2-5 kHz.
    """
    if bit_depth >= 32 or audio.size == 0:
        return audio
    q = float(2 ** (bit_depth - 1))                     # LSB = 1 / q
    rng = np.random.default_rng(seed)
    x = np.asarray(audio, dtype=np.float64)
    noise = (rng.random(x.shape) - rng.random(x.shape)) / q
    if noise_shaping:
        flat = lambda a: np.ascontiguousarray(a.T if a.ndim == 2 else a[None, :])  # noqa: E731
        xs, ns = flat(x), flat(noise)
        out = np.stack([_quantize_shaped(xs[c], ns[c], q, 0.7) for c in range(xs.shape[0])])
        return (out.T if x.ndim == 2 else out[0]).astype(np.float32)
    y = np.clip(np.round((x + noise) * q) / q, -1.0, (q - 1.0) / q)
    return y.astype(np.float32)


def export_lufs_report(report: dict) -> str:
    """Format the master report for printing."""
    lines = ["master report"]
    for key in (
        "input_lufs", "output_lufs", "loudness_gain_db",
        "input_peak_db", "output_peak_db", "output_true_peak_db",
        "input_crest_db", "output_crest_db", "crest_lost_db",
    ):
        if key in report:
            lines.append(f"  {key:<20} {report[key]}")
    for stage in report.get("stages", []):
        lines.append(f"  after {stage['stage']:<16} peak {stage['peak_db']:+.2f} dBFS")
    if "warning" in report:
        lines.append(f"  WARNING: {report['warning']}")
    return "\n".join(lines)
