"""Mixdown: run each stem through its chain, pan, and sum to a stereo bus."""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE
from .analysis import peak_db
from .cleanup import clean_stem
from .config import DEFAULT_CONFIG
from .dynamics import apply_gain_db, compress_with_settings, normalize_peak
from .eq import apply_eq_chain
from .io_utils import Stem, pad_to_common_length
from .panning import pan_mono_to_stereo


def process_stem(stem: Stem, settings: dict, cfg: dict, do_cleanup: bool = True) -> Stem:
    """Full per-stem chain: cleanup -> trim -> EQ -> compress -> pan.

    Order matters. Cleanup first so the compressor isn't reacting to noise. EQ
    before compression so the compressor responds to the tone you've decided on.
    Pan last, once the signal is final.
    """
    s = clean_stem(stem, cfg) if do_cleanup else stem
    audio = s.audio

    audio = apply_gain_db(audio, settings["gain_db"])
    audio = apply_eq_chain(
        audio,
        hpf_hz=settings["hpf_hz"],
        lpf_hz=settings["lpf_hz"],
        bands=settings["eq"],
        sr=s.sr,
    )
    audio = compress_with_settings(audio, settings["comp"], s.sr)
    audio = pan_mono_to_stereo(audio, settings["pan"])

    out = s.copy_with(audio.astype(np.float32))
    out.meta["processed"] = True
    out.meta["post_peak_db"] = round(peak_db(audio), 2)
    return out


def sum_bus(stems: list[Stem]) -> np.ndarray:
    """Sum stereo stems. No auto-attenuation: gain staging is the plan's job."""
    if not stems:
        raise ValueError("nothing to sum")
    stems = pad_to_common_length(stems)
    n = stems[0].n_samples
    bus = np.zeros((n, 2), dtype=np.float64)
    for s in stems:
        a = s.audio
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        bus[: a.shape[0], :] += a
    return bus.astype(np.float32)


def mixdown(
    stems: list[Stem],
    plan: dict,
    cfg: dict | None = None,
    do_cleanup: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, list[Stem], dict]:
    """Process and sum every stem.

    Returns (stereo bus, processed stems, report). The bus is left at whatever
    level the sum produced, then trimmed to the configured headroom so the
    master chain has room to work.
    """
    cfg = cfg or DEFAULT_CONFIG
    processed: list[Stem] = []
    report: dict = {"stems": [], "warnings": []}

    for stem in stems:
        settings = plan["stems"].get(stem.name)
        if settings is None:
            report["warnings"].append(f"no plan entry for {stem.name}, skipped")
            continue
        if settings.get("mute"):
            report["warnings"].append(f"{stem.name} muted by the plan")
            continue

        out = process_stem(stem, settings, cfg, do_cleanup=do_cleanup)
        processed.append(out)
        entry = {
            "name": out.name,
            "role": out.role,
            "peak_db_after": out.meta["post_peak_db"],
            "gain_db": settings["gain_db"],
            "pan": settings["pan"],
        }
        report["stems"].append(entry)
        if verbose:
            print(
                f"  [stem] {out.name:<20} {out.role:<14} "
                f"gain {settings['gain_db']:+6.2f} dB  "
                f"pan {settings['pan']:+.2f}  "
                f"peak {out.meta['post_peak_db']:+6.2f} dBFS"
            )

    if not processed:
        raise RuntimeError("every stem was skipped or muted, nothing to mix")

    bus = sum_bus(processed)
    report["bus_peak_db_raw"] = round(peak_db(bus), 2)

    if report["bus_peak_db_raw"] > 0.0:
        report["warnings"].append(
            f"bus summed to {report['bus_peak_db_raw']:+.2f} dBFS before trim"
        )

    headroom = cfg["mixdown"]["headroom_db"]
    bus = normalize_peak(bus, target_db=headroom)
    report["bus_peak_db_trimmed"] = round(peak_db(bus), 2)
    report["headroom_target_db"] = headroom

    if verbose:
        print(
            f"  [bus ] summed peak {report['bus_peak_db_raw']:+.2f} dBFS "
            f"-> trimmed to {report['bus_peak_db_trimmed']:+.2f} dBFS"
        )

    return bus, processed, report


def export_stems(
    processed: list[Stem], directory: str, sr: int = SAMPLE_RATE
) -> list[str]:
    """Write the processed stems out, so they can be dropped into a real DAW.

    This is the honest escape hatch: if the auto-mix is 80% right, take these
    into Reaper/Logic and finish by hand.
    """
    from .io_utils import write_audio
    from pathlib import Path

    Path(directory).mkdir(parents=True, exist_ok=True)
    return [
        write_audio(str(Path(directory) / f"{s.name}_processed.wav"), s.audio, sr)
        for s in processed
    ]


def add_send_buses(processed: list[Stem], plan: dict) -> np.ndarray:
    """Shared reverb/delay sends, summed in parallel with the dry bus.

    # TODO: depends on reverb.py. A mix of close-miked dry sources needs shared
    # space to sound like one performance rather than four separate recordings.
    """
    pass  # TODO
