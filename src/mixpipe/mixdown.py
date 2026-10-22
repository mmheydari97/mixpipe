"""Mixdown: run each stem through its chain, pan, and sum to a stereo bus."""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE
from .analysis import peak_db
from .cleanup import clean_stem
from .config import DEFAULT_CONFIG
from .dynamics import (
    apply_gain_db, compress_with_settings, normalize_peak, transient_shaper,
)
from .eq import apply_eq_chain, deesser
from .fx import color, send_returns
from .io_utils import Stem, pad_to_common_length
from .panning import pan_mono_to_stereo
from .plan import SEND_OFF_DB, validate_send_buses


def process_stem(stem: Stem, settings: dict, cfg: dict, do_cleanup: bool = True,
                 cleaned: Stem | None = None) -> Stem:
    """Full per-stem chain: cleanup -> trim -> EQ -> shape -> compress -> de-ess ->
    drive/chorus (fx.color) -> fader -> pan.

    Order matters. `gain_db` is a trim into the chain (so the compressor sees a
    consistent level); `fader_db` sets the level after it. Cleanup first so the compressor isn't reacting to noise. EQ
    before compression so the compressor responds to the tone you've decided on.
    The transient shaper goes before the compressor, which would otherwise level
    the attacks it is trying to move. De-essing goes after it, because
    compression lifts sibilance. Pan last, once the signal is final.

    `cleaned` is this stem's already-cleaned audio (cleanup does not depend on the
    plan, so a session that re-renders many times computes it once).
    """
    s = cleaned if cleaned is not None else (clean_stem(stem, cfg) if do_cleanup else stem)
    audio = s.audio

    audio = apply_gain_db(audio, settings["gain_db"])
    audio = apply_eq_chain(
        audio,
        hpf_hz=settings["hpf_hz"],
        lpf_hz=settings["lpf_hz"],
        bands=settings["eq"],
        sr=s.sr,
    )
    tr = settings.get("transient") or {}
    if abs(tr.get("attack", 0.0)) > 1e-3 or abs(tr.get("sustain", 0.0)) > 1e-3:
        audio = transient_shaper(audio, tr.get("attack", 0.0), tr.get("sustain", 0.0), s.sr)
    audio = compress_with_settings(audio, settings["comp"], s.sr)
    ds = settings.get("deess") or {}
    if ds.get("max_reduction_db", 0.0) > 0.0:
        audio = deesser(
            audio, s.sr, freq=ds.get("freq_hz", 6500.0),
            max_reduction_db=ds["max_reduction_db"],
        )
    audio = color(audio, s.sr, settings)        # the effects the splitter shares (fx.py)
    # the fader: level after the chain, so a level change isn't undone by the compressor
    if abs(settings.get("fader_db", 0.0)) > 1e-6:
        audio = apply_gain_db(audio, settings["fader_db"])
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
    cleaned: dict[str, Stem] | None = None,
    bpm: float | None = None,
) -> tuple[np.ndarray, list[Stem], dict]:
    """Process and sum every stem.

    Returns (stereo bus, processed stems, report). The bus is left at whatever
    level the sum produced, then trimmed to the configured headroom so the
    master chain has room to work. `bpm` (the tempo as rendered) sets a delay
    that is synced to a note value.
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

        out = process_stem(stem, settings, cfg, do_cleanup=do_cleanup,
                           cleaned=(cleaned or {}).get(stem.name))
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
    sends = add_send_buses(processed, plan, cfg, bpm=bpm)
    if sends.shape[0] > bus.shape[0]:           # reverb/delay tails run past the dry mix
        bus = np.pad(bus, ((0, sends.shape[0] - bus.shape[0]), (0, 0)))
    bus[: sends.shape[0]] += sends
    report["send_peak_db"] = round(peak_db(sends), 2) if np.any(sends) else None
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
        if report["send_peak_db"] is not None:
            print(f"  [send] reverb/delay returns peak {report['send_peak_db']:+.2f} dBFS")
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


def add_send_buses(
    processed: list[Stem], plan: dict, cfg: dict | None = None, bpm: float | None = None,
) -> np.ndarray:
    """Shared reverb/delay returns, to be summed in parallel with the dry bus:
    fx.send_returns fed by each stem at its own send level
    (`plan["stems"][n]["sends"]`, dB, -60 = off). Returns a stereo array as long
    as the longest stem or its reverb tail, whichever is longer; all zeros when
    the buses are off or nothing is sent.
    """
    cfg = cfg or DEFAULT_CONFIG
    n = max((s.n_samples for s in processed), default=0)
    sr = processed[0].sr if processed else SAMPLE_RATE
    if not processed or not cfg.get("sends", {}).get("enabled", True):
        return np.zeros((n, 2), dtype=np.float32)
    sources = []
    for s in processed:
        levels = plan.get("stems", {}).get(s.name, {}).get("sends", {})
        sources.append((s.audio, float(levels.get("reverb_db", SEND_OFF_DB)),
                        float(levels.get("delay_db", SEND_OFF_DB))))
    return send_returns(sources, validate_send_buses(plan.get("send_buses"), cfg), sr, bpm=bpm)
