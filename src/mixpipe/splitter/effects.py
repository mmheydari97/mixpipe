"""Effects in the splitter: the mixer's own (mixpipe/fx.py, validated by the
mixer plan's rules), rendered on the server per stem, so the page plays them
and every download includes them.

A stem with effects becomes one file: the stem through its tone (bass/treble
shelves), drive and chorus, plus its reverb and echo. Reverb and echo are
linear, so the per-stem tails add up to exactly the shared room the mixer
uses, and the stem's fader, mute and solo in the browser apply to its tail
too, like a post-fader send.

  out/splitter/<song_id>/fx/<source>/<stem>-<hash>.flac

<source> is "orig" or the version the effects were made from ("x070+02-fine");
<hash> covers the stem's settings, and the room/echo settings and tempo when it
sends to them, so changing one knob re-renders one stem. The newest KEEP files
are kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import fx
from ..config import DEFAULT_CONFIG
from ..eq import apply_eq_chain
from ..plan import _clamp as clamp
from ..plan import validate_fx, validate_send_buses
from .pipeline import STEM_NAMES, stem_path

KEEP = 48
VERSION = 1          # bump when the processing changes, so old renders aren't reused
SOURCE_RE = re.compile(r"^(orig|x\d{3}[+-]\d{2}-(fast|fine))$")
FILE_RE = re.compile(r"^(" + "|".join(STEM_NAMES) + r")-[0-9a-f]{16}\.flac$")


def validate(settings: dict | None) -> dict:
    """The page's effect settings, clamped by the mixer plan's limits:
    {"stems": {name: {bass_db, treble_db, drive, chorus, reverb_db, delay_db}},
     "send_buses": the plan's reverb/delay bus settings}."""
    settings = settings if isinstance(settings, dict) else {}
    given = settings.get("stems") if isinstance(settings.get("stems"), dict) else {}
    stems = {}
    for name in STEM_NAMES:
        e = given.get(name) if isinstance(given.get(name), dict) else {}
        inserts = validate_fx({"drive": {"amount": e.get("drive", 0.0)},
                               "chorus": {"mix": e.get("chorus", 0.0)}})
        stems[name] = {
            "bass_db": clamp(e.get("bass_db", 0.0), "tone_db"),
            "treble_db": clamp(e.get("treble_db", 0.0), "tone_db"),
            "drive": inserts["drive"]["amount"],
            "chorus": inserts["chorus"]["mix"],
            "reverb_db": clamp(e.get("reverb_db", fx.OFF_DB), "send_db", fx.OFF_DB),
            "delay_db": clamp(e.get("delay_db", fx.OFF_DB), "send_db", fx.OFF_DB),
        }
    return {"stems": stems, "send_buses": validate_send_buses(settings.get("send_buses"), DEFAULT_CONFIG)}


def _sends(e: dict) -> tuple[bool, bool]:
    return e["reverb_db"] > fx.OFF_DB, e["delay_db"] > fx.OFF_DB


def active(e: dict) -> bool:
    """Does this stem have any effect on?"""
    return (abs(e["bass_db"]) > 0.05 or abs(e["treble_db"]) > 0.05 or e["drive"] > 1e-3
            or e["chorus"] > 1e-3 or any(_sends(e)))


def apply(audio: np.ndarray, sr: int, e: dict, buses: dict, bpm: float | None = None) -> np.ndarray:
    """One stem through its effects: tone, drive, chorus (as the mixer's chain
    runs them), plus its own reverb and echo, at its length."""
    y = apply_eq_chain(audio, hpf_hz=0.0, lpf_hz=0.0, bands=fx.tone_bands(e["bass_db"], e["treble_db"]), sr=sr)
    y = fx.color(y, sr, {"drive": {"amount": e["drive"]}, "chorus": {"mix": e["chorus"]}})
    if any(_sends(e)):
        y = fx.as_stereo(np.asarray(y, dtype=np.float32))
        y = y + fx.send_returns([(y, e["reverb_db"], e["delay_db"])], buses, sr, bpm=bpm, tail=False)[: len(y)]
    return np.asarray(y, dtype=np.float32)


def _key(e: dict, buses: dict, bpm: float | None) -> str:
    reverb, delay = _sends(e)
    synced = delay and bool(buses["delay"].get("sync"))
    what = {"v": VERSION, "stem": e, "reverb": buses["reverb"] if reverb else None,
            "delay": buses["delay"] if delay else None, "bpm": round(bpm, 2) if synced and bpm else None}
    return hashlib.sha1(json.dumps(what, sort_keys=True).encode()).hexdigest()[:16]


def source_name(folder: Path | None) -> str:
    """"orig" for the stems, "<key>-<quality>" for a version's folder."""
    return "orig" if folder is None else f"{folder.parent.name}-{folder.name}"


def render(song_dir: Path, folder: Path | None, settings: dict, bpm: float | None = None) -> dict[str, Path]:
    """The stems with effects, made from `folder` (a version's stems, None for
    the originals) and cached. Returns {name: file} for the stems that have
    any; the others play as they are. `settings` comes from validate()."""
    import soundfile as sf

    song_dir = Path(song_dir)
    out_dir = song_dir / "fx" / source_name(folder)
    wanted = {n: e for n, e in settings["stems"].items() if active(e)}
    paths = {n: out_dir / f"{n}-{_key(e, settings['send_buses'], bpm)}.flac" for n, e in wanted.items()}

    def make(name: str) -> None:
        path = paths[name]
        if path.exists():
            os.utime(path)                     # recently used: kept by prune()
            return
        src = folder / f"{name}.flac" if folder else stem_path(song_dir, name)
        audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
        y = apply(audio, sr, wanted[name], settings["send_buses"], bpm)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial.flac")
        sf.write(str(tmp), np.clip(y, -1, 1), sr, format="FLAC", subtype="PCM_16")
        tmp.replace(path)

    with ThreadPoolExecutor(max(1, len(wanted)), thread_name_prefix="fx") as pool:
        list(pool.map(make, wanted))
    prune(song_dir)
    return paths


def prune(song_dir: Path, keep: int = KEEP) -> None:
    """Keep the newest `keep` renders of a song."""
    files = sorted((song_dir / "fx").glob("*/*.flac"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)
