"""Versions of a song's stems at another speed and/or key, made in the
background and cached on disk so the browser can play them as they are.

  out/splitter/<song_id>/versions/<key>/<quality>/<stem>.flac

`key` is e.g. "x070+02" (70% speed, 2 semitones up). Each version gets a
"fast" pass first, so the change is heard within seconds, then a "fine" one
that replaces it (see stretch.qualities). Only the newest KEEP versions of a
song are kept.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from . import stretch
from .pipeline import STEM_NAMES, stem_path

KEEP = 3
KEY_RE = re.compile(r"^x\d{3}[+-]\d{2}$")


def version_key(speed: float, semitones: int) -> str | None:
    """None for the original (100%, same key)."""
    if speed == 1 and semitones == 0:
        return None
    return f"x{round(speed * 100):03d}{semitones:+03d}"


def version_dir(song_dir: Path, key: str) -> Path:
    return Path(song_dir) / "versions" / key


def stems_dir(song_dir: Path, key: str, quality: str) -> Path:
    return version_dir(song_dir, key) / quality


def ready_quality(song_dir: Path, key: str) -> str | None:
    """The best finished pass of a version: "fine", "fast" or None."""
    for quality in ("fine", "fast"):
        if stems_dir(song_dir, key, quality).is_dir():
            return quality
    return None


def make(song_dir: Path, speed: float, semitones: int, quality: str,
         cancelled: Callable[[], bool] | None = None,
         progress: Callable[[float], None] | None = None) -> Path:
    """Stretch every stem (in parallel, one Rubber Band process each) into
    versions/<key>/<quality>/. The folder only appears once it's complete.
    `progress(fraction)` gets the average over the stems."""
    import soundfile as sf

    vdir = version_dir(song_dir, version_key(speed, semitones))
    tmp = vdir / f".{quality}-partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    done = dict.fromkeys(STEM_NAMES, 0.0)

    def stem_progress(name: str) -> Callable[[float], None]:
        def report(frac: float) -> None:
            done[name] = frac
            if progress:
                progress(sum(done.values()) / len(done))
        return report

    def one(name: str) -> None:
        y, sr = sf.read(str(stem_path(song_dir, name)), dtype="float32", always_2d=True)
        z = stretch.stretch(y, sr, speed, semitones, quality,
                            formant=name == "vocals" and semitones != 0, cancelled=cancelled,
                            progress=stem_progress(name))
        sf.write(str(tmp / f"{name}.flac"), np.clip(z, -1, 1), sr, format="FLAC", subtype="PCM_16")

    try:
        with ThreadPoolExecutor(len(STEM_NAMES)) as pool:
            list(pool.map(one, STEM_NAMES))
        out = vdir / quality
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if quality == "fine":
        shutil.rmtree(vdir / "fast", ignore_errors=True)
    return out


def prune(song_dir: Path, keep: int = KEEP, protect: tuple[str, ...] = ()) -> None:
    """Delete all but the `keep` most recently requested versions."""
    root = Path(song_dir) / "versions"
    if not root.is_dir():
        return
    dirs = sorted((d for d in root.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs[keep:]:
        if d.name not in protect:
            shutil.rmtree(d, ignore_errors=True)


class Maker:
    """Makes versions on one background thread. Each song wants one version at
    a time: asking for another one cancels unfinished work on the previous."""

    def __init__(self):
        self.lock = threading.Lock()
        self.wanted: dict[str, str | None] = {}   # song dir -> key
        self.busy: set[tuple[str, str]] = set()   # (song dir, key) queued or running
        self.errors: dict[tuple[str, str], str] = {}
        self.running: dict[tuple[str, str], tuple[str, float]] = {}  # -> (quality, fraction)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="versions")

    def request(self, song_dir: Path, speed: float, semitones: int) -> dict:
        speed, semitones = stretch.check(speed, semitones)
        key = version_key(speed, semitones)
        song = str(song_dir)
        if key is not None:
            stretch.check_available(semitones)
        with self.lock:
            self.wanted[song] = key
            if key is None:
                return self.status(song_dir, None)
            vdir = version_dir(song_dir, key)
            vdir.mkdir(parents=True, exist_ok=True)
            os.utime(vdir)  # recently used: prune() keeps it
            self.errors.pop((song, key), None)
            self._submit(song_dir, key, speed, semitones)
        return self.status(song_dir, key)

    def _submit(self, song_dir: Path, key: str, speed: float, semitones: int) -> None:
        # with self.lock held
        if ready_quality(song_dir, key) == "fine" or (str(song_dir), key) in self.busy:
            return
        self.busy.add((str(song_dir), key))
        self.pool.submit(self._run, song_dir, key, speed, semitones)

    def status(self, song_dir: Path, key: str | None) -> dict:
        if key is None:
            return {"key": None, "ready": None, "working": False, "error": None,
                    "making": None, "progress": None}
        song = str(song_dir)
        making, frac = self.running.get((song, key), (None, None))
        return {"key": key, "ready": ready_quality(song_dir, key),
                "working": (song, key) in self.busy, "error": self.errors.get((song, key)),
                # the pass being made now ("fast" / "fine") and how far along it is
                "making": making, "progress": None if frac is None else round(frac, 3)}

    def _run(self, song_dir: Path, key: str, speed: float, semitones: int) -> None:
        song = str(song_dir)

        def cancelled() -> bool:
            return self.wanted.get(song) != key

        try:
            for quality in stretch.qualities():
                if cancelled():
                    break
                if ready_quality(song_dir, key) not in (quality, "fine"):
                    self.running[(song, key)] = (quality, 0.0)
                    make(song_dir, speed, semitones, quality, cancelled,
                         progress=lambda f, q=quality: self.running.__setitem__((song, key), (q, f)))
            prune(song_dir, protect=(key,))
        except stretch.Cancelled:
            pass
        except Exception as exc:  # shown in the page; playback carries on
            self.errors[(song, key)] = str(exc) or type(exc).__name__
        finally:
            with self.lock:
                self.busy.discard((song, key))
                self.running.pop((song, key), None)
                # asked for again while it was being cancelled: start over
                if not cancelled() and (song, key) not in self.errors and song_dir.exists():
                    self._submit(song_dir, key, speed, semitones)
