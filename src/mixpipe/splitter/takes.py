"""Takes: you, recorded in the browser over the backing, kept with the song.

  out/splitter/<song_id>/takes/takes.json        [{"n", "name", "created", "start_s", "speed",
                                                   "semitones", "latency_ms", "raw"}]
  out/splitter/<song_id>/takes/<n>.raw.<ext>     the recording as the browser sent it
  out/splitter/<song_id>/takes/<n>.flac          the take on the song's own timeline
  out/splitter/<song_id>/takes/<n>/<key>-<q>.flac  that at another speed/key (cached)

A take is kept like a stem: as long as the song, at its original speed and key,
starting where you started recording. You may record at 70% and two semitones
up; the server takes that back to 100% and the original key with stretch.py
(the code every speed and key change uses), and moves it earlier by the audio
latency the page measured, so it lines up with what you heard. It then plays,
stretches and downloads exactly as the stems do. Nothing here goes in stems/.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from .. import stretch
from ..io_utils import decode
from . import versions
from .pipeline import load_info

PREFIX = "take:"     # a take's name among the stems: "take:1"


def _dir(song_dir: Path) -> Path:
    return Path(song_dir) / "takes"


def load(song_dir: Path) -> list[dict]:
    path = _dir(song_dir) / "takes.json"
    return json.loads(path.read_text()) if path.exists() else []


def _save(song_dir: Path, takes: list[dict]) -> None:
    path = _dir(song_dir) / "takes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(takes, indent=1))
    tmp.replace(path)


def find(song_dir: Path, n: int) -> dict:
    for take in load(song_dir):
        if take["n"] == n:
            return take
    raise KeyError(f"no take {n}")


def name_of(n: int) -> str:
    return f"{PREFIX}{n}"


def number(name: str) -> int | None:
    """1 for "take:1", None for anything else."""
    if not name.startswith(PREFIX) or not name[len(PREFIX):].isdigit():
        return None
    return int(name[len(PREFIX):])


def _place(song_dir: Path, take: dict) -> None:
    """Make <n>.flac from the raw recording: latency off the front, back to the
    song's speed and key, at its place in a song-length track."""
    import soundfile as sf

    info = load_info(song_dir)
    sr = info["sr"]
    audio, rec_sr = decode(_dir(song_dir) / take["raw"])
    if rec_sr != sr:
        from ..io_utils import to_sample_rate

        audio = to_sample_rate(audio, rec_sr, sr)
    audio = audio if audio.ndim == 2 else np.stack([audio, audio], axis=1)
    lag = int(round(take["latency_ms"] / 1000 * sr))
    audio = audio[lag:] if lag >= 0 else np.concatenate([np.zeros((-lag, 2), np.float32), audio])
    if take["speed"] != 1 or take["semitones"]:
        stretch.check_available(take["semitones"])
        audio = stretch.stretch(audio, sr, 1 / take["speed"], -take["semitones"], "fine", formant=True)
    n = int(round(info["duration"] * sr))
    track = np.zeros((n, 2), dtype=np.float32)
    at = int(round(take["start_s"] * sr))
    if at < n:
        part = audio[: n - at]
        track[at: at + len(part)] = part
    sf.write(str(_dir(song_dir) / f"{take['n']}.flac"), np.clip(track, -1, 1), sr, format="FLAC", subtype="PCM_16")
    shutil.rmtree(_dir(song_dir) / str(take["n"]), ignore_errors=True)   # its other versions are stale


def add(song_dir: Path, data: bytes, filename: str, start_s: float, speed: float = 1.0,
        semitones: int = 0, latency_ms: float = 0.0) -> dict:
    """Keep a recording as the song's next take. `start_s` is where in the song
    (as recorded) the recording started, `speed`/`semitones` what the page was
    playing at, `latency_ms` how late the recording is against what was heard."""
    speed, semitones = stretch.check(speed, semitones)
    takes = load(song_dir)
    n = max((t["n"] for t in takes), default=0) + 1
    ext = Path(filename).suffix.lower() or ".webm"
    if not ext[1:].isalnum() or len(ext) > 6:
        ext = ".webm"
    _dir(song_dir).mkdir(parents=True, exist_ok=True)
    raw = f"{n}.raw{ext}"
    (_dir(song_dir) / raw).write_bytes(data)
    take = {"n": n, "name": f"Take {n}", "created": time.time(), "start_s": max(0.0, float(start_s)),
            "speed": speed, "semitones": semitones, "latency_ms": float(latency_ms), "raw": raw}
    try:
        _place(song_dir, take)
    except Exception:
        (_dir(song_dir) / raw).unlink(missing_ok=True)
        raise
    _save(song_dir, takes + [take])
    return take


def nudge(song_dir: Path, n: int, ms: float) -> dict:
    """Move a take `ms` later (negative: earlier) and place it again."""
    takes = load(song_dir)
    take = next((t for t in takes if t["n"] == n), None)
    if take is None:
        raise KeyError(f"no take {n}")
    take["latency_ms"] -= float(ms)
    _place(song_dir, take)
    _save(song_dir, takes)
    return take


def delete(song_dir: Path, n: int) -> None:
    takes = load(song_dir)
    take = next((t for t in takes if t["n"] == n), None)
    if take is None:
        raise KeyError(f"no take {n}")
    for p in (_dir(song_dir) / take["raw"], _dir(song_dir) / f"{n}.flac"):
        p.unlink(missing_ok=True)
    shutil.rmtree(_dir(song_dir) / str(n), ignore_errors=True)
    _save(song_dir, [t for t in takes if t["n"] != n])


def path(song_dir: Path, n: int, key: str | None = None, quality: str = "fine") -> Path:
    """The take as a stem of version `key` (None: the original speed and key),
    stretched like the stems the first time it's asked for."""
    import soundfile as sf

    find(song_dir, n)
    base = _dir(song_dir) / f"{n}.flac"
    if not key:
        return base
    out = _dir(song_dir) / str(n) / f"{key}-{quality}.flac"
    if not out.exists():
        audio, sr = sf.read(str(base), dtype="float32", always_2d=True)
        semi = versions.semitones_of(key)
        stretch.check_available(semi)
        z = stretch.stretch(audio, sr, versions.speed_of(key), semi, quality, formant=bool(semi))
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".flac", delete=False) as tmp:
            sf.write(tmp.name, np.clip(z, -1, 1), sr, format="FLAC", subtype="PCM_16")
        Path(tmp.name).replace(out)
    return out


def paths(song_dir: Path, key: str | None = None, quality: str = "fine") -> dict[str, Path]:
    """{"take:<n>": file} for every take, as stems of version `key`."""
    return {name_of(t["n"]): path(song_dir, t["n"], key, quality) for t in load(song_dir)}
