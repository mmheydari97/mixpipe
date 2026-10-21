"""A folder-per-song store shared by the song tools (chord finder, splitter).

Each song lives in <root>/<song_id>/:

  meta.json    id, title, created, input file names, source (songs from a link)
  inputs/      the uploaded or downloaded files, untouched
  status.json  job state, for the UI to poll

Each tool adds its own files next to these. Ids are derived from the content
(or the link), so adding the same song twice finds the existing folder.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable

from .fetch import FetchError, song_key
from .io_utils import UPLOAD_EXTS

Progress = Callable[[str, float, str], None]


class JobCancelled(Exception):
    """Raised inside a job (from its progress callback) once the user cancelled it."""


def _slug(text: str) -> str:
    s = re.sub(r"[^\w-]+", "-", text.strip().lower(), flags=re.UNICODE).strip("-")
    return s[:40] or "song"


def _safe_name(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE) or "audio"


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def read_meta(song_dir: Path) -> dict:
    return json.loads((Path(song_dir) / "meta.json").read_text())


def new_song(files: list[tuple[str, bytes]], root: Path, title: str | None = None) -> str:
    """Store uploaded files as a new song. Returns its id."""
    if not files:
        raise ValueError("no files uploaded")
    for name, _ in files:
        if Path(name).suffix.lower() not in UPLOAD_EXTS:
            raise ValueError(f"{name}: unsupported file type")
    digest = hashlib.sha1()
    for name, data in files:
        digest.update(name.encode())
        digest.update(data)
    title = title or Path(files[0][0]).stem
    song_id = f"{_slug(title)}-{digest.hexdigest()[:6]}"
    song_dir = Path(root) / song_id
    (song_dir / "inputs").mkdir(parents=True, exist_ok=True)
    names = []
    for name, data in files:
        safe = _safe_name(name)
        (song_dir / "inputs" / safe).write_bytes(data)
        names.append(safe)
    _write_json(song_dir / "meta.json", {
        "id": song_id, "title": title, "created": time.time(), "inputs": names,
    })
    set_status(song_dir, "queued", "decode", 0.0, "waiting")
    return song_id


def new_song_from_url(source: dict, root: Path, title: str | None = None) -> str:
    """Register a song from a resolved link (see fetch.resolve). The audio is
    downloaded later by the tool's job (see download_inputs). Returns its id."""
    title = title or source.get("title") or "song"
    slug = re.sub(r"-{2,}", "-", _slug(title))  # "Artist - Song" -> artist-song
    song_id = f"{slug}-{song_key(source)[:6]}"
    song_dir = Path(root) / song_id
    (song_dir / "inputs").mkdir(parents=True, exist_ok=True)
    meta_path = song_dir / "meta.json"
    if not meta_path.exists():
        _write_json(meta_path, {
            "id": song_id, "title": title, "created": time.time(), "inputs": [],
            "source": source,
        })
        set_status(song_dir, "queued", "download", 0.0, "waiting")
    return song_id


def download_inputs(song_dir: Path, report: Progress,
                    downloader: Callable | None = None) -> dict:
    """The 'download' stage: fetch a link song's audio if it isn't here yet.
    Returns the (possibly updated) meta."""
    song_dir = Path(song_dir)
    meta = read_meta(song_dir)
    source = meta.get("source")
    if source and not meta["inputs"]:
        report("download", 0.0, f"fetching from {source['site']}")
        if downloader is None:
            from .fetch import download as downloader
        path = downloader(source, song_dir / "inputs",
                          progress=lambda f, msg: report("download", f, msg))
        meta["inputs"] = [path.name]
        _write_json(song_dir / "meta.json", meta)
        report("download", 1.0, f"downloaded {path.stat().st_size / 1e6:.1f} MB")
    return meta


def list_songs(root: Path) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        return []
    songs = []
    for meta_path in root.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        meta["status"] = get_status(meta_path.parent)
        songs.append(meta)
    return sorted(songs, key=lambda m: m.get("created", 0), reverse=True)


def set_status(song_dir: Path, state: str, stage: str, progress: float,
               message: str = "", error: str | None = None) -> None:
    _write_json(Path(song_dir) / "status.json", {
        "state": state, "stage": stage, "progress": round(progress, 3),
        "message": message, "error": error, "updated": time.time(),
    })


def get_status(song_dir: Path) -> dict:
    path = Path(song_dir) / "status.json"
    if not path.exists():
        return {"state": "unknown"}
    status = json.loads(path.read_text())
    meta_path = Path(song_dir) / "meta.json"
    status["from_url"] = meta_path.exists() and "source" in json.loads(meta_path.read_text())
    return status


def run_job(song_dir: Path, work: Callable[[Progress], object], done_stage: str,
            cancelled: Callable[[], bool] | None = None):
    """Run work(progress) with status.json bookkeeping; never raises. Returns
    work's result, or None if it failed (the error is in status.json) or was
    cancelled (state "cancelled").

    `cancelled()` is polled every time the job reports progress, so a job stops at
    its next progress report. A step that doesn't report while it runs (a model
    call) finishes first."""
    song_dir = Path(song_dir)
    last = {"t": 0.0}

    def progress(stage: str, frac: float, msg: str) -> None:
        if cancelled is not None and cancelled():
            raise JobCancelled()
        now = time.monotonic()
        if frac in (0.0, 1.0) or now - last["t"] > 0.5:
            last["t"] = now
            set_status(song_dir, "running", stage, frac, msg)

    try:
        if cancelled is not None and cancelled():
            raise JobCancelled()
        result = work(progress)
    except JobCancelled:
        st = get_status(song_dir)
        set_status(song_dir, "cancelled", st.get("stage", "decode"), st.get("progress", 0.0),
                   "cancelled")
        return None
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        st = get_status(song_dir)
        error = str(exc) if isinstance(exc, FetchError) else f"{type(exc).__name__}: {exc}"
        set_status(song_dir, "error", st.get("stage", "decode"), st.get("progress", 0.0),
                   "failed", error=error)
        return None
    set_status(song_dir, "done", done_stage, 1.0, "done")
    return result
