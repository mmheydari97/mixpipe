"""Pieces shared by the song tool web apps (chord finder, splitter): the song
library routes and the one-at-a-time background worker.

Each app passes its own job (what runs on a song), the file that marks a song
as finished, and parse_options() for the extra fields its upload form sends.
"""

from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import fetch, songstore

STATIC = Path(__file__).parent / "static"
_ID = re.compile(r"^[\w-]+$", re.UNICODE)


def as_bool(value, default: bool) -> bool:
    """Form fields arrive as "true"/"false" strings, JSON as booleans."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class SongService:
    """The store root plus a single worker thread: the GPU models don't share
    well, so songs are processed one at a time."""

    def __init__(self, root: str | Path, job: Callable[[Path, dict], object],
                 done_file: str, done_stage: str, run_inline: bool = False,
                 name: str = "songs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.job = job
        self.done_file = done_file
        self.done_stage = done_stage
        self.run_inline = run_inline
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    def song_dir(self, song_id: str) -> Path:
        if not _ID.match(song_id):
            raise HTTPException(400, "bad song id")
        d = self.root / song_id
        if not (d / "meta.json").exists():
            raise HTTPException(404, "no such song")
        return d

    def result_path(self, song_id: str) -> Path:
        p = self.song_dir(song_id) / self.done_file
        if not p.exists():
            raise HTTPException(409, "song is not processed yet")
        return p

    def start(self, song_id: str, options: dict, rerun: bool) -> dict:
        d = self.root / song_id
        if (d / self.done_file).exists() and not rerun:
            # same song added again: keep the existing (possibly edited) result
            songstore.set_status(d, "done", self.done_stage, 1.0, "already done")
        else:
            needs_download = not songstore.read_meta(d)["inputs"]
            songstore.set_status(d, "queued", "download" if needs_download else "decode",
                                 0.0, "waiting")
            if self.run_inline:
                self.job(d, options)
            else:
                self.worker.submit(self.job, d, options)
        return {"id": song_id}


def add_song_routes(app: FastAPI, svc: SongService,
                    parse_options: Callable[[dict], dict], resolver=None) -> None:
    """GET/POST /api/songs, POST /api/songs/link, status and delete.

    parse_options(fields) turns the form/JSON fields other than files, url and
    rerun into the job's options, raising HTTPException(400) on bad input."""

    @app.get("/api/songs")
    def songs():
        return songstore.list_songs(svc.root)

    @app.post("/api/songs")
    async def upload(request: Request):
        form = await request.form()
        files = [f for f in form.getlist("files") if hasattr(f, "read")]
        fields = {k: v for k, v in form.multi_items() if k != "files"}
        rerun = as_bool(fields.pop("rerun", None), False)
        options = parse_options(fields)
        data = [(f.filename or "audio", await f.read()) for f in files]
        try:
            song_id = songstore.new_song(data, svc.root)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return svc.start(song_id, options, rerun)

    @app.post("/api/songs/link")
    def from_link(fields: dict):
        # plain def: FastAPI runs it in a thread, so the lookup doesn't block the loop
        fields = dict(fields)
        url = str(fields.pop("url", "") or "")
        rerun = as_bool(fields.pop("rerun", None), False)
        options = parse_options(fields)
        try:
            source = (resolver or fetch.resolve)(url)
        except fetch.FetchError as exc:
            raise HTTPException(400, str(exc)) from exc
        song_id = songstore.new_song_from_url(source, svc.root)
        return svc.start(song_id, options, rerun)

    @app.get("/api/songs/{song_id}/status")
    def status(song_id: str):
        return songstore.get_status(svc.song_dir(song_id))

    @app.delete("/api/songs/{song_id}")
    def delete_song(song_id: str):
        shutil.rmtree(svc.song_dir(song_id))
        return {"ok": True}


def add_page(app: FastAPI, page: str) -> None:
    """Serve static/<page> at / and the shared static folder at /static."""

    @app.get("/")
    def index():
        return FileResponse(STATIC / page)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
