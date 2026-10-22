"""Pieces shared by the web apps (chord finder, splitter, mixer): the song
library routes, the background job queue and the OpenRouter model catalogue.

Each app passes its own job (what runs on a song), the file that marks a song
as finished, and parse_options() for the extra fields its upload form sends.
"""

from __future__ import annotations

import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .. import fetch, songstore

STATIC = Path(__file__).parent / "static"
# Browsers otherwise cache the page's JS/CSS heuristically and keep running an
# old script after an update; no-cache makes them revalidate (a cheap 304).
NO_CACHE = {"Cache-Control": "no-cache"}
_ID = re.compile(r"^[\w-]+$", re.UNICODE)


def attachment(filename: str) -> dict:
    """Headers that make the browser save a download as `filename`. The real
    name goes in filename* (RFC 5987): headers are latin-1, so a song title in
    Persian would otherwise fail; the plain filename= is an ASCII stand-in."""
    from urllib.parse import quote

    fallback = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    return {"Content-Disposition":
            f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"}


def as_bool(value, default: bool) -> bool:
    """Form fields arrive as "true"/"false" strings, JSON as booleans."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class JobQueue:
    """A small job queue over a thread pool that knows who is waiting.

    One worker by default: the models (Demucs, Whisper) each want the machine, so
    songs are done one after another. `workers` > 1 runs that many at once, which
    pays off only with memory to spare. Jobs are keyed (by song id) so the page can
    show "2 ahead" and cancel one: a job still waiting is dropped, and a running
    one is told to stop at its next progress report (`is_cancelled`)."""

    def __init__(self, workers: int = 1, name: str = "songs"):
        self.workers = max(1, int(workers))
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=name)
        self._lock = threading.Lock()
        self._waiting: list[str] = []
        self._running: set[str] = set()
        self._cancelled: set[str] = set()

    def submit(self, key: str, fn: Callable, *args, on_cancel: Callable | None = None):
        """Queue fn(*args) under `key`. `on_cancel()` runs instead of fn if the job
        is cancelled while it waits."""
        with self._lock:
            self._cancelled.discard(key)
            self._waiting.append(key)

        def run():
            with self._lock:
                if key in self._waiting:
                    self._waiting.remove(key)
                dropped = key in self._cancelled
                if not dropped:
                    self._running.add(key)
            if dropped:
                with self._lock:
                    self._cancelled.discard(key)
                if on_cancel:
                    on_cancel()
                return None
            try:
                return fn(*args)
            finally:
                with self._lock:
                    self._running.discard(key)
                    self._cancelled.discard(key)

        return self._pool.submit(run)

    def ahead(self, key: str) -> int | None:
        """How many jobs must finish (or are running) before `key` starts; None if
        it isn't waiting."""
        with self._lock:
            if key not in self._waiting:
                return None
            busy = len(self._running) if len(self._running) >= self.workers else 0
            return self._waiting.index(key) + busy

    def cancel(self, key: str) -> str | None:
        """"queued" (dropped before it started), "running" (asked to stop) or None."""
        with self._lock:
            if key in self._waiting:
                self._cancelled.add(key)
                self._waiting.remove(key)
                return "queued"
            if key in self._running:
                self._cancelled.add(key)
                return "running"
        return None

    def is_cancelled(self, key: str) -> bool:
        with self._lock:
            return key in self._cancelled

    def is_waiting(self, key: str) -> bool:
        with self._lock:
            return key in self._waiting


class SongService:
    """The store root plus a job queue (one worker unless asked for more)."""

    def __init__(self, root: str | Path, job: Callable[[Path, dict], object],
                 done_file: str, done_stage: str, run_inline: bool = False,
                 name: str = "songs", workers: int = 1):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.job = job
        self.done_file = done_file
        self.done_stage = done_stage
        self.run_inline = run_inline
        self.queue = JobQueue(workers, name)

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

    def cancelled(self, song_id: str) -> Callable[[], bool]:
        """What a job passes to songstore.run_job to notice it was cancelled."""
        return lambda: self.queue.is_cancelled(song_id)

    def submit(self, song_id: str, fn: Callable, *args) -> None:
        """Queue any job on a song (analysis, melody...), or run it now when inline."""
        d = self.root / song_id
        if self.run_inline:
            fn(*args)
        else:
            self.queue.submit(song_id, fn, *args,
                              on_cancel=lambda: songstore.set_status(
                                  d, "cancelled", songstore.get_status(d).get("stage", "decode"),
                                  0.0, "cancelled"))

    def start(self, song_id: str, options: dict, rerun: bool) -> dict:
        d = self.root / song_id
        if (d / self.done_file).exists() and not rerun:
            # same song added again: keep the existing (possibly edited) result
            songstore.set_status(d, "done", self.done_stage, 1.0, "already done")
        else:
            needs_download = not songstore.read_meta(d)["inputs"]
            songstore.set_status(d, "queued", "download" if needs_download else "decode",
                                 0.0, "waiting")
            self.submit(song_id, self.job, d, options)
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
        st = songstore.get_status(svc.song_dir(song_id))
        if st.get("state") == "queued":
            st["ahead"] = svc.queue.ahead(song_id)       # jobs before this one, if it is waiting
        return st

    @app.post("/api/songs/{song_id}/cancel")
    def cancel(song_id: str):
        """Stop a queued job at once, or a running one at its next progress report."""
        d = svc.song_dir(song_id)
        what = svc.queue.cancel(song_id)
        if what == "queued":
            songstore.set_status(d, "cancelled", songstore.get_status(d).get("stage", "decode"),
                                 0.0, "cancelled")
        return {"id": song_id, "cancelled": what}

    @app.delete("/api/songs/{song_id}")
    def delete_song(song_id: str):
        shutil.rmtree(svc.song_dir(song_id))
        return {"ok": True}


def add_openrouter_routes(app: FastAPI,
                          decorate: Callable[[str, list[dict]], list[dict]] | None = None) -> None:
    """The model picker's data (createModelPicker in static/common.js):
    GET /api/openrouter/status says whether a key is set, and
    GET /api/openrouter/models?kind=chat|transcription returns the public catalogue.
    `decorate(kind, models)` adds what an app knows about the models."""
    from .. import openrouter

    @app.get("/api/openrouter/status")
    def or_status():
        return {"has_key": openrouter.api_key() is not None}

    @app.get("/api/openrouter/models")
    def or_models(kind: str = "chat"):
        try:
            models = openrouter.list_models(kind)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"couldn't reach OpenRouter: {exc}") from exc
        return decorate(kind, models) if decorate else models


class _RevalidatedStatic(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


def page_html(page: str) -> HTMLResponse:
    """static/<page>, with /static/x.js -> /static/x.js?v=<mtime>: an edited file
    gets a new URL, so even a copy the browser already cached can't be reused by
    mistake."""
    def versioned(m: re.Match) -> str:
        f = STATIC / m.group(1)
        return f'/static/{m.group(1)}?v={int(f.stat().st_mtime)}"' if f.is_file() else m.group(0)

    html = re.sub(r'/static/([\w.-]+\.(?:js|css))"', versioned, (STATIC / page).read_text())
    return HTMLResponse(html, headers=NO_CACHE)


def mount_static(app: FastAPI) -> None:
    """The shared static folder at /static."""
    app.mount("/static", _RevalidatedStatic(directory=STATIC), name="static")


def add_page(app: FastAPI, page: str) -> None:
    """Serve static/<page> at / and the shared static folder at /static."""

    @app.get("/")
    def index():
        return page_html(page)

    mount_static(app)
