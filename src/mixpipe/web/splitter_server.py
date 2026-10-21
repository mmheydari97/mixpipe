"""Local web UI for the stem splitter.

    python -m mixpipe.cli split-ui           # http://127.0.0.1:8766

Same song library plumbing as the chord finder (web/common.py); songs are
separated one at a time in a background worker and the page polls /status.
Speed/key versions (splitter/versions.py) are made on a second worker, so the
page asks for one with POST /versions and polls until it's ready.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .. import songstore
from ..splitter import pipeline, render, stretch, versions
from .common import SongService, add_page, add_song_routes


class RenderIn(BaseModel):
    gains: dict[str, float] = {}
    speed: float = 1.0
    semitones: float = 0
    fmt: str = "wav"


class VersionIn(BaseModel):
    speed: float = 1.0
    semitones: float = 0


def parse_options(fields: dict) -> dict:
    return {}  # nothing to choose: one model, every stem


def create_app(
    root: str | Path = pipeline.DEFAULT_ROOT,
    separator=None,
    run_inline: bool = False,
    resolver=None,
    downloader=None,
    workers: int = 1,
) -> FastAPI:
    """Build the app. `separator`/`resolver`/`downloader` are injectable for
    tests, and `run_inline` separates synchronously instead of in the worker."""
    app = FastAPI(title="mixpipe splitter")

    def job(d: Path, options: dict):
        return pipeline.run_job(d, options, cancelled=svc.cancelled(d.name),
                                separator=separator, downloader=downloader)

    svc = SongService(root, job, done_file="stems.json", done_stage="separate",
                      run_inline=run_inline, name="splitter", workers=workers)
    add_song_routes(app, svc, parse_options, resolver=resolver)
    maker = versions.Maker()

    @app.get("/api/songs/{song_id}")
    def get_song(song_id: str):
        svc.result_path(song_id)
        d = svc.song_dir(song_id)
        meta = songstore.read_meta(d)
        return {**pipeline.load_info(d), "id": song_id, "title": meta["title"],
                "source": meta.get("source")}

    @app.get("/api/songs/{song_id}/stems/{name}")
    def stem(song_id: str, name: str):
        if name not in pipeline.STEM_NAMES:
            raise HTTPException(404, "no such stem")
        svc.result_path(song_id)
        return FileResponse(pipeline.stem_path(svc.song_dir(song_id), name),
                            media_type="audio/flac")

    def version_key(key: str) -> str:
        if not versions.KEY_RE.match(key):
            raise HTTPException(404, "no such version")
        return key

    @app.post("/api/songs/{song_id}/versions")
    def request_version(song_id: str, body: VersionIn):
        """Start making the stems at this speed/key (100% and 0 = the original,
        which also cancels unfinished work). Returns its status."""
        svc.result_path(song_id)
        try:
            return maker.request(svc.song_dir(song_id), body.speed, body.semitones)
        except (ValueError, stretch.StretchError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/songs/{song_id}/versions/{key}")
    def version_status(song_id: str, key: str):
        return maker.status(svc.song_dir(song_id), version_key(key))

    @app.get("/api/songs/{song_id}/versions/{key}/{quality}/{name}")
    def version_stem(song_id: str, key: str, quality: str, name: str):
        if quality not in ("fast", "fine") or name not in pipeline.STEM_NAMES:
            raise HTTPException(404, "no such stem")
        path = versions.stems_dir(svc.song_dir(song_id), version_key(key), quality) / f"{name}.flac"
        if not path.exists():
            raise HTTPException(404, "version not ready")
        return FileResponse(path, media_type="audio/flac")

    @app.post("/api/songs/{song_id}/render")
    def render_mix(song_id: str, body: RenderIn):
        svc.result_path(song_id)
        d = svc.song_dir(song_id)
        unknown = set(body.gains) - set(pipeline.STEM_NAMES)
        if unknown:
            raise HTTPException(400, f"unknown stems: {', '.join(sorted(unknown))}")
        try:
            speed, semitones = stretch.check(body.speed, body.semitones)
            data = render.render(d, body.gains, speed, semitones, body.fmt)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        muted = [n for n in pipeline.STEM_NAMES if body.gains.get(n, 1.0) <= 0]
        suffix = ("-no-" + "-".join(muted)) if muted and len(muted) < len(pipeline.STEM_NAMES) else ""
        if speed != 1.0:
            suffix += f"-{round(speed * 100)}pct"
        if semitones:
            suffix += f"-{'up' if semitones > 0 else 'down'}{abs(semitones)}st"
        name = re.sub(r"[^\w.-]+", "_", f"{song_id}{suffix}.{body.fmt}")
        return Response(data, media_type=render.FORMATS[body.fmt], headers={
            "Content-Disposition": f'attachment; filename="{name}"'})

    add_page(app, "splitter.html")
    return app
