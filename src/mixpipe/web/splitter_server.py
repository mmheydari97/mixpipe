"""Local web UI for the stem splitter.

    python -m mixpipe.cli split-ui           # http://127.0.0.1:8766

Same song library plumbing as the chord finder (web/common.py); songs are
separated one at a time in a background worker and the page polls /status.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .. import songstore
from ..splitter import pipeline, render
from .common import SongService, add_page, add_song_routes


class RenderIn(BaseModel):
    gains: dict[str, float] = {}
    speed: float = 1.0
    fmt: str = "wav"


def parse_options(fields: dict) -> dict:
    return {}  # nothing to choose: one model, every stem


def create_app(
    root: str | Path = pipeline.DEFAULT_ROOT,
    separator=None,
    run_inline: bool = False,
    resolver=None,
    downloader=None,
) -> FastAPI:
    """Build the app. `separator`/`resolver`/`downloader` are injectable for
    tests, and `run_inline` separates synchronously instead of in the worker."""
    app = FastAPI(title="mixpipe splitter")

    def job(d: Path, options: dict):
        return pipeline.run_job(d, options, separator=separator, downloader=downloader)

    svc = SongService(root, job, done_file="stems.json", done_stage="separate",
                      run_inline=run_inline, name="splitter")
    add_song_routes(app, svc, parse_options, resolver=resolver)

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

    @app.post("/api/songs/{song_id}/render")
    def render_mix(song_id: str, body: RenderIn):
        svc.result_path(song_id)
        d = svc.song_dir(song_id)
        unknown = set(body.gains) - set(pipeline.STEM_NAMES)
        if unknown:
            raise HTTPException(400, f"unknown stems: {', '.join(sorted(unknown))}")
        try:
            data = render.render(d, body.gains, body.speed, body.fmt)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        muted = [n for n in pipeline.STEM_NAMES if body.gains.get(n, 1.0) <= 0]
        suffix = ("-no-" + "-".join(muted)) if muted and len(muted) < len(pipeline.STEM_NAMES) else ""
        if body.speed != 1.0:
            suffix += f"-{round(body.speed * 100)}pct"
        name = re.sub(r"[^\w.-]+", "_", f"{song_id}{suffix}.{body.fmt}")
        return Response(data, media_type=render.FORMATS[body.fmt], headers={
            "Content-Disposition": f'attachment; filename="{name}"'})

    add_page(app, "splitter.html")
    return app
