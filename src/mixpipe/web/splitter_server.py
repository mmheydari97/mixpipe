"""Local web UI for the stem splitter.

    python -m mixpipe.cli split-ui           # http://127.0.0.1:8766

Same song library plumbing as the chord finder (web/common.py); songs are
separated one at a time in a background worker and the page polls /status.
Speed/key versions (splitter/versions.py) are made on a second worker, so the
page asks for one with POST /versions and polls until it's ready.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .. import songstore, stretch
from ..splitter import analysis, effects, pipeline, render, takes, versions
from .common import SongService, add_page, add_song_routes, attachment


class RenderIn(BaseModel):
    gains: dict[str, float] = {}
    speed: float = 1.0
    semitones: float = 0
    fmt: str = "wav"
    fx: dict | None = None          # the page's effect settings (splitter/effects.py)
    tempo: float | None = None      # the song's BPM as recorded, for echoes locked to it


class StemsIn(BaseModel):
    speed: float = 1.0
    semitones: float = 0
    fmt: str = "flac"
    fx: dict | None = None
    tempo: float | None = None


class NudgeIn(BaseModel):
    ms: float                       # later (+) or earlier (-)


class AnalyseIn(BaseModel):
    lyrics: bool = False


class FxIn(BaseModel):
    version: str | None = None      # the version playing (its key), None = the original stems
    quality: str | None = None      # "fast" | "fine"
    settings: dict = {}
    tempo: float | None = None


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
    chords_root: str | Path = "out/chords",
    chords_service=None,
) -> FastAPI:
    """Build the app. `separator`/`resolver`/`downloader` are injectable for
    tests, and `run_inline` separates synchronously instead of in the worker.
    Key, tempo, beats, chords and lyrics come from the chord finder's store at
    `chords_root`, through its `chords_service` (the hub passes the chord
    finder's, so both queue on one worker; standalone, one is made here)."""
    app = FastAPI(title="mixpipe splitter")
    chords_root = Path(chords_root)
    chords: dict = {"svc": chords_service}

    def chords_svc():
        if chords["svc"] is None:
            from .server import chords_service as make

            chords["svc"] = make(chords_root, run_inline=run_inline, workers=workers)
        return chords["svc"]

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
                "source": meta.get("source"), "takes": takes.load(d)}

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

    def download(make, fmt: str, filename: str, media_type: str | None = None) -> Response:
        """Run a render, turning bad input into 400 and tool failures into 500."""
        try:
            data = make()
        except (ValueError, stretch.StretchError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        return Response(data, media_type=media_type or render.FORMATS[fmt], headers=attachment(filename))

    def checked(speed: float, semitones: float) -> tuple[float, int]:
        try:
            return stretch.check(speed, semitones)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/songs/{song_id}/render")
    def render_mix(song_id: str, body: RenderIn):
        svc.result_path(song_id)
        d = svc.song_dir(song_id)
        known = set(pipeline.STEM_NAMES) | {takes.name_of(t["n"]) for t in takes.load(d)}
        unknown = set(body.gains) - known
        if unknown:
            raise HTTPException(400, f"unknown stems: {', '.join(sorted(unknown))}")
        speed, semitones = checked(body.speed, body.semitones)
        muted = [n for n in pipeline.STEM_NAMES if body.gains.get(n, 1.0) <= 0]
        part = ("-no-" + "-".join(muted)) if muted and len(muted) < len(pipeline.STEM_NAMES) else ""
        return download(lambda: render.render(d, body.gains, speed, semitones, body.fmt, body.fx, body.tempo),
                        body.fmt,
                        render.download_name(song_id, body.fmt, speed, semitones, part))

    @app.post("/api/songs/{song_id}/stems/{name}/render")
    def render_stem(song_id: str, name: str, body: StemsIn):
        """One stem (or take) at full level, at the page's speed and key."""
        svc.result_path(song_id)
        d = svc.song_dir(song_id)
        if name not in pipeline.STEM_NAMES:
            if takes.number(name) is None:
                raise HTTPException(404, "no such stem")
            take_of(d, name)                      # 404 when there's no such take
        speed, semitones = checked(body.speed, body.semitones)
        return download(lambda: render.render_stem(d, name, speed, semitones, body.fmt, body.fx, body.tempo),
                        body.fmt,
                        render.download_name(song_id, body.fmt, speed, semitones, f"-{name}"))

    def take_of(d: Path, name: str) -> dict:
        try:
            return takes.find(d, takes.number(name))
        except KeyError:
            raise HTTPException(404, "no such take") from None

    @app.post("/api/songs/{song_id}/stems.zip")
    def stems_zip(song_id: str, body: StemsIn):
        """Every stem at full level, one file each, in a zip."""
        svc.result_path(song_id)
        d = svc.song_dir(song_id)
        speed, semitones = checked(body.speed, body.semitones)
        return download(lambda: render.stems_zip(d, song_id, speed, semitones, body.fmt, body.fx, body.tempo),
                        body.fmt,
                        render.download_name(song_id, "zip", speed, semitones, "-stems"),
                        media_type="application/zip")

    @app.post("/api/songs/{song_id}/takes")
    async def add_take(song_id: str, request: Request):
        """A recording from the page (multipart: file, start_s, speed, semitones,
        latency_ms), kept as the song's next take."""
        svc.result_path(song_id)
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(400, "no recording")

        def num(key, default):
            try:
                return float(form.get(key, default))
            except (TypeError, ValueError):
                raise HTTPException(400, f"{key} must be a number") from None

        data = await upload.read()
        if not data:
            raise HTTPException(400, "the recording is empty")
        try:
            return takes.add(svc.song_dir(song_id), data, upload.filename or "take.webm", num("start_s", 0),
                             num("speed", 1), num("semitones", 0), num("latency_ms", 0))
        except (ValueError, stretch.StretchError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:                   # couldn't decode it
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/songs/{song_id}/takes/{n}")
    def take_file(song_id: str, n: int, version: str | None = None, quality: str = "fine"):
        """A take as a stem of the version playing (none: the original speed and key)."""
        d = svc.song_dir(song_id)
        if version is not None:
            version_key(version)
        if quality not in ("fast", "fine"):
            raise HTTPException(400, "quality is fast or fine")
        try:
            return FileResponse(takes.path(d, n, version, quality), media_type="audio/flac")
        except KeyError:
            raise HTTPException(404, "no such take") from None

    @app.patch("/api/songs/{song_id}/takes/{n}")
    def nudge_take(song_id: str, n: int, body: NudgeIn):
        """Move a take later or earlier, when it doesn't sit with the backing."""
        try:
            return takes.nudge(svc.song_dir(song_id), n, body.ms)
        except KeyError:
            raise HTTPException(404, "no such take") from None

    @app.delete("/api/songs/{song_id}/takes/{n}")
    def delete_take(song_id: str, n: int):
        try:
            takes.delete(svc.song_dir(song_id), n)
        except KeyError:
            raise HTTPException(404, "no such take") from None
        return {"ok": True}

    def shift_of(shift: int) -> int:
        if abs(shift) > stretch.MAX_SEMITONES:
            raise HTTPException(400, f"shift is at most {stretch.MAX_SEMITONES} semitones")
        return shift

    @app.post("/api/songs/{song_id}/analyse")
    def analyse(song_id: str, body: AnalyseIn):
        """Have the chord finder find the key, beats and chords (and the lyrics
        when asked), from this song's stems."""
        svc.result_path(song_id)
        return analysis.start(svc.song_dir(song_id), chords_root, chords_svc(), lyrics=body.lyrics)

    @app.get("/api/songs/{song_id}/analysis")
    def get_analysis(song_id: str, shift: int = 0):
        """Key (as heard, `shift` semitones from the recording), tempo, beats,
        meter: what the chord finder found, or how far it has got."""
        svc.result_path(song_id)
        return analysis.status(svc.song_dir(song_id), chords_root, shift_of(shift))

    @app.get("/api/songs/{song_id}/sheet")
    def get_sheet(song_id: str, shift: int = 0):
        """The chord finder's chords and lyrics, named in the key you hear."""
        svc.result_path(song_id)
        view = analysis.sheet_view(svc.song_dir(song_id), chords_root, shift_of(shift))
        if view is None:
            raise HTTPException(409, "the chord finder hasn't analysed this song yet")
        return view

    @app.get("/api/fx/defaults")
    def fx_defaults():
        """The room and echo a song starts with (the mixer's send-bus defaults)."""
        return effects.validate({})["send_buses"]

    @app.post("/api/songs/{song_id}/fx")
    def render_fx(song_id: str, body: FxIn):
        """The stems of the version playing, with their effects (made now, or
        from the cache). Returns {"stems": {name: url}} for the stems that have
        any; the others play as they are."""
        svc.result_path(song_id)
        d = svc.song_dir(song_id)
        folder = None
        if body.version:
            if body.quality not in ("fast", "fine"):
                raise HTTPException(400, "quality is fast or fine")
            folder = versions.stems_dir(d, version_key(body.version), body.quality)
            if not folder.is_dir():
                raise HTTPException(404, "version not ready")
        speed = versions.speed_of(body.version)
        paths = effects.render(d, folder, effects.validate(body.settings),
                               bpm=body.tempo * speed if body.tempo else None)
        return {"stems": {n: f"/api/songs/{song_id}/fx/{p.parent.name}/{p.name}" for n, p in paths.items()}}

    @app.get("/api/songs/{song_id}/fx/{source}/{file}")
    def fx_file(song_id: str, source: str, file: str):
        if not effects.SOURCE_RE.match(source) or not effects.FILE_RE.match(file):
            raise HTTPException(404, "no such file")
        path = svc.song_dir(song_id) / "fx" / source / file
        if not path.exists():
            raise HTTPException(404, "no such file")
        return FileResponse(path, media_type="audio/flac")

    add_page(app, "splitter.html")
    return app
