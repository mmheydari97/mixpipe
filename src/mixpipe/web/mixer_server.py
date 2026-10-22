"""Local web UI for mixing by chat.

    python -m mixpipe.cli mix-ui            # http://127.0.0.1:8767

Start a session from uploaded stems or from a song the splitter already
separated; the first mix is made in the background, then every chat message
("the acoustic guitar is too loud") revises the plan and re-renders it
(`session.MixSession`). Versions are kept, so any earlier mix can be gone back to.
A turn renders a preview slice for long songs; Export renders the whole song.
Which OpenRouter model answers can be picked when starting a mix and changed
between turns (the picker in static/common.js reads /api/openrouter/models).
AI edits go to a local ACE-Step server (`acestep.py`); /api/ai-edit/status says
whether it is running.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import acestep, openrouter, songstore
from ..config import load_config
from ..io_utils import AUDIO_EXTS, UPLOAD_EXTS
from ..session import MixSession, llm_choice
from .common import JobQueue, add_openrouter_routes, add_page, as_bool

_ID = re.compile(r"^[\w-]+$")
MAX_STEMS = 24


class ChatIn(BaseModel):
    message: str


class SetIn(BaseModel):
    adjustments: list[dict]
    label: str = "manual edit"


class ExportIn(BaseModel):
    bit_depth: int = 24
    full: bool = True


class AiEditIn(BaseModel):
    prompt: str
    mode: str = "repaint"
    start: float = 0.0
    end: float | None = None        # None: to the end of the clip
    strength: float | None = None   # cover only
    seed: int | None = None


class ModelIn(BaseModel):
    model: str | None = None        # an OpenRouter model id; None or "" for the keyword rules


def create_app(
    root: str | Path = "out/mixer",
    splitter_root: str | Path = "out/splitter",
    llm=None,
    use_llm: bool | None = None,
    run_inline: bool = False,
    session_options: dict | None = None,
    editor=None,
) -> FastAPI:
    """Build the app. `llm(system, payload) -> text` replaces the real model (tests);
    `use_llm=False` forces the keyword rules; `run_inline` makes the first mix
    synchronously instead of on the worker; `editor` replaces ACE-Step (tests)."""
    app = FastAPI(title="mixpipe mixer")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    splitter_root = Path(splitter_root)
    queue = JobQueue(1, "mixer")
    sessions: dict[str, dict] = {}
    lock = threading.Lock()

    def get(sid: str) -> dict:
        if not _ID.match(sid) or sid not in sessions:
            raise HTTPException(404, "no such session")
        return sessions[sid]

    def ready(sid: str) -> MixSession:
        entry = get(sid)
        if entry["state"] != "ready":
            raise HTTPException(409, f"session is {entry['state']}")
        return entry["session"]

    def start(sid: str, stems_dir: Path, title: str, notes: str, opts: dict) -> None:
        entry = {"id": sid, "title": title, "state": "queued", "error": None,
                 "created": time.time(), "session": None, "stems_dir": str(stems_dir)}
        with lock:
            sessions[sid] = entry

        def job():
            entry["state"] = "running"
            try:
                kwargs = {"use_llm": use_llm, **(session_options or {}), **opts}
                if use_llm is False:          # the server was started without the model: that wins
                    kwargs["use_llm"] = False
                s = MixSession(str(stems_dir), out_dir=str(root / sid / "versions"), llm=llm, **kwargs)
                s.editor = editor
                s.first_mix(notes)
                entry["session"] = s
                entry["state"] = "ready"
            except Exception as exc:  # noqa: BLE001 - shown on the page
                entry["state"], entry["error"] = "error", f"{type(exc).__name__}: {exc}"

        if run_inline:
            job()
        else:
            queue.submit(sid, job)

    def checked_model(model) -> str:
        """An OpenRouter model id the page asked for, or a 400 that says why not."""
        if use_llm is False:
            raise HTTPException(400, "this server was started with the keyword rules only")
        model = str(model).strip()
        if not openrouter.valid_model_id(model):
            raise HTTPException(400, f"not an OpenRouter model id: {model[:80]!r}")
        if llm is None and openrouter.api_key() is None:
            raise HTTPException(400, "OPENROUTER_API_KEY is not set (see .env_template)")
        return model

    def options(fields) -> dict:
        def num(key):
            v = fields.get(key)
            try:
                return float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                raise HTTPException(400, f"{key} must be a number")
        out = {}
        if num("start") is not None or num("duration") is not None:
            out["start_s"], out["duration_s"] = num("start") or 0.0, num("duration")
        if fields.get("llm") not in (None, ""):
            out["use_llm"] = as_bool(fields.get("llm"), True)
        if fields.get("model") not in (None, "") and out.get("use_llm", True):
            out.update(provider="openrouter", model=checked_model(fields["model"]), use_llm=True)
        out["align"] = as_bool(fields.get("align"), True)
        return out

    @app.get("/api/config")
    def config():
        """What a new session would use, and what the model picker can offer."""
        cfg = load_config(None)
        choice = None if use_llm is False else llm_choice(cfg)
        if llm is not None:
            choice = ("test", "injected")
        default = choice[1] if choice and choice[0] == "openrouter" else cfg["advisor"].get("openrouter_model")
        return {
            "llm": {"provider": choice[0], "model": choice[1]} if choice else None,
            "openrouter": None if use_llm is False else {
                "has_key": llm is not None or openrouter.api_key() is not None,
                "default_model": default,
            },
        }

    @app.get("/api/splitter-songs")
    def splitter_songs():
        out = []
        if splitter_root.is_dir():
            for d in sorted(splitter_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                stems = d / "stems"
                if stems.is_dir() and any(p.suffix.lower() in AUDIO_EXTS for p in stems.iterdir()):
                    try:
                        title = songstore.read_meta(d).get("title", d.name)
                    except Exception:  # noqa: BLE001 - a half-written song just shows its id
                        title = d.name
                    out.append({"id": d.name, "title": title,
                                "stems": sorted(p.stem for p in stems.iterdir() if p.suffix.lower() in AUDIO_EXTS)})
        return out

    @app.post("/api/sessions")
    async def create(request: Request):
        sid = uuid.uuid4().hex[:10]
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/"):
            form = await request.form()
            files = form.getlist("files")
            if not files:
                raise HTTPException(400, "no stems uploaded")
            if len(files) > MAX_STEMS:
                raise HTTPException(400, f"at most {MAX_STEMS} stems")
            for f in files:                # check everything before anything is written
                name = Path(f.filename or "").name
                if not name or Path(name).suffix.lower() not in UPLOAD_EXTS:
                    raise HTTPException(400, f"not an audio file: {name or '?'}")
            fields = {k: v for k, v in form.items() if k != "files"}
            opts = options(fields)
            stems_dir = root / sid / "stems"
            stems_dir.mkdir(parents=True)
            for f in files:
                name = Path(f.filename).name
                ext = Path(name).suffix.lower()
                data = await f.read()
                if ext in AUDIO_EXTS:
                    (stems_dir / name).write_bytes(data)
                else:                      # mp3, m4a ...: decode to a WAV the mixer reads
                    from ..io_utils import decode, write_audio

                    tmp = stems_dir / f".upload{ext}"
                    tmp.write_bytes(data)
                    audio, sr = decode(tmp)
                    write_audio(str(stems_dir / (Path(name).stem + ".wav")), audio, sr)
                    tmp.unlink()
            title = str(fields.get("title") or ", ".join(Path(f.filename).stem for f in files[:3])
                        + ("…" if len(files) > 3 else ""))
        else:
            fields = await request.json()
            song = str(fields.get("splitter_song", ""))
            if not _ID.match(song) or not (splitter_root / song / "stems").is_dir():
                raise HTTPException(404, "no such splitter song")
            stems_dir = splitter_root / song / "stems"
            try:
                title = songstore.read_meta(splitter_root / song).get("title", song)
            except Exception:  # noqa: BLE001
                title = song
            opts = options(fields)
        start(sid, stems_dir, title, str(fields.get("notes") or ""), opts)
        return {"id": sid}

    @app.get("/api/sessions")
    def list_sessions():
        return [{"id": e["id"], "title": e["title"], "state": e["state"], "created": e["created"]}
                for e in sorted(sessions.values(), key=lambda e: -e["created"])]

    def state(sid: str) -> dict:
        e = get(sid)
        out = {"id": sid, "title": e["title"], "state": e["state"], "error": e["error"]}
        if e["state"] in ("queued", "running"):
            out["ahead"] = queue.ahead(sid)
        s: MixSession | None = e["session"]
        if s is not None:
            out.update({
                "current": s.current["n"] if s.current else None,
                "versions": [s.summary(v) for v in s.versions],
                "turns": s.turns,
                "llm": {"provider": s.provider, "model": s.model} if s.use_llm else None,
                "tempo_bpm": s.tempo_bpm,
                "preview": {"start_s": s.start_s, "duration_s": s.duration_s,
                            "full_duration_s": round(s.full_duration, 2)} if s.preview else None,
            })
        return out

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str):
        return state(sid)

    @app.post("/api/sessions/{sid}/chat")
    def chat(sid: str, body: ChatIn):
        s = ready(sid)
        if not body.message.strip():
            raise HTTPException(400, "empty message")
        turn = s.chat(body.message)
        return {"turn": s.summary(turn) if turn.get("n") else
                {k: turn.get(k) for k in ("message", "reply", "source", "changes", "not_understood",
                                          "rejected", "llm_error")},
                **state(sid)}

    @app.post("/api/sessions/{sid}/set")
    def set_params(sid: str, body: SetIn):
        s = ready(sid)
        v = s.set_params(body.adjustments, body.label)
        return {"turn": s.summary(v) if v else None, **state(sid)}

    @app.get("/api/ai-edit/status")
    def ai_edit_status():
        if editor is not None:
            return {"url": "injected", "ok": True, "model": "test", "error": None}
        return acestep.status()

    @app.post("/api/sessions/{sid}/ai-edit")
    def ai_edit(sid: str, body: AiEditIn):
        s = ready(sid)
        if editor is None and not acestep.status()["ok"]:
            raise HTTPException(503, f"ACE-Step isn't running at {acestep.base_url()} "
                                     "(start it with `make acestep`, see README)")
        try:
            v = s.ai_edit(body.prompt, mode=body.mode, start_s=body.start, end_s=body.end,
                          strength=body.strength, seed=body.seed)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - the model's failure, shown on the page
            raise HTTPException(502, f"ACE-Step: {exc}") from exc
        return {"turn": s.summary(v), **state(sid)}

    @app.post("/api/sessions/{sid}/model")
    def set_model(sid: str, body: ModelIn):
        """Switch the model for the next turns (or, with no model, to the keyword rules)."""
        s = ready(sid)
        s.set_model(checked_model(body.model) if body.model else None)
        return state(sid)

    @app.post("/api/sessions/{sid}/undo")
    def undo(sid: str):
        ready(sid).undo()
        return state(sid)

    @app.post("/api/sessions/{sid}/redo")
    def redo(sid: str):
        ready(sid).redo()
        return state(sid)

    @app.post("/api/sessions/{sid}/goto/{n}")
    def goto(sid: str, n: int):
        try:
            ready(sid).goto(n)
        except KeyError:
            raise HTTPException(404, "no such version")
        return state(sid)

    @app.get("/api/sessions/{sid}/audio/{n}.wav")
    def audio(sid: str, n: int):
        try:
            v = ready(sid).version(n)
        except KeyError:
            raise HTTPException(404, "no such version")
        return FileResponse(v["wav"], media_type="audio/wav")

    @app.get("/api/sessions/{sid}/plan/{n}")
    def plan(sid: str, n: int):
        try:
            return ready(sid).version(n)["plan"]
        except KeyError:
            raise HTTPException(404, "no such version")

    @app.post("/api/sessions/{sid}/export")
    def export(sid: str, body: ExportIn):
        s = ready(sid)
        if body.bit_depth not in (16, 24):
            raise HTTPException(400, "bit_depth is 16 or 24")
        n = s.current["n"]
        path = root / sid / f"export_v{n}_{body.bit_depth}bit{'' if body.full else '_preview'}.wav"
        s.export(str(path), bit_depth=body.bit_depth, full=body.full)
        title = re.sub(r"[^\w.-]+", "_", get(sid)["title"])[:60] or "mix"
        return FileResponse(path, media_type="audio/wav", filename=f"{title}_v{n}.wav")

    @app.delete("/api/sessions/{sid}")
    def delete(sid: str):
        get(sid)
        with lock:
            sessions.pop(sid, None)
        return {"ok": True}

    add_openrouter_routes(app)
    add_page(app, "mixer.html")
    return app
