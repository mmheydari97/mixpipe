"""Local web UI for the chord finder.

    python -m mixpipe.cli chords-ui          # http://127.0.0.1:8765

Songs are analysed in a background queue, one at a time unless `workers` says
otherwise (the GPU models don't share well); the page polls /status while a job
runs and can cancel it.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from ..chords import pipeline
from ..chords.align import lines_to_text, reanchor, word_diff
from ..chords.harmonica import (add_note, merge_notes, note_from_display, set_fingering,
                                split_note, to_harp_text)
from ..chords.sheet import (apply_settings, custom_from_display, load_sheet, save_sheet,
                            to_chordpro, to_text, view)
from .common import SongService, add_openrouter_routes, add_page, add_song_routes, as_bool

# Python's mimetypes guesses e.g. audio/mp4a-latm for .m4a, which Safari won't play
AUDIO_TYPES = {
    ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".aac": "audio/aac", ".mp3": "audio/mpeg",
    ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg",
    ".webm": "audio/webm",
}

POLISH_PROMPT = """\
You correct speech-recognition errors in song lyrics. The user message is a \
machine transcript of a sung vocal, one lyric line per row.

Fix misheard words, spelling and obvious punctuation. Keep the original language \
and wording wherever it is plausible. Keep exactly the same number of lines in \
the same order. Do not add, remove, merge or translate lines. Do not add titles, \
section labels, notes or commentary. Return only the corrected lyrics."""


def parse_options(fields: dict) -> dict:
    """The analysis options sent with an upload or a link."""
    stt = fields.get("stt") or "mlx"
    stt_model = fields.get("stt_model") or None
    if stt not in ("mlx", "openrouter"):
        raise HTTPException(400, "stt must be mlx or openrouter")
    if stt == "openrouter" and not stt_model:
        raise HTTPException(400, "pick an OpenRouter transcription model")
    return {"language": str(fields.get("language") or "auto"), "stt": stt,
            "stt_model": stt_model, "separate": as_bool(fields.get("separate"), True),
            "retime": as_bool(fields.get("retime"), True),
            "key_changes": "sensitive" if fields.get("key_changes") == "sensitive" else "normal",
            "recognizer": "neural" if fields.get("recognizer") == "neural" else "template"}


class LyricsIn(BaseModel):
    text: str


class ChordIn(BaseModel):
    selected: int | None = None
    custom: str | None = None


class NoteIn(BaseModel):
    selected: int | None = None
    custom: str | None = None
    fingering: int | None = None
    auto_fingering: bool = False


class NoteEdit(BaseModel):
    action: str                      # split | merge | add
    id: str | None = None            # split: the note
    ids: list[str] | None = None     # merge: neighbouring notes
    at: float | None = None          # split: where (seconds); default the middle
    start: float | None = None       # add
    end: float | None = None
    note: str | None = None          # add: as typed on the tab, e.g. "C5" or "Bb"


class MelodyIn(BaseModel):
    fast: bool = False               # keep grace notes and quick runs


class PolishIn(BaseModel):
    model: str


class SettingsIn(BaseModel):
    transpose: int | None = None
    capo: int | None = None
    time_signature: str | None = None
    downbeat: int | None = None
    shift_downbeat: int | None = None
    reset_meter: bool = False
    harmonica_octave: int | None = None
    harmonica_type: str | None = None
    harmonica_key: str | None = None
    instrument: str | None = None


def create_app(
    root: str | Path = pipeline.DEFAULT_ROOT,
    separator=None,
    transcriber=None,
    run_inline: bool = False,
    resolver=None,
    downloader=None,
    melody_extractor=None,
    retimer=None,
    neural_provider=None,
    workers: int = 1,
) -> FastAPI:
    """Build the app. `separator`/`transcriber`/`resolver`/`downloader`/
    `melody_extractor`/`retimer` are injectable for tests, and `run_inline` runs analysis
    synchronously instead of in the worker thread."""
    root = Path(root)
    app = FastAPI(title="mixpipe chords")
    sheet_lock = threading.Lock()

    def job(d: Path, options: dict):
        return pipeline.run_job(d, options, cancelled=svc.cancelled(d.name),
                                separator=separator, transcriber=transcriber,
                                root=root, downloader=downloader,
                                melody_extractor=melody_extractor, retimer=retimer,
                                neural_provider=neural_provider)

    svc = SongService(root, job, done_file="sheet.json", done_stage="align",
                      run_inline=run_inline, name="chords", workers=workers)
    song_dir, sheet_path = svc.song_dir, svc.result_path
    add_song_routes(app, svc, parse_options, resolver=resolver)

    @app.get("/api/songs/{song_id}")
    def get_song(song_id: str):
        return view(load_sheet(sheet_path(song_id)))

    @app.put("/api/songs/{song_id}/lyrics")
    def put_lyrics(song_id: str, body: LyricsIn):
        p = sheet_path(song_id)
        with sheet_lock:
            sheet = load_sheet(p)
            sheet["lines"] = reanchor(sheet["lines"], body.text)
            save_sheet(sheet, p)
        return view(sheet)

    @app.patch("/api/songs/{song_id}/chords/{chord_id}")
    def patch_chord(song_id: str, chord_id: str, body: ChordIn):
        p = sheet_path(song_id)
        with sheet_lock:
            sheet = load_sheet(p)
            chord = next((c for c in sheet["chords"] if c["id"] == chord_id), None)
            if chord is None:
                raise HTTPException(404, "no such chord")
            if body.selected is not None:
                if not 0 <= body.selected < len(chord["candidates"]):
                    raise HTTPException(400, "selected out of range")
                chord["selected"] = body.selected
                chord["custom"] = None
            if body.custom is not None:
                # typed as displayed (after transpose/capo); stored at concert pitch
                text = body.custom.strip()
                chord["custom"] = custom_from_display(text, sheet, chord) if text else None
            save_sheet(sheet, p)
        return chord

    @app.patch("/api/songs/{song_id}/notes/{note_id}")
    def patch_note(song_id: str, note_id: str, body: NoteIn):
        """Edit one melody note. `custom` is typed as shown on the harmonica tab
        ("" reverts, "rest" removes the note); returns the whole view."""
        p = sheet_path(song_id)
        with sheet_lock:
            sheet = load_sheet(p)
            notes = (sheet.get("melody") or {}).get("notes", [])
            note = next((n for n in notes if n["id"] == note_id), None)
            if note is None:
                raise HTTPException(404, "no such note")
            try:
                if body.selected is not None:
                    if not 0 <= body.selected < len(note["candidates"]):
                        raise ValueError("selected out of range")
                    note["selected"] = body.selected
                    note["custom"] = None
                    note["fingering"] = None
                if body.custom is not None:
                    note["custom"] = note_from_display(body.custom, sheet, note)
                    note["fingering"] = None
                if body.auto_fingering:
                    set_fingering(note, None, sheet)
                elif body.fingering is not None:
                    set_fingering(note, body.fingering, sheet)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            save_sheet(sheet, p)
        return view(sheet)

    @app.post("/api/songs/{song_id}/notes")
    def edit_notes(song_id: str, body: NoteEdit):
        """Split, merge or add melody notes; returns the whole view."""
        p = sheet_path(song_id)
        with sheet_lock:
            sheet = load_sheet(p)
            try:
                if body.action == "split":
                    if not body.id:
                        raise ValueError("say which note to split")
                    split_note(sheet, body.id, body.at)
                elif body.action == "merge":
                    merge_notes(sheet, body.ids or [])
                elif body.action == "add":
                    if body.start is None or body.end is None or not body.note:
                        raise ValueError("a new note needs start, end and a pitch")
                    add_note(sheet, body.start, body.end, body.note)
                else:
                    raise ValueError("action must be split, merge or add")
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            save_sheet(sheet, p)
        return view(sheet)

    @app.post("/api/songs/{song_id}/melody")
    def find_melody(song_id: str, body: MelodyIn | None = None):
        """Find (or re-find) the melody for the harmonica tab; poll /status."""
        d = song_dir(song_id)
        sheet_path(song_id)
        fast = bool(body and body.fast)
        pipeline.set_status(d, "queued", "decode", 0.0, "waiting")
        svc.submit(song_id, lambda: pipeline.run_melody_job(
            d, lock=sheet_lock, extractor=melody_extractor, fast=fast,
            cancelled=svc.cancelled(song_id)))
        return {"id": song_id}

    @app.patch("/api/songs/{song_id}/settings")
    def patch_settings(song_id: str, body: SettingsIn):
        p = sheet_path(song_id)
        with sheet_lock:
            sheet = load_sheet(p)
            try:
                apply_settings(sheet, **body.model_dump())
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            save_sheet(sheet, p)
        return view(sheet)

    @app.get("/api/songs/{song_id}/audio")
    def audio(song_id: str):
        f = pipeline.playback_file(song_dir(song_id))
        if f is None or not f.exists():
            raise HTTPException(404, "no playback audio yet")
        return FileResponse(f, media_type=AUDIO_TYPES.get(f.suffix.lower()))

    @app.get("/api/songs/{song_id}/export")
    def export(song_id: str, fmt: str = "cho"):
        sheet = load_sheet(sheet_path(song_id))
        if fmt == "cho":
            body, ext = to_chordpro(sheet), "cho"
        elif fmt == "txt":
            body, ext = to_text(sheet), "txt"
        elif fmt == "harp":
            body, ext = to_harp_text(sheet), "harp.txt"
        else:
            raise HTTPException(400, "fmt must be cho, txt or harp")
        return PlainTextResponse(body, headers={
            "Content-Disposition": f'attachment; filename="{song_id}.{ext}"'})

    @app.post("/api/songs/{song_id}/polish")
    def polish(song_id: str, body: PolishIn):
        from .. import openrouter

        if openrouter.api_key() is None:
            raise HTTPException(400, "OPENROUTER_API_KEY is not set on the server")
        sheet = load_sheet(sheet_path(song_id))
        old = lines_to_text(sheet["lines"])
        if not old.strip():
            raise HTTPException(400, "no lyrics to polish")
        try:
            new = openrouter.chat(
                [{"role": "system", "content": POLISH_PROMPT},
                 {"role": "user", "content": old}],
                model=body.model, max_tokens=4000,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
        new = new.strip().strip("`").strip()
        return {"old": old, "text": new, "diff": word_diff(old, new)}

    # ---- OpenRouter catalogue ---------------------------------------------------

    def with_word_timing(kind: str, models: list[dict]) -> list[dict]:
        if kind != "transcription":
            return models
        caps = pipeline.openrouter_caps(root)       # what earlier songs found out per model
        return [{**m, "word_timing": caps.get(m["id"])} for m in models]

    add_openrouter_routes(app, with_word_timing)

    add_page(app, "index.html")
    return app
