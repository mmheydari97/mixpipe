"""Song store and the analysis pipeline: decode -> separate -> transcribe ->
chords -> align.

Each song lives in out/chords/<song_id>/:

  meta.json        id, title, created, input file names
  inputs/          the uploaded files, untouched
  mix.flac         playback audio, only when the inputs can't be served as-is
  separated/       Demucs stems (mono FLAC), cached
  transcript.json  cached, keyed by backend/model/language
  chords.json      cached, keyed by RECOGNIZER_VERSION
  sheet.json       the editable result (see sheet.py)
  status.json      job state, for the UI to poll

Intermediate files are reused on re-runs, so retrying after a failure or
re-transcribing with another model doesn't redo separation.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .ingest import UPLOAD_EXTS, WORK_SR, Ingested, ingest
from .sheet import SHEET_VERSION, load_sheet, save_sheet

DEFAULT_ROOT = Path("out/chords")
RECOGNIZER_VERSION = 4
STAGES = ("decode", "separate", "transcribe", "chords", "align")
BROWSER_PLAYABLE = {".mp3", ".m4a", ".aac", ".mp4", ".wav", ".flac", ".ogg", ".opus", ".webm"}

DEFAULT_OPTIONS = {
    "language": "auto",   # "auto", an ISO-639-1 code, or "none" (no lyrics)
    "stt": "mlx",         # "mlx" | "openrouter"
    "stt_model": None,    # OpenRouter model id when stt == "openrouter"
    "separate": True,     # run Demucs on a single full-mix upload
}

Progress = Callable[[str, float, str], None]


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


def _slug(text: str) -> str:
    s = re.sub(r"[^\w-]+", "-", text.strip().lower(), flags=re.UNICODE).strip("-")
    return s[:40] or "song"


def _safe_name(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE) or "audio"


def new_song(files: list[tuple[str, bytes]], root: Path = DEFAULT_ROOT,
             title: str | None = None) -> str:
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


def list_songs(root: Path = DEFAULT_ROOT) -> list[dict]:
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
    return json.loads(path.read_text())


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _caps_path(root: Path) -> Path:
    return Path(root) / "openrouter_caps.json"


def openrouter_caps(root: Path = DEFAULT_ROOT) -> dict:
    """{model_id: "word" | "approx"} learned from past transcriptions."""
    path = _caps_path(root)
    return json.loads(path.read_text()) if path.exists() else {}


def _record_cap(root: Path, model: str, timing: str) -> None:
    caps = openrouter_caps(root)
    caps[model] = timing
    Path(root).mkdir(parents=True, exist_ok=True)
    _write_json(_caps_path(root), caps)


# --------------------------------------------------------------------------- #
# audio cache helpers
# --------------------------------------------------------------------------- #


def _write_mono(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    sf.write(str(path), np.clip(mono, -1, 1).astype(np.float32), sr, format="FLAC",
             subtype="PCM_16")


def _read_mono(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def playback_file(song_dir: Path) -> Path | None:
    meta = json.loads((Path(song_dir) / "meta.json").read_text())
    inputs = meta["inputs"]
    if len(inputs) == 1 and Path(inputs[0]).suffix.lower() in BROWSER_PLAYABLE:
        return Path(song_dir) / "inputs" / inputs[0]
    mix = Path(song_dir) / "mix.flac"
    return mix if mix.exists() else None


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def analyze(
    song_dir: str | Path,
    options: dict | None = None,
    progress: Progress | None = None,
    separator: Callable | None = None,
    transcriber: Callable | None = None,
    root: Path | None = None,
) -> dict:
    """Run every stage for one song and write sheet.json. Returns the sheet.

    `separator(audio, sr, progress=cb) -> list[Stem]` and
    `transcriber(vocal, sr, backend=, model=, language=) -> transcript` can be
    injected (tests use fakes so no models are downloaded).
    """
    from .recognize import recognize
    from .transcribe import has_vocals, transcribe, vocal_activity

    song_dir = Path(song_dir)
    root = Path(root) if root else song_dir.parent
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    meta = json.loads((song_dir / "meta.json").read_text())
    report = progress or (lambda stage, frac, msg: None)

    # ---- decode ---------------------------------------------------------------
    report("decode", 0.0, "decoding audio")
    ing: Ingested = ingest([song_dir / "inputs" / n for n in meta["inputs"]])
    if playback_file(song_dir) is None:
        _write_stereo(song_dir / "mix.flac", ing.mix, ing.sr)
    report("decode", 1.0, f"{len(ing.mix) / ing.sr:.0f} s of audio")

    # ---- separate -------------------------------------------------------------
    harmony, vocal, bass = ing.harmony, ing.vocal, ing.bass
    if ing.mode == "mix":
        mono_mix = ing.mix.mean(axis=1)
        if opts["separate"]:
            sep_dir = song_dir / "separated"
            needed = ("vocals", "bass", "other")
            if not all((sep_dir / f"{n}.flac").exists() for n in needed):
                report("separate", 0.0, "separating vocals / bass / other (Demucs)")
                if separator is None:
                    from ..separation import separate as separator
                stems = separator(ing.mix, ing.sr,
                                  progress=lambda f: report("separate", f, "separating"))
                for st in stems:
                    _write_mono(sep_dir / f"{st.name}.flac", st.audio, ing.sr)
            vocal = _read_mono(sep_dir / "vocals.flac")
            bass = _read_mono(sep_dir / "bass.flac")
            harmony = bass + _read_mono(sep_dir / "other.flac")
            report("separate", 1.0, "separated")
        else:
            harmony, vocal, bass = mono_mix, mono_mix, None
            report("separate", 1.0, "skipped")
    else:
        report("separate", 1.0, "using uploaded stems")
        if harmony is None:
            raise ValueError("no harmonic stems: every upload looks like vocals or drums")

    # ---- transcribe -----------------------------------------------------------
    transcript = {"backend": "none", "model": None, "language": None,
                  "timing": "none", "lines": []}
    if opts["language"] != "none" and vocal is not None:
        regions = vocal_activity(vocal, ing.sr)
        if not has_vocals(regions):
            report("transcribe", 1.0, "no vocals found")
        else:
            cache_key = {"stt": opts["stt"], "stt_model": opts["stt_model"],
                         "language": opts["language"]}
            cache = song_dir / "transcript.json"
            cached = json.loads(cache.read_text()) if cache.exists() else None
            if cached and cached.get("cache_key") == cache_key:
                transcript = cached
                report("transcribe", 1.0, "using cached transcript")
            else:
                label = "OpenRouter " + str(opts["stt_model"]) if opts["stt"] == "openrouter" \
                    else "local Whisper (first run downloads ~1.6 GB)"
                report("transcribe", 0.1, f"transcribing with {label}")
                fn = transcriber or transcribe
                transcript = fn(vocal, ing.sr, backend=opts["stt"], model=opts["stt_model"],
                                language=opts["language"])
                transcript["cache_key"] = cache_key
                _write_json(cache, transcript)
                if transcript["backend"] == "openrouter" and transcript["lines"]:
                    _record_cap(root, transcript["model"], transcript["timing"])
                report("transcribe", 1.0, f"{sum(len(ln) for ln in transcript['lines'])} words")
    else:
        report("transcribe", 1.0, "skipped (instrumental)")

    # ---- chords ---------------------------------------------------------------
    chords_path = song_dir / "chords.json"
    cached = json.loads(chords_path.read_text()) if chords_path.exists() else None
    if cached and cached.get("version") == RECOGNIZER_VERSION:
        chords = cached
        report("chords", 1.0, "using cached chords")
    else:
        report("chords", 0.1, "detecting beats and chords")
        chords = recognize(harmony, ing.sr, bass=bass)
        chords["version"] = RECOGNIZER_VERSION
        _write_json(chords_path, chords)
        report("chords", 1.0, f"{len(chords['segments'])} chord changes, key {chords['key_label']}")

    # ---- align ----------------------------------------------------------------
    report("align", 0.5, "placing chords on lyrics")
    playback = playback_file(song_dir)
    sheet = {
        "version": SHEET_VERSION,
        "id": meta["id"],
        "title": meta["title"],
        "created": meta["created"],
        "audio": playback.relative_to(song_dir).as_posix() if playback else None,
        "duration": round(len(ing.mix) / ing.sr, 3),
        "key": chords["key"],
        "key_label": chords["key_label"],
        "key_regions": chords.get("key_regions", []),
        "tempo": chords["tempo"],
        "beat_period": chords["beat_period"],
        "beat_times": chords.get("beat_times", []),
        "meter": {**{k: chords["meter"][k]
                     for k in ("time_signature", "beats_per_bar", "downbeat")},
                  "detected": dict(chords["meter"])},
        "transpose": 0,
        "capo": 0,
        "transcript": {k: transcript.get(k) for k in ("backend", "model", "language", "timing")},
        "lines": [{"words": ln, "gap_before": False} for ln in transcript["lines"]],
        "chords": chords["segments"],
    }
    save_sheet(sheet, song_dir / "sheet.json")
    report("align", 1.0, "done")
    return sheet


def _write_stereo(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), np.clip(audio, -1, 1).astype(np.float32), sr, format="FLAC",
             subtype="PCM_16")


def run_job(song_dir: Path, options: dict | None = None, **kwargs) -> dict | None:
    """analyze() with status.json bookkeeping; never raises."""
    song_dir = Path(song_dir)
    last = {"t": 0.0}

    def progress(stage: str, frac: float, msg: str) -> None:
        now = time.monotonic()
        if frac in (0.0, 1.0) or now - last["t"] > 0.5:
            last["t"] = now
            set_status(song_dir, "running", stage, frac, msg)

    try:
        sheet = analyze(song_dir, options, progress=progress, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        st = get_status(song_dir)
        set_status(song_dir, "error", st.get("stage", "decode"), st.get("progress", 0.0),
                   "failed", error=f"{type(exc).__name__}: {exc}")
        return None
    set_status(song_dir, "done", "align", 1.0, "done")
    return sheet


__all__ = ["DEFAULT_ROOT", "STAGES", "analyze", "run_job", "new_song", "list_songs",
           "get_status", "set_status", "playback_file", "openrouter_caps",
           "load_sheet", "save_sheet", "WORK_SR"]
