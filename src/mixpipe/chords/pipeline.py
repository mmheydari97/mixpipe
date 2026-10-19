"""Song store and the analysis pipeline: (download ->) decode -> separate ->
transcribe -> chords -> melody -> align.

Each song lives in out/chords/<song_id>/ (folder layout and ids: songstore.py):

  meta.json        id, title, created, input file names, source (songs from a link)
  inputs/          the uploaded or downloaded files, untouched
  mix.flac         playback audio, only when the inputs can't be served as-is
  separated/       Demucs stems (mono FLAC), cached
  transcript.json  cached, keyed by backend/model/language
  chords.json      cached, keyed by RECOGNIZER_VERSION
  melody.json      cached melody notes for the harmonica tab, keyed by MELODY_VERSION
  sheet.json       the editable result (see sheet.py)
  status.json      job state, for the UI to poll

Intermediate files are reused on re-runs, so retrying after a failure or
re-transcribing with another model doesn't redo separation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .. import songstore
from ..songstore import (Progress, _write_json, download_inputs, get_status, list_songs,
                         new_song, new_song_from_url, set_status)
from .ingest import WORK_SR, Ingested, ingest
from .sheet import SHEET_VERSION, load_sheet, save_sheet

DEFAULT_ROOT = Path("out/chords")
RECOGNIZER_VERSION = 4
MELODY_VERSION = 1
STAGES = ("download", "decode", "separate", "transcribe", "chords", "melody", "align")
BROWSER_PLAYABLE = {".mp3", ".m4a", ".aac", ".mp4", ".wav", ".flac", ".ogg", ".opus", ".webm"}

DEFAULT_OPTIONS = {
    "language": "auto",   # "auto", an ISO-639-1 code, or "none" (no lyrics)
    "stt": "mlx",         # "mlx" | "openrouter"
    "stt_model": None,    # OpenRouter model id when stt == "openrouter"
    "separate": True,     # run Demucs on a single full-mix upload
}

# --------------------------------------------------------------------------- #
# OpenRouter capabilities learned from past runs
# --------------------------------------------------------------------------- #


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
    downloader: Callable | None = None,
    melody_extractor: Callable | None = None,
) -> dict:
    """Run every stage for one song and write sheet.json. Returns the sheet.

    `separator(audio, sr, progress=cb) -> list[Stem]`,
    `transcriber(vocal, sr, backend=, model=, language=) -> transcript`,
    `downloader(source, dest_dir, progress=cb) -> Path` and
    `melody_extractor(vocal, sr, regions, fallback=, key=, progress=) -> {"notes"}`
    can be injected (tests use fakes so no models or songs are downloaded).
    """
    from .recognize import recognize
    from .transcribe import has_vocals, transcribe, vocal_activity

    song_dir = Path(song_dir)
    root = Path(root) if root else song_dir.parent
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    report = progress or (lambda stage, frac, msg: None)

    # ---- download (songs from a link) -----------------------------------------
    meta = download_inputs(song_dir, report, downloader)
    source = meta.get("source")

    # ---- decode ---------------------------------------------------------------
    report("decode", 0.0, "decoding audio")
    ing: Ingested = ingest([song_dir / "inputs" / n for n in meta["inputs"]])
    if playback_file(song_dir) is None:
        _write_stereo(song_dir / "mix.flac", ing.mix, ing.sr)
    report("decode", 1.0, f"{len(ing.mix) / ing.sr:.0f} s of audio")

    # ---- separate -------------------------------------------------------------
    harmony, vocal, bass = _sources(song_dir, ing, opts["separate"], separator, report)

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

    # ---- melody (harmonica tab) -------------------------------------------------
    melody = find_melody(song_dir, vocal, _lead(harmony, bass), ing.sr, chords["key"],
                         melody_extractor, report)

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
        "melody": {"notes": melody["notes"], "octave": 0},
    }
    if source:
        sheet["source"] = {k: source.get(k) for k in ("url", "page_url", "site", "note")}
    save_sheet(sheet, song_dir / "sheet.json")
    report("align", 1.0, "done")
    return sheet


def _sources(song_dir: Path, ing: Ingested, separate: bool, separator, report):
    """(harmony, vocal, bass) mono signals: Demucs stems of a full mix (cached
    in separated/), the uploaded stems, or the mix itself when not separating."""
    if ing.mode != "mix":
        report("separate", 1.0, "using uploaded stems")
        if ing.harmony is None:
            raise ValueError("no harmonic stems: every upload looks like vocals or drums")
        return ing.harmony, ing.vocal, ing.bass
    if not separate:
        mono_mix = ing.mix.mean(axis=1)
        report("separate", 1.0, "skipped")
        return mono_mix, mono_mix, None
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
    report("separate", 1.0, "separated")
    return bass + _read_mono(sep_dir / "other.flac"), vocal, bass


def _lead(harmony, bass):
    """The instruments minus the bass line, where a lead melody is most likely."""
    if harmony is None or bass is None or len(bass) != len(harmony):
        return harmony
    return harmony - bass


def find_melody(song_dir: Path, vocal, lead, sr: int, key: str | None,
                extractor: Callable | None, report, force: bool = False) -> dict:
    """Melody notes for the harmonica tab, cached in melody.json. `lead` (the
    instruments) is used where nobody sings."""
    from .transcribe import has_vocals, vocal_activity

    path = song_dir / "melody.json"
    cached = json.loads(path.read_text()) if path.exists() and not force else None
    if cached and cached.get("version") == MELODY_VERSION:
        report("melody", 1.0, "using cached melody")
        return cached
    report("melody", 0.05, "finding the melody")
    regions = vocal_activity(vocal, sr) if vocal is not None else []
    if not has_vocals(regions):
        regions = []
    if extractor is None:
        from .melody import extract_melody as extractor
    melody = extractor(vocal if regions else None, sr, regions, fallback=lead, key=key,
                       progress=lambda f, msg: report("melody", f, msg))
    melody["version"] = MELODY_VERSION
    _write_json(path, melody)
    report("melody", 1.0, f"{len(melody['notes'])} melody notes")
    return melody


def _write_stereo(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), np.clip(audio, -1, 1).astype(np.float32), sr, format="FLAC",
             subtype="PCM_16")


def run_job(song_dir: Path, options: dict | None = None, **kwargs) -> dict | None:
    """analyze() with status.json bookkeeping; never raises."""
    return songstore.run_job(
        song_dir, lambda progress: analyze(song_dir, options, progress=progress, **kwargs),
        done_stage="align")


def run_melody_job(song_dir: Path, lock=None, extractor: Callable | None = None) -> dict | None:
    """(Re-)find the melody of an analysed song and store it in sheet.json
    without touching chord or lyric edits. Uses the cached Demucs stems when
    the song was separated; never re-runs separation. Never raises."""
    import contextlib

    song_dir = Path(song_dir)

    def work(report):
        meta = json.loads((song_dir / "meta.json").read_text())
        report("decode", 0.0, "decoding audio")
        ing = ingest([song_dir / "inputs" / n for n in meta["inputs"]])
        report("decode", 1.0, "decoded")
        separated = (song_dir / "separated" / "vocals.flac").exists()
        harmony, vocal, bass = _sources(song_dir, ing, separated, None, lambda *a: None)
        key = load_sheet(song_dir / "sheet.json").get("key")
        melody = find_melody(song_dir, vocal, _lead(harmony, bass), ing.sr, key, extractor,
                             report, force=True)
        with lock or contextlib.nullcontext():
            sheet = load_sheet(song_dir / "sheet.json")
            octave = (sheet.get("melody") or {}).get("octave", 0)
            sheet["melody"] = {"notes": melody["notes"], "octave": octave}
            save_sheet(sheet, song_dir / "sheet.json")
        return sheet

    return songstore.run_job(song_dir, work, done_stage="melody")


__all__ = ["DEFAULT_ROOT", "STAGES", "analyze", "run_job", "run_melody_job", "new_song",
           "new_song_from_url", "list_songs", "get_status", "set_status", "playback_file", "openrouter_caps",
           "load_sheet", "save_sheet", "WORK_SR"]
