"""A split song's key, tempo, beats, chords and lyrics: the chord finder's.

The splitter doesn't analyse songs itself. It hands a split song to the chord
finder's store (songstore.copy_song: the same id, title and input, so nothing
is downloaded again) and seeds the chord finder's separated/ cache from its
own six stems (vocals; bass; guitar + keys + other), so the chord finder skips
Demucs. The chord finder's job then writes sheet.json, which the splitter
reads for its key and BPM readout, its metronome (beats, meter) and its chords
and lyrics view. A song the chord finder already has is simply linked by id.

Lyrics are optional: the first run skips them (language "none"); asking for
them later reruns the job, and only the transcription is new (the chords are
cached).
"""

from __future__ import annotations

import copy
from pathlib import Path

from .. import songstore
from .pipeline import load_info, stem_path

# the chord finder's three cached stems, from the splitter's six
_SEED = {"vocals": ("vocals",), "bass": ("bass",), "other": ("guitar", "piano", "other")}


def chords_dir(split_dir: Path, chords_root: Path) -> Path:
    return Path(chords_root) / Path(split_dir).name


def link(split_dir: Path, chords_root: Path) -> str:
    """Make sure the chord finder has this song, with its stems seeded.
    Returns the chord finder's song id (the same as the splitter's)."""
    import soundfile as sf

    from ..chords.pipeline import write_mono

    song_id = songstore.copy_song(split_dir, chords_root)
    sep = chords_dir(split_dir, chords_root) / "separated"
    if not all((sep / f"{n}.flac").exists() for n in _SEED):
        sr = load_info(split_dir)["sr"]
        for name, parts in _SEED.items():
            mono = None
            for part in parts:
                y, _ = sf.read(str(stem_path(split_dir, part)), dtype="float32", always_2d=True)
                y = y.mean(axis=1)
                mono = y if mono is None else mono + y
            write_mono(sep / f"{name}.flac", mono, sr)
    return song_id


def start(split_dir: Path, chords_root: Path, svc, lyrics: bool = False) -> dict:
    """Link the song and run the chord finder on it (`svc` is the chord
    finder's SongService). Without lyrics only key, beats and chords are made."""
    song_id = link(split_dir, chords_root)
    done = (chords_dir(split_dir, chords_root) / "sheet.json").exists()
    svc.start(song_id, {"language": "auto" if lyrics else "none"}, rerun=done and lyrics)
    return status(split_dir, chords_root)


def _sheet(split_dir: Path, chords_root: Path) -> dict | None:
    from ..chords.sheet import load_sheet

    path = chords_dir(split_dir, chords_root) / "sheet.json"
    return load_sheet(path) if path.exists() else None


def status(split_dir: Path, chords_root: Path, shift: int = 0) -> dict:
    """What the chord finder knows about the song, with the key as heard after
    moving it `shift` semitones:
    {"state": "none" | queued | running | error | "done", "job": status,
     "key", "key_label", "original_key_label", "tempo", "beat_times", "meter",
     "bars": [{"start", "beats"}] (the chord finder's bar grid), "lyrics": bool}."""
    from ..chords.align import grid_for
    from ..chords.theory import key_label

    cdir = chords_dir(split_dir, chords_root)
    if not (cdir / "meta.json").exists():
        return {"state": "none"}
    job = songstore.get_status(cdir)
    sheet = _sheet(split_dir, chords_root)
    if sheet is None:
        return {"state": job.get("state", "queued"), "job": job}
    lyrics = any(line.get("words") for line in sheet.get("lines") or [])
    grid = grid_for(sheet)
    return {
        "state": "done", "job": job, "chords_id": cdir.name,
        "key": sheet.get("key"), "key_label": key_label(sheet.get("key"), shift),
        "original_key_label": sheet.get("key_label"), "tempo": sheet.get("tempo"),
        "beat_times": sheet.get("beat_times") or [], "meter": sheet.get("meter") or {},
        "bars": grid.bars() if grid else [], "lyrics": lyrics,
    }


def sheet_view(split_dir: Path, chords_root: Path, shift: int = 0) -> dict | None:
    """The chord finder's sheet as its page shows it (sheet.view), with chords
    named in the key you hear: moved `shift` semitones, capo off."""
    from ..chords.sheet import apply_settings, view

    sheet = _sheet(split_dir, chords_root)
    if sheet is None:
        return None
    heard = copy.deepcopy(sheet)
    t = int(shift) % 12                    # the same chord names: +12 is +0, -1 is +11
    apply_settings(heard, transpose=t - 12 if t > 6 else t, capo=0)
    return view(heard)

