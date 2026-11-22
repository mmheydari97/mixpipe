"""The six-stem split (vocals / guitar / drums / bass / keys / other) of a chord
finder song, made only when something asks for it: the Vocals and Guitar
sliders under the player, and the guitar tab. Nothing is split up front.

One separation per song, and one implementation of it (the splitter's):

- if the splitter already has the song (same id, see songstore.copy_song), its
  stems are used where they are;
- otherwise splitter.pipeline.split runs on the chord finder's own song folder,
  writing out/chords/<id>/stems/ and stems.json there, so the song doesn't
  appear in the splitter's library.

stems/backing.flac (drums + bass + keys + other) is written next to them, so the
chord finder plays three tracks: vocals, guitar and the rest. The job keeps its
own status file (STATUS), so the song's own status stays "done" meanwhile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .. import songstore
from ..songstore import Progress
from ..splitter.pipeline import STEM_NAMES, load_info, split, stem_path

STATUS = "stems-status.json"
PLAY = ("vocals", "guitar", "backing")       # what the chord finder's player loads
BACKING = ("drums", "bass", "piano", "other")


def _complete(d: Path) -> bool:
    return bool(load_info(d)) and all(stem_path(d, n).exists() for n in STEM_NAMES)


def stems_dir(song_dir: Path, splitter_root: Path | None = None) -> Path | None:
    """The folder holding the song's six finished stems, or None."""
    song_dir = Path(song_dir)
    if splitter_root is not None and _complete(Path(splitter_root) / song_dir.name):
        return Path(splitter_root) / song_dir.name
    return song_dir if _complete(song_dir) else None


def backing_path(song_dir: Path) -> Path:
    return Path(song_dir) / "stems" / "backing.flac"


def stem_file(song_dir: Path, name: str, splitter_root: Path | None = None) -> Path | None:
    """One of the six stems or "backing", if it has been made."""
    if name == "backing":
        p = backing_path(song_dir)
        return p if p.exists() else None
    if name not in STEM_NAMES:
        raise ValueError(f"no stem called {name}")
    d = stems_dir(song_dir, splitter_root)
    return stem_path(d, name) if d else None


def ready(song_dir: Path, splitter_root: Path | None = None) -> bool:
    return stems_dir(song_dir, splitter_root) is not None and backing_path(song_dir).exists()


def ensure(song_dir: str | Path, splitter_root: Path | None = None,
           progress: Progress | None = None, separator: Callable | None = None) -> dict:
    """Make (or find) the six stems and the backing; returns {name: Path} for
    the six stems and "backing"."""
    import soundfile as sf

    song_dir = Path(song_dir)
    report = progress or (lambda stage, frac, msg: None)
    d = stems_dir(song_dir, splitter_root)
    if d is None:
        meta = songstore.read_meta(song_dir)
        if len(meta["inputs"]) > 1:
            raise ValueError("this song was added as separate stems, so it can't be split again")
        split(song_dir, report, separator=separator)
        d = song_dir
    else:
        report("separate", 1.0, "using the stems already made")
    out = {n: stem_path(d, n) for n in STEM_NAMES}
    backing = backing_path(song_dir)
    if not backing.exists():
        report("separate", 0.98, "mixing the backing")
        total, sr = None, None
        for name in BACKING:
            y, sr = sf.read(str(out[name]), dtype="float32", always_2d=True)
            total = y if total is None else total[: len(y)] + y[: len(total)]
        backing.parent.mkdir(parents=True, exist_ok=True)
        tmp = backing.with_suffix(".part.flac")
        sf.write(str(tmp), np.clip(total, -1, 1), sr, format="FLAC", subtype="PCM_16")
        tmp.replace(backing)                     # never a half-written file to serve
    out["backing"] = backing
    gain_db(song_dir, splitter_root)          # measured now, so the page doesn't wait for it
    report("separate", 1.0, "ready")
    return out


def run_job(song_dir: Path, splitter_root: Path | None = None, cancelled: Callable | None = None,
            separator: Callable | None = None) -> dict | None:
    """ensure() with its own status file (STATUS); never raises."""
    return songstore.run_job(
        song_dir, lambda progress: ensure(song_dir, splitter_root, progress, separator),
        done_stage="separate", cancelled=cancelled, status_file=STATUS)


HEADROOM = 0.95     # the loudest the three tracks may add up to in the page


def gain_db(song_dir: Path, splitter_root: Path | None = None) -> float:
    """How much the page turns the three tracks down so they can't clip.

    Separated stems don't add back up to exactly the recording: their sum peaks
    higher (1-2 dB on loud masters), and played at full level in the browser
    that clips. The peak of vocals + guitar + backing is measured once and kept
    in stems/playalong.json."""
    import json

    import soundfile as sf

    path = Path(song_dir) / "stems" / "playalong.json"
    if path.exists():
        return json.loads(path.read_text())["gain_db"]
    total = None
    for name in PLAY:
        y, _ = sf.read(str(stem_file(song_dir, name, splitter_root)), dtype="float32", always_2d=True)
        total = y if total is None else total[: len(y)] + y[: len(total)]
    peak = float(np.max(np.abs(total))) if total is not None and total.size else 0.0
    gain = min(0.0, 20 * np.log10(HEADROOM / peak)) if peak > 0 else 0.0
    path.write_text(json.dumps({"peak": round(peak, 4), "gain_db": round(float(gain), 2)}))
    return round(float(gain), 2)


def status(song_dir: Path, splitter_root: Path | None = None) -> dict:
    """{"ready": bool, "gain_db" (when ready), **the job's status}."""
    st = songstore.get_status(song_dir, STATUS)
    ok = ready(song_dir, splitter_root)
    out = {**st, "ready": ok}
    if ok:
        out["gain_db"] = gain_db(song_dir, splitter_root)
    return out
