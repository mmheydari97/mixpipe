"""(download ->) decode -> separate, for one song.

Each song lives in out/splitter/<song_id>/ (folder layout and ids: songstore.py):

  meta.json, inputs/, status.json   see songstore
  stems/<name>.flac                 stereo, 16-bit, one per STEMS entry
  stems.json                        names, duration, model: marks the song done
  versions/                         stems at other speeds/keys (versions.py)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from .. import songstore
from ..io_utils import WORK_SR, decode, normalize_shape, to_sample_rate
from ..songstore import Progress, _write_json, download_inputs

DEFAULT_ROOT = Path("out/splitter")
MODEL = "htdemucs_6s"
STAGES = ("download", "decode", "separate")
# (file name, label in the UI); file names are Demucs' source names
STEMS = (("vocals", "Vocals"), ("guitar", "Guitar"), ("drums", "Drums"),
         ("bass", "Bass"), ("piano", "Keys"), ("other", "Other"))
STEM_NAMES = tuple(name for name, _ in STEMS)


def stem_path(song_dir: Path, name: str) -> Path:
    return Path(song_dir) / "stems" / f"{name}.flac"


def load_info(song_dir: Path) -> dict | None:
    path = Path(song_dir) / "stems.json"
    return json.loads(path.read_text()) if path.exists() else None


def _stereo(audio: np.ndarray) -> np.ndarray:
    audio = normalize_shape(audio)
    return np.stack([audio, audio], axis=1) if audio.ndim == 1 else audio[:, :2]


def split(song_dir: str | Path, progress: Progress | None = None,
          separator: Callable | None = None, downloader: Callable | None = None) -> dict:
    """Separate one song into STEMS and write stems.json. Returns its content.

    `separator(audio, sr, model=, progress=cb) -> list[Stem]` and
    `downloader(source, dest_dir, progress=cb) -> Path` can be injected (tests
    use fakes so nothing is downloaded).
    """
    import soundfile as sf

    song_dir = Path(song_dir)
    report = progress or (lambda stage, frac, msg: None)
    meta = download_inputs(song_dir, report, downloader)

    info = load_info(song_dir)
    if info and all(stem_path(song_dir, n).exists() for n in STEM_NAMES):
        report("separate", 1.0, "using cached stems")
        return info

    report("decode", 0.0, "decoding audio")
    if len(meta["inputs"]) != 1:
        raise ValueError("the splitter takes one file: the full song")
    audio, sr = decode(song_dir / "inputs" / meta["inputs"][0])
    audio = to_sample_rate(_stereo(audio), sr, WORK_SR).astype(np.float32)
    duration = len(audio) / WORK_SR
    report("decode", 1.0, f"{duration:.0f} s of audio")

    report("separate", 0.0, "separating vocals / guitar / drums / bass / keys / other (Demucs)")
    if separator is None:
        from ..separation import separate as separator
    stems = {st.name: st.audio for st in separator(
        audio, WORK_SR, model=MODEL, progress=lambda f: report("separate", f, "separating"))}
    missing = [n for n in STEM_NAMES if n not in stems]
    if missing:
        raise RuntimeError(f"separator returned no {', '.join(missing)} stem")
    (song_dir / "stems").mkdir(exist_ok=True)
    shutil.rmtree(song_dir / "versions", ignore_errors=True)  # made from the old stems
    for name in STEM_NAMES:
        y = _stereo(stems[name])[: len(audio)]
        if len(y) < len(audio):  # keep every stem the same length, for sync
            y = np.pad(y, ((0, len(audio) - len(y)), (0, 0)))
        sf.write(str(stem_path(song_dir, name)), np.clip(y, -1, 1), WORK_SR,
                 format="FLAC", subtype="PCM_16")
    info = {"model": MODEL, "sr": WORK_SR, "duration": round(duration, 3),
            "stems": [{"name": n, "label": label} for n, label in STEMS]}
    _write_json(song_dir / "stems.json", info)
    report("separate", 1.0, "separated")
    return info


def run_job(song_dir: Path, options: dict | None = None, cancelled=None,
            **kwargs) -> dict | None:
    """split() with status.json bookkeeping; never raises. `cancelled()` stops the
    job at its next progress report."""
    return songstore.run_job(song_dir, lambda progress: split(song_dir, progress, **kwargs),
                             done_stage="separate", cancelled=cancelled)
