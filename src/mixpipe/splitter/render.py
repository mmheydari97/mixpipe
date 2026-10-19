"""Render the mix you hear in the browser to a file: stems x gains, optionally
at another speed and/or key (Rubber Band, see stretch.py), as WAV or MP3."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import stretch, versions
from .pipeline import STEM_NAMES, stem_path

FORMATS = {"wav": "audio/wav", "mp3": "audio/mpeg"}


def _fit(audio: np.ndarray) -> np.ndarray:
    """Scaled down only if it would clip."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio * (0.99 / peak) if peak > 0.99 else audio


def mix(song_dir: Path, gains: dict[str, float], folder: Path | None = None) -> tuple[np.ndarray, int]:
    """Sum the stems (from `folder`, default the song's stems/) with the given
    gains (missing names = 1.0, so leaving a stem out keeps it)."""
    import soundfile as sf

    def path(name: str) -> Path:
        return folder / f"{name}.flac" if folder else stem_path(song_dir, name)

    out, sr = None, None
    for name in STEM_NAMES:
        g = float(gains.get(name, 1.0))
        if g <= 0:
            continue
        y, sr = sf.read(str(path(name)), dtype="float32", always_2d=True)
        out = y * g if out is None else out + y * g
    if out is None:  # everything muted
        info = sf.info(str(path(STEM_NAMES[0])))
        return np.zeros((info.frames, 2), dtype=np.float32), info.samplerate
    return _fit(out).astype(np.float32), sr


def render(song_dir: Path, gains: dict[str, float], speed: float = 1.0,
           semitones: int = 0, fmt: str = "wav") -> bytes:
    """The mix as WAV or MP3 bytes. With a speed or key change it's mixed from
    the finished high-quality version when there is one (what the page plays),
    otherwise the mix is stretched with Rubber Band's fine engine."""
    import soundfile as sf

    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    speed, semitones = stretch.check(speed, semitones)
    song_dir = Path(song_dir)
    key = versions.version_key(speed, semitones)
    if key and versions.ready_quality(song_dir, key) == "fine":
        audio, sr = mix(song_dir, gains, versions.stems_dir(song_dir, key, "fine"))
    else:
        audio, sr = mix(song_dir, gains)
        if key:
            stretch.check_available(semitones)
            audio = _fit(stretch.stretch(audio, sr, speed, semitones, "fine"))
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    if fmt == "wav":
        return buf.getvalue()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("MP3 needs ffmpeg (brew install ffmpeg)")
    # a real file, not a pipe: ffmpeg can't fill in sizes on a pipe
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"mix.{fmt}"
        proc = subprocess.run([ffmpeg, "-v", "error", "-f", "wav", "-i", "pipe:0",
                               "-b:a", "256k", str(out)],
                              input=buf.getvalue(), capture_output=True, check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:200]}")
        return out.read_bytes()
