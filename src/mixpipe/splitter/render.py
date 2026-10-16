"""Render the mix you hear in the browser to a file: stems x gains, optionally
slowed down (same pitch), as WAV or MP3."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .pipeline import STEM_NAMES, stem_path

FORMATS = {"wav": "audio/wav", "mp3": "audio/mpeg"}
MIN_SPEED, MAX_SPEED = 0.5, 1.0


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("slowing down and MP3 need ffmpeg (brew install ffmpeg)")
    return ffmpeg


def has_filter(name: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True)
    return any(line.split()[1:2] == [name] for line in out.stdout.splitlines())


def mix(song_dir: Path, gains: dict[str, float]) -> tuple[np.ndarray, int]:
    """Sum the stems with the given gains (missing names = 1.0, so leaving a
    stem out keeps it). Scaled down only if the sum would clip."""
    import soundfile as sf

    out, sr = None, None
    for name in STEM_NAMES:
        g = float(gains.get(name, 1.0))
        if g <= 0:
            continue
        y, sr = sf.read(str(stem_path(song_dir, name)), dtype="float32", always_2d=True)
        out = y * g if out is None else out + y * g
    if out is None:  # everything muted
        info = sf.info(str(stem_path(song_dir, STEM_NAMES[0])))
        return np.zeros((info.frames, 2), dtype=np.float32), info.samplerate
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out.astype(np.float32), sr


def render(song_dir: Path, gains: dict[str, float], speed: float = 1.0,
           fmt: str = "wav") -> bytes:
    """The mix as WAV or MP3 bytes. speed < 1 slows it down without changing
    pitch (ffmpeg rubberband, or atempo if ffmpeg was built without it)."""
    import soundfile as sf

    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
    audio, sr = mix(Path(song_dir), gains)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    wav = buf.getvalue()
    if speed == 1.0 and fmt == "wav":
        return wav

    cmd = [_ffmpeg(), "-v", "error", "-f", "wav", "-i", "pipe:0"]
    if speed != 1.0:
        stretch = f"rubberband=tempo={speed}" if has_filter("rubberband") else f"atempo={speed}"
        cmd += ["-af", stretch]
    cmd += ["-c:a", "pcm_s16le"] if fmt == "wav" else ["-b:a", "256k"]
    # a real file, not a pipe: ffmpeg can't fill in WAV sizes on a pipe
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"mix.{fmt}"
        proc = subprocess.run(cmd + [str(out)], input=wav, capture_output=True, check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:200]}")
        return out.read_bytes()
