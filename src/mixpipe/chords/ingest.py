"""Decode uploads and decide what plays the harmony and what carries the lyrics.

  one file               -> a full mix; separate it (Demucs) unless told not to
  several, some "vocal"  -> vocal stems feed lyrics, the rest (minus drums) chords
  several, no vocal      -> everything (minus drums) feeds chords, no lyrics

Roles come from filenames via io_utils.infer_role (vox.wav, bass.wav, ...).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io_utils import infer_role, normalize_shape, read_audio, to_sample_rate

WORK_SR = 44_100  # Demucs' native rate; everything is brought here first

UPLOAD_EXTS = {
    ".wav", ".wave", ".flac", ".aiff", ".aif", ".ogg", ".mp3", ".m4a", ".aac",
    ".mp4", ".opus", ".webm", ".wma",
}


@dataclass
class Ingested:
    mix: np.ndarray                  # (n, 2) everything summed, for playback/separation
    sr: int
    mode: str                        # "mix" | "stems"
    harmony: np.ndarray | None = None  # mono, drums and vocals excluded (stems mode)
    vocal: np.ndarray | None = None    # mono (stems mode)
    bass: np.ndarray | None = None     # mono (stems mode)
    roles: dict[str, str] | None = None


def decode(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any audio file: soundfile first, ffmpeg for everything else (m4a...)."""
    path = Path(path)
    try:
        return read_audio(path)
    except Exception:  # noqa: BLE001 - fall through to ffmpeg
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"{path.name}: can't decode this format without ffmpeg (brew install ffmpeg)"
        )
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2",
         "-ar", str(WORK_SR), "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"{path.name}: ffmpeg could not decode it: "
                           f"{proc.stderr.decode(errors='replace')[:200]}")
    audio = np.frombuffer(proc.stdout, dtype="<f4").reshape(-1, 2)
    return audio.copy(), WORK_SR


def _stereo(audio: np.ndarray) -> np.ndarray:
    return np.stack([audio, audio], axis=1) if audio.ndim == 1 else audio


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def _sum(tracks: list[np.ndarray], n: int) -> np.ndarray:
    out = np.zeros((n,) + tracks[0].shape[1:], dtype=np.float32)
    for t in tracks:
        out[: len(t)] += t
    return out


def ingest(paths: list[str | Path], role_table: dict[str, str] | None = None) -> Ingested:
    if not paths:
        raise ValueError("no audio files given")
    tracks: list[tuple[str, np.ndarray]] = []
    for p in paths:
        audio, sr = decode(p)
        audio = to_sample_rate(normalize_shape(audio), sr, WORK_SR)
        tracks.append((Path(p).name, _stereo(audio).astype(np.float32)))

    n = max(len(a) for _, a in tracks)
    mix = _sum([a for _, a in tracks], n)
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 1.0:
        mix /= peak

    if len(tracks) == 1:
        return Ingested(mix=mix, sr=WORK_SR, mode="mix")

    roles = {name: infer_role(name, role_table) for name, _ in tracks}
    vocal = [_mono(a) for name, a in tracks if roles[name] == "vocal"]
    bass = [_mono(a) for name, a in tracks if roles[name] == "bass"]
    harmony = [_mono(a) for name, a in tracks if roles[name] not in ("vocal", "drums")]
    return Ingested(
        mix=mix,
        sr=WORK_SR,
        mode="stems",
        harmony=_sum(harmony, n) if harmony else None,
        vocal=_sum(vocal, n) if vocal else None,
        bass=_sum(bass, n) if bass else None,
        roles=roles,
    )
