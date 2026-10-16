"""Audio file IO and format normalization.

Everything downstream assumes: float32/float64 numpy arrays, shape (n_samples,)
for mono or (n_samples, 2) for stereo, values nominally in [-1, 1], sample rate
equal to mixpipe.SAMPLE_RATE.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import SAMPLE_RATE

try:  # optional, better IO
    import soundfile as sf

    HAVE_SOUNDFILE = True
except ImportError:  # pragma: no cover
    sf = None
    HAVE_SOUNDFILE = False


AUDIO_EXTS = {".wav", ".wave", ".flac", ".aiff", ".aif", ".ogg"}
# what decode() takes (via ffmpeg for the compressed ones)
UPLOAD_EXTS = AUDIO_EXTS | {".mp3", ".m4a", ".aac", ".mp4", ".opus", ".webm", ".wma"}
WORK_SR = 44_100  # Demucs' native rate; the song tools bring everything here first


@dataclass
class Stem:
    """One track in the mix."""

    name: str
    audio: np.ndarray  # (n,) mono or (n, 2) stereo
    sr: int
    role: str = "default"
    path: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.audio.shape[0]

    @property
    def channels(self) -> int:
        return 1 if self.audio.ndim == 1 else self.audio.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sr

    def mono(self) -> np.ndarray:
        """Mono view for analysis purposes."""
        return self.audio if self.audio.ndim == 1 else self.audio.mean(axis=1)

    def copy_with(self, audio: np.ndarray) -> "Stem":
        return Stem(
            name=self.name,
            audio=audio,
            sr=self.sr,
            role=self.role,
            path=self.path,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Stem {self.name!r} role={self.role} "
            f"{self.duration:.2f}s {self.sr}Hz {self.channels}ch>"
        )


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def _read_wave_stdlib(path: str) -> tuple[np.ndarray, int]:
    """Fallback WAV reader for when soundfile isn't installed.

    Handles 8/16/24/32-bit PCM. Does not handle float WAV or compressed WAV;
    install soundfile for those.
    """
    with wave.open(path, "rb") as w:
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 1:  # unsigned
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int32)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(
            f"{path}: unsupported sample width {width * 8}-bit. "
            "Install soundfile (pip install soundfile)."
        )

    if n_ch > 1:
        data = data.reshape(-1, n_ch)
    return data, sr


def read_audio(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Read any supported audio file to float array + sample rate."""
    path = str(path)
    if HAVE_SOUNDFILE:
        audio, sr = sf.read(path, always_2d=False, dtype="float32")
        return audio, sr
    if not path.lower().endswith((".wav", ".wave")):
        raise RuntimeError(
            f"{path}: only WAV is supported without soundfile installed."
        )
    return _read_wave_stdlib(path)


def decode(path: str | os.PathLike) -> tuple[np.ndarray, int]:
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


def to_sample_rate(audio: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Polyphase resample. Exact ratio, no drift."""
    if sr == target:
        return audio
    from math import gcd

    g = gcd(int(sr), int(target))
    up, down = target // g, sr // g
    axis = 0
    return resample_poly(audio, up, down, axis=axis).astype(np.float32)


def normalize_shape(audio: np.ndarray, force_mono: bool = False) -> np.ndarray:
    """Collapse >2 channels to stereo, optionally fold to mono."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        if audio.shape[1] > 2:
            audio = audio[:, :2]
        if audio.shape[1] == 1:
            audio = audio[:, 0]
    if force_mono and audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def infer_role(filename: str, role_table: dict[str, str] | None = None) -> str:
    """Guess the instrument role from the filename.

    Longest matching key wins, so `electric_lead` beats `electric`.
    """
    from .config import DEFAULT_CONFIG

    table = role_table if role_table is not None else DEFAULT_CONFIG["roles"]
    stem_name = Path(filename).stem.lower()
    best, best_len = "default", 0
    for key, role in table.items():
        if key in stem_name and len(key) > best_len:
            best, best_len = role, len(key)
    return best


def load_stems(
    directory: str | os.PathLike,
    force_mono: bool = True,
    role_table: dict[str, str] | None = None,
) -> list[Stem]:
    """Load every audio file in a directory as a Stem, normalized and resampled."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"stem directory not found: {directory}")

    paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTS
    )
    if not paths:
        raise FileNotFoundError(
            f"no audio files in {directory} "
            f"(looking for {', '.join(sorted(AUDIO_EXTS))})"
        )

    stems: list[Stem] = []
    for p in paths:
        audio, sr = read_audio(p)
        audio = normalize_shape(audio, force_mono=force_mono)
        audio = to_sample_rate(audio, sr, SAMPLE_RATE)
        stems.append(
            Stem(
                name=p.stem,
                audio=audio,
                sr=SAMPLE_RATE,
                role=infer_role(p.name, role_table),
                path=str(p),
                meta={"source_sr": sr},
            )
        )
    return stems


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #


def write_audio(
    path: str | os.PathLike,
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    subtype: str = "PCM_24",
) -> str:
    """Write a float array to disk. Clips to [-1, 1) to avoid integer wrap."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 0.9999999)

    if HAVE_SOUNDFILE:
        sf.write(path, audio.astype(np.float32), sr, subtype=subtype)
        return path

    # stdlib fallback: 24-bit PCM WAV
    n_ch = 1 if audio.ndim == 1 else audio.shape[1]
    ints = np.round(audio * 8388607.0).astype(np.int32).reshape(-1)
    b = np.empty((ints.size, 3), dtype=np.uint8)
    b[:, 0] = ints & 0xFF
    b[:, 1] = (ints >> 8) & 0xFF
    b[:, 2] = (ints >> 16) & 0xFF
    with wave.open(path, "wb") as w:
        w.setnchannels(n_ch)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(b.tobytes())
    return path


def pad_to_common_length(stems: list[Stem]) -> list[Stem]:
    """Zero-pad every stem to the longest one so they can be summed."""
    if not stems:
        return stems
    n = max(s.n_samples for s in stems)
    out = []
    for s in stems:
        if s.n_samples == n:
            out.append(s)
            continue
        pad = n - s.n_samples
        width = ((0, pad),) if s.audio.ndim == 1 else ((0, pad), (0, 0))
        out.append(s.copy_with(np.pad(s.audio, width)))
    return out
