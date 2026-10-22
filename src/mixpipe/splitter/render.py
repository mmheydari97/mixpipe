"""Render what you hear in the browser to a file: the mix (stems x gains), one
stem, or every stem in a zip, optionally at another speed and/or key (Rubber
Band, see mixpipe/stretch.py) and with the stems' effects (effects.py), as WAV,
FLAC or MP3. `fx` is the page's effect settings (effects.validate() shape) and
`tempo` the song's BPM as recorded, for echoes locked to it."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .. import stretch
from . import effects, takes, versions
from .pipeline import STEM_NAMES, stem_path

FORMATS = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg"}


def _fit(audio: np.ndarray) -> np.ndarray:
    """Scaled down only if it would clip."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio * (0.99 / peak) if peak > 0.99 else audio


def _check_format(fmt: str) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")


def encode(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    """16-bit WAV or FLAC, or 256 kbit/s MP3 through ffmpeg."""
    import soundfile as sf

    _check_format(fmt)
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="FLAC" if fmt == "flac" else "WAV",
             subtype="PCM_16")
    if fmt != "mp3":
        return buf.getvalue()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("MP3 needs ffmpeg (brew install ffmpeg)")
    # a real file, not a pipe: ffmpeg can't fill in sizes on a pipe
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp3"
        proc = subprocess.run([ffmpeg, "-v", "error", "-f", "wav", "-i", "pipe:0",
                               "-b:a", "256k", str(out)],
                              input=buf.getvalue(), capture_output=True, check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:200]}")
        return out.read_bytes()


def download_name(song_id: str, ext: str, speed: float = 1.0, semitones: int = 0,
                  part: str = "") -> str:
    """e.g. "<id>-vocals-80pct-up2st.flac": what it is and how it was changed."""
    name = f"{song_id}{part}"
    if speed != 1.0:
        name += f"-{round(speed * 100)}pct"
    if semitones:
        name += f"-{'up' if semitones > 0 else 'down'}{abs(semitones)}st"
    return re.sub(r"[^\w.-]+", "_", f"{name}.{ext}")


def mix(song_dir: Path, gains: dict[str, float], folder: Path | None = None,
        paths: dict[str, Path] | None = None) -> tuple[np.ndarray, int]:
    """Sum the stems (from `folder`, default the song's stems/; `paths` overrides
    single stems, e.g. with their effects, and adds takes) with the given gains
    (a stem left out keeps 1.0; a take is only in the mix when its gain is given)."""
    import soundfile as sf

    def path(name: str) -> Path:
        if paths and name in paths:
            return paths[name]
        return folder / f"{name}.flac" if folder else stem_path(song_dir, name)

    out, sr = None, None
    extra = [n for n in (paths or {}) if n not in STEM_NAMES and n in gains]
    for name in list(STEM_NAMES) + extra:
        g = float(gains.get(name, 1.0))
        if g <= 0:
            continue
        y, sr = sf.read(str(path(name)), dtype="float32", always_2d=True)
        out = y * g if out is None else out + y * g
    if out is None:  # everything muted
        info = sf.info(str(path(STEM_NAMES[0])))
        return np.zeros((info.frames, 2), dtype=np.float32), info.samplerate
    return _fit(out).astype(np.float32), sr


def _with_fx(song_dir: Path, folder: Path | None, fx: dict | None, tempo: float | None,
             speed: float) -> dict[str, Path]:
    """The stems of `folder` that have effects, rendered (and cached) by effects.py."""
    if not fx:
        return {}
    bpm = tempo * speed if tempo else None
    return effects.render(song_dir, folder, effects.validate(fx), bpm)


def _takes(song_dir: Path, gains: dict[str, float], key: str | None) -> dict[str, Path]:
    """The takes the mix has a gain for, as stems of version `key`."""
    return {name: takes.path(song_dir, takes.number(name), key) for name, g in gains.items()
            if takes.number(name) is not None and float(g) > 0}


def render(song_dir: Path, gains: dict[str, float], speed: float = 1.0,
           semitones: int = 0, fmt: str = "wav", fx: dict | None = None,
           tempo: float | None = None) -> bytes:
    """The mix as a file. With a speed or key change it's mixed from the
    finished high-quality version when there is one (what the page plays),
    otherwise the mix is stretched with Rubber Band's fine engine (keeping
    formants when only the voice is in it)."""
    _check_format(fmt)
    speed, semitones = stretch.check(speed, semitones)
    song_dir = Path(song_dir)
    key = versions.version_key(speed, semitones)
    if key and versions.ready_quality(song_dir, key) == "fine":
        folder = versions.stems_dir(song_dir, key, "fine")
        paths = {**_with_fx(song_dir, folder, fx, tempo, speed), **_takes(song_dir, gains, key)}
        audio, sr = mix(song_dir, gains, folder, paths)
    else:
        paths = {**_with_fx(song_dir, None, fx, tempo, 1.0), **_takes(song_dir, gains, None)}
        audio, sr = mix(song_dir, gains, paths=paths)
        if key:
            stretch.check_available(semitones)
            heard = [n for n in STEM_NAMES if float(gains.get(n, 1.0)) > 0]
            voice = bool(heard) and all(versions.is_voice(n) for n in heard)
            audio = _fit(stretch.stretch(audio, sr, speed, semitones, "fine", formant=voice))
    return encode(audio, sr, fmt)


def render_stem(song_dir: Path, name: str, speed: float = 1.0, semitones: int = 0,
                fmt: str = "flac", fx: dict | None = None, tempo: float | None = None) -> bytes:
    """One stem (or take) at full level, as render() makes the mix."""
    if name not in STEM_NAMES and takes.number(name) is None:
        raise ValueError(f"no stem called {name}")
    gains = {n: float(n == name) for n in STEM_NAMES}
    if takes.number(name) is not None:
        gains[name] = 1.0
    return render(song_dir, gains, speed, semitones, fmt, fx, tempo)


def _read(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    import soundfile as sf

    return {n: sf.read(str(p), dtype="float32", always_2d=True)[0] for n, p in paths.items()}


def stems_zip(song_dir: Path, song_id: str, speed: float = 1.0, semitones: int = 0,
              fmt: str = "flac", fx: dict | None = None, tempo: float | None = None) -> bytes:
    """Every stem at full level in one zip (stored, not compressed: the audio
    already is), from the finished version when there is one, otherwise
    stretched now, all stems at once."""
    _check_format(fmt)
    speed, semitones = stretch.check(speed, semitones)
    song_dir = Path(song_dir)
    key = versions.version_key(speed, semitones)
    if key and versions.ready_quality(song_dir, key) == "fine":
        folder = versions.stems_dir(song_dir, key, "fine")
        sr, stems = versions.load_stems(song_dir, folder)
        stems.update(_read({**_with_fx(song_dir, folder, fx, tempo, speed), **takes.paths(song_dir, key)}))
    else:
        sr, stems = versions.load_stems(song_dir)
        stems.update(_read({**_with_fx(song_dir, None, fx, tempo, 1.0), **takes.paths(song_dir)}))
        if key:
            stretch.check_available(semitones)
            stems = stretch.stretch_stems(stems, sr, speed, semitones, "fine",
                                          formant=lambda n: versions.is_voice(n) or takes.number(n) is not None)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, audio in stems.items():
            part = f"-take{takes.number(name)}" if takes.number(name) is not None else f"-{name}"
            arcname = download_name(song_id, fmt, speed, semitones, part=part)
            zf.writestr(arcname, encode(audio, sr, fmt))
    return buf.getvalue()
