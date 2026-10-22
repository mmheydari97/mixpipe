"""Speed and key changes with Rubber Band: the one place MixPipe calls it.

Used by the splitter (speed/key versions and downloads), the mixer (tempo and
key from the chat) and tsm.pitch_shift. The browser can change speed by itself
(playbackRate + preservesPitch), but its quick time-stretch smears and warbles
away from 100%, and it can't change the key at all. Rubber Band does both much
more cleanly. In order of preference:

- the `rubberband` command-line tool (installed with Homebrew's ffmpeg):
  "fast" uses its R2 engine (~4 s for six 4-minute stems), "fine" its R3
  engine (~20 s, clearly cleaner);
- ffmpeg's rubberband filter (R2 only, so a single "fine" pass);
- ffmpeg's atempo (speed only, rougher).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

MIN_SPEED, MAX_SPEED = 0.5, 3.0
MAX_SEMITONES = 12
# share of the run spent in Rubber Band's first ("studying") pass, measured on a
# 4.5-minute stem: R2 0.3 s of 2.9 s, R3 0.1 s of 18 s
STUDY_SHARE = {"fast": 0.12, "fine": 0.01}
_PERCENT = re.compile(rb"^\s*(\d{1,3})%\s*$")


class StretchError(RuntimeError):
    pass


class Cancelled(Exception):
    """The caller's cancelled() turned true while Rubber Band was running."""


def check(speed: float, semitones: float) -> tuple[float, int]:
    """Validated (speed, semitones): speed in MIN_SPEED..MAX_SPEED, a whole
    number of semitones (= guitar frets) within +-MAX_SEMITONES."""
    speed = round(float(speed), 2)
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
    if float(semitones) != int(semitones) or abs(int(semitones)) > MAX_SEMITONES:
        raise ValueError(f"key change must be a whole number of semitones "
                         f"between -{MAX_SEMITONES} and +{MAX_SEMITONES}")
    return speed, int(semitones)


@lru_cache(maxsize=1)
def _ffmpeg_filters() -> frozenset[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return frozenset()
    out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True)
    return frozenset(line.split()[1] for line in out.stdout.splitlines() if len(line.split()) > 1)


def has_rubberband() -> bool:
    """The rubberband tool itself (both engines, formant preservation)."""
    return shutil.which("rubberband") is not None


def qualities() -> tuple[str, ...]:
    """The passes a version gets, quickest first ("fast" is skipped when only
    one engine is available)."""
    if has_rubberband():
        return ("fast", "fine")
    return ("fine",) if shutil.which("ffmpeg") else ()


def check_available(semitones: float) -> None:
    if has_rubberband() or "rubberband" in _ffmpeg_filters():
        return
    if semitones:
        raise StretchError("changing the key needs Rubber Band: brew install rubberband")
    if not shutil.which("ffmpeg"):
        raise StretchError("changing speed needs ffmpeg: brew install ffmpeg")


def _command(src: Path, dst: Path, speed: float, semitones: float, quality: str,
             formant: bool) -> list[str]:
    rubberband = shutil.which("rubberband")
    if rubberband:
        # no -q: its stderr carries the progress (see _watch_progress)
        cmd = [rubberband, "-3" if quality == "fine" else "-2", "--centre-focus",
               "--tempo", str(speed), "--pitch", f"{semitones:g}"]
        return cmd + (["--formant"] if formant else []) + [str(src), str(dst)]
    check_available(semitones)
    if "rubberband" in _ffmpeg_filters():
        af = f"rubberband=tempo={speed}:pitch={2 ** (semitones / 12)}:pitchq=quality"
        af += ":formant=preserved" if formant else ""
    else:
        af = f"atempo={speed}"
    return [shutil.which("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-af", af,
            "-c:a", "pcm_f32le", str(dst)]


def _watch_progress(stream, quality: str, progress: Callable[[float], None] | None,
                    tail: list[str]) -> None:
    """Read Rubber Band's stderr: "Pass 1: Studying..." and "Pass 2:
    Processing...", each counting 0%..100% on \r-separated lines. Reports the
    overall fraction; other lines are kept in `tail` for error messages."""
    study = STUDY_SHARE.get(quality, 0.05)
    second_pass = False
    buf = b""
    for chunk in iter(lambda: stream.read1(4096), b""):
        buf += chunk
        *lines, buf = re.split(rb"[\r\n]", buf)
        for line in lines:
            m = _PERCENT.match(line)
            if m:
                pct = min(int(m.group(1)), 100) / 100
                if progress:
                    progress(study + (1 - study) * pct if second_pass else study * pct)
            elif line.strip():
                second_pass = second_pass or line.startswith(b"Pass 2")
                tail[:] = (tail + [line.decode(errors="replace").strip()])[-5:]


def stretch(audio: np.ndarray, sr: int, speed: float, semitones: float = 0,
            quality: str = "fine", formant: bool = False,
            cancelled: Callable[[], bool] | None = None,
            progress: Callable[[float], None] | None = None) -> np.ndarray:
    """audio (frames x channels, float) at `speed` and moved by `semitones`
    (fractions of a semitone are fine here; check() is for what a person picks).

    The result is exactly len(audio) / speed frames long, so stems stay the
    same length (and in sync), and always 2-D. It isn't clipped: Rubber Band can overshoot 1.0
    a little. `formant=True` keeps a voice's character when the key moves.
    `cancelled()` is polled while Rubber Band runs; when it turns true the
    process is killed and Cancelled is raised. `progress(fraction)` is called
    as it goes (the rubberband tool only; ffmpeg reports none).
    """
    import soundfile as sf

    if speed == 1 and semitones == 0:
        return audio
    with tempfile.TemporaryDirectory(prefix="mixpipe-stretch-") as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        sf.write(str(src), audio, sr, subtype="FLOAT")  # float in -> float out, no wrap-around
        proc = subprocess.Popen(_command(src, dst, speed, semitones, quality, formant),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tail: list[str] = []
        reader = threading.Thread(target=_watch_progress, daemon=True,
                                  args=(proc.stderr, quality, progress, tail))
        reader.start()
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancelled and cancelled():
                    proc.kill()
                    proc.wait()
                    raise Cancelled() from None
        reader.join(timeout=5)
        if proc.returncode != 0 or not dst.exists():
            raise StretchError(f"{Path(proc.args[0]).name} failed: {' / '.join(tail)[-200:]}")
        out, _ = sf.read(str(dst), dtype="float32", always_2d=True)
    if progress:
        progress(1.0)
    n = round(len(audio) / speed)
    if len(out) < n:
        out = np.pad(out, ((0, n - len(out)), (0, 0)))
    return out[:n]


def stretch_stems(stems: dict[str, np.ndarray], sr: int, speed: float, semitones: float = 0,
                  quality: str = "fine", formant: Callable[[str], bool] = lambda name: False,
                  cancelled: Callable[[], bool] | None = None,
                  progress: Callable[[float], None] | None = None,
                  keep_loudness: bool = False) -> dict[str, np.ndarray]:
    """Every stem through stretch(), in parallel (one Rubber Band process each).

    Each comes back len/speed frames long in its input's shape (a mono stem
    stays 1-D). `formant(name)` picks the stems that keep their formants when
    the key moves: the voices. `progress(fraction)` gets the average.
    `keep_loudness` turns each result back to its input's loudness: R3 softens
    sharp attacks, which can take a plucked part 1-2 dB down, and a mix's
    balance shouldn't move because its tempo did."""
    if not stems:
        return {}
    done = dict.fromkeys(stems, 0.0)

    def report(name: str) -> Callable[[float], None]:
        def update(frac: float) -> None:
            done[name] = frac
            if progress:
                progress(sum(done.values()) / len(done))
        return update

    def one(name: str) -> tuple[str, np.ndarray]:
        y = np.asarray(stems[name], dtype=np.float32)
        z = stretch(y if y.ndim == 2 else y[:, None], sr, speed, semitones, quality,
                    formant=bool(semitones) and formant(name), cancelled=cancelled,
                    progress=report(name))
        z = z[:, 0] if y.ndim == 1 else z
        if keep_loudness:
            from .analysis import integrated_lufs

            mono = lambda a: a.mean(axis=1) if a.ndim == 2 else a  # noqa: E731
            lift = integrated_lufs(mono(y), sr) - integrated_lufs(mono(z), sr)
            if np.isfinite(lift):
                z = z * np.float32(10 ** (lift / 20))
        return name, z

    with ThreadPoolExecutor(len(stems), thread_name_prefix="stretch") as pool:
        return dict(pool.map(one, stems))
