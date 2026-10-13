"""Speech-to-text for sung vocals, with word timestamps.

Backends, all returning the same transcript shape:

  mlx            mlx-whisper on the Apple Silicon GPU (default; free, offline)
  faster-whisper CPU fallback when MLX isn't installed
  openrouter     any OpenRouter transcription model. Word timing only comes back
                 from OpenAI-compatible upstreams; otherwise times are estimated
                 and the transcript is marked timing="approx".

Transcript: {"backend", "model", "language", "timing": "word"|"approx"|"none",
             "lines": [[{"text", "start", "end"}, ...], ...]}

Whisper hallucinates on silence and on instrumental breaks ("Thank you for
watching"), so words are dropped unless they overlap audible vocal activity on
the separated vocal stem.
"""

from __future__ import annotations

import io
import re

import numpy as np

from ..io_utils import to_sample_rate

WHISPER_SR = 16_000
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
FW_MODEL = "large-v3-turbo"

LINE_GAP_S = 0.7        # silence that always starts a new lyric line
SOFT_GAP_S = 0.25       # silence that starts a new line once the line is long
MAX_LINE_WORDS = 12
ACTIVITY_SLACK_S = 0.4  # words this close to vocal activity still count
MIN_LINE_COVERAGE = 0.3 # lines mostly outside vocal activity are hallucinations


# --------------------------------------------------------------------------- #
# vocal activity
# --------------------------------------------------------------------------- #


def vocal_activity(vocal: np.ndarray, sr: int, frame_s: float = 0.05) -> list[tuple[float, float]]:
    """Regions where the vocal stem is audibly active, as (start_s, end_s)."""
    mono = vocal if vocal.ndim == 1 else vocal.mean(axis=1)
    hop = max(1, int(frame_s * sr))
    n = len(mono) // hop
    if n == 0:
        return []
    frames = mono[: n * hop].reshape(n, hop)
    db = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    loud = np.percentile(db, 95)
    threshold = max(loud - 30.0, -55.0)
    active = db > threshold

    regions: list[list[float]] = []
    for i in np.flatnonzero(active):
        t = i * frame_s
        if regions and t - regions[-1][1] <= 0.3:
            regions[-1][1] = t + frame_s
        else:
            regions.append([t, t + frame_s])
    return [(a, b) for a, b in regions if b - a >= 0.2]


def has_vocals(regions: list[tuple[float, float]], min_total_s: float = 3.0) -> bool:
    return sum(b - a for a, b in regions) >= min_total_s


def _in_activity(t0: float, t1: float, regions: list[tuple[float, float]]) -> bool:
    return any(t0 < b + ACTIVITY_SLACK_S and t1 > a - ACTIVITY_SLACK_S for a, b in regions)


# --------------------------------------------------------------------------- #
# shaping
# --------------------------------------------------------------------------- #


def _word(text: str, start: float, end: float) -> dict:
    return {"text": text.strip(), "start": round(float(start), 3),
            "end": round(float(max(end, start)), 3)}


def split_lines(segments: list[list[dict]]) -> list[list[dict]]:
    """Turn recogniser segments into lyric lines: break at segment boundaries,
    at long gaps, and at moderate gaps once a line gets long."""
    lines: list[list[dict]] = []
    for seg in segments:
        line: list[dict] = []
        for w in seg:
            if not w["text"]:
                continue
            if line:
                gap = w["start"] - line[-1]["end"]
                if gap >= LINE_GAP_S or (len(line) >= MAX_LINE_WORDS and gap >= SOFT_GAP_S):
                    lines.append(line)
                    line = []
            line.append(w)
        if line:
            lines.append(line)
    return lines


def _coverage(t0: float, t1: float, regions) -> float:
    """Fraction of [t0, t1] that overlaps vocal activity."""
    if t1 <= t0:
        return 1.0 if _in_activity(t0, t1, regions) else 0.0
    overlap = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in regions)
    return overlap / (t1 - t0)


def filter_hallucinations(lines: list[list[dict]], regions) -> list[list[dict]]:
    """Drop lines that mostly sit outside vocal activity, then stray words.

    Judged per line first because Whisper often starts a line's first word early
    (at the end of the previous segment), so per-word tests alone either keep
    "Thank you." tails or eat real first words.
    """
    if not regions:
        return lines
    out = []
    for line in lines:
        if _coverage(line[0]["start"], line[-1]["end"], regions) < MIN_LINE_COVERAGE:
            continue
        kept = [w for w in line if _in_activity(w["start"], w["end"], regions)]
        if kept:
            out.append(kept)
    return out


def _spread(tokens: list[str], start: float, end: float) -> list[dict]:
    """Give tokens times inside [start, end], proportional to their length."""
    if not tokens:
        return []
    weights = np.array([max(1, len(t)) for t in tokens], dtype=float)
    edges = start + (end - start) * np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
    return [_word(t, edges[i], edges[i + 1]) for i, t in enumerate(tokens)]


def _voiced_mapper(regions: list[tuple[float, float]]):
    """Map "seconds of voiced time" to absolute time, so text with no timing at
    all can be laid out where somebody is actually singing."""
    cum = np.concatenate([[0.0], np.cumsum([b - a for a, b in regions])])

    def to_abs(v: float) -> float:
        k = int(np.clip(np.searchsorted(cum, v, side="right") - 1, 0, len(regions) - 1))
        return regions[k][0] + min(v - cum[k], regions[k][1] - regions[k][0])

    return to_abs, float(cum[-1])


_SENTENCE = re.compile(r"(?<=[.!?。！？؟])\s+")


def normalize_remote(resp: dict, regions: list[tuple[float, float]], duration: float) -> tuple[list[list[dict]], str]:
    """Normalise an OpenRouter transcription response to (lines, timing)."""
    words = resp.get("words") or []
    segments = [s for s in (resp.get("segments") or []) if "start" in s and "end" in s]

    if words:
        ws = [_word(w.get("word", w.get("text", "")), w["start"], w["end"]) for w in words]
        if segments:
            groups: list[list[dict]] = [[] for _ in segments]
            for w in ws:
                mid = 0.5 * (w["start"] + w["end"])
                k = next((i for i, s in enumerate(segments) if mid < s["end"]), len(segments) - 1)
                groups[k].append(w)
            return split_lines([g for g in groups if g]), "word"
        return split_lines([ws]), "word"

    if segments and any(s.get("text") for s in segments):
        lines = [_spread(s.get("text", "").split(), s["start"], s["end"]) for s in segments]
        return [ln for ln in lines if ln], "approx"

    text = (resp.get("text") or "").strip()
    if not text:
        return [], "none"
    raw_lines = [ln.strip() for part in text.split("\n") for ln in _SENTENCE.split(part)]
    raw_lines = [ln for ln in raw_lines if ln]
    regions = regions or [(0.0, duration)]
    to_abs, total = _voiced_mapper(regions)
    weights = np.array([max(1, len(ln)) for ln in raw_lines], dtype=float)
    edges = total * np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
    lines = []
    for i, ln in enumerate(raw_lines):
        tokens = ln.split()
        local = _spread(tokens, edges[i], edges[i + 1])
        for w in local:  # voiced time -> absolute time
            w["start"], w["end"] = round(to_abs(w["start"]), 3), round(to_abs(w["end"]), 3)
        lines.append(local)
    return lines, "approx"


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #


def _whisper_input(vocal: np.ndarray, sr: int) -> np.ndarray:
    mono = vocal if vocal.ndim == 1 else vocal.mean(axis=1)
    audio = to_sample_rate(mono.astype(np.float32), sr, WHISPER_SR)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio * (0.9 / peak) if peak > 0 else audio).astype(np.float32)


def _from_whisper_segments(segments) -> list[list[dict]]:
    out = []
    for seg in segments:
        words = seg["words"] if isinstance(seg, dict) else seg.words
        out.append([
            _word(w["word"] if isinstance(w, dict) else w.word,
                  w["start"] if isinstance(w, dict) else w.start,
                  w["end"] if isinstance(w, dict) else w.end)
            for w in (words or [])
        ])
    return out


def transcribe_mlx(audio16k: np.ndarray, language: str | None, model: str = MLX_MODEL):
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio16k,
        path_or_hf_repo=model,
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        verbose=None,
    )
    return _from_whisper_segments(result.get("segments", [])), result.get("language")


def transcribe_faster_whisper(audio16k: np.ndarray, language: str | None, model: str = FW_MODEL):
    from faster_whisper import WhisperModel

    fw = WhisperModel(model, device="cpu", compute_type="int8")
    segments, info = fw.transcribe(
        audio16k, language=language, word_timestamps=True,
        condition_on_previous_text=False, vad_filter=True,
    )
    return _from_whisper_segments(list(segments)), info.language


def transcribe_openrouter(audio16k: np.ndarray, language: str | None, model: str,
                          regions: list[tuple[float, float]]):
    import soundfile as sf

    from .. import openrouter

    buf = io.BytesIO()
    sf.write(buf, audio16k, WHISPER_SR, format="FLAC", subtype="PCM_16")
    resp = openrouter.transcribe(buf.getvalue(), "flac", model=model, language=language)
    lines, timing = normalize_remote(resp, regions, len(audio16k) / WHISPER_SR)
    return lines, resp.get("language") or language, timing


def transcribe(
    vocal: np.ndarray,
    sr: int,
    backend: str = "mlx",
    model: str | None = None,
    language: str | None = None,
) -> dict:
    """Transcribe a (separated) vocal signal into lyric lines with word times."""
    language = None if language in (None, "", "auto") else language
    regions = vocal_activity(vocal, sr)
    audio16k = _whisper_input(vocal, sr)
    timing = "word"

    if backend == "openrouter":
        if not model:
            raise ValueError("pick an OpenRouter transcription model")
        lines, lang, timing = transcribe_openrouter(audio16k, language, model, regions)
    else:
        try:
            import mlx_whisper  # noqa: F401

            backend, model = "mlx", model or MLX_MODEL
            segments, lang = transcribe_mlx(audio16k, language, model)
        except ImportError:
            try:
                import faster_whisper  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "no local speech-to-text: pip install mlx-whisper "
                    "(Apple Silicon) or faster-whisper"
                ) from exc
            backend, model = "faster-whisper", FW_MODEL
            segments, lang = transcribe_faster_whisper(audio16k, language, model)
        lines = split_lines(segments)

    lines = filter_hallucinations(lines, regions)
    return {
        "backend": backend,
        "model": model,
        "language": lang,
        "timing": timing if lines else "none",
        "lines": lines,
    }
