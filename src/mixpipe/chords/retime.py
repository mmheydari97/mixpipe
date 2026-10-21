"""Fix word positions for text that came without trustworthy timing.

Some OpenRouter transcription models return only text (or segment-level times),
so the words are spread evenly over the sung stretch and chords land on the wrong
ones. Two local fixes, tried in this order, work for text from any source:

  1. Whisper anchors. Local Whisper (the default backend, free and offline) is run
     on the vocal for its word timestamps. Its words are matched to the given
     text (difflib on normalised tokens); every word that matches takes Whisper's
     time, and the words in between are spread across the gap that is left
     (`align.reanchor`, the same machinery lyric edits use). Used when at least
     MIN_MATCH of the words match.
  2. Onset snapping. With no local Whisper, or too little in common with its
     output, every word's estimated start is moved to the nearest sung onset in
     the vocal stem: a dynamic programme picks one onset (or none) per word so that
     the total shift is small, strong onsets are preferred, and order is kept.

Either way the result is marked `timing = "aligned"` (Whisper anchors) or
`"snapped"` (onsets) instead of `"approx"`, and `transcript["retimed"]` says how.
"""

from __future__ import annotations

import difflib
from typing import Callable

import numpy as np

from .align import _norm, reanchor

MIN_MATCH = 0.35        # share of the words Whisper must agree on to use its anchors
SNAP_WINDOW_S = 0.35    # a word start may move this far to meet an onset
SNAP_MIN_GAP_S = 0.06   # two word starts closer than this are merged away
SHIFT_WEIGHT = 0.45     # cost of moving a word the whole window
STRENGTH_WEIGHT = 0.7   # cost of snapping to a feeble onset
NO_SNAP_COST = 0.55     # cost of leaving a word where it is
ONSET_HOP = 160         # at 16 kHz: 10 ms


def match_fraction(timed_lines: list[list[dict]] | list[dict], text: str) -> float:
    """Share of the words in `text` that also occur, in order, in the timed lines."""
    old = [w for ln in timed_lines for w in (ln["words"] if isinstance(ln, dict) else ln)]
    new = text.split()
    if not new or not old:
        return 0.0
    sm = difflib.SequenceMatcher(None, [_norm(w["text"]) for w in old],
                                 [_norm(t) for t in new], autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(new)


def whisper_lines(vocal: np.ndarray, sr: int, language: str | None) -> list[list[dict]] | None:
    """Timed lyric lines from a local Whisper, or None when none is installed."""
    from . import transcribe as tr

    lang = None if language in (None, "", "auto") else language
    audio = tr._whisper_input(vocal, sr)
    try:
        import mlx_whisper  # noqa: F401

        segments, _ = tr.transcribe_mlx(audio, lang)
    except ImportError:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return None
        segments, _ = tr.transcribe_faster_whisper(audio, lang)
    lines = tr.split_lines(segments)
    return tr.filter_hallucinations(lines, tr.vocal_activity(vocal, sr))


def _onsets(vocal: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(times in s, strengths 0-1) of sung onsets in the vocal stem."""
    import librosa

    from ..io_utils import to_sample_rate

    y = vocal if vocal.ndim == 1 else vocal.mean(axis=1)
    y = to_sample_rate(y.astype(np.float32), sr, 16_000)
    env = librosa.onset.onset_strength(y=y, sr=16_000, hop_length=ONSET_HOP)
    if env.size < 5 or env.max() <= 0:
        return np.zeros(0), np.zeros(0)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=8, post_avg=8,
                                   delta=0.08 * float(env.max()), wait=5)
    if len(peaks) == 0:
        return np.zeros(0), np.zeros(0)
    times = peaks * ONSET_HOP / 16_000.0
    strength = np.clip(env[peaks] / (np.percentile(env[peaks], 90) + 1e-9), 0.0, 1.0)
    return times, strength


def snap_to_onsets(lines: list[dict], onset_times: np.ndarray, onset_strength: np.ndarray) -> list[dict]:
    """Move word starts onto nearby onsets; ends follow. Returns new lines.

    One DP over all words in time order. For word i with estimated start s_i the
    options are "stay" (cost NO_SNAP_COST) or any onset o within SNAP_WINDOW_S of
    s_i (cost SHIFT_WEIGHT * |o - s_i| / window + STRENGTH_WEIGHT * (1 - strength)). A word must start at least
    SNAP_MIN_GAP_S after the one before it."""
    words = [w for ln in lines for w in ln["words"]]
    if not words or len(onset_times) == 0:
        return lines
    starts = np.array([w["start"] for w in words])
    cands: list[list[tuple[float, float]]] = []     # per word: (position, cost); first = stay
    for s in starts:
        lo, hi = np.searchsorted(onset_times, [s - SNAP_WINDOW_S, s + SNAP_WINDOW_S])
        opts = [(float(s), NO_SNAP_COST)]
        for k in range(lo, hi):
            opts.append((float(onset_times[k]),
                         SHIFT_WEIGHT * abs(onset_times[k] - s) / SNAP_WINDOW_S
                         + STRENGTH_WEIGHT * (1.0 - float(onset_strength[k]))))
        cands.append(opts)

    INF = float("inf")
    cost = [np.array([c for _, c in cands[0]])]
    back: list[np.ndarray] = [np.zeros(len(cands[0]), dtype=int)]
    for i in range(1, len(words)):
        cur = np.full(len(cands[i]), INF)
        arg = np.zeros(len(cands[i]), dtype=int)
        for a, (pos, c) in enumerate(cands[i]):
            for b, (ppos, _) in enumerate(cands[i - 1]):
                if pos - ppos >= SNAP_MIN_GAP_S - 1e-9 and cost[-1][b] + c < cur[a]:
                    cur[a], arg[a] = cost[-1][b] + c, b
        if not np.isfinite(cur).any():            # nothing fits after the previous word: stay put
            cur[0], arg[0] = cost[-1].min() + NO_SNAP_COST, int(np.argmin(cost[-1]))
        cost.append(cur)
        back.append(arg)
    choice = int(np.argmin(cost[-1]))
    chosen = [0] * len(words)
    for i in range(len(words) - 1, -1, -1):
        chosen[i] = choice
        choice = int(back[i][choice])
    new_starts = np.array([cands[i][chosen[i]][0] for i in range(len(words))])

    out, k = [], 0
    for ln in lines:
        moved = []
        for w in ln["words"]:
            nxt = new_starts[k + 1] if k + 1 < len(words) else np.inf
            start = float(new_starts[k])
            # a word pushed later keeps its length; one pulled earlier keeps its end
            end = min(w["end"] + max(start - w["start"], 0.0), float(nxt) - 0.01)
            moved.append({**w, "start": round(start, 3), "end": round(max(end, start + 0.04), 3)})
            k += 1
        out.append({**ln, "words": moved})
    return out


def retime(transcript: dict, vocal: np.ndarray, sr: int,
           whisper: Callable | None = None) -> dict:
    """Return the transcript with better word timing, or unchanged if nothing
    helps. `transcript["lines"]` is the pipeline's [[word, ...], ...]. `whisper`
    (vocal, sr, language) -> timed lines | None can be injected for tests."""
    lines = transcript.get("lines") or []
    if not lines:
        return transcript
    text = "\n".join(" ".join(w["text"] for w in ln) for ln in lines)
    timed = None
    try:
        timed = (whisper or whisper_lines)(vocal, sr, transcript.get("language"))
    except Exception:  # noqa: BLE001 - a missing/broken local Whisper just means "try onsets"
        timed = None

    if timed and match_fraction(timed, text) >= MIN_MATCH:
        # one line of the text per original line, so the lines come back as they were
        shaped = reanchor([{"words": ln, "gap_before": False} for ln in timed], text)
        new_lines = [ln["words"] for ln in shaped]
        if [len(ln) for ln in new_lines] == [len(ln) for ln in lines]:
            return {**transcript, "lines": new_lines, "timing": "aligned", "retimed": "whisper"}

    times, strength = _onsets(vocal, sr)
    if len(times):
        snapped = snap_to_onsets([{"words": ln, "gap_before": False} for ln in lines],
                                 times, strength)
        return {**transcript, "lines": [ln["words"] for ln in snapped],
                "timing": "snapped", "retimed": "onsets"}
    return transcript
