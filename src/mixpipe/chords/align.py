"""Put chords on words, and keep word timing alive through lyric edits.

Chords are anchored to *time*, words carry times, and the sheet is rebuilt from
both on demand. A lyric edit therefore can never lose a chord: it only changes
which word the chord lands on.
"""

from __future__ import annotations

import bisect
import difflib
import math
import re


def chord_label(chord: dict) -> str:
    """The label the user currently sees: custom text, or the selected candidate."""
    if chord.get("custom"):
        return chord["custom"]
    cands = chord.get("candidates") or []
    if not cands:
        return "N.C."
    idx = chord.get("selected", 0) or 0
    return cands[min(max(idx, 0), len(cands) - 1)]["label"]


def _visible(chords: list[dict]) -> list[dict]:
    return sorted((c for c in chords if chord_label(c) != "N.C."), key=lambda c: c["start"])


class BarGrid:
    """Bars laid over the beat grid.

    A plain grid has `beats_per_bar` beats per bar with the bar starting at beat
    index `downbeat`. With `sections` ([{"beat", "beats_per_bar"}], the beat of each
    section's first downbeat) the meter can change along the way: each section
    runs in its own meter until the next one starts (a last short bar is allowed),
    and bars are numbered straight through. Beats before the first or after the
    last tracked beat are extrapolated at `beat_period`."""

    def __init__(self, beat_times: list[float], beats_per_bar: int = 4,
                 downbeat: int = 0, beat_period: float = 0.5, sections: list[dict] | None = None):
        self.bt = list(beat_times) or [0.0]
        self.period = beat_period if beat_period > 0 else 0.5
        if sections:
            secs = sorted(({"beat": int(sec["beat"]), "m": max(1, int(sec["beats_per_bar"]))}
                           for sec in sections), key=lambda sec: sec["beat"])
        else:
            m = max(1, int(beats_per_bar))
            secs = [{"beat": int(downbeat) % m, "m": m}]
        self.secs = secs
        self.m, self.p = secs[0]["m"], secs[0]["beat"] % secs[0]["m"]
        self.bars_before = [0]
        for a, b in zip(secs, secs[1:]):
            self.bars_before.append(self.bars_before[-1] + max(1, -(-(b["beat"] - a["beat"]) // a["m"])))

    def beat_index(self, t: float) -> int:
        bt = self.bt
        if t < bt[0] - 0.08:
            return -int(math.ceil((bt[0] - 0.08 - t) / self.period))
        if t > bt[-1] + self.period:
            return len(bt) - 1 + int((t - bt[-1] + 0.08) // self.period)
        return bisect.bisect_right(bt, t + 0.08) - 1

    def _section_of_beat(self, beat: int) -> int:
        return max(0, bisect.bisect_right([sec["beat"] for sec in self.secs], beat) - 1)

    def beat_in_bar(self, beat: int) -> tuple[int, int]:
        """(position of beat index `beat` in its bar, 0 = downbeat; the bar's beats)."""
        sec = self.secs[self._section_of_beat(beat)]
        return (beat - sec["beat"]) % sec["m"], sec["m"]

    def bar_of(self, t: float) -> int:
        beat = self.beat_index(t)
        k = self._section_of_beat(beat)
        return self.bars_before[k] + (beat - self.secs[k]["beat"]) // self.secs[k]["m"]

    def chord_bar(self, t: float) -> int:
        """The bar a chord that starts at `t` is written in. A change that lands
        within a half beat before a bar line anticipates it (a push), so it is
        written in the bar it leads into."""
        k = self.bar_of(t)
        gap = self.bar_start(k + 1) - t
        return k + 1 if 0.12 * self.period < gap <= 0.6 * self.period else k

    def beat_time(self, idx: int) -> float:
        if idx < 0:
            return self.bt[0] + idx * self.period
        if idx >= len(self.bt):
            return self.bt[-1] + (idx - len(self.bt) + 1) * self.period
        return self.bt[idx]

    def bar_start(self, k: int) -> float:
        s = max(0, bisect.bisect_right(self.bars_before, k) - 1)
        return self.beat_time(self.secs[s]["beat"] + (k - self.bars_before[s]) * self.secs[s]["m"])

    def bars(self) -> list[dict]:
        """Every bar from the first downbeat to the last tracked beat:
        [{"start": the downbeat's time, "beats": the bar's beats}]. A metronome
        accents these and counts in with the bar's length."""
        out, k = [], 0
        while True:
            start = self.bar_start(k)
            if start > self.bt[-1] + 1e-6:
                return out
            s = max(0, bisect.bisect_right(self.bars_before, k) - 1)
            out.append({"start": round(start, 4), "beats": self.secs[s]["m"]})
            k += 1

    def section_changes(self) -> list[dict]:
        """Where the meter changes after the first section: [{"bar", "time", "beats_per_bar"}]."""
        return [{"bar": self.bars_before[i], "time": self.bar_start(self.bars_before[i]),
                 "beats_per_bar": sec["m"]} for i, sec in enumerate(self.secs) if i > 0]


def grid_for(sheet: dict) -> BarGrid | None:
    """The sheet's bar grid, or None for sheets made before bars existed."""
    meter = sheet.get("meter")
    if not meter or not sheet.get("beat_times"):
        return None
    sections = meter.get("sections") or None
    if sections and len(sections) < 2:
        sections = None
    return BarGrid(sheet["beat_times"], meter.get("beats_per_bar", 4),
                   meter.get("downbeat", 0), sheet.get("beat_period", 0.5), sections=sections)


def syllable_index(spans: list[dict], t: float, tol: float) -> int:
    """Which syllable a chord at time `t` belongs to: the last one that starts no
    later than t + tol (the same rule that puts a chord on a word)."""
    idx = 0
    for k, sp in enumerate(spans):
        if sp["start"] <= t + tol:
            idx = k
    return idx


def build_rows(lines: list[dict], chords: list[dict], beat_period: float = 0.5,
               grid: BarGrid | None = None, bars_per_row: int = 4,
               chords_per_row: int = 8, syllabify=None) -> list[dict]:
    """Interleave lyric rows and instrumental rows in time order.

    A chord lands on the last word that starts no later than (chord start + tol).
    A chord change shortly before a line (within ~1 beat) lands on its first
    word. Anything else between lines becomes an instrumental row.

    `syllabify(word) -> [{"text", "start", "end", "c"}]` splits a word that carries
    a chord into syllables: such a word gets "syl" (the spans) and "at" (chord id
    -> index of the syllable it sits over), so a change in the middle of a held
    word shows over the right syllable. Words of one syllable get neither.

    With a bar grid, instrumental rows are cut into bars (`bars_per_row` per row;
    each bar lists the changes inside it, and `hold` = the chord still sounding
    on its downbeat when no change lands there),
    and lyric words carry `bar` when a new bar begins while they are sung.
    """
    tol = max(0.15, 0.35 * beat_period)
    lead = max(0.3, 1.0 * beat_period)
    lines = [ln for ln in lines if ln.get("words")]
    visible = _visible(chords)
    by_id = {c["id"]: c for c in visible}

    on_word: dict[tuple[int, int], list[str]] = {}
    gaps: dict[int, list[str]] = {}  # gap index k = after line k-1 (0 = intro)

    for c in visible:
        t = c["start"]
        k = -1
        for i, ln in enumerate(lines):
            if ln["words"][0]["start"] - lead <= t:
                k = i
            else:
                break
        if k >= 0 and t <= lines[k]["words"][-1]["end"] + tol:
            words = lines[k]["words"]
            j = 0
            for idx, w in enumerate(words):
                if w["start"] <= t + tol:
                    j = idx
            on_word.setdefault((k, j), []).append(c["id"])
        else:
            gaps.setdefault(k + 1, []).append(c["id"])

    def sounding_at(t: float) -> str | None:
        for c in visible:
            if c["start"] <= t + 1e-3 < c["end"]:
                return c["id"]
        return None

    rows: list[dict] = []

    def add_gap(k: int, until: float | None) -> None:
        ids = gaps.get(k, [])
        if not ids:
            return
        if grid is None:
            for a in range(0, len(ids), chords_per_row):
                rows.append({"type": "instrumental", "chords": ids[a:a + chords_per_row]})
            return
        last = by_id[ids[-1]]
        end = last["end"] if until is None else min(last["end"], until)
        k0 = grid.chord_bar(by_id[ids[0]]["start"])
        k1 = max(k0, grid.bar_of(max(end - 1e-3, by_id[ids[-1]]["start"])),
                 grid.chord_bar(by_id[ids[-1]]["start"]))
        bars = []
        for bar in range(k0, k1 + 1):
            inside = [cid for cid in ids if grid.chord_bar(by_id[cid]["start"]) == bar]
            # the chord still sounding on the downbeat, unless a change lands on it
            # (or arrives just ahead of it: a push)
            t0 = grid.bar_start(bar)
            lead = t0 - by_id[inside[0]]["start"] if inside else 0.0
            on_beat = inside and (abs(lead) <= 0.12 or 0 < lead <= 0.6 * grid.period)
            hold = None if on_beat else sounding_at(t0)
            bars.append({"bar": bar, "chords": inside, "hold": hold})
        for a in range(0, len(bars), bars_per_row):
            chunk = bars[a:a + bars_per_row]
            rows.append({"type": "instrumental",
                         "chords": [cid for b in chunk for cid in b["chords"]],
                         "bars": chunk})

    def next_line_start(i: int) -> float | None:
        return lines[i]["words"][0]["start"] - lead if i < len(lines) else None

    add_gap(0, next_line_start(0))
    for i, ln in enumerate(lines):
        words = []
        for j, w in enumerate(ln["words"]):
            word = {"i": j, "text": w["text"], "start": w["start"], "end": w["end"],
                    "chords": on_word.get((i, j), [])}
            if syllabify is not None and word["chords"]:
                spans = syllabify(w)
                if len(spans) > 1:
                    word["syl"] = spans
                    word["at"] = {cid: syllable_index(spans, by_id[cid]["start"], tol)
                                  for cid in word["chords"]}
            if grid is not None:
                nxt = ln["words"][j + 1]["start"] if j + 1 < len(ln["words"]) else w["end"] + tol
                a, b = grid.bar_of(w["start"] - tol), grid.bar_of(nxt - tol)
                if b > a:  # a bar line falls while this word is sung
                    word["bar"] = b
            words.append(word)
        carry = None if words[0]["chords"] else sounding_at(words[0]["start"])
        rows.append({"type": "lyric", "line": i, "gap_before": bool(ln.get("gap_before")),
                     "carry": carry, "words": words})
        add_gap(i + 1, next_line_start(i + 1))
    return rows


# --------------------------------------------------------------------------- #
# lyric editing
# --------------------------------------------------------------------------- #

_PUNCT = re.compile(r"[^\w']+", re.UNICODE)


def _norm(token: str) -> str:
    n = _PUNCT.sub("", token.lower())
    return n or token.lower()


def lines_to_text(lines: list[dict]) -> str:
    out = []
    for ln in lines:
        if ln.get("gap_before") and out:
            out.append("")
        out.append(" ".join(w["text"] for w in ln["words"]))
    return "\n".join(out)


def reanchor(lines: list[dict], new_text: str) -> list[dict]:
    """Apply an edited lyric text to timed lines.

    Words that survive the edit keep their timestamps (matched with difflib on
    normalised tokens). Replaced words share the time span of what they replace;
    inserted words are squeezed between their neighbours. Line breaks follow the
    new text; a blank line marks a stanza break (gap_before on the next line).
    """
    old = [w for ln in lines for w in ln["words"]]

    new_lines_tokens: list[list[str]] = []
    gap_flags: list[bool] = []
    pending_gap = False
    for raw in new_text.replace("\r\n", "\n").split("\n"):
        tokens = raw.split()
        if not tokens:
            pending_gap = bool(new_lines_tokens)
            continue
        new_lines_tokens.append(tokens)
        gap_flags.append(pending_gap)
        pending_gap = False

    new_flat = [t for ln in new_lines_tokens for t in ln]
    times: list[tuple[float, float, bool] | None] = [None] * len(new_flat)

    sm = difflib.SequenceMatcher(None, [_norm(w["text"]) for w in old],
                                 [_norm(t) for t in new_flat], autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            for k in range(b1 - b0):
                w = old[a0 + k]
                times[b0 + k] = (w["start"], w["end"], bool(w.get("approx")))
        elif op == "replace":
            start, end = old[a0]["start"], old[a1 - 1]["end"]
            _fill(times, new_flat, b0, b1, start, end)
        # inserts are filled below once their neighbours are known; deletes drop

    # inserted words: interpolate between the nearest timed neighbours
    i = 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        prev_end = times[i - 1][1] if i > 0 else (old[0]["start"] if old else 0.0)
        next_start = times[j][0] if j < len(times) else (old[-1]["end"] if old else prev_end)
        _fill(times, new_flat, i, j, prev_end, max(prev_end, next_start))
        i = j

    out, k = [], 0
    for tokens, gap in zip(new_lines_tokens, gap_flags):
        words = []
        for tok in tokens:
            s, e, approx = times[k]
            word = {"text": tok, "start": round(s, 3), "end": round(e, 3)}
            if approx:
                word["approx"] = True
            words.append(word)
            k += 1
        out.append({"words": words, "gap_before": gap})
    return out


def _fill(times, tokens, b0, b1, start, end) -> None:
    n = b1 - b0
    if n <= 0:
        return
    weights = [max(1, len(tokens[b])) for b in range(b0, b1)]
    total = float(sum(weights))
    t = start
    for k, b in enumerate(range(b0, b1)):
        dur = (end - start) * weights[k] / total
        times[b] = (t, t + dur, True)
        t += dur


def word_diff(old_text: str, new_text: str) -> list[list[str]]:
    """[[op, text]] with op in equal/insert/delete, for showing a proposed edit."""
    a = [tok for line in old_text.split("\n") for tok in line.split() + ["\n"]]
    b = [tok for line in new_text.split("\n") for tok in line.split() + ["\n"]]
    out: list[list[str]] = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            out.append(["equal", " ".join(a[a0:a1])])
        else:
            if a1 > a0:
                out.append(["delete", " ".join(a[a0:a1])])
            if b1 > b0:
                out.append(["insert", " ".join(b[b0:b1])])
    return out
