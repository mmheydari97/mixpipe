"""The guitar tab: the recorded guitar, note by note, as staff notation and
tablature (guitar_transcribe.py finds the notes).

Stored in sheet["guitar"]:

    {"notes": [note], "lock": None | {"centre", "apan", "pitch", "as"}}

A note is an editable note (notes.py) plus what was measured on it: "amp",
"pan", "centre", "bend", "vibrato", and the user's pins "fingering" (index into
its string/fret positions) and "part" ("lead" | "rhythm" | None = automatic).
Pitches are stored as they sound on the recording; the song's transpose moves
what is shown, and the capo moves the frets.

What the view works out, every time, from the notes:

- **Strums vs picked notes.** Notes that start together (within STRUM_WINDOW)
  form a group. Four or more notes over a wide range, struck as a sweep (or
  five and more at once), are a strum: shown as the chord it plays. Two or
  three notes together, arpeggios and single lines are picked notes and are
  written out note by note.
- **Lead vs rhythm.** In each stretch of a couple of bars the picked notes are
  split in two groups by where they sit in the stereo picture (a centred lead
  vs guitars panned left and right) and how high they are. When the groups
  really differ in the stereo picture, there are two guitars: the lead is the
  more centred, higher group, or the one nearest the locked part ("this is
  the solo"). Otherwise one guitar plays and all its picked notes are shown.
- **Fingering.** A Viterbi pass over the shown notes picks one string/fret
  per note: notes struck together on different strings within a four-fret
  stretch, and the hand moving as little as it can.
- **Rhythm.** Onsets are snapped to the song's beat grid, in sixteenths or
  eighth-note triplets per beat (sixteenths only in 6/8), and written as notes,
  rests and ties per bar for the staff and the tab.
- **Solo sections.** Stretches with chords but no sung words (two bars or more).
"""

from __future__ import annotations

import bisect
import itertools
import math

from . import notes as notes_mod
from .align import chord_label, grid_for
from .notes import REST, nearest_octave, note_midi, parse_note
from .theory import (GUITAR_STANDARD, INSTRUMENTS, key_names, key_short, note_label,
                     transpose_label)

GUITAR_VERSION = 3
FRETS = 22
STRUM_WINDOW = 0.05     # notes starting this close together are struck together
CHORD_SPAN = 4          # frets one hand covers (not counting open strings)
WINDOW_BARS = 2         # lead/rhythm is decided per this many bars
WINDOW_S = 4.0          # ... or per this many seconds when there is no bar grid
TWO_GUITARS = 0.12      # stereo distance between the groups that means two guitars
MIN_GROUP = 3           # notes each group needs before it counts as a part
TICKS = 12              # per beat: sixteenths (3) and triplet eighths (4) both fit
PARTS = ("lead", "rhythm", "both")
SOLO_S = 8.0            # an instrumental stretch this long is listed as a solo
PREFIX = "g"


# --------------------------------------------------------------------------- #
# the instrument: tuning, capo, transpose
# --------------------------------------------------------------------------- #


def tuning_of(sheet: dict) -> list[int]:
    """Open-string pitches, string 6 first, as the song's guitar is tuned
    (the chord finder's instrument: standard, half or whole step down)."""
    inst = sheet.get("instrument")
    offset = INSTRUMENTS[inst]["tuning"] if inst in INSTRUMENTS and inst.startswith("guitar") else 0
    return [m + offset for m in GUITAR_STANDARD]


def played(note: dict, sheet: dict) -> int | None:
    """The pitch the tab plays: as recorded, plus the song's transpose."""
    m = note_midi(note)
    return None if m is None else m + int(sheet.get("transpose", 0))


def positions(midi: int | None, tuning: list[int], capo: int = 0) -> list[dict]:
    """Every place to fret `midi`: [{"string" (1 = top), "fret" (above the capo)}],
    top string first."""
    if midi is None:
        return []
    out = []
    for i, open_ in enumerate(tuning):
        fret = midi - open_ - capo
        if 0 <= fret <= FRETS - capo:
            out.append({"string": 6 - i, "fret": fret})
    return sorted(out, key=lambda p: p["string"])


# --------------------------------------------------------------------------- #
# strums, parts
# --------------------------------------------------------------------------- #


def onset_groups(notes: list[dict]) -> list[list[dict]]:
    """Notes struck together: each group starts with its earliest note and takes
    every note starting within STRUM_WINDOW of it."""
    groups: list[list[dict]] = []
    for n in sorted(notes, key=lambda n: (n["start"], n.get("midi", 0))):
        if groups and n["start"] - groups[-1][0]["start"] <= STRUM_WINDOW:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def is_strum(group: list[dict]) -> bool:
    """Four or more notes over a wide range, struck as a sweep across the strings
    (or five or more at once). Fewer are picked: double stops, triads, pinches."""
    if len(group) < 4:
        return False
    pitches = [n["_midi"] for n in group]
    if max(pitches) - min(pitches) < 10:
        return False
    if len(group) >= 5:
        return True
    starts = [n["start"] for n in group]
    if max(starts) - min(starts) < 0.012:
        return False                     # all at once: a pinch, fingers not a pick
    order = sorted(range(len(group)), key=lambda i: starts[i])
    by_time = [pitches[i] for i in order]
    ups = sum(b > a for a, b in zip(by_time, by_time[1:]))
    downs = sum(b < a for a, b in zip(by_time, by_time[1:]))
    return max(ups, downs) >= len(group) - 2


def _features(n: dict) -> tuple[float, float, float]:
    return (n.get("centre", 1.0), abs(n.get("pan", 0.0)), n["_midi"] / 24.0)


def _two_means(pts: list[tuple[float, float, float]]) -> list[int]:
    """Two groups (labels 0/1) by k-means, started from the most and least
    centred points."""
    score = [p[0] - p[1] for p in pts]
    c = [pts[score.index(max(score))], pts[score.index(min(score))]]
    labels = [0] * len(pts)
    for _ in range(12):
        labels = [0 if math.dist(p, c[0]) <= math.dist(p, c[1]) else 1 for p in pts]
        for k in (0, 1):
            mine = [p for p, l in zip(pts, labels) if l == k]
            if mine:
                c[k] = tuple(sum(v) / len(mine) for v in zip(*mine))
    return labels


def _profile(group: list[dict]) -> dict:
    f = [_features(n) for n in group]
    return {"centre": sum(x[0] for x in f) / len(f), "apan": sum(x[1] for x in f) / len(f),
            "pitch": sum(x[2] for x in f) / len(f)}


def _distance(a: dict, b: dict) -> float:
    return math.dist((a["centre"], a["apan"], a["pitch"]), (b["centre"], b["apan"], b["pitch"]))


def windows(sheet: dict, end: float) -> list[tuple[float, float]]:
    """The stretches lead/rhythm is decided in."""
    grid = grid_for(sheet)
    if grid is None:
        n = max(1, math.ceil(end / WINDOW_S))
        return [(k * WINDOW_S, (k + 1) * WINDOW_S) for k in range(n)]
    out, k = [], 0
    start = min(0.0, grid.bar_start(0))
    while start < end:
        k += WINDOW_BARS
        stop = grid.bar_start(k)
        out.append((start, stop if stop > start else start + WINDOW_S))
        start = out[-1][1]
    return out


def assign_parts(sheet: dict, notes: list[dict]) -> dict:
    """Decide strum / lead / rhythm for the given (pitched) notes.

    Returns {"part": {id: "lead" | "rhythm"}, "strum": {id: group index},
    "strums": [{"start", "end", "ids", "chord"}], "windows": [{"start", "end", "two"}]}."""
    groups = onset_groups(notes)
    strum_of, strums = {}, []
    for g in groups:
        if is_strum(g):
            idx = len(strums)
            strums.append({"start": g[0]["start"], "end": max(n["end"] for n in g),
                           "ids": [n["id"] for n in g], "chord": chord_at(sheet, g[0]["start"])})
            for n in g:
                strum_of[n["id"]] = idx
    lock = (sheet.get("guitar") or {}).get("lock")
    part: dict[str, str] = {}
    info = []
    picked = sorted((n for n in notes if n["id"] not in strum_of), key=lambda n: n["start"])
    starts = [n["start"] for n in picked]
    end = max((n["end"] for n in notes), default=0.0)
    for a, b in windows(sheet, end + 0.01):
        here = picked[bisect.bisect_left(starts, a):bisect.bisect_left(starts, b)]
        two = False
        if len(here) >= 2 * MIN_GROUP:
            labels = _two_means([_features(n) for n in here])
            g0 = [n for n, l in zip(here, labels) if l == 0]
            g1 = [n for n, l in zip(here, labels) if l == 1]
            if len(g0) >= MIN_GROUP and len(g1) >= MIN_GROUP:
                p0, p1 = _profile(g0), _profile(g1)
                two = math.dist((p0["centre"], p0["apan"]), (p1["centre"], p1["apan"])) >= TWO_GUITARS
        if two:
            if lock:
                lead_is_0 = (_distance(p0, lock) <= _distance(p1, lock)) == (lock.get("as", "lead") == "lead")
            else:
                lead_is_0 = (p0["centre"] - p0["apan"] + p0["pitch"] / 2
                             >= p1["centre"] - p1["apan"] + p1["pitch"] / 2)
            for n in g0:
                part[n["id"]] = "lead" if lead_is_0 else "rhythm"
            for n in g1:
                part[n["id"]] = "rhythm" if lead_is_0 else "lead"
        else:
            for n in here:
                part[n["id"]] = "lead"
        info.append({"start": round(a, 3), "end": round(b, 3), "two": two})
    for n in notes:
        if n["id"] in strum_of:
            part[n["id"]] = "rhythm"
        if n.get("part") in ("lead", "rhythm"):
            part[n["id"]] = n["part"]             # the user's pin wins
    return {"part": part, "strum": strum_of, "strums": strums, "windows": info}


def lock_part(sheet: dict, note_id: str, as_part: str = "lead") -> dict:
    """"This is the solo": remember the stereo position and register of the
    note's group, so every stretch with two guitars follows that guitar.
    `as_part` "rhythm" marks the note's guitar as the rhythm part instead."""
    if as_part not in ("lead", "rhythm"):
        raise ValueError("as must be lead or rhythm")
    g = _guitar(sheet)
    notes = _pitched(sheet)
    note = next((n for n in notes if n["id"] == note_id), None)
    if note is None:
        raise ValueError("no such note")
    g["lock"] = None
    parts = assign_parts(sheet, notes)
    mine = parts["part"].get(note_id)
    near = [n for n in notes if abs(n["start"] - note["start"]) <= WINDOW_S
            and parts["part"].get(n["id"]) == mine and n["id"] not in parts["strum"]]
    g["lock"] = {**_profile(near or [note]), "as": as_part, "note": note_id}
    return g["lock"]


# --------------------------------------------------------------------------- #
# fingering
# --------------------------------------------------------------------------- #


def _chord_options(pitches: list[int], tuning: list[int], capo: int,
                   pinned: dict[int, dict]) -> list[tuple[float, int | None, list[dict]]]:
    """Ways to fret notes struck together: one string each, within CHORD_SPAN
    frets. [(cost, hand position or None when all open, [position per note])]."""
    per = []
    for i, m in enumerate(pitches):
        opts = positions(m, tuning, capo)
        if i in pinned and pinned[i] in opts:
            opts = [pinned[i]]
        per.append(opts or [None])
    out = []
    for combo in itertools.islice(itertools.product(*per), 4000):
        placed = [p for p in combo if p is not None]
        strings = [p["string"] for p in placed]
        if len(set(strings)) != len(strings):
            continue
        fretted = [p["fret"] for p in placed if p["fret"] > 0]
        span = max(fretted) - min(fretted) if fretted else 0
        if span > CHORD_SPAN:
            continue
        hand = min(fretted) if fretted else None
        cost = 0.25 * span + 0.02 * sum(fretted) + 2.0 * (len(combo) - len(placed))
        cost += 0.15 * sum(1 for p in placed if p["fret"] == 0 and hand is not None and hand > 4)
        out.append((cost, hand, list(combo)))
    out.sort(key=lambda o: o[0])
    return out[:40] or [(10.0, None, [None] * len(pitches))]


def _move(a: int | None, b: int | None) -> float:
    if a is None or b is None:
        return 0.0
    d = abs(a - b)
    return 0.0 if d <= 2 else 0.6 * (d - 2) + 0.4


def fingerings(events: list[list[dict]], sheet: dict) -> dict[str, dict | None]:
    """{note id: {"string", "fret"} or None (out of reach)} for notes grouped by
    onset (`events`), moving the hand as little as possible (Viterbi)."""
    tuning, capo = tuning_of(sheet), int(sheet.get("capo", 0))
    layers = []
    for ev in events:
        pitches = [n["_play"] for n in ev]
        pinned = {}
        for i, n in enumerate(ev):
            opts = positions(n["_play"], tuning, capo)
            if n.get("fingering") is not None and 0 <= n["fingering"] < len(opts):
                pinned[i] = opts[n["fingering"]]
        layers.append(_chord_options(pitches, tuning, capo, pinned))
    out: dict[str, dict | None] = {}
    if not layers:
        return out
    cost = [o[0] for o in layers[0]]
    back = []
    for prev, cur in zip(layers, layers[1:]):
        new, ptr = [], []
        for c, hand, _ in cur:
            best = min(range(len(prev)), key=lambda j: cost[j] + _move(prev[j][1], hand))
            new.append(cost[best] + _move(prev[best][1], hand) + c)
            ptr.append(best)
        cost, back = new, back + [ptr]
    k = min(range(len(cost)), key=cost.__getitem__)
    choice = [k]
    for ptr in reversed(back):
        k = ptr[k]
        choice.append(k)
    choice.reverse()
    for ev, layer, k in zip(events, layers, choice):
        for n, pos in zip(ev, layer[k][2]):
            out[n["id"]] = pos
    return out


# --------------------------------------------------------------------------- #
# rhythm: notes on the beat grid, written per bar
# --------------------------------------------------------------------------- #


def _beat_value(sheet: dict) -> int:
    """The note value of one tracked beat: 8 in 6/8, else 4."""
    return 8 if (sheet.get("meter") or {}).get("beats_per_bar") == 6 else 4


def _subdivisions(beat_value: int) -> tuple[int, ...]:
    return (2,) if beat_value == 8 else (4, 3)


# (code, dots, ticks) with a quarter note = TICKS; in 6/8 (an eighth-note beat) x2
_VALUES = [("w", 0, 48), ("h", 1, 36), ("h", 0, 24), ("q", 1, 18), ("q", 0, 12),
           ("8", 1, 9), ("8", 0, 6), ("16", 0, 3)]
_TRIPLETS = [("q", 0, 8), ("8", 0, 4)]     # inside a triplet beat (3 in the time of 2)


def _pieces(length: int, at: int, triplet: bool, scale: int) -> list[tuple[str, int, int, bool]]:
    """Note values for `length` ticks starting `at` ticks into a beat:
    [(code, dots, ticks, in a triplet)]. Whole beats from a beat line use the
    plain values; a piece inside a beat never crosses its end."""
    values = [(c, d, t * scale) for c, d, t in _VALUES] + ([("32", 0, 3)] if scale == 2 else [])
    out = []
    while length > 0:
        on_beat = at % TICKS == 0
        if triplet and not (on_beat and length >= TICKS):
            room = min(length, TICKS - at % TICKS)
            table = [(c, d, t, True) for c, d, t in _TRIPLETS]
        else:
            room = length if on_beat else min(length, TICKS - at % TICKS)
            table = [(c, d, t, False) for c, d, t in values]
        piece = next((v for v in table if v[2] <= room), None)
        if piece is None:                  # an odd remainder: the smallest value
            piece = (table[-1][0], 0, room, table[-1][3])
        out.append(piece)
        length -= piece[2]
        at += piece[2]
    return out


class _Clock:
    """Song seconds <-> beat ticks, over the sheet's beat grid."""

    def __init__(self, sheet: dict):
        self.grid = grid_for(sheet)
        bt = list(sheet.get("beat_times") or [])
        self.period = sheet.get("beat_period") or 0.5
        self.bt = bt

    def beat_time(self, i: int) -> float:
        return self.grid.beat_time(i)

    def locate(self, t: float) -> tuple[int, float]:
        """(beat index, fraction 0..1 into it)."""
        bt = self.bt
        if t < bt[0]:
            i = -int(math.ceil((bt[0] - t) / self.period))
        elif t >= bt[-1]:
            i = len(bt) - 1 + int((t - bt[-1]) // self.period)
        else:
            i = bisect.bisect_right(bt, t) - 1
        a, b = self.beat_time(i), self.beat_time(i + 1)
        return i, min(max((t - a) / (b - a), 0.0), 1.0)


def quantize(events: list[dict], sheet: dict) -> list[dict]:
    """Snap each event's start and end to the grid: adds "tick" and "end_tick"
    (absolute: beat index * TICKS + position) and the beats' subdivision."""
    clock = _Clock(sheet)
    subs = _subdivisions(_beat_value(sheet))
    by_beat: dict[int, list[float]] = {}
    for ev in events:
        i, f = clock.locate(ev["start"])
        by_beat.setdefault(i, []).append(f)
    choice = {}
    for i, fracs in by_beat.items():
        def err(s):
            return sum(abs(f * s - round(f * s)) / s for f in fracs)
        best = min(subs, key=lambda s: (err(s) - (0.02 * len(fracs) if s == subs[0] else 0), -s))
        choice[i] = best
    out = []
    for ev in events:
        i, f = clock.locate(ev["start"])
        s = choice[i]
        tick = i * TICKS + round(f * s) * (TICKS // s)
        j, g = clock.locate(ev["end"])
        se = choice.get(j, subs[0])
        end_tick = j * TICKS + round(g * se) * (TICKS // se)
        out.append({**ev, "tick": tick, "end_tick": max(end_tick, tick + TICKS // se)})
    out.sort(key=lambda e: e["tick"])
    merged: list[dict] = []
    for ev in out:                       # two events snapped to one tick are one chord
        if merged and merged[-1]["tick"] == ev["tick"]:
            m = merged[-1]
            have = {x["midi"] for x in m["notes"]}
            m["notes"] = sorted(m["notes"] + [x for x in ev["notes"] if x["midi"] not in have],
                                key=lambda x: x["midi"])
            m["end_tick"] = max(m["end_tick"], ev["end_tick"])
        else:
            merged.append(ev)
    return [{**ev, "triplet": choice.get(ev["tick"] // TICKS) == 3} for ev in merged]


def measures(events: list[dict], sheet: dict, names: list[str]) -> list[dict]:
    """Bars of notes and rests for the staff and the tab, from quantize()'s
    events. Each bar: {"index", "start" (s), "beats", "chords": [{"tick", "label"}],
    "items": [item]}, an item being {"dur" ("q", "8"...), "dots", "rest", "tie" (to
    the next item), "triplet", "beat" (index, for grouping triplets), "t" (s),
    "keys", "notes": [{"id", "midi", "key", "string", "fret", "tech"}], "strum"}."""
    grid = grid_for(sheet)
    if grid is None or not events:
        return []
    clock = _Clock(sheet)
    scale = 2 if _beat_value(sheet) == 8 else 1
    first_bar = grid.bar_of(clock.beat_time(events[0]["tick"] // TICKS) + 0.001)
    last_bar = grid.bar_of(clock.beat_time(max(e["end_tick"] for e in events) // TICKS) + 0.001)
    bars = []
    for k in range(first_bar, last_bar + 1):
        start = grid.bar_start(k)
        b0 = grid.beat_index(start + 0.001)
        _, m = grid.beat_in_bar(b0)
        bars.append({"index": k, "start": round(start, 3), "beats": m,
                     "t0": b0 * TICKS, "t1": (b0 + m) * TICKS, "items": [], "chords": []})

    def bar_for(tick):
        return next((b for b in bars if b["t0"] <= tick < b["t1"]), None)

    def time_of(tick):
        i, f = divmod(tick, TICKS)
        a = clock.beat_time(i)
        return round(a + f / TICKS * (clock.beat_time(i + 1) - a), 3)

    def emit(a, b, ev):
        """Items for ticks a..b (a note event, or a rest when ev is None)."""
        made = []
        while a < b:
            bar = bar_for(a)
            if bar is None:
                break
            stop = min(b, bar["t1"])
            pos = a
            while pos < stop:              # beat by beat, so triplets stay in their beat
                beat_end = (pos // TICKS + 1) * TICKS
                chunk_end = stop if pos % TICKS == 0 and not (ev and ev["triplet"]) else min(stop, beat_end)
                for code, dots, ticks, trip in _pieces(chunk_end - pos, pos % TICKS,
                                                       bool(ev and ev["triplet"]), scale):
                    item = {"dur": code, "dots": dots, "rest": ev is None, "triplet": trip,
                            "beat": pos // TICKS, "t": time_of(pos), "tie": False}
                    if ev is not None:
                        item.update(notes=ev["notes"], strum=ev.get("strum") if not made else None,
                                    keys=[n["key"] for n in ev["notes"]])
                    bar["items"].append(item)
                    made.append(item)
                    pos += ticks
            a = stop
        if ev is not None:
            for item in made[:-1]:
                item["tie"] = True         # one note written as several values
        return made

    cursor = bars[0]["t0"]
    for i, ev in enumerate(events):
        if ev["tick"] > cursor:
            emit(cursor, ev["tick"], None)
        nxt = events[i + 1]["tick"] if i + 1 < len(events) else None
        stop = ev["end_tick"] if nxt is None else min(max(ev["end_tick"], ev["tick"] + 1), nxt)
        if nxt is not None and nxt - stop < TICKS // 4:
            stop = nxt                     # a tiny gap is legato, not a rest
        emit(ev["tick"], stop, ev)
        cursor = max(cursor, stop)
    if cursor < bars[-1]["t1"]:
        emit(cursor, bars[-1]["t1"], None)
    for c in sheet.get("chords", []):
        label = chord_label(c)
        if label == "N.C.":
            continue
        i, f = clock.locate(c["start"])
        tick = i * TICKS + round(f * 4) * 3
        bar = bar_for(tick)
        if bar is not None:
            bar["chords"].append({"tick": tick - bar["t0"], "t": time_of(tick),
                                  "label": _display(label, sheet)})
    for bar in bars:
        del bar["t0"], bar["t1"]
    return bars


# --------------------------------------------------------------------------- #
# sections, chords, labels
# --------------------------------------------------------------------------- #


def chord_at(sheet: dict, t: float) -> str | None:
    best = None
    for c in sheet.get("chords", []):
        if c["start"] <= t + 0.1 and chord_label(c) != "N.C.":
            best = c
    return _display(chord_label(best), sheet) if best else None


def _display(label: str, sheet: dict) -> str:
    shift = int(sheet.get("transpose", 0))
    return label if shift == 0 else transpose_label(label, shift, key_names(sheet.get("key"), shift))


def sections(sheet: dict, end: float) -> list[dict]:
    """Instrumental stretches (no sung words for two bars or more, and at least
    3 s) and the sung ones between them: [{"start", "end", "kind"}]."""
    words = sorted((w["start"], w["end"]) for line in sheet.get("lines", [])
                   for w in line.get("words", []))
    gap = max(3.0, 2 * len_bar(sheet))
    out, cursor = [], 0.0
    for a, b in words:
        if a - cursor >= gap:
            if out and out[-1]["kind"] == "sung":
                out[-1]["end"] = round(cursor, 3)
            out.append({"start": round(cursor, 3), "end": round(a, 3), "kind": "instrumental"})
            out.append({"start": round(a, 3), "end": round(b, 3), "kind": "sung"})
        elif out and out[-1]["kind"] == "sung":
            out[-1]["end"] = round(b, 3)
        else:
            out.append({"start": round(cursor, 3), "end": round(b, 3), "kind": "sung"})
        cursor = max(cursor, b)
    if end - cursor >= gap or not words:
        out.append({"start": round(cursor, 3), "end": round(end, 3), "kind": "instrumental"})
    elif out:
        out[-1]["end"] = round(max(out[-1]["end"], end), 3)
    return out


def len_bar(sheet: dict) -> float:
    meter = sheet.get("meter") or {}
    return float(sheet.get("beat_period") or 0.5) * int(meter.get("beats_per_bar") or 4)


def vexflow_key(midi: int, names: list[str]) -> str:
    """Written pitch for the treble-8vb staff: an octave above the sound, as
    "f#/4"."""
    written = midi + 12
    return f"{names[written % 12].lower()}/{written // 12 - 1}"


def _tech(n: dict) -> list[str]:
    out = []
    bend = n.get("bend") or 0
    if bend >= 1.5:
        out.append("full")
    elif bend >= 0.75:
        out.append("½")
    if n.get("vibrato"):
        out.append("~")
    return out


# --------------------------------------------------------------------------- #
# store, edits, view
# --------------------------------------------------------------------------- #


def _guitar(sheet: dict) -> dict:
    g = sheet.get("guitar")
    if not g or g.get("notes") is None:
        raise ValueError("this song has no guitar tab yet")
    return g


def from_transcription(found: list[dict], old: dict | None = None) -> dict:
    """sheet["guitar"] from guitar_transcribe.transcribe's notes; keeps the old lock."""
    notes = []
    for i, n in enumerate(found):
        notes.append({"id": f"{PREFIX}{i}", "start": n["start"], "end": n["end"],
                      "candidates": _candidates(n["midi"], n.get("amp", 1.0), n.get("alt")),
                      "selected": 0,
                      "custom": None, "fingering": None, "part": None,
                      **{k: n[k] for k in ("amp", "pan", "centre", "bend", "vibrato") if k in n}})
    return {"notes": notes, "lock": (old or {}).get("lock")}


def _candidates(midi: int, amp: float, alt: int | None = None) -> list[dict]:
    """The heard pitch, the other pitch of a bend/trill, then the octave errors a
    transcriber makes most."""
    options = [(midi, amp)] + ([(alt, amp * 0.5)] if alt is not None else []) + \
        [(midi - 12, amp * 0.3), (midi + 12, amp * 0.2)]
    return [{"label": note_label(m), "midi": m, "conf": round(c, 3)}
            for m, c in options if 28 <= m <= 100]


def _pitched(sheet: dict) -> list[dict]:
    """The notes that aren't rests, with their recorded ("_midi") and played
    ("_play") pitch."""
    out = []
    for n in _guitar(sheet)["notes"]:
        m = note_midi(n)
        if m is not None:
            out.append({**n, "_midi": m, "_play": m + int(sheet.get("transpose", 0))})
    return out


def note_from_display(text: str, sheet: dict, note: dict) -> str | None:
    """A note typed as the tab shows it ("F#", "A4", "rest"), stored at recorded
    pitch. No octave: the one nearest the note's current pitch."""
    text = (text or "").strip()
    if not text:
        return None
    if text.lower() in (REST, "x", "×", "-"):
        return REST
    parsed = parse_note(text)
    if parsed is None:
        raise ValueError(f"{text!r} isn't a note name (try E, F#, Bb3)")
    shift = int(sheet.get("transpose", 0))
    value, octave = parsed
    if octave is None:
        ref = note_midi(note)
        ref = (ref if ref is not None else 64) + shift
        value = nearest_octave(value, ref)
    return note_label(value - shift)


def edit_note(sheet: dict, note_id: str, selected: int | None = None, custom: str | None = None,
              fingering: int | None = None, auto_fingering: bool = False,
              part: str | None = None, auto_part: bool = False) -> None:
    """One note's pitch choice, typed pitch ("" back to heard, "rest" removes
    it), pinned string/fret and pinned part. Raises ValueError."""
    note = notes_mod.find(_guitar(sheet)["notes"], note_id)
    if selected is not None:
        if not 0 <= selected < len(note["candidates"]):
            raise ValueError("selected out of range")
        note["selected"], note["custom"], note["fingering"] = selected, None, None
    if custom is not None:
        note["custom"] = note_from_display(custom, sheet, note)
        note["fingering"] = None
    if auto_fingering:
        note["fingering"] = None
    elif fingering is not None:
        opts = positions(played(note, sheet), tuning_of(sheet), int(sheet.get("capo", 0)))
        if not 0 <= fingering < len(opts):
            raise ValueError("fingering out of range")
        note["fingering"] = fingering
    if auto_part:
        note["part"] = None
    elif part is not None:
        if part not in ("lead", "rhythm"):
            raise ValueError("part must be lead or rhythm")
        note["part"] = part


def split_note(sheet: dict, note_id: str, at: float | None = None) -> str:
    return notes_mod.split(_guitar(sheet)["notes"], note_id, at, reset=("fingering",), prefix=PREFIX)


def merge_notes(sheet: dict, note_ids: list[str]) -> str:
    g = _guitar(sheet)
    g["notes"], merged = notes_mod.merge(g["notes"], note_ids, neighbours=False)
    return merged


def add_note(sheet: dict, start: float, end: float, text: str, part: str | None = None) -> str:
    """A note the transcriber missed; guitar notes may overlap (chords)."""
    probe = {"candidates": [], "custom": None, "selected": 0}
    label = note_from_display(text, sheet, probe)
    if label == REST or label is None:
        raise ValueError(f"{text!r} isn't a note name (try E, F#, Bb3)")
    return notes_mod.add(_guitar(sheet)["notes"], start, end, label, monophonic=False,
                         prefix=PREFIX, part=part if part in ("lead", "rhythm") else None,
                         amp=1.0, pan=0.0, centre=1.0, bend=0.0, vibrato=False)


def delete_note(sheet: dict, note_id: str) -> None:
    notes_mod.find(_guitar(sheet)["notes"], note_id)["custom"] = REST


def guitar_view(sheet: dict, part: str = "lead") -> dict | None:
    """What the guitar tab draws for one part ("lead", "rhythm" or "both")."""
    if (sheet.get("guitar") or {}).get("notes") is None:
        return None
    if part not in PARTS:
        raise ValueError("part must be lead, rhythm or both")
    notes = _pitched(sheet)
    transpose = int(sheet.get("transpose", 0))
    names = key_names(sheet.get("key"), transpose)
    parts = assign_parts(sheet, notes)
    shown = [n for n in notes if part == "both" or parts["part"].get(n["id"]) == part]
    shown_ids = {n["id"] for n in shown}
    groups = onset_groups(shown)
    fing = fingerings(groups, sheet)
    tuning, capo = tuning_of(sheet), int(sheet.get("capo", 0))
    out_notes = []
    for n in sorted(notes, key=lambda n: (n["start"], n["_midi"])):
        pos = fing.get(n["id"])
        opts = positions(n["_play"], tuning, capo)
        out_notes.append({
            "id": n["id"], "start": n["start"], "end": n["end"], "midi": n["_play"],
            "label": note_label(n["_play"], names), "part": parts["part"].get(n["id"]),
            "pinned_part": n.get("part"), "strum": parts["strum"].get(n["id"]),
            "shown": n["id"] in shown_ids,
            "string": pos["string"] if pos else None, "fret": pos["fret"] if pos else None,
            "positions": opts, "fingering": n.get("fingering"),
            "candidates": [{"label": note_label(c["midi"] + transpose, names), "midi": c["midi"] + transpose,
                            "conf": c["conf"]} for c in n["candidates"]],
            "selected": n.get("selected", 0), "custom": n.get("custom"),
            "tech": _tech(n), "amp": n.get("amp"), "centre": n.get("centre"), "pan": n.get("pan")})
    by_id = {n["id"]: n for n in out_notes}
    events = []
    for g in groups:
        strum = parts["strum"].get(g[0]["id"])
        uniq = {}
        for n in sorted(g, key=lambda n: -n.get("amp", 0)):
            uniq.setdefault(n["_play"], n)          # one pitch once, the strongest
        evn = [{"id": n["id"], "midi": n["_play"], "key": vexflow_key(n["_play"], names),
                "string": by_id[n["id"]]["string"], "fret": by_id[n["id"]]["fret"],
                "tech": by_id[n["id"]]["tech"]}
               for n in sorted(uniq.values(), key=lambda n: n["_play"])]
        events.append({"start": g[0]["start"], "end": max(n["end"] for n in g), "notes": evn,
                       "strum": parts["strums"][strum]["chord"] if strum is not None else None})
    end = max((n["end"] for n in notes), default=0.0)
    bars = measures(quantize(events, sheet), sheet, names) if grid_for(sheet) else []
    meter = sheet.get("meter") or {}
    beats = int(meter.get("beats_per_bar") or 4)
    secs = sections(sheet, max(end, float(sheet.get("duration") or 0)))
    return {
        "part": part,
        "notes": [x for x in out_notes if x["shown"]],
        "hidden": sum(1 for x in out_notes if not x["shown"]),
        "strums": parts["strums"],
        "windows": parts["windows"],
        "two_guitars": any(w["two"] for w in parts["windows"]),
        "lock": (sheet.get("guitar") or {}).get("lock"),
        "sections": secs,
        "solos": [x for x in secs if x["kind"] == "instrumental" and x["end"] - x["start"] >= SOLO_S],
        "measures": bars,
        "time_signature": meter.get("time_signature") or f"{beats}/4",
        "key": key_short(sheet.get("key"), transpose),
        "tuning": [note_label(m) for m in tuning],
        "capo": capo,
        "_events": events,
    }


def to_tab_text(sheet: dict, part: str = "lead", width: int = 72) -> str:
    """Plain-text tab of one part, a column per note group, rows of `width`."""
    view = guitar_view(sheet, part)
    if view is None:
        return ""
    names = ["e", "B", "G", "D", "A", "E"]
    head = [f"{sheet.get('title', '')} - guitar ({part})", f"Tuning: {' '.join(view['tuning'])}"
            + (f", capo {view['capo']}" if view["capo"] else ""), ""]
    cols = []
    for ev in view["_events"]:
        col = {s: "" for s in range(1, 7)}
        for n in ev["notes"]:
            if n["string"]:
                col[n["string"]] = str(n["fret"]) + ("b" if "½" in n["tech"] or "full" in n["tech"] else "")
        w = max(len(v) for v in col.values()) or 1
        cols.append(({s: v.ljust(w, "-") if v else "-" * w for s, v in col.items()}, ev["strum"]))
    rows, line, size = [], [], 0
    for col, strum in cols:
        w = len(col[1]) + 1
        if size + w > width and line:
            rows.append(line)
            line, size = [], 0
        line.append((col, strum))
        size += w
    if line:
        rows.append(line)
    out = head
    for row in rows:
        marks = "".join((strum or "").ljust(len(c[1]) + 1)[: len(c[1]) + 1] for c, strum in row)
        if marks.strip():
            out.append("  " + marks)
        for s in range(1, 7):
            out.append(f"{names[s - 1]}|" + "".join(c[s] + "-" for c, _ in row) + "|")
        out.append("")
    return "\n".join(out)
