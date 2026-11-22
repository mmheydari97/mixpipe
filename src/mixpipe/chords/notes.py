"""Note lists the user can edit: the harmonica tab's melody and the guitar tab's
notes share these. A note is

    {"id", "start", "end", "candidates": [{"label", "midi", "conf"}], "selected",
     "custom": a label typed by the user, or REST, or None, ...instrument fields}

Each instrument decides what a typed name means (its octave and transpose) and
what else a note carries; splitting, merging, adding and reading the pitch are
the same for both.
"""

from __future__ import annotations

import re

from .align import chord_label

REST = "rest"          # a note's custom label when the user removed it
MIN_NOTE_S = 0.05      # edits may not make a note shorter than this
OVERLAP_S = 0.02       # a new note may overlap a neighbour by this much

_NOTE = re.compile(r"^\s*([A-Ga-g])([#b♯♭]?)\s*(-?\d)?\s*$")
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}


def pretty(name: str) -> str:
    """"Bb" -> "B♭", "F#" -> "F♯"."""
    return name.replace("#", "♯").replace("b", "♭") if len(name) > 1 else name


def parse_note(text: str) -> tuple[int, int | None] | None:
    """"Bb4" -> (70, 4); "f#" -> (6, None): the MIDI number when an octave is
    given, else the pitch class. None if it isn't a note name."""
    m = _NOTE.match(text or "")
    if not m:
        return None
    semis = _NATURAL[m.group(1).upper()] + _ACC[m.group(2)]
    if m.group(3) is None:
        return semis % 12, None
    octave = int(m.group(3))
    return 12 * (octave + 1) + semis, octave


def note_midi(note: dict) -> int | None:
    """Pitch of a note as the user currently sees it (before any instrument
    shift); None for a rest."""
    label = chord_label(note)
    if label == REST:
        return None
    if note.get("custom"):
        parsed = parse_note(label)
        return parsed[0] if parsed and parsed[1] is not None else None
    cands = note.get("candidates") or []
    if not cands:
        return None
    return cands[min(max(note.get("selected", 0) or 0, 0), len(cands) - 1)]["midi"]


def nearest_octave(pc: int, ref: int) -> int:
    """The MIDI note with pitch class `pc` nearest `ref` (lower on a tie)."""
    return min((pc + 12 * k for k in range(0, 11)), key=lambda m: (abs(m - ref), m))


def new_id(notes: list[dict], prefix: str = "n") -> str:
    pat = re.compile(rf"{re.escape(prefix)}(\d+)")
    used = [int(m.group(1)) for n in notes if (m := pat.fullmatch(n.get("id", "")))]
    return f"{prefix}{max(used, default=-1) + 1}"


def find(notes: list[dict], note_id: str) -> dict:
    note = next((n for n in notes if n["id"] == note_id), None)
    if note is None:
        raise ValueError("no such note")
    return note


def split(notes: list[dict], note_id: str, at: float | None = None,
          reset: tuple[str, ...] = ("fingering",), prefix: str = "n") -> str:
    """Cut a note in two at time `at` (default: its middle), e.g. a repeated
    pitch the tracker held as one note. Both halves keep the pitch choice; the
    `reset` fields go back to automatic. Returns the id of the second half.
    Raises ValueError."""
    note = find(notes, note_id)
    at = 0.5 * (note["start"] + note["end"]) if at is None else float(at)
    if not note["start"] + MIN_NOTE_S <= at <= note["end"] - MIN_NOTE_S:
        raise ValueError(f"a note can only be split inside itself, leaving at least "
                         f"{MIN_NOTE_S * 1000:.0f} ms on each side")
    second = {**{k: (list(v) if isinstance(v, list) else v) for k, v in note.items()},
              "id": new_id(notes, prefix), "start": round(at, 3)}
    second["candidates"] = [dict(c) for c in note.get("candidates", [])]
    for k in reset:
        second[k] = None
    note["end"] = round(at, 3)
    note["fingering"] = None
    notes.append(second)
    notes.sort(key=lambda n: n["start"])
    return second["id"]


def merge(notes: list[dict], note_ids: list[str], reset: tuple[str, ...] = ("fingering",),
          neighbours: bool = True) -> tuple[list[dict], str]:
    """Join notes into one (a note the tracker split in two, or a trill to
    smooth out). The longest of them decides the pitch; with `neighbours` they
    must follow each other. Returns (the new list, the merged note's id).
    Raises ValueError."""
    ordered = sorted(notes, key=lambda n: n["start"])
    chosen = [find(ordered, i) for i in dict.fromkeys(note_ids)]
    if len(chosen) < 2:
        raise ValueError("pick at least two notes to merge")
    chosen.sort(key=lambda n: n["start"])
    if neighbours:
        idx = [ordered.index(n) for n in chosen]
        if idx != list(range(idx[0], idx[0] + len(idx))):
            raise ValueError("only neighbouring notes can be merged")
    keeper = max(chosen, key=lambda n: n["end"] - n["start"])
    first = chosen[0]
    merged = {**keeper, "id": first["id"], "start": first["start"],
              "end": max(n["end"] for n in chosen)}
    for k in reset:
        merged[k] = None
    drop = {n["id"] for n in chosen}
    out = [n for n in notes if n["id"] not in drop] + [merged]
    out.sort(key=lambda n: n["start"])
    return out, merged["id"]


def add(notes: list[dict], start: float, end: float, label: str, monophonic: bool = True,
        prefix: str = "n", **fields) -> str:
    """Add a note the tracker missed; `label` is a full note name at the
    stored pitch ("C5"). A monophonic line may not get overlapping notes.
    Returns its id. Raises ValueError."""
    start, end = round(float(start), 3), round(float(end), 3)
    if end - start < MIN_NOTE_S:
        raise ValueError(f"a note must last at least {MIN_NOTE_S * 1000:.0f} ms")
    if start < 0:
        raise ValueError("a note can't start before the song does")
    parsed = parse_note(label) if label and label != REST else None
    if parsed is None or parsed[1] is None:
        raise ValueError(f"{label!r} isn't a note name (try C, F#, Bb4)")
    if monophonic:
        for n in notes:
            if note_midi(n) is not None and start < n["end"] - OVERLAP_S and end > n["start"] + OVERLAP_S:
                raise ValueError(f"that overlaps the note at {n['start']:.1f} s")
    note = {"id": new_id(notes, prefix), "start": start, "end": end, "selected": 0, "custom": None,
            "fingering": None, "source": "user", **fields,
            "candidates": [{"label": label, "midi": parsed[0], "conf": 1.0}]}
    notes.append(note)
    notes.sort(key=lambda n: n["start"])
    return note["id"]
