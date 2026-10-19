"""The harmonica tab: a 12-hole C chromatic harmonica (Hohner Chromonica 48 /
270 style) in solo tuning, and the melody laid out for it.

Solo tuning repeats one 4-hole block per octave, starting at middle C:

    hole   1  2  3  4 | 5  6  7  8 | 9 10 11 12
    blow   C  E  G  C | C  E  G  C | C  E  G  C
    draw   D  F  A  B | D  F  A  B | D  F  A  B

Pressing the button raises a hole by a semitone, so the range is C4 to C#7
(hole 12 blow + button). Some notes have several fingerings (F = 2 draw or 2
blow + button; C5 = 4 blow, 5 blow or 4 draw + button). Without a manual choice
the tab uses no button when it can, then the hole nearest the previous note.

Tab notation: "3↑" blow, "3↓" draw, "●" button in ("3↓●").

Melody notes are stored at sung (concert) pitch in sheet["melody"]; the
song's transpose setting and an octave shift (automatic fit + the user's
offset) give the pitch played on the harmonica. The sung notes decide the
octave; the rough notes guessed from the instruments (often a bass or guitar
register) get their own automatic fit, plus the same user offset.
"""

from __future__ import annotations

import bisect
import re

from .align import build_rows, chord_label, grid_for
from .theory import NEUTRAL, key_names, key_label

HOLES = 12
LOWEST = 60            # C4, hole 1 blow
BLOW = (0, 4, 7, 12)   # semitones above the block's C, per position in the block
DRAW = (2, 5, 9, 11)
REST = "rest"          # a note's custom label when the user removed it
MAX_OCTAVE = 3
NOTES_PER_ROW = 12     # notes per instrumental row when there are no bars

_NOTE = re.compile(r"^\s*([A-Ga-g])([#b♯♭]?)\s*(-?\d)?\s*$")
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}


# --------------------------------------------------------------------------- #
# the instrument
# --------------------------------------------------------------------------- #


def tab_text(hole: int, breath: str, button: bool) -> str:
    return f"{hole}{'↑' if breath == 'blow' else '↓'}{'●' if button else ''}"


def _all_positions() -> list[dict]:
    out = []
    for hole in range(1, HOLES + 1):
        block, pos = divmod(hole - 1, 4)
        for breath, table in (("blow", BLOW), ("draw", DRAW)):
            for button in (False, True):
                out.append({"hole": hole, "breath": breath, "button": button,
                            "midi": LOWEST + 12 * block + table[pos] + int(button),
                            "tab": tab_text(hole, breath, button)})
    return out


_POSITIONS = _all_positions()
HIGHEST = max(p["midi"] for p in _POSITIONS)   # C#7


def fingerings(midi: int | None) -> list[dict]:
    """Every way to play `midi`, no-button ones first, then by hole.
    [] when the harmonica can't play it."""
    if midi is None:
        return []
    found = [{k: p[k] for k in ("hole", "breath", "button", "tab")}
             for p in _POSITIONS if p["midi"] == midi]
    return sorted(found, key=lambda f: (f["button"], f["hole"]))


def choose_fingering(options: list[dict], prev_hole: int | None) -> int:
    """Index of the default fingering: no button when possible, then the hole
    nearest the previous note (the lowest hole when there is none)."""
    if not options:
        return -1
    pool = [i for i, f in enumerate(options) if not f["button"]] or list(range(len(options)))
    if prev_hole is None:
        return pool[0]
    return min(pool, key=lambda i: (abs(options[i]["hole"] - prev_hole), options[i]["hole"]))


def in_range(midi: int | None) -> bool:
    return midi is not None and LOWEST <= midi <= HIGHEST


def fit_octave(notes: list[tuple[int, float]]) -> int:
    """Octave shift (-3..3) that puts the most playing time in range.
    `notes` is [(midi, duration_s)]. Ties go to the smallest shift."""
    if not notes:
        return 0
    best, best_score = 0, -1.0
    for k in sorted(range(-MAX_OCTAVE, MAX_OCTAVE + 1), key=abs):
        score = sum(d for m, d in notes if in_range(m + 12 * k))
        if score > best_score + 1e-9:
            best, best_score = k, score
    return best


# --------------------------------------------------------------------------- #
# note names
# --------------------------------------------------------------------------- #


def note_label(midi: int, names: list[str] = NEUTRAL) -> str:
    """72 -> "C5" (scientific pitch, octave 4 starts at middle C)."""
    return f"{names[midi % 12]}{midi // 12 - 1}"


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
    """Sung pitch of a note as the user currently sees it; None for a rest."""
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


# --------------------------------------------------------------------------- #
# sheet integration
# --------------------------------------------------------------------------- #


def _melody(sheet: dict) -> dict | None:
    mel = sheet.get("melody")
    return mel if mel and mel.get("notes") is not None else None


def octave_shifts(sheet: dict) -> dict:
    """Octave shifts for the current notes and transpose: "auto" (fit of the
    sung notes), "total" (auto + the user's offset) and "instr" (the same for
    notes guessed from the instruments)."""
    mel = _melody(sheet) or {"notes": []}
    transpose = int(sheet.get("transpose", 0))
    timed = {"vocal": [], "instr": []}
    for n in mel["notes"]:
        if (m := note_midi(n)) is not None:
            timed["instr" if n.get("source") == "instr" else "vocal"].append(
                (m + transpose, n["end"] - n["start"]))
    instr_auto = fit_octave(timed["instr"])
    auto = fit_octave(timed["vocal"]) if timed["vocal"] else instr_auto
    user = int(mel.get("octave", 0))
    return {"auto": auto, "total": auto + user, "instr": instr_auto + user}


def played_shift(sheet: dict, note: dict | None = None) -> int:
    """Semitones from sung pitch to the pitch played on the harmonica."""
    shifts = octave_shifts(sheet)
    octaves = shifts["instr"] if note and note.get("source") == "instr" else shifts["total"]
    return int(sheet.get("transpose", 0)) + 12 * octaves


def note_from_display(text: str, sheet: dict, note: dict) -> str | None:
    """A note typed as shown on the tab ("Bb", "F#5", "rest"), stored back at
    sung pitch. Without an octave, the one nearest the note's current pitch is
    used. None reverts to the detected candidates. Raises ValueError."""
    text = (text or "").strip()
    if not text:
        return None
    if text.lower() in (REST, "x", "×", "-"):
        return REST
    parsed = parse_note(text)
    if parsed is None:
        raise ValueError(f"{text!r} isn't a note name (try C, F#, Bb4)")
    shift = played_shift(sheet, note)
    value, octave = parsed
    if octave is None:
        ref = note_midi(note)
        if ref is None:
            ref = note["candidates"][0]["midi"] if note.get("candidates") else LOWEST + 12
        ref += shift
        value = min((value + 12 * k for k in range(0, 11)), key=lambda m: (abs(m - ref), m))
    return note_label(value - shift)


def set_fingering(note: dict, index: int | None, sheet: dict) -> None:
    """Pin one of the note's fingerings (None = automatic). Raises ValueError."""
    if index is None:
        note["fingering"] = None
        return
    midi = note_midi(note)
    options = fingerings(None if midi is None else midi + played_shift(sheet, note))
    if not 0 <= index < len(options):
        raise ValueError("fingering out of range")
    note["fingering"] = index


def _key_at(sheet: dict):
    """t -> the key sounding at time t (a chord's own key after a modulation)."""
    chords = sorted(sheet.get("chords", []), key=lambda c: c["start"])
    starts = [c["start"] for c in chords]

    def at(t: float) -> str | None:
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and chords[i].get("key"):
            return chords[i]["key"]
        return sheet.get("key")
    return at


def harmonica_view(sheet: dict) -> dict | None:
    """The melody as shown on the harmonica tab, or None before it was found.

    Each note gains "midi" (played pitch, None for a rest), "sung" (pitch in
    the recording), "display" (note name), "name" (with octave), "fingerings", "fingering_index", "tab",
    "in_range", "rest", and per-candidate "display" / "tab". "rows" places the
    notes on the lyrics like the chord sheet does.
    """
    mel = _melody(sheet)
    if mel is None:
        return None
    transpose = int(sheet.get("transpose", 0))
    shifts = octave_shifts(sheet)
    key_at = _key_at(sheet)

    def describe(midi: int | None, names: list[str]) -> dict:
        if midi is None:
            return {"midi": None, "display": "×", "name": "rest", "tab": None, "in_range": True}
        options = fingerings(midi)
        return {"midi": midi, "display": pretty(names[midi % 12]),
                "name": pretty(note_label(midi, names)),
                "tab": options[0]["tab"] if options else None, "in_range": bool(options)}

    notes, prev_hole = [], None
    playable = total_time = 0.0
    for n in sorted(mel["notes"], key=lambda n: n["start"]):
        names = key_names(key_at(n["start"]), transpose)
        shift = transpose + 12 * shifts["instr" if n.get("source") == "instr" else "total"]
        sung = note_midi(n)
        midi = None if sung is None else sung + shift
        out = {**n, **describe(midi, names), "rest": sung is None, "sung": sung}
        options = fingerings(midi)
        pinned = n.get("fingering")
        idx = pinned if isinstance(pinned, int) and 0 <= pinned < len(options) \
            else choose_fingering(options, prev_hole)
        out["fingerings"] = options
        out["fingering_index"] = idx
        if idx >= 0:
            out["tab"] = options[idx]["tab"]
            prev_hole = options[idx]["hole"]
        out["candidates"] = [{**c, **{k: v for k, v in describe(c["midi"] + shift, names).items()
                                      if k in ("display", "name", "tab", "in_range")},
                              "played_midi": c["midi"] + shift}
                             for c in n.get("candidates", [])]
        if sung is not None:
            total_time += n["end"] - n["start"]
            playable += (n["end"] - n["start"]) if out["in_range"] else 0.0
        notes.append(out)

    return {
        "notes": notes,
        "rows": build_rows(sheet.get("lines", []), notes, sheet.get("beat_period", 0.5),
                           grid=grid_for(sheet), chords_per_row=NOTES_PER_ROW),
        "octave": shifts["total"],
        "auto_octave": shifts["auto"],
        "instr_octave": shifts["instr"],
        "user_octave": int(mel.get("octave", 0)),
        "in_range": round(playable / total_time, 3) if total_time else 1.0,
        "key_label": key_label(sheet.get("key"), transpose),
        "range": [note_label(LOWEST), note_label(HIGHEST)],
    }


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def _token(n: dict) -> tuple[str, str]:
    """(name line, tab line) text for one note."""
    if n["rest"]:
        return "", ""
    return n["display"], n["tab"] or "--"


def to_harp_text(sheet: dict) -> str:
    """Monospace harmonica tab: note names and tabs over each lyric line."""
    hv = harmonica_view(sheet)
    out = [sheet.get("title", "Untitled"),
           "C chromatic harmonica, 12 holes, solo tuning   ↑ blow   ↓ draw   ● button in"]
    if hv is None:
        out += ["", "(no melody found yet)"]
        return "\n".join(out) + "\n"
    meta = []
    if hv["key_label"]:
        meta.append(f"Key: {hv['key_label']}")
    if hv["octave"]:
        meta.append(f"Melody moved {abs(hv['octave'])} octave{'s' if abs(hv['octave']) > 1 else ''} "
                    f"{'up' if hv['octave'] > 0 else 'down'} to fit")
    if hv["in_range"] < 1.0:
        meta.append(f"{round(hv['in_range'] * 100)}% playable (-- = out of range)")
    if meta:
        out.append("   ".join(meta))
    out.append("")

    notes = {n["id"]: n for n in hv["notes"]}
    for row in hv["rows"]:
        if row["type"] == "instrumental":
            def cell(ids: list[str]) -> str:
                toks = [_token(notes[i]) for i in ids if not notes[i]["rest"]]
                return "  ".join(f"{a} {b}" for a, b in toks) or "-"
            if "bars" in row:
                out.append("| " + " | ".join(cell(b["chords"]) for b in row["bars"]) + " |")
            else:
                out.append(cell(row["chords"]))
            continue
        if row["gap_before"]:
            out.append("")
        names, tabs, lyric = "", "", ""
        for w in row["words"]:
            if lyric:
                lyric += " "
            width = max(len(names), len(tabs))
            if width:  # the previous word's notes ran past it: start this word after them
                lyric = lyric.ljust(width + 1)
            col = len(lyric)
            for cid in w["chords"]:
                a, b = _token(notes[cid])
                if not a:
                    continue
                names = names.ljust(col) + a
                tabs = tabs.ljust(col) + b
                col += max(len(a), len(b)) + 1
            lyric += w["text"]
        if names.strip():
            out.append(names.rstrip())
            out.append(tabs.rstrip())
        out.append(lyric)
    return "\n".join(out).rstrip() + "\n"
