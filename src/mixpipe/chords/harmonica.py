"""The harmonica tab: a melody laid out for a harmonica.

Two instruments, in any key:

  chromatic12  12-hole chromatic (Hohner Chromonica 48 / 270 style) in solo
               tuning. One 4-hole block repeats per octave, starting at middle C
               for a C harp:

                   hole   1  2  3  4 | 5  6  7  8 | 9 10 11 12
                   blow   C  E  G  C | C  E  G  C | C  E  G  C
                   draw   D  F  A  B | D  F  A  B | D  F  A  B

               The slide button raises a hole a semitone, so every semitone from
               C4 to C#7 is playable. Some notes have several fingerings (F = 2
               draw or 2 blow + button; C5 = 4 blow, 5 blow or 4 draw + button).

  diatonic10   10-hole Richter-tuned diatonic. Blow and draw give a major scale
               (a C harp: blow C E G C E G C E G C, draw D G B D F A B D F A);
               the missing notes are made by *bending* a draw note on holes 1-6
               and 10, or a blow note on holes 7-10 (tab: one prime per
               semitone, 3↓′ = half step, 3↓″ = whole step, 3↓‴ = 1½ steps).
               Overblows and overdraws are not modelled, so a few semitones
               (e.g. E♭ in the low octave) are not playable.

The harp's key moves the whole layout: G sits a fourth below C (G3), D♭ a
semitone above (D♭4), and so on, using the usual harp keys.

Without a manual choice the tab picks the fingering with no bend and no button
when it can, then the hole nearest the previous note.

Tab notation: "3↑" blow, "3↓" draw, "●" button in ("3↓●"), primes for bends.

Melody notes are stored at sung (concert) pitch in sheet["melody"]; the
song's transpose setting and an octave shift (automatic fit + the user's
offset) give the pitch played on the harmonica. The sung notes decide the
octave; the rough notes guessed from the instruments (often a bass or guitar
register) get their own automatic fit, plus the same user offset.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from .align import build_rows, chord_label, grid_for
from .syllables import make_syllabifier
from .theory import NEUTRAL, key_names, key_label, note_pc, parse_key

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

# ---- the Richter diatonic layout, semitones above C4 -------------------------
DIATONIC_BLOW = (0, 4, 7, 12, 16, 19, 24, 28, 31, 36)
DIATONIC_DRAW = (2, 7, 11, 14, 17, 21, 23, 26, 29, 33)
DRAW_BENDS = {1: 1, 2: 2, 3: 3, 4: 1, 5: 1, 6: 1, 10: 1}   # hole -> deepest bend (semitones)
BLOW_BENDS = {7: 1, 8: 1, 9: 1, 10: 2}
BEND_MARKS = {1: "′", 2: "″", 3: "‴"}

KINDS = {
    "chromatic12": "12-hole chromatic, solo tuning",
    "diatonic10": "10-hole diatonic (Richter)",
}
DEFAULT_KIND = "chromatic12"
HARP_KEYS = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# harp tonic relative to the song's tonic -> the usual position names
POSITIONS = {0: "1st (straight harp)", 7: "2nd (cross harp)", 2: "3rd (slant harp)",
             9: "4th", 4: "5th", 11: "6th", 5: "12th"}


def key_offset(pc: int) -> int:
    """Semitones a harp in key `pc` sits above a C harp: G, A♭, A, B♭ and B go
    down (G3...), D♭ to F up, F♯ up."""
    return (pc + 5) % 12 - 5


def tab_text(hole: int, breath: str, button: bool, bend: int = 0) -> str:
    return (f"{hole}{'↑' if breath == 'blow' else '↓'}"
            f"{BEND_MARKS.get(bend, '′' * bend) if bend else ''}{'●' if button else ''}")


class Instrument:
    """One harmonica: kind + key, and every way to sound each pitch on it."""

    def __init__(self, kind: str = DEFAULT_KIND, key: int | str = 0):
        if kind not in KINDS:
            raise ValueError(f"unknown harmonica type {kind!r} (try {', '.join(KINDS)})")
        pc = key if isinstance(key, int) else note_pc(str(key))
        if pc is None:
            raise ValueError(f"{key!r} isn't a harmonica key (try C, G, Bb, F#)")
        self.kind, self.pc = kind, pc % 12
        self.shift = key_offset(self.pc)
        self.positions = self._build()
        self._by_midi: dict[int, list[dict]] = {}
        for p in self.positions:
            self._by_midi.setdefault(p["midi"], []).append(p)
        for midi, found in self._by_midi.items():
            found.sort(key=lambda f: (f["bend"], f["button"], f["hole"], f["breath"]))
        self.lowest = min(self._by_midi)
        self.highest = max(self._by_midi)

    # ---- layout ----
    def _build(self) -> list[dict]:
        out = []
        if self.kind == "chromatic12":
            for hole in range(1, HOLES + 1):
                block, pos = divmod(hole - 1, 4)
                for breath, table in (("blow", BLOW), ("draw", DRAW)):
                    for button in (False, True):
                        out.append(self._pos(hole, breath, button, 0,
                                             LOWEST + 12 * block + table[pos] + int(button)))
        else:
            for hole in range(1, 11):
                blow, draw = DIATONIC_BLOW[hole - 1], DIATONIC_DRAW[hole - 1]
                out.append(self._pos(hole, "blow", False, 0, LOWEST + blow))
                out.append(self._pos(hole, "draw", False, 0, LOWEST + draw))
                for k in range(1, DRAW_BENDS.get(hole, 0) + 1):
                    out.append(self._pos(hole, "draw", False, k, LOWEST + draw - k))
                for k in range(1, BLOW_BENDS.get(hole, 0) + 1):
                    out.append(self._pos(hole, "blow", False, k, LOWEST + blow - k))
        return out

    def _pos(self, hole: int, breath: str, button: bool, bend: int, midi: int) -> dict:
        return {"hole": hole, "breath": breath, "button": button, "bend": bend,
                "midi": midi + self.shift, "tab": tab_text(hole, breath, button, bend)}

    # ---- queries ----
    @property
    def name(self) -> str:
        return f"{HARP_KEYS[self.pc]} {KINDS[self.kind]}"

    @property
    def short_name(self) -> str:
        return f"{HARP_KEYS[self.pc]} {'chromatic' if self.kind == 'chromatic12' else 'diatonic'}"

    def fingerings(self, midi: int | None) -> list[dict]:
        """Every way to play `midi`: unbent and no-button first, then by hole.
        [] when the harmonica can't play it."""
        if midi is None:
            return []
        return [{k: p[k] for k in ("hole", "breath", "button", "bend", "tab")}
                for p in self._by_midi.get(midi, [])]

    def playable(self, midi: int | None) -> bool:
        return midi is not None and midi in self._by_midi

    def to_dict(self) -> dict:
        return {"type": self.kind, "key": HARP_KEYS[self.pc]}


DEFAULT_INSTRUMENT = Instrument()
_CACHE: dict[tuple[str, int], Instrument] = {(DEFAULT_KIND, 0): DEFAULT_INSTRUMENT}


def get_instrument(kind: str = DEFAULT_KIND, key: int | str = 0) -> Instrument:
    pc = key if isinstance(key, int) else note_pc(str(key))
    if pc is None:
        raise ValueError(f"{key!r} isn't a harmonica key (try C, G, Bb, F#)")
    ident = (kind, pc % 12)
    if ident not in _CACHE:
        _CACHE[ident] = Instrument(kind, pc)
    return _CACHE[ident]


def instrument_of(sheet: dict) -> Instrument:
    """The harmonica a sheet is set to (12-hole C chromatic by default)."""
    mel = sheet.get("melody") or {}
    spec = mel.get("instrument") or {}
    try:
        return get_instrument(spec.get("type", DEFAULT_KIND), spec.get("key", "C"))
    except ValueError:
        return DEFAULT_INSTRUMENT


# Module-level conveniences for the default (C chromatic) harmonica.
_POSITIONS = DEFAULT_INSTRUMENT.positions
HIGHEST = DEFAULT_INSTRUMENT.highest           # C#7


def fingerings(midi: int | None, instrument: Instrument | None = None) -> list[dict]:
    """Every way to play `midi` (on the C chromatic unless told otherwise)."""
    return (instrument or DEFAULT_INSTRUMENT).fingerings(midi)


def choose_fingering(options: list[dict], prev_hole: int | None) -> int:
    """Index of the default fingering: no bend and no button when possible, then
    the hole nearest the previous note (the lowest hole when there is none)."""
    if not options:
        return -1
    cost = lambda f: (f.get("bend", 0), f["button"])  # noqa: E731
    best = min(cost(f) for f in options)
    pool = [i for i, f in enumerate(options) if cost(f) == best]
    if prev_hole is None:
        return pool[0]
    return min(pool, key=lambda i: (abs(options[i]["hole"] - prev_hole), options[i]["hole"]))


def in_range(midi: int | None, instrument: Instrument | None = None) -> bool:
    """Can the harmonica play this pitch at all?"""
    return (instrument or DEFAULT_INSTRUMENT).playable(midi)


def fit_octave(notes: list[tuple[int, float]], instrument: Instrument | None = None) -> int:
    """Octave shift (-3..3) that puts the most playing time on notes the
    harmonica can play. `notes` is [(midi, duration_s)]. Ties go to the smallest
    shift."""
    if not notes:
        return 0
    inst = instrument or DEFAULT_INSTRUMENT
    best, best_score = 0, -1.0
    for k in sorted(range(-MAX_OCTAVE, MAX_OCTAVE + 1), key=abs):
        score = sum(d for m, d in notes if inst.playable(m + 12 * k))
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
    inst = instrument_of(sheet)
    timed = {"vocal": [], "instr": []}
    for n in mel["notes"]:
        if (m := note_midi(n)) is not None:
            timed["instr" if n.get("source") == "instr" else "vocal"].append(
                (m + transpose, n["end"] - n["start"]))
    instr_auto = fit_octave(timed["instr"], inst)
    auto = fit_octave(timed["vocal"], inst) if timed["vocal"] else instr_auto
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
            ref = note["candidates"][0]["midi"] if note.get("candidates") else instrument_of(sheet).lowest + 12
        ref += shift
        value = min((value + 12 * k for k in range(0, 11)), key=lambda m: (abs(m - ref), m))
    return note_label(value - shift)


def set_fingering(note: dict, index: int | None, sheet: dict) -> None:
    """Pin one of the note's fingerings (None = automatic). Raises ValueError."""
    if index is None:
        note["fingering"] = None
        return
    midi = note_midi(note)
    options = fingerings(None if midi is None else midi + played_shift(sheet, note),
                         instrument_of(sheet))
    if not 0 <= index < len(options):
        raise ValueError("fingering out of range")
    note["fingering"] = index


# --------------------------------------------------------------------------- #
# editing the note list
# --------------------------------------------------------------------------- #

MIN_NOTE_S = 0.05      # edits may not make a note shorter than this
OVERLAP_S = 0.02       # a new note may overlap a neighbour by this much


def _notes(sheet: dict) -> list[dict]:
    mel = _melody(sheet)
    if mel is None:
        raise ValueError("this song has no melody yet")
    return mel["notes"]


def _new_id(notes: list[dict]) -> str:
    used = [int(n["id"][1:]) for n in notes if re.fullmatch(r"n\d+", n.get("id", ""))]
    return f"n{max(used, default=-1) + 1}"


def _find(notes: list[dict], note_id: str) -> dict:
    note = next((n for n in notes if n["id"] == note_id), None)
    if note is None:
        raise ValueError("no such note")
    return note


def split_note(sheet: dict, note_id: str, at: float | None = None) -> str:
    """Cut a note in two at time `at` (default: its middle), e.g. a repeated
    pitch the tracker held as one note. Both halves keep the pitch choice;
    returns the id of the second half. Raises ValueError."""
    notes = _notes(sheet)
    note = _find(notes, note_id)
    at = 0.5 * (note["start"] + note["end"]) if at is None else float(at)
    if not note["start"] + MIN_NOTE_S <= at <= note["end"] - MIN_NOTE_S:
        raise ValueError(f"a note can only be split inside itself, leaving at least "
                         f"{MIN_NOTE_S * 1000:.0f} ms on each side")
    second = {**{k: (list(v) if isinstance(v, list) else v) for k, v in note.items()},
              "id": _new_id(notes), "start": round(at, 3), "fingering": None}
    second["candidates"] = [dict(c) for c in note.get("candidates", [])]
    note["end"] = round(at, 3)
    note["fingering"] = None
    notes.append(second)
    notes.sort(key=lambda n: n["start"])
    return second["id"]


def merge_notes(sheet: dict, note_ids: list[str]) -> str:
    """Join neighbouring notes into one (a note the tracker split in two, or
    a trill to smooth out). The longest of them decides the pitch. Returns the
    id of the first. Raises ValueError."""
    notes = sorted(_notes(sheet), key=lambda n: n["start"])
    chosen = [_find(notes, i) for i in dict.fromkeys(note_ids)]
    if len(chosen) < 2:
        raise ValueError("pick at least two notes to merge")
    chosen.sort(key=lambda n: n["start"])
    idx = [notes.index(n) for n in chosen]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError("only neighbouring notes can be merged")
    keeper = max(chosen, key=lambda n: n["end"] - n["start"])
    first = chosen[0]
    merged = {**{k: v for k, v in keeper.items()}, "id": first["id"],
              "start": first["start"], "end": chosen[-1]["end"], "fingering": None}
    drop = {n["id"] for n in chosen}
    mel = _melody(sheet)
    mel["notes"] = [n for n in mel["notes"] if n["id"] not in drop] + [merged]
    mel["notes"].sort(key=lambda n: n["start"])
    return merged["id"]


def add_note(sheet: dict, start: float, end: float, text: str) -> str:
    """Add a note the tracker missed. `text` is typed as shown on the tab
    ("C5", "Bb"; no octave = the one nearest the melody). Returns its id.
    Raises ValueError."""
    notes = _notes(sheet)
    start, end = round(float(start), 3), round(float(end), 3)
    if end - start < MIN_NOTE_S:
        raise ValueError(f"a note must last at least {MIN_NOTE_S * 1000:.0f} ms")
    if start < 0:
        raise ValueError("a note can't start before the song does")
    for n in notes:
        if note_midi(n) is not None and start < n["end"] - OVERLAP_S and end > n["start"] + OVERLAP_S:
            raise ValueError(f"that overlaps the note at {n['start']:.1f} s")
    probe = {"id": "probe", "start": start, "end": end, "candidates": [], "selected": 0,
             "custom": None}
    near = [n for n in notes if note_midi(n) is not None]
    if near:
        ref = min(near, key=lambda n: abs(n["start"] - start))
        probe["candidates"] = ref["candidates"]
    label = note_from_display(text, sheet, probe)
    parsed = parse_note(label) if label else None
    if label == REST or parsed is None or parsed[1] is None:
        raise ValueError(f"{text!r} isn't a note name (try C, F#, Bb4)")
    note = {"id": _new_id(notes), "start": start, "end": end, "selected": 0, "custom": None,
            "fingering": None, "source": "user",
            "candidates": [{"label": label, "midi": parsed[0], "conf": 1.0}]}
    notes.append(note)
    notes.sort(key=lambda n: n["start"])
    return note["id"]


def position_name(sheet: dict, instrument: Instrument) -> str | None:
    """"2nd (cross harp)" and so on: where the song's key sits on this harp."""
    parsed = parse_key(sheet.get("key"))
    if parsed is None:
        return None
    tonic = (parsed[0] + int(sheet.get("transpose", 0))) % 12
    return POSITIONS.get((tonic - instrument.pc) % 12)


def suggest_instruments(sheet: dict, limit: int = 4) -> list[dict]:
    """The diatonic harp keys that suit this melody best.

    For each of the 12 keys, the melody (as transposed) is moved to the octave
    that puts the most playing time on playable notes, and scored by the share
    of time that is playable at all and the share playable without bending.
    Best first (unbent share, then playable share). The 12-hole chromatic plays
    everything, so it is not ranked.
    """
    mel = _melody(sheet)
    if mel is None:
        return []
    transpose = int(sheet.get("transpose", 0))
    notes = [(m + transpose, n["end"] - n["start"])
             for n in mel["notes"] if n.get("source") != "instr" and (m := note_midi(n)) is not None]
    if not notes:
        notes = [(m + transpose, n["end"] - n["start"])
                 for n in mel["notes"] if (m := note_midi(n)) is not None]
    total = sum(d for _, d in notes)
    if total <= 0:
        return []
    current = instrument_of(sheet)
    out = []
    for pc in range(12):
        inst = get_instrument("diatonic10", pc)
        octave = fit_octave(notes, inst)
        playable = sum(d for m, d in notes if inst.playable(m + 12 * octave))
        unbent = sum(d for m, d in notes
                     if any(f["bend"] == 0 for f in inst.fingerings(m + 12 * octave)))
        out.append({"type": "diatonic10", "key": HARP_KEYS[pc], "name": inst.short_name,
                    "playable": round(playable / total, 3), "unbent": round(unbent / total, 3),
                    "octave": octave, "position": position_name(sheet, inst),
                    "current": inst.kind == current.kind and inst.pc == current.pc})
    out.sort(key=lambda r: (-r["unbent"], -r["playable"], r["key"]))
    return out[:limit]


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
    inst = instrument_of(sheet)

    def describe(midi: int | None, names: list[str]) -> dict:
        if midi is None:
            return {"midi": None, "display": "×", "name": "rest", "tab": None, "in_range": True}
        options = fingerings(midi, inst)
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
        options = fingerings(midi, inst)
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
                           grid=grid_for(sheet), chords_per_row=NOTES_PER_ROW,
                           syllabify=make_syllabifier(
                               (sheet.get("transcript") or {}).get("language"),
                               sheet.get("vocal_env"))),
        "octave": shifts["total"],
        "auto_octave": shifts["auto"],
        "instr_octave": shifts["instr"],
        "user_octave": int(mel.get("octave", 0)),
        "in_range": round(playable / total_time, 3) if total_time else 1.0,
        "key_label": key_label(sheet.get("key"), transpose),
        "range": [note_label(inst.lowest), note_label(inst.highest)],
        "instrument": {**inst.to_dict(), "name": inst.name, "short_name": inst.short_name,
                       "diatonic": inst.kind == "diatonic10"},
        "suggestions": suggest_instruments(sheet),
        "fast": bool(mel.get("fast")),
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
    inst = instrument_of(sheet)
    legend = ("↑ blow   ↓ draw   ′ ″ ‴ bend a half / whole / 1½ steps" if inst.kind == "diatonic10"
              else "↑ blow   ↓ draw   ● button in")
    out = [sheet.get("title", "Untitled"), f"{inst.name} harmonica   {legend}"]
    if hv is None:
        out += ["", "(no melody found yet)"]
        return "\n".join(out) + "\n"
    meta = []
    position = position_name(sheet, inst)
    if position and inst.kind == "diatonic10":
        meta.append(f"Position: {position}")
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
