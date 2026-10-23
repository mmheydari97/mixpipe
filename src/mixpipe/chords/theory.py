"""Small music-theory helpers: note spelling, chord-label parsing, transposition,
and capo suggestions. Pure Python, no audio dependencies.
"""

from __future__ import annotations

import re

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NEUTRAL = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_FLAT_MAJOR = {5, 10, 3, 8, 1, 6}   # F Bb Eb Ab Db Gb
_FLAT_MINOR = {2, 7, 0, 5, 10, 3}   # D G C F Bb Eb

_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}
_LABEL = re.compile(r"^([A-G])([#b♯♭]?)([^/]*)(?:/([A-G])([#b♯♭]?))?$")

TIME_SIGNATURES = {2: "2/4", 3: "3/4", 4: "4/4", 6: "6/8"}


def note_names_for_key(tonic: int, mode: str) -> list[str]:
    """Sharps in sharp keys, flats in flat keys, common names in C / Am."""
    if (mode == "maj" and tonic == 0) or (mode == "min" and tonic == 9):
        return NEUTRAL
    flat = tonic in (_FLAT_MAJOR if mode == "maj" else _FLAT_MINOR)
    return FLAT if flat else SHARP


def note_pc(name: str) -> int | None:
    m = re.match(r"^([A-G])([#b♯♭]?)$", name.strip())
    if not m:
        return None
    return (_NATURAL[m.group(1)] + _ACC[m.group(2)]) % 12


def parse_key(key: str | None) -> tuple[int, str] | None:
    """"Eb:maj" -> (3, "maj")."""
    if not key or ":" not in key:
        return None
    tonic, mode = key.split(":", 1)
    pc = note_pc(tonic)
    return None if pc is None else (pc, mode)


def parse_label(label: str) -> tuple[int, str, int | None] | None:
    """"F#m7/C#" -> (6, "m7", 1). The quality text is kept verbatim.
    None for anything that isn't a chord name (N.C., free text)."""
    m = _LABEL.match((label or "").strip())
    if not m:
        return None
    root = (_NATURAL[m.group(1)] + _ACC[m.group(2)]) % 12
    bass = (_NATURAL[m.group(4)] + _ACC[m.group(5)]) % 12 if m.group(4) else None
    return root, m.group(3), bass


# suffix -> intervals above the root, root first (the chord recogniser's templates)
CHORD_TONES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
}
_EXTRA_TONES = {"m7b5": (0, 3, 6, 10), "ø": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
                "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "9": (0, 4, 7, 10, 2),
                "add9": (0, 4, 7, 2), "m9": (0, 3, 7, 10, 2), "maj9": (0, 4, 7, 11, 2),
                "5": (0, 7), "+": (0, 4, 8)}


def chord_intervals(quality: str) -> tuple[int, ...]:
    """Intervals above the root for a chord suffix, root first. Typed suffixes
    outside the table fall back to the nearest triad ("m11" -> minor)."""
    q = quality.strip()
    if q in CHORD_TONES:
        return CHORD_TONES[q]
    if q in _EXTRA_TONES:
        return _EXTRA_TONES[q]
    if q.startswith(("dim", "°")):
        return CHORD_TONES["dim"]
    if q.startswith("aug"):
        return CHORD_TONES["aug"]
    if q.startswith("sus"):
        return CHORD_TONES["sus2" if q.startswith("sus2") else "sus4"]
    if q.startswith(("m", "-")) and not q.startswith("maj"):
        return CHORD_TONES["m"]
    return CHORD_TONES[""]


def chord_pcs(label: str) -> tuple[int, ...] | None:
    """"Am7" -> (9, 0, 4, 7): the chord's pitch classes, root first (a slash bass
    is added last). None for anything that isn't a chord name."""
    parsed = parse_label(label)
    if parsed is None:
        return None
    root, quality, bass = parsed
    pcs = list(dict.fromkeys((root + iv) % 12 for iv in chord_intervals(quality)))
    if bass is not None and bass not in pcs:
        pcs.append(bass)
    return tuple(pcs)


def transpose_label(label: str, shift: int, names: list[str]) -> str:
    """Move a chord label by `shift` semitones, spelled with `names`.
    Labels that don't parse (N.C., free text) come back unchanged."""
    parsed = parse_label(label)
    if parsed is None:
        return label
    root, quality, bass = parsed
    out = names[(root + shift) % 12] + quality
    if bass is not None:
        out += "/" + names[(bass + shift) % 12]
    return out


def key_names(key: str | None, shift: int = 0) -> list[str]:
    """Spelling for a key moved by `shift` semitones (NEUTRAL if unknown)."""
    parsed = parse_key(key)
    if parsed is None:
        return NEUTRAL
    tonic, mode = parsed
    return note_names_for_key((tonic + shift) % 12, mode)


def key_label(key: str | None, shift: int = 0) -> str | None:
    parsed = parse_key(key)
    if parsed is None:
        return None
    tonic, mode = parsed
    t = (tonic + shift) % 12
    return f"{note_names_for_key(t, mode)[t]} {'major' if mode == 'maj' else 'minor'}"


def key_short(key: str | None, shift: int = 0) -> str | None:
    """ChordPro-style key: "D", "F#m"."""
    parsed = parse_key(key)
    if parsed is None:
        return None
    tonic, mode = parsed
    t = (tonic + shift) % 12
    return note_names_for_key(t, mode)[t] + ("m" if mode == "min" else "")


# --------------------------------------------------------------------------- #
# instruments, capo
# --------------------------------------------------------------------------- #

# Open-position shapes, as (root pc, quality). Slash bass is ignored. "Easy"
# shapes are played open; "medium" ones need a barre or a stretch and count half.
# A shape is named by what it is *fingered as*: on a ukulele in C tuning the
# "C" shape is open 0003.
EASY_SHAPES = {   # guitar, standard tuning
    (0, ""), (2, ""), (4, ""), (7, ""), (9, ""),                  # C D E G A
    (9, "m"), (4, "m"), (2, "m"),                                 # Am Em Dm
    (9, "7"), (11, "7"), (0, "7"), (2, "7"), (4, "7"), (7, "7"),  # A7 B7 C7 D7 E7 G7
    (9, "m7"), (4, "m7"), (2, "m7"),                              # Am7 Em7 Dm7
    (0, "maj7"), (5, "maj7"), (9, "maj7"), (2, "maj7"), (4, "maj7"), (7, "maj7"),
    (9, "sus2"), (9, "sus4"), (2, "sus2"), (2, "sus4"), (4, "sus4"),
    (0, "add9"), (7, "6"), (0, "6"), (9, "6"), (4, "6"),
}
MEDIUM_SHAPES = {
    (5, ""), (5, "sus2"), (11, "m"), (11, "m7"), (6, "m"), (6, "m7"), (5, "7"),
}
# Ukulele, GCEA (soprano / concert / tenor): C Am F G G7 C7 Dm A A7 Am7 are all within
# the first three frets; D, E and the minors on E and B need a stretch or a barre.
EASY_UKE = {
    (0, ""), (9, "m"), (5, ""), (7, ""), (7, "7"), (0, "7"), (2, "m"), (9, ""), (9, "7"),
    (9, "m7"), (2, "m7"), (0, "maj7"), (5, "maj7"), (0, "6"), (5, "sus2"), (9, "sus4"),
    (4, "m7"), (2, "7"),
}
MEDIUM_UKE = {
    (2, ""), (4, "m"), (4, ""), (4, "7"), (10, ""), (7, "m"), (11, "m"), (5, "7"), (5, "m"),
    (7, "maj7"), (2, "maj7"), (11, ""), (9, "maj7"), (7, "sus4"), (0, "sus2"), (2, "sus4"),
}
# Baritone ukulele, DGBE: the top four strings of a guitar, so the open shapes that
# need only those strings; anything with a root on the missing low strings is a barre.
EASY_BARITONE = {
    (7, ""), (0, ""), (2, ""), (4, "m"), (9, "m"), (2, "m"), (9, ""), (4, ""), (4, "7"),
    (7, "7"), (2, "7"), (9, "7"), (0, "maj7"), (7, "maj7"), (4, "m7"), (9, "m7"), (2, "m7"),
    (2, "sus2"), (2, "sus4"), (7, "sus4"), (0, "7"), (2, "maj7"), (9, "sus2"),
}
MEDIUM_BARITONE = {(5, ""), (11, "m"), (11, "7"), (5, "maj7"), (4, "sus4"), (0, "add9"), (7, "6")}

# id -> profile. `tuning` is how many semitones the instrument sounds above the
# shapes' own names: a ukulele in D tuning (A D F# B) fingers C shapes that sound D.
INSTRUMENTS: dict[str, dict] = {
    "guitar": {"label": "Guitar", "easy": EASY_SHAPES, "medium": MEDIUM_SHAPES,
               "tuning": 0, "max_capo": 7},
    "guitar_half_down": {"label": "Guitar, half step down (E♭)", "easy": EASY_SHAPES,
                         "medium": MEDIUM_SHAPES, "tuning": -1, "max_capo": 7},
    "guitar_whole_down": {"label": "Guitar, whole step down (D)", "easy": EASY_SHAPES,
                          "medium": MEDIUM_SHAPES, "tuning": -2, "max_capo": 7},
    "ukulele": {"label": "Ukulele (GCEA)", "easy": EASY_UKE, "medium": MEDIUM_UKE,
                "tuning": 0, "max_capo": 7},
    "ukulele_d": {"label": "Ukulele, D tuning (ADF♯B)", "easy": EASY_UKE, "medium": MEDIUM_UKE,
                  "tuning": 2, "max_capo": 7},
    "baritone_ukulele": {"label": "Baritone ukulele (DGBE)", "easy": EASY_BARITONE,
                         "medium": MEDIUM_BARITONE, "tuning": 0, "max_capo": 7},
}
DEFAULT_INSTRUMENT = "guitar"
CAPO_FRET_COST = 0.015   # per fret: prefer lower capos when shapes are equal
MIN_CAPO_GAIN = 0.15     # only suggest a capo that clearly beats the current one


def instrument_profile(instrument: str | None) -> dict:
    """The profile for an instrument id (the guitar for anything unknown)."""
    return INSTRUMENTS.get(instrument or DEFAULT_INSTRUMENT, INSTRUMENTS[DEFAULT_INSTRUMENT])


def instrument_choices() -> list[dict]:
    return [{"id": k, "label": v["label"], "tuning": v["tuning"]} for k, v in INSTRUMENTS.items()]


def _easy_fraction(chords: list[tuple[int, str, float]], shift: int,
                   instrument: str | None = None) -> float:
    prof = instrument_profile(instrument)
    total = sum(d for _, _, d in chords)
    if total <= 0:
        return 0.0
    score = 0.0
    for r, q, d in chords:
        shape = ((r + shift) % 12, q)
        score += d if shape in prof["easy"] else 0.5 * d if shape in prof["medium"] else 0.0
    return score / total


def suggest_capo(chords: list[tuple[int, str, float]], transpose: int = 0,
                 current_capo: int = 0, max_capo: int | None = None,
                 instrument: str | None = None) -> dict | None:
    """The capo fret that puts the most playing time on open shapes, or None
    when nothing clearly beats the current capo.

    `chords` is [(sounding root pc, quality, duration_s)]. Shapes are the
    sounding chord moved by (transpose - capo - the instrument's tuning offset).
    Open shapes count fully, barre shapes like F and Bm half; each fret costs a
    little so lower capos win ties. `instrument` picks the shape tables (guitar,
    ukulele...).
    """
    if not chords:
        return None
    prof = instrument_profile(instrument)
    tuning = prof["tuning"]
    top = prof["max_capo"] if max_capo is None else max_capo
    frac = lambda capo: _easy_fraction(chords, transpose - capo - tuning, instrument)  # noqa: E731
    current = frac(current_capo) - CAPO_FRET_COST * current_capo
    best_capo, best = current_capo, current
    for capo in range(top + 1):
        score = frac(capo) - CAPO_FRET_COST * capo
        if score > best + 1e-9:
            best_capo, best = capo, score
    if best_capo == current_capo or best - current < MIN_CAPO_GAIN:
        return None
    return {"capo": best_capo, "easy": round(frac(best_capo), 3)}


def easy_fraction(chords: list[tuple[int, str, float]], transpose: int, capo: int,
                  instrument: str | None = None) -> float:
    tuning = instrument_profile(instrument)["tuning"]
    return round(_easy_fraction(chords, transpose - capo - tuning, instrument), 3)
