"""The chord sheet document: storage, view model, settings, and exports.

sheet.json is the source of truth for one song:

  {"version": 2, "id", "title", "created", "audio", "duration",
   "key", "key_label", "key_regions": [{"start", "end", "key", "key_label"}],
   "tempo", "beat_period", "beat_times",
   "meter": {"time_signature", "beats_per_bar", "downbeat", "detected": {...}},
   "transpose": 0, "capo": 0,
   "recognizer": "template" | "btc",
   "transcript": {"backend", "model", "language", "timing", "retimed"},
   "lines":  [{"words": [{"text", "start", "end", "approx"?}], "gap_before"}],
   "chords": [{"id", "start", "end", "beats", "candidates", "selected", "custom"}],
   "melody": {"octave": 0, "notes": [{"id", "start", "end", "candidates": [{"label",
              "midi", "conf"}], "selected", "custom", "fingering", "source"}]}}

"key" is the key the song starts in (what the overview shows). If the song
modulates, each chord carries its own "key", which drives its spelling, and the
view marks the chord where the key changes.

Chord labels are always stored at concert (sounding) pitch. Transpose and capo
only change what is displayed and exported: shapes = sounding + transpose - capo.
Rows (chords placed on words, bars) are derived on demand by align.build_rows.

"melody" feeds the harmonica tab (harmonica.py); sheets from before it existed
have none until the melody is found for them. Notes are stored at sung pitch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .align import build_rows, chord_label, grid_for
from .harmonica import MAX_OCTAVE, harmonica_view
from .syllables import make_syllabifier
from .theory import (DEFAULT_INSTRUMENT, INSTRUMENTS, TIME_SIGNATURES, easy_fraction,
                     instrument_choices, key_label, key_names, key_short, parse_label,
                     suggest_capo, transpose_label)

SHEET_VERSION = 3
TS_CHOICES = ("2/4", "3/4", "4/4", "6/8")
COMPOUND_PULSE_BPM = 140.0  # 6/8 with a pulse faster than this is counted in eighths


def load_sheet(path: str | Path) -> dict:
    return migrate(json.loads(Path(path).read_text()))


def save_sheet(sheet: dict, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sheet, indent=1, ensure_ascii=False))
    os.replace(tmp, path)  # atomic, so a crash never leaves half a sheet


def migrate(sheet: dict) -> dict:
    """Fill fields added after version 1 so older sheets keep working."""
    sheet.setdefault("transpose", 0)
    sheet.setdefault("capo", 0)
    sheet.setdefault("beat_times", [])
    sheet.setdefault("key_regions", [])
    meter = sheet.setdefault("meter", {"beats_per_bar": 4, "downbeat": 0, "detected": None})
    meter.setdefault("time_signature", TIME_SIGNATURES.get(meter["beats_per_bar"], "4/4"))
    return sheet


# --------------------------------------------------------------------------- #
# transposition / capo
# --------------------------------------------------------------------------- #


def instrument_id(sheet: dict) -> str:
    inst = sheet.get("instrument")
    return inst if inst in INSTRUMENTS else DEFAULT_INSTRUMENT


def shape_shift(sheet: dict) -> int:
    """Semitones from sounding pitch to the chord shapes shown: the transpose, less
    the capo, less how far the instrument is tuned above its shapes' names."""
    tuning = INSTRUMENTS[instrument_id(sheet)]["tuning"]
    return int(sheet.get("transpose", 0)) - int(sheet.get("capo", 0)) - tuning


def chord_key(chord: dict, sheet: dict) -> str | None:
    """The key a chord sits in: its own (after a modulation) or the song's."""
    return chord.get("key") or sheet.get("key")


def display_label(chord: dict, sheet: dict) -> str:
    shift = shape_shift(sheet)
    label = chord_label(chord)
    if shift == 0:
        return label
    return transpose_label(label, shift, key_names(chord_key(chord, sheet), shift))


def custom_from_display(text: str, sheet: dict, chord: dict | None = None) -> str:
    """A chord typed in display (shape) space, stored back at concert pitch."""
    shift = shape_shift(sheet)
    if shift == 0 or parse_label(text) is None:
        return text
    key = chord_key(chord, sheet) if chord else sheet.get("key")
    return transpose_label(text, -shift, key_names(key, 0))


def key_changes(sheet: dict) -> dict[str, str]:
    """{chord id: new key (sounding, after transpose)} for each visible chord
    where the key differs from the previous one. Empty when the song stays put."""
    transpose = int(sheet.get("transpose", 0))
    out, prev = {}, None
    for c in sorted(sheet["chords"], key=lambda c: c["start"]):
        if chord_label(c) == "N.C.":
            continue
        key = chord_key(c, sheet)
        if prev is not None and key != prev:
            out[c["id"]] = key_label(key, transpose)
        prev = key
    return out


def _durations(sheet: dict) -> list[tuple[int, str, float]]:
    out = []
    for c in sheet["chords"]:
        parsed = parse_label(chord_label(c))
        if parsed:
            out.append((parsed[0], parsed[1], c["end"] - c["start"]))
    return out


def beats_per_bar_for(ts: str, sheet: dict) -> int:
    """How many *tracked* beats make a bar of `ts`. 6/8 depends on what the beat
    tracker locked onto: the dotted quarter (2 per bar) or the eighth (6)."""
    if ts != "6/8":
        return {"2/4": 2, "3/4": 3, "4/4": 4}[ts]
    det = sheet["meter"].get("detected") or {}
    if det.get("time_signature") == "6/8":
        return det["beats_per_bar"]
    if det.get("subdivision") == "ternary":
        return 2
    return 6 if (sheet.get("tempo") or 0) >= COMPOUND_PULSE_BPM else 2


def apply_settings(sheet: dict, transpose=None, capo=None, time_signature=None,
                   downbeat=None, shift_downbeat=None, reset_meter=False,
                   harmonica_octave=None, harmonica_type=None, harmonica_key=None,
                   instrument=None) -> dict:
    """Validate and apply display settings in place. Raises ValueError."""
    meter = sheet["meter"]
    if reset_meter and meter.get("detected"):
        detected = meter["detected"]
        meter["time_signature"] = detected.get(
            "time_signature", TIME_SIGNATURES.get(detected["beats_per_bar"], "4/4"))
        meter["beats_per_bar"] = detected["beats_per_bar"]
        meter["downbeat"] = detected["downbeat"]
        meter["sections"] = list(detected.get("sections") or [])
    if transpose is not None:
        if not -11 <= int(transpose) <= 11:
            raise ValueError("transpose must be between -11 and 11")
        sheet["transpose"] = int(transpose)
    if capo is not None:
        if not 0 <= int(capo) <= 11:
            raise ValueError("capo must be between 0 and 11")
        sheet["capo"] = int(capo)
    if instrument is not None:
        if instrument not in INSTRUMENTS:
            raise ValueError(f"instrument must be one of {', '.join(INSTRUMENTS)}")
        sheet["instrument"] = instrument
    if time_signature is not None:
        if time_signature not in TS_CHOICES:
            raise ValueError(f"time_signature must be one of {', '.join(TS_CHOICES)}")
        meter["sections"] = []              # a hand-picked meter replaces the detected changes
        meter["time_signature"] = time_signature
        meter["beats_per_bar"] = beats_per_bar_for(time_signature, sheet)
        # keep bar lines anchored where the detector put the first downbeat
        anchor = (meter.get("detected") or {}).get("downbeat", meter["downbeat"])
        meter["downbeat"] = anchor % meter["beats_per_bar"]
    if downbeat is not None:
        meter["sections"] = []
        meter["downbeat"] = int(downbeat) % meter["beats_per_bar"]
    if shift_downbeat is not None:
        if meter.get("sections"):
            # slide every section's downbeat together, keeping the meter changes
            for sec in meter["sections"]:
                sec["beat"] += int(shift_downbeat)
        meter["downbeat"] = (meter["downbeat"] + int(shift_downbeat)) % meter["beats_per_bar"]
    if harmonica_octave is not None:
        if not sheet.get("melody"):
            raise ValueError("this song has no melody yet")
        if not -MAX_OCTAVE <= int(harmonica_octave) <= MAX_OCTAVE:
            raise ValueError(f"harmonica_octave must be between -{MAX_OCTAVE} and {MAX_OCTAVE}")
        sheet["melody"]["octave"] = int(harmonica_octave)
    if harmonica_type is not None or harmonica_key is not None:
        if not sheet.get("melody"):
            raise ValueError("this song has no melody yet")
        from .harmonica import get_instrument, instrument_of

        current = instrument_of(sheet)
        inst = get_instrument(harmonica_type or current.kind,
                              harmonica_key if harmonica_key is not None else current.pc)
        sheet["melody"]["instrument"] = inst.to_dict()
        for note in sheet["melody"]["notes"]:
            note["fingering"] = None        # a pinned fingering belongs to the old harp
    return sheet


def meter_changes(sheet: dict) -> list[dict]:
    """Meter changes after the first section: [{"bar" (1-based), "time_signature",
    "time"}], empty for a song in one meter."""
    grid = grid_for(sheet)
    if grid is None:
        return []
    return [{"bar": c["bar"] + 1, "time_signature": TIME_SIGNATURES.get(c["beats_per_bar"], "4/4"),
             "time": round(c["time"], 3)} for c in grid.section_changes()]


def time_signature(sheet: dict) -> str:
    return sheet["meter"].get("time_signature") or TIME_SIGNATURES.get(
        sheet["meter"]["beats_per_bar"], "4/4")


# --------------------------------------------------------------------------- #
# view
# --------------------------------------------------------------------------- #


def syllabifier(sheet: dict):
    """The sheet's word -> syllables function (language and, when stored, the
    vocal loudness envelope that places the syllable boundaries)."""
    return make_syllabifier((sheet.get("transcript") or {}).get("language"),
                            sheet.get("vocal_env"))


def _rows(sheet: dict) -> list[dict]:
    return build_rows(sheet["lines"], sheet["chords"], sheet.get("beat_period", 0.5),
                      grid=grid_for(sheet), syllabify=syllabifier(sheet))


def view(sheet: dict) -> dict:
    """Sheet plus derived rows and display labels, as served to the UI."""
    sheet = migrate(sheet)
    shift = shape_shift(sheet)
    names = key_names(sheet.get("key"), shift)
    out = dict(sheet)
    changes = key_changes(sheet)
    out["chords"] = []
    for c in sheet["chords"]:
        c2 = dict(c)
        c2["display"] = display_label(c, sheet)
        cnames = key_names(chord_key(c, sheet), shift) if c.get("key") else names
        c2["candidates"] = [
            {**cand, "display": cand["label"] if shift == 0
             else transpose_label(cand["label"], shift, cnames)}
            for cand in c["candidates"]
        ]
        c2["key_change"] = changes.get(c["id"])
        out["chords"].append(c2)
    out["rows"] = _rows(sheet)
    durations = _durations(sheet)
    transpose = int(sheet.get("transpose", 0))
    out["display"] = {
        "key_label": key_label(sheet.get("key"), transpose),        # sounding, after transpose
        "shape_key_label": key_label(sheet.get("key"), shift),      # what you finger with capo
        "time_signature": time_signature(sheet),
        "meter_changes": meter_changes(sheet),
        "instrument": instrument_id(sheet),
        "instrument_label": INSTRUMENTS[instrument_id(sheet)]["label"],
        "instruments": instrument_choices(),
        "easy_shapes": easy_fraction(durations, transpose, int(sheet.get("capo", 0)),
                                     instrument_id(sheet)),
        "capo_suggestion": suggest_capo(durations, transpose, int(sheet.get("capo", 0)),
                                        instrument=instrument_id(sheet)),
    }
    out.pop("melody", None)  # served as "harmonica" instead
    out.pop("vocal_env", None)  # only used on the server, to place syllables
    out["harmonica"] = harmonica_view(sheet)
    return out


# --------------------------------------------------------------------------- #
# exports
# --------------------------------------------------------------------------- #


def _header(sheet: dict) -> dict:
    transpose, capo = int(sheet.get("transpose", 0)), int(sheet.get("capo", 0))
    shift = shape_shift(sheet)
    inst = instrument_id(sheet)
    return {
        "key": key_short(sheet.get("key"), transpose),
        "capo": capo,
        "shapes": key_short(sheet.get("key"), shift) if shift != transpose else None,
        "instrument": INSTRUMENTS[inst]["label"] if inst != DEFAULT_INSTRUMENT else None,
        "time": time_signature(sheet) if sheet.get("beat_times") else None,
        "tempo": f"{sheet['tempo']:.0f}" if sheet.get("tempo") else None,
    }


def _word_with_chords(word: dict, labels: list[tuple[str, float, str]]) -> list[tuple[int, str]]:
    """Character offsets inside the word where each chord change falls.

    A word split into syllables (`word["syl"]`, `word["at"]`) puts the chord at
    the start of the syllable it sits over. A one-syllable word keeps its first
    chord at the front, and a later change at the front or the end depending on
    which half of the word it falls in; (a chord can't be "in" a single syllable)."""
    text, dur = word["text"], max(word["end"] - word["start"], 1e-6)
    spans, at = word.get("syl"), word.get("at") or {}
    out = []
    for k, (label, start, cid) in enumerate(labels):
        if spans and cid in at:
            pos = spans[at[cid]]["c"]
        elif k == 0:
            pos = 0
        elif spans is None and len(text) > 0:
            pos = 0 if (start - word["start"]) / dur < 0.5 else len(text)
        else:
            pos = round(len(text) * (start - word["start"]) / dur)
        out.append((min(max(pos, 0), len(text)), label))
    return out


def _bar_text(row: dict, label) -> str:
    """"| G | D C | % |" for a bar-aware row, "| G | D |" otherwise."""
    if "bars" not in row:
        return "| " + " | ".join(label(c) for c in row["chords"]) + " |"
    cells = []
    for bar in row["bars"]:
        if bar["chords"]:
            held = [bar["hold"]] if bar["hold"] else []
            cells.append(" ".join(label(c) for c in held + bar["chords"]))
        else:
            cells.append("%" if bar["hold"] else "-")
    return "| " + " | ".join(cells) + " |"


def _row_chord_ids(row: dict) -> list[str]:
    if row["type"] == "instrumental":
        return list(row["chords"])
    return [cid for w in row["words"] for cid in w["chords"]]


def _row_key_change(row: dict, changes: dict[str, str]) -> str | None:
    return next((changes[c] for c in _row_chord_ids(row) if c in changes), None)


def _row_time(row: dict, chords: dict) -> float | None:
    """When a row begins: its first word, or the first chord of an instrumental row."""
    if row["type"] == "lyric":
        return row["words"][0]["start"] if row["words"] else None
    ids = _row_chord_ids(row)
    return chords[ids[0]]["start"] if ids else None


def _due_meter_changes(pending: list[dict], row: dict, chords: dict) -> list[dict]:
    """Pop the meter changes that happen at or before this row starts."""
    t = _row_time(row, chords)
    due = []
    while t is not None and pending and pending[0]["time"] <= t + 0.3:
        due.append(pending.pop(0))
    return due


def to_text(sheet: dict) -> str:
    """Monospace chords-over-lyrics sheet."""
    sheet = migrate(sheet)
    chords = {c["id"]: c for c in sheet["chords"]}
    label = lambda cid: display_label(chords[cid], sheet)  # noqa: E731
    h = _header(sheet)
    out = [sheet.get("title", "Untitled")]
    meta = []
    if h["key"]:
        meta.append(f"Key: {h['key']}")
    if h["instrument"]:
        meta.append(f"Instrument: {h['instrument']}")
    if h["capo"]:
        meta.append(f"Capo {h['capo']} (play {h['shapes']} shapes)")
    elif h["shapes"]:
        meta.append(f"Play {h['shapes']} shapes")
    if h["time"]:
        meta.append(f"Time: {h['time']}")
    if h["tempo"]:
        meta.append(f"Tempo: {h['tempo']}")
    if meta:
        out.append("   ".join(meta))
    out.append("")

    changes = key_changes(sheet)
    pending_meter = meter_changes(sheet)
    for row in _rows(sheet):
        change = _row_key_change(row, changes)
        if change:
            out.append(f"[Key change: {change}]")
        for mc in _due_meter_changes(pending_meter, row, chords):
            out.append(f"[Time: {mc['time_signature']}]")
        if row["type"] == "instrumental":
            out.append(_bar_text(row, label))
            continue
        if row["gap_before"]:
            out.append("")
        lyric, chord_line = "", ""
        for w in row["words"]:
            if lyric:
                lyric += " "
            base = len(lyric)
            labels = [(label(c), chords[c]["start"], c) for c in w["chords"]]
            for pos, lab in _word_with_chords(w, labels):
                col = base + pos
                if chord_line:
                    col = max(col, len(chord_line) + 1)
                chord_line = chord_line.ljust(col) + lab
            lyric += w["text"]
        if chord_line:
            out.append(chord_line.rstrip())
        out.append(lyric)
    return "\n".join(out).rstrip() + "\n"


def to_chordpro(sheet: dict) -> str:
    """ChordPro: {title}, {key}, {capo}, {time}, {tempo}, then [C]inline chords."""
    sheet = migrate(sheet)
    chords = {c["id"]: c for c in sheet["chords"]}
    label = lambda cid: display_label(chords[cid], sheet)  # noqa: E731
    h = _header(sheet)
    out = [f"{{title: {sheet.get('title', 'Untitled')}}}"]
    if h["key"]:
        out.append(f"{{key: {h['key']}}}")
    if h["instrument"]:
        out.append(f"{{comment: Instrument: {h['instrument']}}}")
    if h["capo"]:
        out.append(f"{{capo: {h['capo']}}}")
    if h["time"]:
        out.append(f"{{time: {h['time']}}}")
    if h["tempo"]:
        out.append(f"{{tempo: {h['tempo']}}}")
    out.append("")

    bracket = lambda cid: f"[{label(cid)}]"  # noqa: E731
    changes = key_changes(sheet)
    transpose = int(sheet.get("transpose", 0))
    pending_meter = meter_changes(sheet)
    for row in _rows(sheet):
        change_id = next((c for c in _row_chord_ids(row) if c in changes), None)
        if change_id:
            out.append(f"{{key: {key_short(chord_key(chords[change_id], sheet), transpose)}}}")
        for mc in _due_meter_changes(pending_meter, row, chords):
            out.append(f"{{time: {mc['time_signature']}}}")
        if row["type"] == "instrumental":
            out.append("{comment: Instrumental}")
            out.append(_bar_text(row, bracket) if "bars" in row
                       else " ".join(bracket(c) for c in row["chords"]))
            continue
        if row["gap_before"]:
            out.append("")
        parts = []
        for w in row["words"]:
            labels = [(label(c), chords[c]["start"], c) for c in w["chords"]]
            text, built, last = w["text"], "", 0
            for pos, lab in _word_with_chords(w, labels):
                built += text[last:pos] + f"[{lab}]"
                last = pos
            parts.append(built + text[last:])
        out.append(" ".join(parts))
    return "\n".join(out).rstrip() + "\n"
