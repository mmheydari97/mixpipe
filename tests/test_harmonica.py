"""Tests for the harmonica tab (12-hole C chromatic, solo tuning).

Run with:  PYTHONPATH=src python -m pytest tests/test_harmonica.py -q
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("librosa")

from mixpipe.chords.harmonica import (HIGHEST, LOWEST, choose_fingering, fingerings,  # noqa: E402
                                      fit_octave, harmonica_view, note_from_display,
                                      parse_note, to_harp_text)
from mixpipe.chords.melody import extract_melody  # noqa: E402
from mixpipe.chords.sheet import apply_settings, view  # noqa: E402

SR = 44_100


def tabs(midi):
    return [f["tab"] for f in fingerings(midi)]


# --------------------------------------------------------------------------- #
# the instrument
# --------------------------------------------------------------------------- #

def test_solo_tuning_layout():
    assert tabs(60) == ["1↑"]                    # C4, the lowest note
    assert tabs(61) == ["1↑●"]
    assert tabs(62) == ["1↓"]
    assert tabs(64) == ["2↑"]
    assert tabs(65) == ["2↓", "2↑●"]             # F: draw, or E blow + button
    assert tabs(67) == ["3↑"]
    assert tabs(69) == ["3↓"]
    assert tabs(70) == ["3↓●"]                   # Bb
    assert tabs(71) == ["4↓"]                    # B
    assert tabs(72) == ["4↑", "5↑", "4↓●"]       # C5 three ways
    assert tabs(84) == ["8↑", "9↑", "8↓●"]
    assert tabs(96) == ["12↑", "12↓●"]
    assert tabs(97) == ["12↑●"]                  # C#7, the highest note
    assert (LOWEST, HIGHEST) == (60, 97)
    assert tabs(59) == [] and tabs(98) == [] and tabs(None) == []
    # every semitone in range is playable
    assert all(fingerings(m) for m in range(LOWEST, HIGHEST + 1))


def test_default_fingering_avoids_button_then_nearest_hole():
    c5 = fingerings(72)
    assert c5[choose_fingering(c5, None)]["tab"] == "4↑"
    assert c5[choose_fingering(c5, 3)]["tab"] == "4↑"
    assert c5[choose_fingering(c5, 6)]["tab"] == "5↑"
    f = fingerings(65)
    assert f[choose_fingering(f, 9)]["tab"] == "2↓"  # no button beats a nearer hole
    assert choose_fingering([], 3) == -1


def test_fit_octave():
    low = [(48, 1.0), (50, 1.0), (52, 1.0), (55, 1.0)]       # C3-G3
    assert fit_octave(low) == 1
    assert fit_octave([(60, 1.0), (72, 1.0)]) == 0
    assert fit_octave([(100, 1.0), (103, 1.0)]) == -1
    assert fit_octave([]) == 0


def test_parse_note():
    assert parse_note("C4") == (60, 4)
    assert parse_note("Bb4") == (70, 4)
    assert parse_note("f#") == (6, None)
    assert parse_note("B♭") == (10, None)
    assert parse_note("Cb4") == (59, 4)
    assert parse_note("H") is None and parse_note("Cm") is None


# --------------------------------------------------------------------------- #
# melody extraction
# --------------------------------------------------------------------------- #

def sung(midi, dur, vibrato=0.0):
    t = np.arange(int(dur * SR)) / SR
    f = 440 * 2 ** ((midi - 69) / 12) * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1, t / 0.02) * np.minimum(1, (dur - t) / 0.03)
    return 0.3 * env * sum(0.5 ** (h - 1) * np.sin(h * phase) for h in range(1, 5))


# (midi or None for silence, seconds, vibrato depth)
TUNE = [(None, 1.0, 0), (60, 0.5, 0), (62, 0.5, 0), (64, 0.5, 0), (64, 0.5, 0),
        (None, 0.3, 0), (67, 1.0, 0), (65, 0.4, 0), (64, 0.6, 0.01)]


@pytest.fixture(scope="module")
def tune():
    parts, truth, t = [], [], 0.0
    for midi, dur, vib in TUNE:
        if midi is None:
            parts.append(np.zeros(int(dur * SR)))
        else:
            parts.append(sung(midi, dur, vib))
            truth.append((t, midi))
        t += dur
    return np.concatenate(parts).astype(np.float32), truth, t


def test_extract_melody_notes(tune):
    y, truth, end = tune
    out = extract_melody(y, SR, [(1.0, end)], key="C:maj")
    notes = out["notes"]
    assert [n["candidates"][0]["midi"] for n in notes] == [m for _, m in truth]
    for n, (t, _) in zip(notes, truth):
        assert abs(n["start"] - t) <= 0.05
    for n in notes:
        midis = [c["midi"] for c in n["candidates"]]
        assert len(set(midis)) == 3
        assert n["candidates"][0]["conf"] == max(c["conf"] for c in n["candidates"])
        assert n["source"] == "vocal" and n["selected"] == 0 and n["custom"] is None
    assert [n["id"] for n in notes] == [f"n{i}" for i in range(len(notes))]


def test_instrument_fallback_where_nobody_sings(tune):
    y, _, end = tune
    out = extract_melody(None, SR, [], fallback=y, key="C:maj")
    assert out["notes"] and all(n["source"] == "instr" for n in out["notes"])
    assert out["notes"][0]["candidates"][0]["midi"] == 60


def test_octave_error_is_corrected():
    from mixpipe.chords.melody import fix_octaves
    from mixpipe.chords.theory import NEUTRAL

    def note(m):
        return {"candidates": [{"label": "", "midi": m, "conf": 0.9},
                               {"label": "", "midi": m + 1, "conf": 0.05},
                               {"label": "", "midi": m - 1, "conf": 0.05}]}
    notes = [note(m) for m in (64, 65, 67, 55, 65, 64)]   # G3 is G4 tracked an octave low
    assert fix_octaves(notes, NEUTRAL) == 1
    assert [c["midi"] for c in notes[3]["candidates"]] == [67, 55, 56]
    assert [n["candidates"][0]["midi"] for n in notes] == [64, 65, 67, 67, 65, 64]


# --------------------------------------------------------------------------- #
# sheet view, edits, export
# --------------------------------------------------------------------------- #

def W(text, start, end):
    return {"text": text, "start": start, "end": end}


def N(nid, midi, start, end, label=None):
    names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    cands = [{"label": f"{names[m % 12]}{m // 12 - 1}", "midi": m, "conf": c}
             for m, c in ((midi, 0.8), (midi + 1, 0.15), (midi - 1, 0.05))]
    return {"id": nid, "start": start, "end": end, "candidates": cands, "selected": 0,
            "custom": None, "fingering": None, "source": "vocal"}


@pytest.fixture
def harp_sheet():
    return {
        "version": 3, "id": "s", "title": "Test", "key": "F:maj", "key_label": "F major",
        "tempo": 120, "beat_period": 0.5, "beat_times": [], "transpose": 0, "capo": 0,
        "meter": {"time_signature": "4/4", "beats_per_bar": 4, "downbeat": 0, "detected": None},
        "lines": [{"words": [W("here", 1.0, 1.4), W("comes", 1.5, 1.9), W("the", 2.0, 2.2),
                             W("sun", 2.3, 3.2)], "gap_before": False}],
        "chords": [],
        "melody": {"octave": 0, "notes": [
            N("n0", 55, 1.0, 1.4),   # G3 - sung an octave below the harmonica
            N("n1", 57, 1.5, 1.9),   # A3
            N("n2", 58, 2.0, 2.2),   # Bb3
            N("n3", 60, 2.3, 2.7),   # C4
            N("n4", 60, 2.8, 3.2),
            N("n5", 53, 5.0, 5.5),   # F3, after the line: an instrumental row
        ]},
    }


def test_view_fits_octave_and_tabs(harp_sheet):
    hv = view(harp_sheet)["harmonica"]
    assert "melody" not in view(harp_sheet)
    assert hv["auto_octave"] == 1 and hv["octave"] == 1
    notes = {n["id"]: n for n in hv["notes"]}
    assert [notes[f"n{i}"]["tab"] for i in range(6)] == ["3↑", "3↓", "3↓●", "4↑", "4↑", "2↓"]
    assert notes["n2"]["display"] == "B♭" and notes["n2"]["name"] == "B♭4"
    assert notes["n3"]["fingerings"][1]["tab"] == "5↑"
    assert hv["in_range"] == 1.0
    lyric = next(r for r in hv["rows"] if r["type"] == "lyric")
    assert [w["chords"] for w in lyric["words"]] == [["n0"], ["n1"], ["n2"], ["n3", "n4"]]
    assert any(r["type"] == "instrumental" and r["chords"] == ["n5"] for r in hv["rows"])


def test_no_melody_yet(harp_sheet):
    del harp_sheet["melody"]
    assert view(harp_sheet)["harmonica"] is None
    with pytest.raises(ValueError):
        apply_settings(harp_sheet, harmonica_octave=1)


def test_octave_and_transpose(harp_sheet):
    apply_settings(harp_sheet, harmonica_octave=1)        # one above the automatic fit
    hv = harmonica_view(harp_sheet)
    assert hv["octave"] == 2 and hv["user_octave"] == 1
    assert next(n for n in hv["notes"] if n["id"] == "n0")["tab"] == "7↑"   # G5
    with pytest.raises(ValueError):
        apply_settings(harp_sheet, harmonica_octave=9)
    apply_settings(harp_sheet, harmonica_octave=0, transpose=2)
    hv = harmonica_view(harp_sheet)
    n0 = next(n for n in hv["notes"] if n["id"] == "n0")
    assert n0["midi"] == 55 + 2 + 12 and n0["display"] == "A"


def test_custom_note_round_trips(harp_sheet):
    harp_sheet["transpose"] = 2
    note = harp_sheet["melody"]["notes"][0]          # G3 sung, shown as A4
    assert note_from_display("B", harp_sheet, note) == "A3"     # nearest B to A4 is B4
    assert note_from_display("C#5", harp_sheet, note) == "B3"
    assert note_from_display("rest", harp_sheet, note) == "rest"
    assert note_from_display("", harp_sheet, note) is None
    with pytest.raises(ValueError):
        note_from_display("H2", harp_sheet, note)
    note["custom"] = note_from_display("B", harp_sheet, note)
    n0 = next(n for n in harmonica_view(harp_sheet)["notes"] if n["id"] == "n0")
    assert n0["display"] == "B" and n0["tab"] == "4↓"
    note["custom"] = "rest"
    n0 = next(n for n in harmonica_view(harp_sheet)["notes"] if n["id"] == "n0")
    assert n0["rest"] and n0["tab"] is None


def test_out_of_range_flagged(harp_sheet):
    # F#6: fine on its own, but the low melody moves everything up an octave
    harp_sheet["melody"]["notes"].append(N("n6", 90, 6.0, 6.5))
    hv = harmonica_view(harp_sheet)
    n6 = next(n for n in hv["notes"] if n["id"] == "n6")
    assert not n6["in_range"] and n6["tab"] is None and hv["in_range"] < 1.0


def test_instrument_notes_fit_their_own_octave(harp_sheet):
    bass = N("n6", 40, 6.0, 9.0)        # a long E2 guessed from the instruments
    bass["source"] = "instr"
    harp_sheet["melody"]["notes"].append(bass)
    hv = harmonica_view(harp_sheet)
    assert (hv["octave"], hv["instr_octave"]) == (1, 2)   # the sung notes decide "octave"
    n6 = next(n for n in hv["notes"] if n["id"] == "n6")
    assert n6["in_range"] and n6["tab"] == "2↑"           # E4
    assert note_from_display("F", harp_sheet, bass) == "F2"


def test_harp_text_export(harp_sheet):
    text = to_harp_text(harp_sheet)
    lines = text.splitlines()
    assert lines[0] == "Test"
    i = lines.index("here comes the sun")
    assert lines[i - 2].split() == ["G", "A", "B♭", "C", "C"]
    assert lines[i - 1].split() == ["3↑", "3↓", "3↓●", "4↑", "4↑"]
    assert lines[i - 2].index("A") == lines[i].index("comes")
    assert "F 2↓" in text
    assert "1 octave up" in text


# --------------------------------------------------------------------------- #
# web API
# --------------------------------------------------------------------------- #

def fake_melody(vocal, sr, regions, fallback=None, key=None, progress=None):
    notes = [N("n0", 67, 4.1, 4.4), N("n1", 69, 4.6, 4.9), N("n2", 72, 5.1, 5.4)]
    return {"notes": notes}


def test_api_harmonica_flow(tmp_path, wav_bytes):
    from fastapi.testclient import TestClient

    from mixpipe.web.server import create_app
    from test_chords import fake_separator, fake_transcriber

    t = np.arange(SR * 8) / SR
    y = (0.3 * np.sin(2 * np.pi * 262 * t)).astype(np.float32)
    calls = []

    def melody(*a, **k):
        calls.append(1)
        return fake_melody(*a, **k)

    client = TestClient(create_app(tmp_path, separator=fake_separator,
                                   transcriber=fake_transcriber, melody_extractor=melody,
                                   run_inline=True))
    sid = client.post("/api/songs", files=[("files", ("s.wav", wav_bytes(y, SR), "audio/wav"))],
                      data={"language": "en"}).json()["id"]
    song = client.get(f"/api/songs/{sid}").json()
    hv = song["harmonica"]
    assert [n["tab"] for n in hv["notes"]] == ["3↑", "3↓", "4↑"]
    assert calls == [1]

    v = client.patch(f"/api/songs/{sid}/notes/n2", json={"fingering": 1}).json()
    assert v["harmonica"]["notes"][2]["tab"] == "5↑"
    assert client.patch(f"/api/songs/{sid}/notes/n2", json={"fingering": 7}).status_code == 400
    v = client.patch(f"/api/songs/{sid}/notes/n1", json={"selected": 1}).json()
    assert v["harmonica"]["notes"][1]["midi"] == 70
    v = client.patch(f"/api/songs/{sid}/notes/n1", json={"custom": "F#"}).json()
    assert v["harmonica"]["notes"][1]["midi"] == 66
    assert client.patch(f"/api/songs/{sid}/notes/n1", json={"custom": "zz"}).status_code == 400
    assert client.patch(f"/api/songs/{sid}/notes/n9", json={"selected": 0}).status_code == 404
    v = client.patch(f"/api/songs/{sid}/settings", json={"harmonica_octave": 1}).json()
    assert v["harmonica"]["notes"][0]["tab"] == "7↑"

    r = client.get(f"/api/songs/{sid}/export?fmt=harp")
    assert r.status_code == 200 and "7↑" in r.text
    assert 'filename="' + sid + '.harp.txt"' in r.headers["content-disposition"]

    # a song analysed before the harmonica tab existed: chord edits must survive
    cid = song["chords"][0]["id"]
    client.patch(f"/api/songs/{sid}/chords/{cid}", json={"custom": "Cadd9"})
    path = tmp_path / sid / "sheet.json"
    raw = json.loads(path.read_text())
    del raw["melody"]
    path.write_text(json.dumps(raw))
    assert client.get(f"/api/songs/{sid}").json()["harmonica"] is None
    assert client.post(f"/api/songs/{sid}/melody").status_code == 200
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "done" and st["stage"] == "melody", st
    again = client.get(f"/api/songs/{sid}").json()
    assert again["harmonica"] and len(again["harmonica"]["notes"]) == 3
    assert next(c for c in again["chords"] if c["id"] == cid)["custom"] == "Cadd9"
    assert calls == [1, 1]   # re-detect ignores the cache


# --------------------------------------------------------------------------- #
# other harmonicas: diatonic, other keys
# --------------------------------------------------------------------------- #

from mixpipe.chords.harmonica import (Instrument, add_note, get_instrument,  # noqa: E402
                                      instrument_of, merge_notes, position_name,
                                      split_note, suggest_instruments)


def dtabs(midi, key="C", kind="diatonic10"):
    return [f["tab"] for f in get_instrument(kind, key).fingerings(midi)]


def test_diatonic_layout_matches_a_c_harp_chart():
    c = get_instrument("diatonic10", "C")
    assert (c.lowest, c.highest) == (60, 96)
    assert dtabs(60) == ["1↑"] and dtabs(62) == ["1↓"] and dtabs(64) == ["2↑"]
    assert set(dtabs(67)) == {"2↓", "3↑"}
    assert dtabs(65) == ["2↓″"]                  # F4: 2 draw, whole-step bend
    assert dtabs(66) == ["2↓′"]                  # F#4: half-step bend
    assert dtabs(68) == ["3↓‴"]                  # A♭4: 3 draw, 1½ steps
    assert dtabs(69) == ["3↓″"] and dtabs(70) == ["3↓′"] and dtabs(71) == ["3↓"]
    assert dtabs(61) == ["1↓′"]                  # C#4
    assert dtabs(94) == ["10↑″"] and dtabs(95) == ["10↑′"]   # blow bends on hole 10
    assert dtabs(63) == [] and dtabs(75) == []   # E♭ needs an overblow: not modelled
    assert dtabs(59) == [] and dtabs(97) == []


def test_unbent_notes_are_listed_first():
    f = get_instrument("diatonic10", "C").fingerings(67)       # G4: 3 blow, 2 draw
    assert [x["bend"] for x in f] == [0, 0]
    e5 = get_instrument("diatonic10", "C").fingerings(76)      # E5: 5 blow beats a 5 draw bend
    assert e5[0]["tab"] == "5↑" and e5[1]["bend"] == 1
    assert choose_fingering(e5, 9) == 0          # no bend wins even when the bent hole is nearer


def test_harp_keys_shift_the_layout():
    assert get_instrument("diatonic10", "G").lowest == 55          # G3, a fourth below C
    assert get_instrument("diatonic10", "A").lowest == 57
    assert get_instrument("diatonic10", "Db").lowest == 61
    assert get_instrument("diatonic10", "F#").lowest == 66
    assert get_instrument("chromatic12", "G").highest == HIGHEST - 5
    # the same tab sounds different notes: 4 blow is the key's high C-equivalent
    assert dtabs(72 - 5, "G") == ["4↑"]
    assert get_instrument("diatonic10", 7) is get_instrument("diatonic10", "G")
    with pytest.raises(ValueError):
        get_instrument("tremolo", "C")
    with pytest.raises(ValueError):
        get_instrument("diatonic10", "H")
    assert Instrument("chromatic12", "C").name == "C 12-hole chromatic, solo tuning"


def test_fit_octave_counts_only_playable_notes():
    c = get_instrument("diatonic10", "C")
    # E♭4 and E♭5 need an overblow, E♭6 is a bent 8 blow: a long E♭5 is moved up an octave
    assert fit_octave([(75, 5.0), (74, 1.0)], c) == 1
    assert fit_octave([(62, 1.0), (64, 1.0)], c) == 0           # D4 and E4 are fine where they are
    assert fit_octave([(48 + 2, 1.0), (48 + 4, 1.0)], c) == 1


@pytest.fixture
def cmajor_sheet(harp_sheet):
    """A melody that sits in C major and lies on a C diatonic without bends."""
    harp_sheet["key"], harp_sheet["key_label"] = "C:maj", "C major"
    midis = [60, 64, 67, 72, 67, 64, 60]
    harp_sheet["melody"]["notes"] = [N(f"n{i}", m, 1.0 + 0.4 * i, 1.35 + 0.4 * i)
                                     for i, m in enumerate(midis)]
    return harp_sheet


def test_diatonic_view_and_settings(cmajor_sheet):
    apply_settings(cmajor_sheet, harmonica_type="diatonic10", harmonica_key="C")
    assert instrument_of(cmajor_sheet).name.startswith("C 10-hole diatonic")
    hv = harmonica_view(cmajor_sheet)
    assert hv["instrument"] == {"type": "diatonic10", "key": "C", "name": "C 10-hole diatonic (Richter)",
                                "short_name": "C diatonic", "diatonic": True}
    # G4 is 2 draw or 3 blow: the hole nearest the previous note wins each time
    assert [n["tab"] for n in hv["notes"]] == ["1↑", "2↑", "2↓", "4↑", "3↑", "2↑", "1↑"]
    assert hv["octave"] == 0 and hv["in_range"] == 1.0
    assert hv["range"] == ["C4", "C7"] and hv["fast"] is False
    assert position_name(cmajor_sheet, instrument_of(cmajor_sheet)) == "1st (straight harp)"
    # changing harp resets pinned fingerings
    cmajor_sheet["melody"]["notes"][3]["fingering"] = 0
    apply_settings(cmajor_sheet, harmonica_key="G")
    assert cmajor_sheet["melody"]["notes"][3]["fingering"] is None
    assert instrument_of(cmajor_sheet).kind == "diatonic10" and instrument_of(cmajor_sheet).pc == 7
    with pytest.raises(ValueError):
        apply_settings(cmajor_sheet, harmonica_type="kazoo")
    with pytest.raises(ValueError):
        apply_settings(cmajor_sheet, harmonica_key="Q")


def test_suggestions_rank_harps_by_how_little_they_bend(cmajor_sheet):
    best = suggest_instruments(cmajor_sheet)
    assert best[0]["key"] == "C" and best[0]["unbent"] == 1.0 and best[0]["playable"] == 1.0
    assert best[0]["position"] == "1st (straight harp)"
    # cross harp for a song in C is the F harp
    f = next(r for r in suggest_instruments(cmajor_sheet, limit=12) if r["key"] == "F")
    assert f["position"] == "2nd (cross harp)"
    assert [r["unbent"] for r in best] == sorted((r["unbent"] for r in best), reverse=True)
    assert all(0.0 <= r["playable"] <= 1.0 for r in best)
    del cmajor_sheet["melody"]
    assert suggest_instruments(cmajor_sheet) == []


def test_diatonic_text_export(cmajor_sheet):
    apply_settings(cmajor_sheet, harmonica_type="diatonic10", harmonica_key="C")
    text = to_harp_text(cmajor_sheet)
    assert "C 10-hole diatonic (Richter) harmonica" in text and "bend" in text
    assert "Position: 1st (straight harp)" in text
    # a bent note shows its primes
    cmajor_sheet["melody"]["notes"].append(N("n9", 65, 5.0, 5.4))     # F4
    assert "2↓″" in to_harp_text(cmajor_sheet)


# --------------------------------------------------------------------------- #
# editing notes
# --------------------------------------------------------------------------- #

def ids(sheet):
    return [n["id"] for n in sorted(sheet["melody"]["notes"], key=lambda n: n["start"])]


def test_split_note(harp_sheet):
    new = split_note(harp_sheet, "n3", 2.5)
    notes = {n["id"]: n for n in harp_sheet["melody"]["notes"]}
    assert (notes["n3"]["start"], notes["n3"]["end"]) == (2.3, 2.5)
    assert (notes[new]["start"], notes[new]["end"]) == (2.5, 2.7)
    assert notes[new]["candidates"] == notes["n3"]["candidates"] and new == "n6"
    assert notes[new]["candidates"] is not notes["n3"]["candidates"]
    mid = split_note(harp_sheet, "n4")
    assert abs(next(n for n in harp_sheet["melody"]["notes"] if n["id"] == mid)["start"] - 3.0) < 1e-6
    for bad in (2.3, 2.31, 9.0):
        with pytest.raises(ValueError):
            split_note(harp_sheet, "n0", bad)
    with pytest.raises(ValueError):
        split_note(harp_sheet, "nope")


def test_merge_notes(harp_sheet):
    notes = harp_sheet["melody"]["notes"]
    notes[3]["custom"] = "D4"                     # the longer note decides the pitch... equal: first wins
    merged = merge_notes(harp_sheet, ["n3", "n4"])
    assert merged == "n3"
    n3 = next(n for n in harp_sheet["melody"]["notes"] if n["id"] == "n3")
    assert (n3["start"], n3["end"]) == (2.3, 3.2) and "n4" not in ids(harp_sheet)
    with pytest.raises(ValueError):
        merge_notes(harp_sheet, ["n0", "n2"])      # not neighbours (n1 between)
    with pytest.raises(ValueError):
        merge_notes(harp_sheet, ["n0"])
    # the longer note's pitch wins
    harp_sheet["melody"]["notes"][0]["end"] = 1.1
    harp_sheet["melody"]["notes"][1]["candidates"][0]["midi"] = 56
    assert merge_notes(harp_sheet, ["n0", "n1"]) == "n0"
    assert next(n for n in harp_sheet["melody"]["notes"] if n["id"] == "n0")["candidates"][0]["midi"] == 56


def test_add_note(harp_sheet):
    nid = add_note(harp_sheet, 4.0, 4.5, "D4")
    note = next(n for n in harp_sheet["melody"]["notes"] if n["id"] == nid)
    # typed as shown on the tab: this melody is shown an octave up, so D4 is stored as D3
    assert note["source"] == "user" and note["candidates"][0]["midi"] == 50
    assert nid == "n6" and ids(harp_sheet)[-2] == "n6"
    hv = harmonica_view(harp_sheet)
    shown = next(n for n in hv["notes"] if n["id"] == nid)
    assert shown["display"] == "D" and shown["midi"] == 62
    with pytest.raises(ValueError, match="overlaps"):
        add_note(harp_sheet, 2.4, 2.9, "C")
    with pytest.raises(ValueError, match="at least"):
        add_note(harp_sheet, 7.0, 7.01, "C")
    with pytest.raises(ValueError, match="isn't a note"):
        add_note(harp_sheet, 7.0, 7.5, "zz")
    with pytest.raises(ValueError, match="isn't a note"):
        add_note(harp_sheet, 7.0, 7.5, "rest")
    # octave-less names land near the melody, with the user's transpose in mind
    harp_sheet["transpose"] = 2
    nid2 = add_note(harp_sheet, 8.0, 8.5, "B")
    shown = next(n for n in harmonica_view(harp_sheet)["notes"] if n["id"] == nid2)
    assert shown["display"] == "B"


def test_api_note_editing_and_harmonica_settings(tmp_path, wav_bytes):
    from fastapi.testclient import TestClient

    from mixpipe.web.server import create_app
    from test_chords import fake_separator, fake_transcriber

    t = np.arange(SR * 8) / SR
    y = (0.3 * np.sin(2 * np.pi * 262 * t)).astype(np.float32)
    seen = []

    def melody(vocal, sr, regions, fallback=None, key=None, progress=None, fast=False):
        seen.append(fast)
        return fake_melody(vocal, sr, regions, fallback, key, progress)

    client = TestClient(create_app(tmp_path, separator=fake_separator,
                                   transcriber=fake_transcriber, melody_extractor=melody,
                                   run_inline=True))
    sid = client.post("/api/songs", files=[("files", ("s.wav", wav_bytes(y, SR), "audio/wav"))],
                      data={"language": "en"}).json()["id"]
    post = lambda body: client.post(f"/api/songs/{sid}/notes", json=body)  # noqa: E731

    v = post({"action": "split", "id": "n0", "at": 4.25}).json()
    assert len(v["harmonica"]["notes"]) == 4
    v = post({"action": "merge", "ids": ["n0", "n3"]}).json()
    assert len(v["harmonica"]["notes"]) == 3
    v = post({"action": "add", "start": 6.0, "end": 6.5, "note": "C5"}).json()
    assert v["harmonica"]["notes"][-1]["source"] == "user"
    assert post({"action": "add", "start": 4.2, "end": 4.9, "note": "C5"}).status_code == 400   # overlaps
    assert post({"action": "split"}).status_code == 400
    assert post({"action": "add", "note": "C"}).status_code == 400
    assert post({"action": "explode"}).status_code == 400
    assert post({"action": "split", "id": "n99"}).status_code == 400

    v = client.patch(f"/api/songs/{sid}/settings",
                     json={"harmonica_type": "diatonic10", "harmonica_key": "G"}).json()
    assert v["harmonica"]["instrument"]["short_name"] == "G diatonic"
    assert client.patch(f"/api/songs/{sid}/settings", json={"harmonica_key": "?"}).status_code == 400
    assert client.get(f"/api/songs/{sid}/export?fmt=harp").text.count("G 10-hole diatonic") == 1

    # re-detecting keeps the chosen harmonica; fast notes reach the extractor
    assert client.post(f"/api/songs/{sid}/melody", json={"fast": True}).status_code == 200
    again = client.get(f"/api/songs/{sid}").json()
    assert again["harmonica"]["instrument"]["key"] == "G" and again["harmonica"]["fast"] is True
    assert seen == [False, True]

    # pin a note as passing, then let the tab decide again
    v = client.patch(f"/api/songs/{sid}/notes/n0", json={"main": False}).json()
    n0 = next(n for n in v["harmonica"]["notes"] if n["id"] == "n0")
    assert n0["level"] == 0 and n0["main_pinned"] and n0["hold"] is None
    v = client.patch(f"/api/songs/{sid}/notes/n0", json={"auto_main": True}).json()
    n0 = next(n for n in v["harmonica"]["notes"] if n["id"] == "n0")
    assert n0["level"] >= 1 and not n0["main_pinned"]


# --------------------------------------------------------------------------- #
# melody extraction: polyphonic fallback, fast notes
# --------------------------------------------------------------------------- #

def _tone(midi, dur, amp=0.3, harmonics=6):
    f = 440 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(dur * SR)) / SR
    return sum(amp * 0.6 ** (h - 1) * np.sin(2 * np.pi * f * h * t)
               for h in range(1, harmonics + 1) if f * h < SR / 2)


def _faded(midi, dur):
    n = int(dur * SR)
    env = np.ones(n)
    k = int(0.01 * SR)
    env[:k], env[-k:] = np.linspace(0, 1, k), np.linspace(1, 0, k)
    return _tone(midi, dur) * env


def test_instrument_fallback_follows_a_lead_over_chords():
    tune = [(72, .5), (74, .5), (76, .5), (77, .5), (79, 1.0), (76, .5), (72, .5), (74, .5),
            (71, .5), (72, 1.0)]
    lead = np.concatenate([_faded(m, d) for m, d in tune])
    n = len(lead)
    comp = np.zeros(n)
    step = int(1.25 * SR)
    for i, chord in enumerate([[60, 64, 67], [59, 62, 67], [60, 64, 69], [60, 65, 69]] * 3):
        a, b = i * step, min(n, (i + 1) * step)
        if a >= n:
            break
        for m in chord:
            comp[a:b] += _tone(m, (b - a) / SR, 0.12)[:b - a]
    mix = lead + comp
    mix = (mix / np.abs(mix).max() * 0.8).astype(np.float32)
    out = extract_melody(None, SR, [], fallback=mix, key="C:maj")
    truth, t0 = [], 0.0
    for m, d in tune:
        truth.append((t0, t0 + d, m))
        t0 += d
    hit = total = 0.0
    for a, b, m in truth:
        total += b - a
        for note in out["notes"]:
            if note["end"] > a and note["start"] < b and note["candidates"][0]["midi"] == m:
                hit += min(b, note["end"]) - max(a, note["start"])
    assert hit / total > 0.85
    assert all(n["source"] == "instr" for n in out["notes"])


def test_fast_mode_keeps_quick_notes():
    parts = [np.zeros(int(0.5 * SR))]
    for m, d in [(60, 0.4), (64, 0.07), (62, 0.07), (60, 0.07), (62, 0.4)]:
        parts.append(sung(m, d))
    y = np.concatenate(parts).astype(np.float32)
    regions = [(0.5, len(y) / SR)]
    normal = extract_melody(y, SR, regions, key="C:maj")["notes"]
    fast = extract_melody(y, SR, regions, key="C:maj", fast=True)["notes"]
    assert len(fast) > len(normal)
    assert len(fast) >= 4


# --------------------------------------------------------------------------- #
# main notes, harmonica chords
# --------------------------------------------------------------------------- #

from mixpipe.chords.harmonica import DEFAULT_INSTRUMENT, harp_chords, set_main  # noqa: E402
from mixpipe.chords.theory import chord_pcs  # noqa: E402


def C(cid, label, start, end):
    return {"id": cid, "start": start, "end": end, "candidates": [{"label": label, "conf": 1.0}],
            "selected": 0, "custom": None}


@pytest.fixture
def beat_sheet(harp_sheet):
    """4/4 at 120 bpm from 1.0 s (a downbeat), over a C chord."""
    harp_sheet["key"], harp_sheet["key_label"] = "C:maj", "C major"
    harp_sheet["beat_times"] = [1.0 + 0.5 * i for i in range(9)]
    harp_sheet["chords"] = [C("c0", "C", 1.0, 7.0)]
    harp_sheet["melody"]["notes"] = [
        N("n0", 72, 1.0, 1.9),    # C5 held over beats 1-2: main
        N("n1", 74, 1.9, 2.0),    # D5 quick pickup: passing
        N("n2", 76, 2.0, 2.45),   # E5 on beat 3 (strong)
        N("n3", 77, 2.5, 2.6),    # F5 on beat 4, short: passing
        N("n4", 79, 2.6, 2.95),   # G5 fills beat 4 and is a chord tone: main of the beat
        N("n5", 84, 3.0, 3.4),    # C6 on the next downbeat
        N("n6", 72, 6.0, 6.4),    # after a long rest
    ]
    return harp_sheet


def levels(sheet):
    hv = harmonica_view(sheet)
    return {n["id"]: n["level"] for n in hv["notes"]}, {n["id"]: n["hold"] for n in hv["notes"]}


def test_main_notes_per_beat_and_strong_beat(beat_sheet):
    lv, hold = levels(beat_sheet)
    assert lv == {"n0": 2, "n1": 0, "n2": 2, "n3": 0, "n4": 1, "n5": 2, "n6": 2}
    # hold a main note until the next one of its level...
    assert hold["n0"] == {"beat": 2.0, "strong": 2.0}
    assert hold["n2"] == {"beat": 2.6, "strong": 3.0}
    assert hold["n4"] == {"beat": 3.0}
    assert hold["n1"] is None
    # ...but not through a long rest
    assert hold["n5"]["strong"] == 3.4


def test_pinning_main_notes(beat_sheet):
    notes = {n["id"]: n for n in beat_sheet["melody"]["notes"]}
    set_main(notes["n3"], True)
    set_main(notes["n0"], False)
    lv, _ = levels(beat_sheet)
    assert lv["n3"] == 2 and lv["n0"] == 0
    assert lv["n1"] >= 1           # beat 2 goes to the next best note
    set_main(notes["n0"], None)
    assert levels(beat_sheet)[0]["n0"] == 2
    # edits that make notes start them unpinned
    assert notes["n0"]["main"] is None


def test_chord_pcs():
    assert chord_pcs("Am7") == (9, 0, 4, 7)
    assert chord_pcs("C/G") == (0, 4, 7)
    assert chord_pcs("C/Bb") == (0, 4, 7, 10)
    assert chord_pcs("Fm11") == (5, 8, 0)
    assert chord_pcs("Bbm7b5") == (10, 1, 4, 8)
    assert chord_pcs("N.C.") is None


def chords_on(labels, instrument=DEFAULT_INSTRUMENT, notes=()):
    sheet = {"key": "C:maj", "transpose": 0,
             "chords": [C(f"c{i}", lab, 2.0 * i, 2.0 * i + 2) for i, lab in enumerate(labels)]}
    return {c["label"]: c for c in harp_chords(sheet, instrument, list(notes))}


def test_harp_chords_on_c_chromatic():
    got = chords_on(["C", "Am", "F", "Dm", "Db", "G7", "E"])
    assert (got["C"]["tab"], got["C"]["full"], got["C"]["notes"]) == ("1-3↑", True, ["C", "E", "G"])
    assert (got["Am"]["tab"], got["Am"]["full"], got["Am"]["notes"]) == ("1-2↑", False, ["C", "E"])
    assert (got["F"]["tab"], got["F"]["full"]) == ("2-3↓", False)
    assert (got["Dm"]["tab"], got["Dm"]["full"]) == ("1-3↓", True)
    assert (got["D♭"]["tab"], got["D♭"]["full"], got["D♭"]["button"]) == ("1-3↑●", True, True)
    assert got["G7"]["tab"] == "4-6↓" and got["G7"]["notes"] == ["B", "D", "F"]
    assert got["E"]["tab"] is None                      # no two reeds of E major sit together
    assert all(c["id"] == f"h{c['chord']}" for c in got.values())


def test_harp_chord_sits_by_the_melody():
    note = {"start": 0.0, "end": 1.5, "fingering_index": 0, "fingerings": [{"hole": 10}]}
    assert chords_on(["C"], notes=[note])["C"]["tab"] == "9-11↑"


def test_harp_chords_on_c_diatonic():
    got = chords_on(["G", "C", "Dm"], get_instrument("diatonic10", "C"))
    assert (got["G"]["tab"], got["G"]["full"]) == ("1-3↓", True)
    assert (got["C"]["tab"], got["C"]["full"]) == ("1-3↑", True)
    assert got["Dm"]["full"] and got["Dm"]["breath"] == "draw"


def test_view_and_export_show_chords_and_passing_notes(beat_sheet):
    beat_sheet["chords"] = [C("c0", "C", 1.0, 2.0), C("c1", "Am", 2.0, 7.0)]
    beat_sheet["melody"]["octave"] = -1                # keep it near the lyric test's range
    hv = harmonica_view(beat_sheet)
    assert [c["id"] for c in hv["chords"]] == ["hc0", "hc1"]
    lyric = next(r for r in hv["rows"] if r["type"] == "lyric")
    placed = [i for w in lyric["words"] for i in w["chords"]]
    assert "hc0" in placed and "hc1" in placed
    text = to_harp_text(beat_sheet)
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("here "))
    assert lines[i - 3].startswith("C 1-3↑ ") and "Am 1-2↑~" in lines[i - 3]
    assert lines[i - 3].index("Am") == lines[i - 2].index("E")      # over the note it starts with
    assert "(D)" in lines[i - 2]                       # the passing pickup
    assert "passing note" in text and "holes 1 to 3 together" in text
