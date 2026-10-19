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
