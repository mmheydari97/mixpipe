"""The guitar tab (chords/guitar.py, chords/guitar_transcribe.py): strums vs
picked notes, lead vs rhythm, fingering, rhythm on the beat grid, sections,
edits and the routes. Synthetic notes and a fake transcriber; no model.

Run with:  PYTHONPATH=src:tests python -m pytest tests/test_guitar.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from mixpipe.chords import guitar
from mixpipe.chords.guitar_transcribe import drop_ghosts, merge_bent_pairs, merge_wobbles, stereo_position

BEAT = 0.5


def sheet_with(notes=None, beats=64, bpb=4, words=(), chords=()):
    return {
        "title": "t", "key": "E:min", "transpose": 0, "capo": 0, "duration": beats * BEAT,
        "beat_times": [i * BEAT for i in range(beats)], "beat_period": BEAT,
        "meter": {"beats_per_bar": bpb, "downbeat": 0, "time_signature": f"{bpb}/4"},
        "lines": [{"words": [{"text": "la", "start": a, "end": b} for a, b in words]}] if words else [],
        "chords": [{"id": f"c{i}", "start": a, "end": b, "candidates": [{"label": lab}], "selected": 0}
                   for i, (a, b, lab) in enumerate(chords)],
        "guitar": guitar.from_transcription(notes or []),
    }


def n(start, midi, dur=0.4, amp=0.6, pan=0.0, centre=0.9):
    return {"start": start, "end": start + dur, "midi": midi, "amp": amp, "pan": pan, "centre": centre,
            "bend": 0.0, "vibrato": False}


# ---- transcription clean-up --------------------------------------------------


def test_overtones_are_dropped_but_octaves_struck_together_kept():
    notes = [n(0.0, 40, amp=0.8), n(0.01, 52, amp=0.4), n(0.02, 59, amp=0.3),   # E2 + its 2nd, 3rd harmonics
             n(1.0, 52, amp=0.6), n(1.0, 64, amp=0.6),                          # an octave, equally strong
             n(2.0, 60, dur=0.05, amp=0.2)]                                      # a blip
    kept = drop_ghosts(notes)
    assert [(k["start"], k["midi"]) for k in kept] == [(0.0, 40), (1.0, 52), (1.0, 64)]


def test_a_bend_heard_as_quick_notes_becomes_one_bent_note():
    wobble = [n(1.0 + 0.1 * i, 67 + (i % 2), dur=0.09) for i in range(6)]   # G G# G G# G G#
    run = [n(3.0 + 0.1 * i, m, dur=0.09) for i, m in enumerate((64, 67, 69))]  # a real run stays
    out = merge_wobbles(wobble + run)
    assert len(out) == 4
    bent = out[0]
    assert (bent["midi"], bent["alt"], bent["bend"], bent["vibrato"]) == (67, 68, 1.0, True)
    assert bent["start"] == 1.0 and bent["end"] == pytest.approx(1.59)
    s = sheet_with(out)
    v = guitar.guitar_view(s)
    assert [c["label"] for c in v["notes"][0]["candidates"]][:2] == ["G4", "G#4"]
    assert v["notes"][0]["tech"] == ["½", "~"]


def test_a_bend_heard_as_two_notes_at_once():
    out = merge_bent_pairs([n(1.0, 67, 0.3), n(1.02, 68, 0.2), n(2.0, 64), n(2.0, 67)])
    assert [(x["midi"], x.get("alt"), x.get("bend")) for x in out] == [(67, 68, 1.0), (64, None, 0.0), (67, None, 0.0)]


def test_stereo_position_tells_centre_from_sides():
    sr = 22050
    t = np.arange(sr // 2) / sr
    tone = np.sin(2 * np.pi * 330 * t).astype(np.float32)
    centred = np.stack([tone, tone], axis=1)
    rng = np.random.default_rng(0)
    other = np.sin(2 * np.pi * 330 * t + 1.3).astype(np.float32) * 0.9   # a second take, out of phase
    doubled = np.stack([tone, other], axis=1)
    left = np.stack([tone, tone * 0.1], axis=1) + rng.normal(0, 1e-4, (len(t), 2)).astype(np.float32)
    pan_c, centre_c = stereo_position(centred, sr, 0.0, 0.5, 64)
    _, centre_d = stereo_position(doubled, sr, 0.0, 0.5, 64)
    pan_l, _ = stereo_position(left, sr, 0.0, 0.5, 64)
    assert abs(pan_c) < 0.05 and centre_c > 0.95
    assert centre_d < 0.85
    assert pan_l < -0.6


# ---- strums and parts --------------------------------------------------------


def test_strum_vs_picked():
    sweep = [n(0.0 + 0.008 * i, m) for i, m in enumerate((40, 47, 52, 55, 59, 64))]
    pinch = [n(2.0, m) for m in (40, 55, 59, 64)]              # four strings at once, fingers
    triad = [n(4.0 + 0.005 * i, m) for i, m in enumerate((55, 59, 64))]
    s = sheet_with(sweep + pinch + triad, chords=[(0.0, 8.0, "E:min")])
    parts = guitar.assign_parts(s, guitar._pitched(s))
    assert len(parts["strums"]) == 1 and parts["strums"][0]["chord"] == "E:min"
    assert all(parts["part"][i] == "rhythm" for i in parts["strums"][0]["ids"])
    picked = [k for k, v in parts["part"].items() if k not in parts["strum"]]
    assert len(picked) == 7 and all(parts["part"][k] == "lead" for k in picked)


def two_guitars():
    """A centred lead up high over double-tracked rhythm notes panned out."""
    lead = [n(0.25 + 0.5 * i, 67 + (i % 3) * 2, amp=0.6, pan=0.02, centre=0.92) for i in range(16)]
    rhythm = [n(0.5 * i, 40 + (i % 2) * 7, amp=0.5, pan=(-0.5 if i % 2 else 0.5), centre=0.55)
              for i in range(16)]
    return lead, rhythm


def test_two_guitars_split_into_lead_and_rhythm():
    lead, rhythm = two_guitars()
    s = sheet_with(lead + rhythm)
    parts = guitar.assign_parts(s, guitar._pitched(s))
    ids = {(x["start"], x["_midi"]): x["id"] for x in guitar._pitched(s)}
    assert all(parts["part"][ids[(x["start"], x["midi"])]] == "lead" for x in lead)
    assert all(parts["part"][ids[(x["start"], x["midi"])]] == "rhythm" for x in rhythm)
    assert any(w["two"] for w in parts["windows"])


def test_lock_follows_the_chosen_guitar():
    lead, rhythm = two_guitars()
    s = sheet_with(lead + rhythm)
    low = next(x for x in guitar._pitched(s) if x["_midi"] == 40)
    guitar.lock_part(s, low["id"], "lead")           # "no, THIS is the solo"
    parts = guitar.assign_parts(s, guitar._pitched(s))
    assert parts["part"][low["id"]] == "lead"
    high = next(x for x in guitar._pitched(s) if x["_midi"] >= 67)
    assert parts["part"][high["id"]] == "rhythm"
    assert s["guitar"]["lock"]["as"] == "lead"


def test_a_pinned_part_wins():
    lead, rhythm = two_guitars()
    s = sheet_with(lead + rhythm)
    note = next(x for x in guitar._pitched(s) if x["_midi"] == 40)
    guitar.edit_note(s, note["id"], part="lead")
    assert guitar.assign_parts(s, guitar._pitched(s))["part"][note["id"]] == "lead"


# ---- fingering ---------------------------------------------------------------


def test_positions_and_capo():
    tuning = guitar.tuning_of({})
    assert [p["string"] for p in guitar.positions(64, tuning)] == [1, 2, 3, 4, 5]
    assert guitar.positions(40, tuning) == [{"string": 6, "fret": 0}]
    assert guitar.positions(40, tuning, capo=2) == []
    assert {"string": 1, "fret": 0} in guitar.positions(66, tuning, capo=2)


def test_fingering_keeps_the_hand_in_place():
    # an A minor pentatonic run, A3 to A4
    run = [n(0.5 * i, m) for i, m in enumerate((57, 60, 62, 64, 67, 69))]
    s = sheet_with(run)
    pitched = guitar._pitched(s)
    fing = guitar.fingerings(guitar.onset_groups(pitched), s)
    frets = [fing[x["id"]]["fret"] for x in pitched]
    fretted = [f for f in frets if f > 0]
    assert max(fretted) - min(fretted) <= 4, frets


def test_notes_struck_together_take_one_string_each():
    triad = [n(0.0, m) for m in (64, 67, 71)]          # E4 G4 B4
    s = sheet_with(triad)
    pitched = guitar._pitched(s)
    fing = guitar.fingerings(guitar.onset_groups(pitched), s)
    strings = [fing[x["id"]]["string"] for x in pitched]
    assert len(set(strings)) == 3
    frets = [fing[x["id"]]["fret"] for x in pitched if fing[x["id"]]["fret"] > 0]
    assert not frets or max(frets) - min(frets) <= guitar.CHORD_SPAN


def test_pinned_fingering_is_kept():
    s = sheet_with([n(0.0, 64)])
    note = guitar._pitched(s)[0]
    opts = guitar.positions(64, guitar.tuning_of(s))
    k = next(i for i, p in enumerate(opts) if p["string"] == 2)
    guitar.edit_note(s, note["id"], fingering=k)
    v = guitar.guitar_view(s)
    assert (v["notes"][0]["string"], v["notes"][0]["fret"]) == (2, 5)


# ---- rhythm on the grid -------------------------------------------------------


def _durations(view):
    return [(it["dur"], it["dots"], it["rest"], it["triplet"]) for bar in view["measures"] for it in bar["items"]]


def ticks_of(item, scale=1):
    base = {"w": 48, "h": 24, "q": 12, "8": 6, "16": 3, "32": 1.5}[item["dur"]] * scale
    base *= 1.5 if item["dots"] else 1
    return base * 2 / 3 if item["triplet"] else base


def test_eighths_and_rests_fill_whole_bars():
    s = sheet_with([n(0.0, 64, 0.2), n(0.25, 67, 0.2), n(1.0, 69, 0.45)])   # 8th, 8th, rest, quarter
    v = guitar.guitar_view(s)
    bar = v["measures"][0]
    assert sum(ticks_of(it) for it in bar["items"]) == 48
    kinds = [(it["dur"], it["rest"]) for it in bar["items"]]
    assert kinds[:4] == [("8", False), ("8", False), ("q", True), ("q", False)]
    assert all(sum(ticks_of(it) for it in b["items"]) == 48 for b in v["measures"])


def test_triplets_are_detected_per_beat():
    trip = [n(1.0 + i * BEAT / 3, 64 + i, 0.15) for i in range(3)]
    s = sheet_with(trip)
    v = guitar.guitar_view(s)
    items = [it for bar in v["measures"] for it in bar["items"] if not it["rest"]]
    assert len(items) == 3 and all(it["triplet"] and it["dur"] == "8" for it in items)


def test_a_long_note_over_the_barline_is_tied():
    s = sheet_with([n(1.5, 64, 1.5)])          # beat 3 of bar 1 to beat 2 of bar 2: 3 beats
    v = guitar.guitar_view(s)
    sounding = [it for bar in v["measures"] for it in bar["items"] if not it["rest"]]
    assert len(sounding) >= 2 and all(it["tie"] for it in sounding[:-1]) and not sounding[-1]["tie"]
    assert sum(ticks_of(it) for it in sounding) == 36


def test_six_eight_uses_eighth_beats():
    s = sheet_with([n(i * BEAT, 64, 0.45) for i in range(6)], bpb=6)
    s["meter"]["time_signature"] = "6/8"
    v = guitar.guitar_view(s)
    items = [it for it in v["measures"][0]["items"] if not it["rest"]]
    assert len(items) == 6 and all(it["dur"] == "8" for it in items)
    assert sum(ticks_of(it, 2) for it in v["measures"][0]["items"]) == 6 * 12


# ---- sections, edits, view ------------------------------------------------------


def test_instrumental_sections():
    s = sheet_with(words=[(10.0, 11.0), (11.5, 12.0), (20.0, 21.0)], beats=64)
    secs = guitar.sections(s, 32.0)
    kinds = [(x["kind"], x["start"], x["end"]) for x in secs]
    assert kinds[0] == ("instrumental", 0.0, 10.0)
    assert ("instrumental", 12.0, 20.0) in kinds
    assert kinds[-1] == ("instrumental", 21.0, 32.0)


def test_edits_and_view():
    s = sheet_with([n(0.0, 64), n(1.0, 67, 0.8)])
    first, second = [x["id"] for x in guitar._pitched(s)]
    guitar.edit_note(s, first, custom="F#")
    assert guitar.guitar_view(s)["notes"][0]["label"] == "F#4"
    new = guitar.split_note(s, second, 1.4)
    assert new.startswith("g") and len(guitar._pitched(s)) == 3
    guitar.delete_note(s, new)
    assert len(guitar._pitched(s)) == 2
    added = guitar.add_note(s, 0.0, 0.4, "B3")              # under the first note: chords are fine
    assert any(x["id"] == added for x in guitar._pitched(s))
    s["transpose"] = 2
    labels = [x["label"] for x in guitar.guitar_view(s)["notes"]]
    assert "G#4" in labels or "A♭4" in labels or "Ab4" in labels
    text = guitar.to_tab_text(s)
    assert text.count("|") >= 12 and "e|" in text


def test_capo_moves_frets():
    s = sheet_with([n(0.0, 66)])
    s["capo"] = 2
    v = guitar.guitar_view(s)
    assert v["capo"] == 2 and v["notes"][0]["fret"] == 0     # F#4 = open top string over capo 2


# ---- routes ------------------------------------------------------------------------


def test_routes(tmp_path, wav_bytes):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mixpipe import songstore
    from mixpipe.chords.sheet import save_sheet
    from mixpipe.web.server import create_app
    from test_splitter import SR, fake_separator, song

    root = tmp_path / "chords"
    sid = songstore.new_song([("song.wav", wav_bytes(song(), SR))], root)
    songstore.set_status(root / sid, "done", "align", 1.0, "done")
    sheet = sheet_with()
    del sheet["guitar"]
    save_sheet(sheet, root / sid / "sheet.json")
    calls = []

    def transcriber(mono, sr):
        calls.append(len(mono))
        return [(0.0, 0.4, 64, 0.6, None), (0.5, 0.9, 67, 0.6, [0, 3, 6, 6, 6, 3]), (0.0, 0.4, 76, 0.2, None)]

    app = create_app(root, run_inline=True, splitter_root=tmp_path / "splitter",
                     stem_separator=fake_separator, guitar_transcriber=transcriber)
    c = TestClient(app)
    assert c.get(f"/api/songs/{sid}/guitar").status_code == 409
    assert c.get(f"/api/songs/{sid}").json()["has_guitar"] is False
    st = c.post(f"/api/songs/{sid}/guitar").json()
    assert st["state"] == "done", st
    assert len(calls) == 1
    v = c.get(f"/api/songs/{sid}/guitar").json()
    assert [x["label"] for x in v["notes"]] == ["E4", "G4"]          # the E5 overtone is gone
    assert v["measures"] and v["time_signature"] == "4/4"
    assert c.get(f"/api/songs/{sid}").json()["has_guitar"] is True
    nid = v["notes"][1]["id"]
    v = c.patch(f"/api/songs/{sid}/guitar/notes/{nid}", json={"custom": "A"}).json()
    assert v["notes"][1]["label"] == "A4"
    v = c.post(f"/api/songs/{sid}/guitar/notes", json={"action": "lock", "id": nid}).json()
    assert v["lock"]["as"] == "lead"
    assert c.post(f"/api/songs/{sid}/guitar/notes", json={"action": "nope"}).status_code == 400
    r = c.get(f"/api/songs/{sid}/export?fmt=tab")
    assert r.status_code == 200 and "e|" in r.text
    # a second run uses the cached notes and keeps the lock
    c.post(f"/api/songs/{sid}/guitar")
    assert len(calls) == 1
    assert c.get(f"/api/songs/{sid}/guitar").json()["lock"] is not None
