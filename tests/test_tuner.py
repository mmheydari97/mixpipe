"""The guitar tuner: tunings from the two switches, string detection, the WebSocket."""

from __future__ import annotations

import numpy as np
import pytest

from mixpipe.chords.theory import guitar_tuning, guitar_tunings
from mixpipe.tuner import StringTracker

SR = 48_000
CHUNK = 2048


def hz(midi: float, a4: float = 440.0) -> float:
    return a4 * 2 ** ((midi - 69) / 12)


def pluck(f: float, secs: float = 1.2, amps=(1, .9, .6, .4, .3, .2), seed: int = 0) -> np.ndarray:
    """A plucked-string stand-in: harmonics that die away, upper ones faster, over a little noise."""
    t = np.arange(int(SR * secs)) / SR
    x = sum(a * np.sin(2 * np.pi * f * (k + 1) * t + k) * np.exp(-t * 2.0 * (k + 1) ** .5)
            for k, a in enumerate(amps))
    noise = 0.002 * np.random.default_rng(seed).standard_normal(t.size)
    return (0.3 * x / np.max(np.abs(x)) + noise).astype(np.float32)


def feed(tracker: StringTracker, sig: np.ndarray) -> list[dict]:
    return [tracker.feed(sig[i:i + CHUNK]) for i in range(0, sig.size - CHUNK + 1, CHUNK)]


def notes(readings: list[dict]) -> list[dict]:
    return [r for r in readings if r["state"] == "note"]


def midis(whole_down=False, drop6=False) -> list[int]:
    return [s["midi"] for s in guitar_tuning(whole_down, drop6)["strings"]]


# --------------------------------------------------------------------- tunings

@pytest.mark.parametrize("whole_down,drop6,name,labels", [
    (False, False, "Standard", ["E2", "A2", "D3", "G3", "B3", "E4"]),
    (True, False, "D standard", ["D2", "G2", "C3", "F3", "A3", "D4"]),
    (False, True, "Drop D", ["D2", "A2", "D3", "G3", "B3", "E4"]),
    (True, True, "Drop C", ["C2", "G2", "C3", "F3", "A3", "D4"]),
])
def test_two_switches_make_the_four_tunings(whole_down, drop6, name, labels):
    t = guitar_tuning(whole_down, drop6)
    assert t["name"] == name
    assert [s["label"] for s in t["strings"]] == labels
    assert [s["string"] for s in t["strings"]] == [6, 5, 4, 3, 2, 1]
    assert len(guitar_tunings()) == 4


# ------------------------------------------------------------ string detection

@pytest.mark.parametrize("whole_down,drop6", [(False, False), (True, False), (False, True), (True, True)])
def test_each_string_is_found_in_tune(whole_down, drop6):
    ms = midis(whole_down, drop6)
    for string, m in zip(range(6, 0, -1), ms):
        got = notes(feed(StringTracker(SR, ms), pluck(hz(m))))
        assert len(got) > 15
        assert {r["string"] for r in got} == {string}
        assert abs(np.median([r["cents"] for r in got])) < 2.0


@pytest.mark.parametrize("cents", [20, -20, 8])
def test_how_far_off_is_measured_in_cents(cents):
    ms = midis()
    for m in (ms[0], ms[3], ms[5]):          # low E, G, high E
        got = notes(feed(StringTracker(SR, ms), pluck(hz(m + cents / 100))))
        assert np.median([r["cents"] for r in got]) == pytest.approx(cents, abs=2.0)


def test_a_low_string_with_a_weak_fundamental_is_not_taken_for_its_octave():
    ms = midis(drop6=True)                   # D2 .. and D3 on string 4
    got = notes(feed(StringTracker(SR, ms), pluck(hz(38), amps=(.3, 1, .7, .5, .3, .2))))
    assert got and {r["string"] for r in got} == {6}


def test_ringing_low_string_with_octave_errors_stays_on_its_string():
    ms = midis(drop6=True)
    tr = StringTracker(SR, ms)
    sig = pluck(hz(38))
    first = notes(feed(tr, sig[: 8 * CHUNK]))
    assert first[-1]["string"] == 6
    # the same note keeps ringing, quieter, but now heard an octave up (D3): no new pluck
    t = np.arange(6 * CHUNK) / SR
    octave_up = (0.05 * np.sin(2 * np.pi * hz(50) * t)).astype(np.float32)
    after = notes(feed(tr, octave_up))
    assert after and {r["string"] for r in after} == {6}
    assert abs(np.median([r["cents"] for r in after])) < 2.0


def test_a_blip_from_another_string_does_not_move_the_display():
    ms = midis()
    tr = StringTracker(SR, ms)
    assert notes(feed(tr, pluck(hz(45))[: 10 * CHUNK]))[-1]["string"] == 5
    blip = pluck(hz(50), secs=0.2)[: 3 * CHUNK]          # one or two chunks of D3
    readings = feed(tr, blip[:CHUNK])
    assert all(r["state"] != "note" or r["string"] == 5 for r in readings)


def test_a_new_pluck_on_another_string_takes_over():
    ms = midis()
    tr = StringTracker(SR, ms)
    feed(tr, pluck(hz(45))[: 10 * CHUNK])
    got = notes(feed(tr, pluck(hz(50))))
    assert got[-1]["string"] == 4


def test_a_locked_string_is_measured_however_far_off():
    ms = midis()
    tr = StringTracker(SR, ms, lock=6)
    got = notes(feed(tr, pluck(hz(40 - 4))))       # low E four semitones slack: nearest is "C2"?
    assert got and {r["string"] for r in got} == {6}
    assert np.median([r["cents"] for r in got]) == pytest.approx(-400, abs=3)
    assert got[-1]["semitones_off"] == -4 and got[-1]["heard"] == "C2" and got[-1]["locked"]


def test_without_the_lock_a_very_slack_string_goes_to_its_neighbour():
    # why the lock exists: A2 tuned 3 semitones flat sounds nearer E2 than A2
    got = notes(feed(StringTracker(SR, midis()), pluck(hz(45 - 3))))
    assert got[-1]["string"] == 6


def test_a4_reference_moves_the_targets():
    ms = midis()
    got = notes(feed(StringTracker(SR, ms, a4=432.0), pluck(hz(45, a4=432.0))))
    assert abs(np.median([r["cents"] for r in got])) < 2.0


def test_silence_is_silent():
    readings = feed(StringTracker(SR, midis()), np.zeros(SR // 2, dtype=np.float32))
    assert {r["state"] for r in readings} == {"silent"}


def test_high_sample_rates_still_reach_the_lowest_string():
    sr = 96_000
    t = np.arange(sr) / sr
    sig = (0.3 * np.sin(2 * np.pi * hz(36) * t) + 0.2 * np.sin(2 * np.pi * hz(48) * t)).astype(np.float32)
    tr = StringTracker(sr, midis(True, True))
    got = notes([tr.feed(sig[i:i + 4096]) for i in range(0, sig.size - 4096, 4096)])
    assert got and got[-1]["string"] == 6 and abs(got[-1]["cents"]) < 2.0


# ------------------------------------------------------------------ websocket

def test_websocket_streams_readings():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mixpipe.web.tuner_server import create_app

    c = TestClient(create_app())
    assert c.get("/").status_code == 200
    sig = pluck(hz(38))
    with c.websocket_connect("/ws") as ws:
        ws.send_bytes(sig[:CHUNK].tobytes())           # no sr yet: ignored
        ws.send_json({"sr": SR, "whole_down": False, "drop6": True, "a4": 440, "lock": None})
        got = []
        for i in range(0, 12 * CHUNK, CHUNK):
            ws.send_bytes(sig[i:i + CHUNK].tobytes())
            got.append(ws.receive_json())
        assert got[-1]["state"] == "note" and got[-1]["string"] == 6
        # the switches change the targets: the same D2 is now 2 semitones under standard's E2
        ws.send_json({"whole_down": False, "drop6": False})
        ws.send_json({"lock": 6})
        later = []
        for i in range(12 * CHUNK, 22 * CHUNK, CHUNK):
            ws.send_bytes(sig[i:i + CHUNK].tobytes())
            later.append(ws.receive_json())
        last = notes(later)[-1]
        assert last["string"] == 6 and last["semitones_off"] == -2 and last["locked"]
