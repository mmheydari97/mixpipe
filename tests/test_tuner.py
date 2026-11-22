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
    """Readings of a string: live ones and, once it locked in, the frozen "tuned" ones."""
    return [r for r in readings if r["state"] in ("note", "tuned")]


def ring(f: float, secs: float = 4.0, cents=None) -> np.ndarray:
    """A long-ringing string (an acoustic, or an electric's sustain); `cents(t)`
    bends it, for a peg being turned while it rings."""
    t = np.arange(int(SR * secs)) / SR
    bend = np.zeros(t.size) if cents is None else cents(t)
    phase = 2 * np.pi * np.cumsum(f * 2 ** (bend / 1200)) / SR
    x = sum(a * np.sin((k + 1) * phase + k) * np.exp(-t * 0.8 * (k + 1) ** .5)
            for k, a in enumerate((1, .9, .6, .4, .3, .2, .15, .1)))
    return 0.3 * x / np.max(np.abs(x))


def amp(clean: np.ndarray, hum: float = 0.0, seed: int = 0) -> np.ndarray:
    """Through a high-gain amp: hard distortion, with mains hum (60 Hz and its
    120 Hz) going in alongside the guitar, and a little noise on top."""
    t = np.arange(clean.size) / SR
    h = hum * np.sin(2 * np.pi * 60 * t) + hum / 2 * np.sin(2 * np.pi * 120 * t)
    noise = 0.003 * np.random.default_rng(seed).standard_normal(t.size)
    return (np.tanh(25 * (clean + h)) * 0.5 + noise).astype(np.float32)


def chorus(x: np.ndarray, rate: float = 0.8, depth_ms: float = 3.0) -> np.ndarray:
    """A chorus pedal: the dry signal mixed with a copy whose delay swings, which
    really does move the pitch up and down a few cents."""
    n = np.arange(x.size)
    delay = (7 + depth_ms * np.sin(2 * np.pi * rate * n / SR)) * SR / 1000
    return (0.5 * x + 0.5 * np.interp(n - delay, n, x, left=0)).astype(np.float32)


def tuned(readings: list[dict]) -> list[int]:
    """Indices of the readings that lock a string in (the first "tuned" of each)."""
    return [i for i, r in enumerate(readings) if r["state"] == "tuned" and r["new"]]


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


# --------------------------------------------------- an electric through an amp

@pytest.mark.parametrize("whole_down,drop6,m", [
    (False, False, 40), (False, False, 45), (False, False, 64), (True, True, 36), (True, False, 43)])
def test_hum_through_distortion_does_not_take_over_the_note(whole_down, drop6, m):
    # before the prefilter, A2 read as string 6 at -600 cents for most of the note
    ms = midis(whole_down, drop6)
    got = notes(feed(StringTracker(SR, ms), amp(ring(hz(m)), hum=0.01)))
    assert len(got) > 40
    assert {r["string"] for r in got} == {6 - ms.index(m)}
    assert max(abs(r["cents"]) for r in got) < 5


def test_hum_alone_is_not_a_note():
    t = np.arange(3 * SR) / SR
    for level in (0.01, 0.05):
        hum = level * np.sin(2 * np.pi * 60 * t) + level / 2 * np.sin(2 * np.pi * 120 * t)
        for sig in (hum.astype(np.float32), amp(np.zeros(t.size), hum=level)):
            readings = feed(StringTracker(SR, midis()), sig)
            assert not notes(readings)


def test_the_dying_tail_does_not_drift():
    # an acoustic's tail used to read 8-36 cents sharp; 15 cents flat keeps it from locking in
    ms = midis()
    sig = ring(hz(40 - 0.15))
    sig = (sig + 0.003 * np.random.default_rng(1).standard_normal(sig.size)).astype(np.float32)
    got = [r for r in feed(StringTracker(SR, ms), sig) if r["state"] == "note"]
    assert len(got) > 60
    assert all(abs(r["cents"] + 15) < 3 for r in got)


@pytest.mark.parametrize("rate", [0.8, 2.0])
def test_a_chorus_wobble_locks_in_at_its_centre(rate):
    ms = midis()
    readings = feed(StringTracker(SR, ms), chorus(amp(ring(hz(45)))))
    live = [r["cents"] for r in readings if r["state"] == "note"]
    assert max(live) > 8 and min(live) < -8               # the needle does swing
    first = tuned(readings)
    assert len(first) == 1
    assert abs(readings[first[0]]["cents"]) < 3


# ------------------------------------------------------- plucks after a pause

def room(secs: float, level: float = 0.004, seed: int = 2) -> np.ndarray:
    return level * np.random.default_rng(seed).standard_normal(int(SR * secs))


def picked(f: float, secs: float = 2.0, level: float = 0.3, seed: int = 0) -> np.ndarray:
    """A pluck as a mic hears it: a click of pick noise, and for the first
    ~80 ms the octave louder than the note (picked near the bridge)."""
    x = ring(f, secs) / 0.3
    t = np.arange(x.size) / SR
    x = x + 1.5 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 25)
    n = int(0.01 * SR)
    x[:n] += 0.8 * np.random.default_rng(seed).standard_normal(n) * np.linspace(1, 0, n)
    return level * x / np.max(np.abs(x))


def first_shown(readings: list[dict], string: int, after: float = 0.0) -> float | None:
    """Seconds into the signal at which `string` first shows."""
    for i, r in enumerate(readings):
        if i * CHUNK / SR >= after and r["state"] in ("note", "tuned") and r["string"] == string:
            return i * CHUNK / SR
    return None


def test_a_pluck_after_a_pause_is_heard():
    # the bug: the pluck's own chunk (no pitch yet) counted as more silence, which
    # forgot the pluck, and the whole note was then ignored as "the tail"
    sig = np.concatenate([room(2.0), room(2.0) + picked(hz(45))])
    at = first_shown(feed(StringTracker(SR, midis()), sig.astype(np.float32)), 5)
    assert at is not None and at - 2.0 < 0.2


def test_each_pluck_after_the_last_one_died_is_heard():
    sig = np.concatenate([room(0.5)] + [room(5.0) + np.pad(picked(hz(45 + c / 100), 3.0), (0, 2 * SR))
                                         for c in (-20, -8, 0)])
    readings = feed(StringTracker(SR, midis()), sig.astype(np.float32))
    for k, cents in enumerate((-20, -8, 0)):
        start = 0.5 + 5.0 * k
        at = first_shown(readings, 5, after=start)
        assert at is not None and at - start < 0.2
        live = [r["cents"] for r in readings[int(start * SR / CHUNK):int((start + 0.6) * SR / CHUNK)]
                if r["state"] == "note"]
        assert live and abs(np.median(live) - cents) < 3


def test_a_new_string_every_second_is_followed():
    ms = midis()
    sig = room(7.0)
    for k, m in enumerate(ms):
        i = int((0.3 + k) * SR)
        p = picked(hz(m), 7.0 - 0.3 - k, seed=k)
        sig[i:i + p.size] += p
    readings = feed(StringTracker(SR, ms), sig.astype(np.float32))
    for k in range(6):
        at = first_shown(readings, 6 - k, after=0.3 + k)
        assert at is not None and at - (0.3 + k) < 0.3


# ----------------------------------------------------------------- lock-in

def test_an_in_tune_string_locks_in_and_stays_locked_as_it_dies():
    ms = midis()
    readings = feed(StringTracker(SR, ms), ring(hz(45)).astype(np.float32))
    first = tuned(readings)
    assert len(first) == 1
    assert 0.6 < first[0] * CHUNK / SR < 1.2
    assert readings[first[0]]["string"] == 5 and abs(readings[first[0]]["cents"]) < 2
    after = readings[first[0]:]
    assert {r["state"] for r in after} <= {"tuned", "silent"}
    assert {r["cents"] for r in after if r["state"] == "tuned"} == {readings[first[0]]["cents"]}


def test_a_string_off_by_a_little_never_locks_in():
    for cents in (8, -12, 20):
        readings = feed(StringTracker(SR, midis()), ring(hz(45 + cents / 100)).astype(np.float32))
        assert not tuned(readings)
        assert np.median([r["cents"] for r in notes(readings)]) == pytest.approx(cents, abs=2)


def test_a_turning_peg_locks_in_only_once_it_stops():
    # 20 cents flat, brought up to pitch over 1.5 s, then left alone
    secs = 4.0
    sig = ring(hz(40), secs, cents=lambda t: np.minimum(0.0, -20 + 20 * t / 1.5))
    readings = feed(StringTracker(SR, midis()), sig.astype(np.float32))
    first = tuned(readings)
    assert len(first) == 1 and first[0] * CHUNK / SR > 1.5
    assert abs(readings[first[0]]["cents"]) < 3


def test_a_new_pluck_unlocks_the_string():
    ms = midis()
    tr = StringTracker(SR, ms)
    readings = feed(tr, ring(hz(45), 2.0).astype(np.float32))
    assert tuned(readings)
    later = feed(tr, ring(hz(45 + 0.2), 1.0).astype(np.float32))     # picked again, 20 cents sharp
    live = [r for r in later if r["state"] == "note"]
    assert live and live[-1]["cents"] == pytest.approx(20, abs=2)


def feed_until_tuned(tr: StringTracker, sig: np.ndarray) -> int:
    """Feed chunks until the string locks in; returns the sample it got to."""
    for i in range(0, sig.size - CHUNK + 1, CHUNK):
        if tr.feed(sig[i:i + CHUNK])["state"] == "tuned":
            return i + CHUNK
    raise AssertionError("never locked in")


def chime_echo(n: int, delay: float = 0.1, level: float = 0.3) -> np.ndarray:
    """The page's chime (E7 then B7, as tuner.js plays it) coming back through
    the mic `delay` seconds after the lock-in, loud."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, f in enumerate((2637.0, 3951.1)):
        tt = t - delay - 0.11 * k
        on = tt >= 0
        out[on] += level * np.sin(2 * np.pi * f * tt[on]) * np.exp(-tt[on] * 20)
    return out


def test_after_a_lock_in_the_echo_of_a_chime_is_not_a_pluck():
    tr = StringTracker(SR, midis())
    a2 = ring(hz(45), 3.0)
    at = feed_until_tuned(tr, a2.astype(np.float32))
    rest = a2[at:]
    after = feed(tr, (rest + chime_echo(rest.size)).astype(np.float32))
    assert {r["state"] for r in after} <= {"tuned", "silent"}


def test_a_quick_pluck_of_the_next_string_is_not_lost_to_the_guard():
    # string 4 picked 0.3 s after string 5 locked in, inside the chime guard
    tr = StringTracker(SR, midis())
    a2 = ring(hz(45), 3.0)
    at = feed_until_tuned(tr, a2.astype(np.float32))
    nxt = np.zeros(a2.size - at)
    d3 = ring(hz(50), 2.0)
    start = int(0.3 * SR)
    nxt[start:start + d3.size] = d3[: nxt.size - start]
    got = notes(feed(tr, (0.3 * a2[at:] + nxt).astype(np.float32)))
    assert got and got[-1]["string"] == 4 and abs(got[-1]["cents"]) < 3


def test_moving_the_lock_on_waits_for_the_next_pluck():
    # the page locks string 4 next while string 5 still rings: that isn't string 4 far off
    ms = midis()
    tr = StringTracker(SR, ms, lock=5)
    a2 = ring(hz(45), 3.0).astype(np.float32)
    assert tuned(feed(tr, a2[: int(1.2 * SR)]))
    tr.configure(lock=4, wait=True)
    ringing_on = feed(tr, a2[int(1.2 * SR):])
    assert not notes(ringing_on)
    got = notes(feed(tr, ring(hz(50), 1.0).astype(np.float32)))
    assert got and {r["string"] for r in got} == {4} and abs(got[-1]["cents"]) < 2


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
