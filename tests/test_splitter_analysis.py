"""The splitter's link to the chord finder: a split song's key, tempo, beats,
chords and lyrics come from one chord-finder analysis of its own stems."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

from test_chords import BPM, PROGRESSION, SR, fake_transcriber, strummed  # noqa: E402

from mixpipe import songstore  # noqa: E402
from mixpipe.splitter.pipeline import STEM_NAMES, stem_path  # noqa: E402


def no_separation(*args, **kwargs):
    raise AssertionError("the chord finder separated the song again")


@pytest.fixture(scope="module")
def song_audio() -> np.ndarray:
    y = np.concatenate([strummed(notes, bass) for _, notes, bass in PROGRESSION])
    y = (y / np.abs(y).max() * 0.8).astype(np.float32)
    return np.stack([y, y], axis=1)


@pytest.fixture
def split(tmp_path, wav_bytes, song_audio):
    """A split song (made by hand: the chords are in the guitar and the
    vocals) and a splitter app whose chord finder runs inline with fakes."""
    from fastapi.testclient import TestClient

    from mixpipe.web import server, splitter_server

    split_root, chords_root = tmp_path / "splitter", tmp_path / "chords"
    sid = songstore.new_song([("progression.wav", wav_bytes(song_audio, SR))], split_root, "Progression")
    d = split_root / sid
    (d / "stems").mkdir()
    level = {"guitar": 1.0, "vocals": 1.0}
    for name in STEM_NAMES:
        sf.write(str(stem_path(d, name)), song_audio * level.get(name, 0.0), SR, subtype="PCM_16")
    (d / "stems.json").write_text(
        '{"model": "htdemucs_6s", "sr": %d, "duration": %.3f, "stems": [%s]}'
        % (SR, len(song_audio) / SR, ", ".join(f'{{"name": "{n}", "label": "{n}"}}' for n in STEM_NAMES)))
    songstore.set_status(d, "done", "separate", 1.0, "done")
    svc = server.chords_service(chords_root, run_inline=True, separator=no_separation,
                                transcriber=fake_transcriber)
    client = TestClient(splitter_server.create_app(split_root, run_inline=True, chords_root=chords_root,
                                                   chords_service=svc))
    return client, sid, chords_root


def test_the_chord_finder_analyses_a_split_song_from_its_stems(split):
    client, sid, chords_root = split
    assert client.get(f"/api/songs/{sid}/analysis").json() == {"state": "none"}
    assert client.get(f"/api/songs/{sid}/sheet").status_code == 409

    st = client.post(f"/api/songs/{sid}/analyse", json={}).json()
    assert st["state"] == "done", st
    assert st["chords_id"] == sid and (chords_root / sid / "separated" / "other.flac").exists()
    assert st["key_label"] == "C major" and st["tempo"] == pytest.approx(BPM, rel=0.05)
    assert len(st["beat_times"]) > 20 and st["meter"]["beats_per_bar"] in (2, 4)
    bars = st["bars"]                                     # the chord finder's bar grid
    assert len(bars) >= 6 and all(b["beats"] == st["meter"]["beats_per_bar"] for b in bars)
    assert all(min(abs(b["start"] - t) for t in st["beat_times"]) < 1e-3 for b in bars)   # on beats
    assert st["lyrics"] is False                          # the first run skips them

    # what you hear after moving the song two semitones up
    assert client.get(f"/api/songs/{sid}/analysis?shift=2").json()["key_label"] == "D major"
    first = lambda view: next(c["display"] for c in view["chords"] if c["display"] != "N.C.")  # noqa: E731
    assert first(client.get(f"/api/songs/{sid}/sheet").json()) == "C"
    assert first(client.get(f"/api/songs/{sid}/sheet?shift=2").json()) == "D"
    assert first(client.get(f"/api/songs/{sid}/sheet?shift=-12").json()) == "C"
    assert client.get(f"/api/songs/{sid}/analysis?shift=13").status_code == 400


def test_lyrics_are_added_later_without_redoing_the_chords(split):
    client, sid, chords_root = split
    client.post(f"/api/songs/{sid}/analyse", json={})
    chords_json = chords_root / sid / "chords.json"
    made = chords_json.stat().st_mtime_ns
    st = client.post(f"/api/songs/{sid}/analyse", json={"lyrics": True}).json()
    assert st["lyrics"] is True
    assert chords_json.stat().st_mtime_ns == made            # cached, not recognised again
    words = [w["text"] for line in client.get(f"/api/songs/{sid}/sheet").json()["lines"]
             for w in line["words"]]
    assert words == ["la", "la", "lah"]


def test_a_song_the_chord_finder_already_has_is_linked(split, tmp_path):
    client, sid, chords_root = split
    client.post(f"/api/songs/{sid}/analyse", json={})
    sheet = (chords_root / sid / "sheet.json").read_text()
    again = client.post(f"/api/songs/{sid}/analyse", json={}).json()   # nothing re-run
    assert again["state"] == "done" and (chords_root / sid / "sheet.json").read_text() == sheet
