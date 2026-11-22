"""The chord finder's Vocals / Guitar sliders: the on-demand six-stem split
(chords/stems6.py) and its routes. Fake separator, no models.

Run with:  PYTHONPATH=src python -m pytest tests/test_playalong.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
sf = pytest.importorskip("soundfile")

from mixpipe import songstore  # noqa: E402
from mixpipe.chords import stems6  # noqa: E402
from mixpipe.splitter.pipeline import STEM_NAMES, stem_path  # noqa: E402

from test_splitter import FREQS, SR, fake_separator, song, tone_energy  # noqa: E402


@pytest.fixture
def chord_song(tmp_path, wav_bytes):
    root = tmp_path / "chords"
    sid = songstore.new_song([("song.wav", wav_bytes(song(), SR))], root)
    songstore.set_status(root / sid, "done", "align", 1.0, "done")
    return root / sid


def test_ensure_splits_once_and_mixes_the_backing(chord_song, tmp_path):
    calls = []

    def separator(audio, sr, model=None, progress=None):
        calls.append(model)
        return fake_separator(audio, sr, model, progress)

    paths = stems6.ensure(chord_song, tmp_path / "splitter", separator=separator)
    assert calls == ["htdemucs_6s"]
    assert set(paths) == set(STEM_NAMES) | {"backing"}
    assert stems6.ready(chord_song, tmp_path / "splitter")
    backing, _ = sf.read(str(paths["backing"]), dtype="float32", always_2d=True)
    for name in ("drums", "bass", "piano", "other"):
        assert tone_energy(backing, FREQS[name]) > 0.03, name
    for name in ("vocals", "guitar"):
        assert tone_energy(backing, FREQS[name]) < 0.003, name
    # the song's own status is untouched; the split has its own
    assert songstore.get_status(chord_song)["state"] == "done"
    # the page turns the three tracks down just enough that their sum can't clip
    tracks = [sf.read(str(paths[n]), dtype="float32", always_2d=True)[0] for n in stems6.PLAY]
    peak = float(np.max(np.abs(sum(tracks))))
    g = stems6.status(chord_song, tmp_path / "splitter")["gain_db"]
    assert g <= 0 and peak * 10 ** (g / 20) <= stems6.HEADROOM + 1e-3
    # a second call reuses everything
    stems6.ensure(chord_song, tmp_path / "splitter", separator=separator)
    assert calls == ["htdemucs_6s"]


def test_reuses_the_splitters_stems(chord_song, tmp_path):
    split_root = tmp_path / "splitter"
    songstore.copy_song(chord_song, split_root)
    from mixpipe.splitter.pipeline import split

    split(split_root / chord_song.name, separator=fake_separator)

    def no_separator(*a, **k):
        raise AssertionError("must not separate again")

    paths = stems6.ensure(chord_song, split_root, separator=no_separator)
    assert paths["guitar"] == stem_path(split_root / chord_song.name, "guitar")
    assert not (chord_song / "stems.json").exists()      # nothing re-made here
    assert paths["backing"].parent == chord_song / "stems"


def test_stems_upload_cannot_be_split(tmp_path, wav_bytes):
    root = tmp_path / "chords"
    sid = songstore.new_song([("vocals.wav", wav_bytes(song(), SR)),
                              ("guitar.wav", wav_bytes(song() * 0.5, SR))], root)
    assert stems6.run_job(root / sid, tmp_path / "splitter", separator=fake_separator) is None
    st = stems6.status(root / sid, tmp_path / "splitter")
    assert st["state"] == "error" and "separate stems" in st["error"]


def test_routes(chord_song, tmp_path):
    from fastapi.testclient import TestClient

    from mixpipe.web.server import create_app

    app = create_app(chord_song.parent, run_inline=True, splitter_root=tmp_path / "splitter",
                     stem_separator=fake_separator)
    c = TestClient(app)
    sid = chord_song.name
    assert c.get(f"/api/songs/{sid}/playalong").json()["ready"] is False
    assert c.get(f"/api/songs/{sid}/playalong/guitar").status_code == 409
    st = c.post(f"/api/songs/{sid}/playalong").json()
    assert st["ready"] is True and st["state"] == "done"
    for name in stems6.PLAY:
        r = c.get(f"/api/songs/{sid}/playalong/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/flac"
    assert c.get(f"/api/songs/{sid}/playalong/drums").status_code == 404
    assert c.get(f"/api/songs/{sid}/status").json()["state"] == "done"
