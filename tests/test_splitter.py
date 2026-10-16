"""Tests for the stem splitter.

Run with:  PYTHONPATH=src python -m pytest tests/test_splitter.py -q

Needs requirements-chords.txt (fastapi, soundfile). No model downloads and no
network: separation and link downloads are replaced with fakes.
"""

from __future__ import annotations

import io
import shutil

import numpy as np
import pytest

pytest.importorskip("fastapi")
sf = pytest.importorskip("soundfile")

from mixpipe.io_utils import Stem  # noqa: E402
from mixpipe.splitter.pipeline import STEM_NAMES  # noqa: E402

SR = 44_100
SECONDS = 2.0
# one sine per stem, so a mix shows which stems went into it
FREQS = dict(zip(STEM_NAMES, (220.0, 330.0, 440.0, 110.0, 550.0, 660.0)))


def song() -> np.ndarray:
    t = np.arange(int(SR * SECONDS)) / SR
    mono = sum(0.1 * np.sin(2 * np.pi * f * t) for f in FREQS.values())
    return np.stack([mono, mono * 0.8], axis=1).astype(np.float32)


def fake_separator(audio, sr, model=None, progress=None):
    assert model == "htdemucs_6s"
    t = np.arange(len(audio)) / sr
    out = []
    for name, f in FREQS.items():
        y = (0.1 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        out.append(Stem(name, np.stack([y, y * 0.8], axis=1), sr, name))
    if progress:
        progress(1.0)
    return out


def tone_energy(audio: np.ndarray, f: float) -> float:
    """Magnitude of one frequency in the left channel."""
    left = audio[:, 0] if audio.ndim == 2 else audio
    t = np.arange(len(left)) / SR
    return abs(np.dot(left, np.exp(-2j * np.pi * f * t))) / len(left)


@pytest.fixture
def client(tmp_path, fake_link, wav_bytes):
    from fastapi.testclient import TestClient

    from mixpipe.web.splitter_server import create_app

    calls, resolver, downloader = fake_link(wav_bytes(song(), SR))
    app = create_app(tmp_path, separator=fake_separator, run_inline=True,
                     resolver=resolver, downloader=downloader)
    c = TestClient(app)
    c.calls = calls
    return c


def test_upload_splits_into_six_stereo_stems(client, wav_bytes):
    r = client.post("/api/songs", files=[("files", ("my song.wav", wav_bytes(song(), SR),
                                                    "audio/wav"))])
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "done" and st["stage"] == "separate" and st["from_url"] is False, st

    info = client.get(f"/api/songs/{sid}").json()
    assert info["title"] == "my song"
    assert [s["name"] for s in info["stems"]] == list(STEM_NAMES)
    assert [s["label"] for s in info["stems"]][4] == "Keys"
    assert info["duration"] == pytest.approx(SECONDS, abs=0.01)

    for name in STEM_NAMES:
        r = client.get(f"/api/songs/{sid}/stems/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/flac"
        y, sr = sf.read(io.BytesIO(r.content), always_2d=True)
        assert y.shape == (int(SR * SECONDS), 2) and sr == SR
    assert client.get(f"/api/songs/{sid}/stems/kazoo").status_code == 404
    assert client.get(f"/api/songs/{sid}/stems/..%2Fmeta.json").status_code == 404

    # same file again: stems reused, no new song
    client.post("/api/songs", files=[("files", ("my song.wav", wav_bytes(song(), SR),
                                                "audio/wav"))])
    assert [s["id"] for s in client.get("/api/songs").json()] == [sid]


def test_one_file_only(client, wav_bytes):
    wav = wav_bytes(song(), SR)
    r = client.post("/api/songs", files=[("files", ("a.wav", wav, "audio/wav")),
                                         ("files", ("b.wav", wav, "audio/wav"))])
    sid = r.json()["id"]
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "error" and "one file" in st["error"]
    assert client.post("/api/songs", files=[]).status_code in (400, 422)


def test_link_flow(client):
    r = client.post("/api/songs/link", json={"url": "https://www.youtube.com/playlist?list=x"})
    assert r.status_code == 400 and "single song" in r.json()["detail"]

    sid = client.post("/api/songs/link", json={"url": "https://youtu.be/abc"}).json()["id"]
    assert sid.startswith("some-band-some-song-")
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "done" and st["from_url"] is True, st
    assert client.get(f"/api/songs/{sid}").json()["source"]["site"] == "Youtube"

    again = client.post("/api/songs/link", json={"url": "https://www.youtube.com/watch?v=abc"})
    assert again.json()["id"] == sid and client.calls["download"] == 1


def test_render_applies_gains(client, wav_bytes):
    sid = client.post("/api/songs", files=[("files", ("s.wav", wav_bytes(song(), SR),
                                                      "audio/wav"))]).json()["id"]
    r = client.post(f"/api/songs/{sid}/render", json={"gains": {"vocals": 0, "guitar": 0.5}})
    assert r.status_code == 200, r.text
    assert "no-vocals" in r.headers["content-disposition"]
    mix, sr = sf.read(io.BytesIO(r.content), always_2d=True)
    assert mix.shape == (int(SR * SECONDS), 2)
    full = tone_energy(mix, FREQS["drums"])
    assert tone_energy(mix, FREQS["vocals"]) < 0.01 * full   # muted
    assert tone_energy(mix, FREQS["guitar"]) == pytest.approx(0.5 * full, rel=0.05)

    assert client.post(f"/api/songs/{sid}/render",
                       json={"gains": {"kazoo": 1}}).status_code == 400
    assert client.post(f"/api/songs/{sid}/render", json={"speed": 2}).status_code == 400
    assert client.post(f"/api/songs/{sid}/render", json={"fmt": "ogg"}).status_code == 400


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_render_slower_and_mp3(client, wav_bytes):
    sid = client.post("/api/songs", files=[("files", ("s.wav", wav_bytes(song(), SR),
                                                      "audio/wav"))]).json()["id"]
    r = client.post(f"/api/songs/{sid}/render", json={"speed": 0.5})
    assert r.status_code == 200, r.text
    assert "50pct" in r.headers["content-disposition"]
    slow, _ = sf.read(io.BytesIO(r.content), always_2d=True)
    assert len(slow) / SR == pytest.approx(2 * SECONDS, rel=0.05)
    # same pitch: the guitar tone is still at 330 Hz, not at 165 Hz (no stem uses 165)
    assert tone_energy(slow, FREQS["guitar"]) > 5 * tone_energy(slow, FREQS["guitar"] / 2)

    r = client.post(f"/api/songs/{sid}/render", json={"fmt": "mp3"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    assert r.content[:3] == b"ID3" or r.content[0] == 0xFF


def test_download_failure_then_retry(client):
    client.calls["fail"] = True
    sid = client.post("/api/songs/link", json={"url": "https://youtu.be/abc"}).json()["id"]
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "error" and st["stage"] == "download"
    assert client.get(f"/api/songs/{sid}").status_code == 409
    client.calls["fail"] = False
    client.post("/api/songs/link", json={"url": "https://youtu.be/abc", "rerun": True})
    assert client.get(f"/api/songs/{sid}/status").json()["state"] == "done"
