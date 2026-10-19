"""Tests for the stem splitter.

Run with:  PYTHONPATH=src python -m pytest tests/test_splitter.py -q

Needs requirements-chords.txt (fastapi, soundfile). No model downloads and no
network: separation and link downloads are replaced with fakes.
"""

from __future__ import annotations

import io
import shutil
import time

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


needs_rubberband = pytest.mark.skipif(
    shutil.which("rubberband") is None and shutil.which("ffmpeg") is None,
    reason="needs rubberband or ffmpeg")


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
    assert client.post(f"/api/songs/{sid}/render", json={"speed": 3.5}).status_code == 400
    assert client.post(f"/api/songs/{sid}/render", json={"speed": 0.4}).status_code == 400
    assert client.post(f"/api/songs/{sid}/render", json={"fmt": "ogg"}).status_code == 400


@needs_rubberband
def test_render_speed_and_mp3(client, wav_bytes):
    sid = client.post("/api/songs", files=[("files", ("s.wav", wav_bytes(song(), SR),
                                                      "audio/wav"))]).json()["id"]
    r = client.post(f"/api/songs/{sid}/render", json={"speed": 0.5})
    assert r.status_code == 200, r.text
    assert "50pct" in r.headers["content-disposition"]
    slow, _ = sf.read(io.BytesIO(r.content), always_2d=True)
    assert len(slow) / SR == pytest.approx(2 * SECONDS, rel=0.05)
    # same pitch: the guitar tone is still at 330 Hz, not at 165 Hz (no stem uses 165)
    assert tone_energy(slow, FREQS["guitar"]) > 5 * tone_energy(slow, FREQS["guitar"] / 2)

    r = client.post(f"/api/songs/{sid}/render", json={"speed": 3})
    assert r.status_code == 200, r.text
    assert "300pct" in r.headers["content-disposition"]
    fast, _ = sf.read(io.BytesIO(r.content), always_2d=True)
    assert len(fast) / SR == pytest.approx(SECONDS / 3, rel=0.08)
    # same pitch: still 330 Hz, not 990 Hz (no stem uses 990)
    assert tone_energy(fast, FREQS["guitar"]) > 5 * tone_energy(fast, FREQS["guitar"] * 3)

    r = client.post(f"/api/songs/{sid}/render", json={"fmt": "mp3"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    assert r.content[:3] == b"ID3" or r.content[0] == 0xFF


def upload(client, wav_bytes) -> str:
    return client.post("/api/songs", files=[("files", ("s.wav", wav_bytes(song(), SR),
                                                       "audio/wav"))]).json()["id"]


def wait_version(client, sid, key, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/songs/{sid}/versions/{key}").json()
        if not st["working"]:
            return st
        time.sleep(0.05)
    raise AssertionError(f"version {key} not ready in {timeout} s")


@needs_rubberband
def test_version_changes_speed_and_key(client, wav_bytes):
    sid = upload(client, wav_bytes)
    r = client.post(f"/api/songs/{sid}/versions", json={"speed": 2, "semitones": 12})
    assert r.status_code == 200, r.text
    key = r.json()["key"]
    assert key == "x200+12"
    assert set(r.json()) >= {"making", "progress"}
    seen = []
    while True:  # the pass being made and its progress, for the page's readout
        st = client.get(f"/api/songs/{sid}/versions/{key}").json()
        if st["progress"] is not None:
            seen.append((st["making"], st["progress"]))
        if not st["working"]:
            break
        time.sleep(0.01)
    assert st["ready"] == "fine" and st["error"] is None, st
    assert st["making"] is None and st["progress"] is None
    assert all(m in ("fast", "fine") and 0 <= f <= 1 for m, f in seen), seen
    r = client.get(f"/api/songs/{sid}/versions/{key}/fine/guitar")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/flac"
    y, _ = sf.read(io.BytesIO(r.content), always_2d=True)
    assert len(y) == round(SR * SECONDS / 2)                   # twice as fast
    g = FREQS["guitar"]
    assert tone_energy(y, 2 * g) > 5 * tone_energy(y, g)       # an octave up
    # every stem the same length, so they stay in sync
    lengths = {len(sf.read(io.BytesIO(client.get(
        f"/api/songs/{sid}/versions/{key}/fine/{n}").content))[0]) for n in STEM_NAMES}
    assert lengths == {len(y)}

    assert client.post(f"/api/songs/{sid}/versions", json={}).json()["key"] is None
    for bad in ({"semitones": 13}, {"semitones": 1.5}, {"speed": 3.5}):
        assert client.post(f"/api/songs/{sid}/versions", json=bad).status_code == 400
    assert client.get(f"/api/songs/{sid}/versions/x200+12/fine/kazoo").status_code == 404
    assert client.get(f"/api/songs/{sid}/versions/..%2F..%2Fmeta.json").status_code == 404
    assert client.get(f"/api/songs/{sid}/versions/x050+00/fine/guitar").status_code == 404


@pytest.mark.skipif(shutil.which("rubberband") is None, reason="needs the rubberband tool")
def test_stretch_reports_progress():
    from mixpipe.splitter import stretch

    seen = []
    stretch.stretch(song(), SR, 0.7, 2, "fast", progress=seen.append)
    assert len(seen) > 10 and seen[-1] == 1.0
    assert all(a <= b for a, b in zip(seen, seen[1:]))   # never goes backwards


@needs_rubberband
def test_newer_version_request_supersedes(client, wav_bytes, tmp_path):
    sid = upload(client, wav_bytes)
    first = client.post(f"/api/songs/{sid}/versions", json={"speed": 0.5}).json()["key"]
    second = client.post(f"/api/songs/{sid}/versions", json={"speed": 0.7, "semitones": -2}).json()["key"]
    assert wait_version(client, sid, second)["ready"] == "fine"
    assert wait_version(client, sid, first)["ready"] != "fine"   # cancelled, not finished

    # only the newest versions are kept
    for speed in (0.6, 0.8, 0.9, 1.1):
        key = client.post(f"/api/songs/{sid}/versions", json={"speed": speed}).json()["key"]
        wait_version(client, sid, key)
    kept = sorted(d.name for d in (tmp_path / sid / "versions").iterdir())
    assert len(kept) <= 3 and "x110+00" in kept, kept


@needs_rubberband
def test_render_key_change(client, wav_bytes):
    sid = upload(client, wav_bytes)
    guitar_only = {n: 0 for n in STEM_NAMES if n != "guitar"}
    r = client.post(f"/api/songs/{sid}/render", json={"gains": guitar_only, "semitones": -12})
    assert r.status_code == 200, r.text
    assert "down12st" in r.headers["content-disposition"]
    y, _ = sf.read(io.BytesIO(r.content), always_2d=True)
    assert len(y) == int(SR * SECONDS)
    g = FREQS["guitar"]
    assert tone_energy(y, g / 2) > 5 * tone_energy(y, g)       # an octave down
    assert client.post(f"/api/songs/{sid}/render", json={"semitones": 0.5}).status_code == 400


def test_download_failure_then_retry(client):
    client.calls["fail"] = True
    sid = client.post("/api/songs/link", json={"url": "https://youtu.be/abc"}).json()["id"]
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "error" and st["stage"] == "download"
    assert client.get(f"/api/songs/{sid}").status_code == 409
    client.calls["fail"] = False
    client.post("/api/songs/link", json={"url": "https://youtu.be/abc", "rerun": True})
    assert client.get(f"/api/songs/{sid}/status").json()["state"] == "done"


def test_page_assets_are_versioned_and_revalidated(client):
    """A browser must never run a stale script after an update."""
    r = client.get("/")
    assert r.headers["cache-control"] == "no-cache"
    assert '/static/splitter.js?v=' in r.text and '/static/common.js?v=' in r.text
    assert client.get("/static/splitter.js").headers["cache-control"] == "no-cache"
