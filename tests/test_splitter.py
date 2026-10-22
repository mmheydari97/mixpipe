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


def test_stem_download_is_that_stem_at_full_level(client, wav_bytes):
    sid = upload(client, wav_bytes)
    r = client.post(f"/api/songs/{sid}/stems/guitar/render", json={})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/flac"
    assert f'filename="{sid}-guitar.flac"' in r.headers["content-disposition"]
    stem, sr = sf.read(io.BytesIO(r.content), always_2d=True)
    assert sr == SR and stem.shape == (int(SR * SECONDS), 2)
    assert tone_energy(stem, FREQS["guitar"]) == pytest.approx(0.05, rel=0.05)   # 0.1 sine, unity
    assert tone_energy(stem, FREQS["drums"]) < 0.01 * tone_energy(stem, FREQS["guitar"])

    wav = client.post(f"/api/songs/{sid}/stems/vocals/render", json={"fmt": "wav"})
    assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"
    assert client.post(f"/api/songs/{sid}/stems/kazoo/render", json={}).status_code == 404
    assert client.post(f"/api/songs/{sid}/stems/guitar/render", json={"fmt": "ogg"}).status_code == 400
    assert client.post(f"/api/songs/{sid}/stems/guitar/render", json={"speed": 9}).status_code == 400


def test_all_stems_as_a_zip(client, wav_bytes):
    import zipfile

    sid = upload(client, wav_bytes)
    r = client.post(f"/api/songs/{sid}/stems.zip", json={"fmt": "wav"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert f'filename="{sid}-stems.zip"' in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
        assert names == [f"{sid}-{n}.wav" for n in STEM_NAMES]
        for name in STEM_NAMES:
            stem, _ = sf.read(io.BytesIO(zf.read(f"{sid}-{name}.wav")), always_2d=True)
            others = [f for n, f in FREQS.items() if n != name]
            assert tone_energy(stem, FREQS[name]) > 50 * max(tone_energy(stem, f) for f in others)


def test_download_names_survive_any_script(client, wav_bytes):
    from urllib.parse import quote

    sid = client.post("/api/songs", files=[("files", ("آهنگ.wav", wav_bytes(song(), SR),
                                                      "audio/wav"))]).json()["id"]
    assert not sid.isascii()
    for url in (f"/api/songs/{sid}/render", f"/api/songs/{sid}/stems/bass/render",
                f"/api/songs/{sid}/stems.zip"):
        r = client.post(url, json={})
        assert r.status_code == 200, (url, r.text)
        cd = r.headers["content-disposition"]
        assert f"filename*=UTF-8''{quote(sid)}" in cd and 'filename="' in cd


@needs_rubberband
def test_stems_zip_follows_speed(client, wav_bytes):
    import zipfile

    sid = upload(client, wav_bytes)
    r = client.post(f"/api/songs/{sid}/stems.zip", json={"speed": 0.5})
    assert r.status_code == 200, r.text
    assert "-50pct.zip" in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert len(zf.namelist()) == len(STEM_NAMES)
        drums, _ = sf.read(io.BytesIO(zf.read(f"{sid}-drums-50pct.flac")), always_2d=True)
    assert len(drums) / SR == pytest.approx(2 * SECONDS, rel=0.02)
    assert tone_energy(drums, FREQS["drums"]) > 5 * tone_energy(drums, FREQS["drums"] / 2)


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
    from mixpipe import stretch

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


# --------------------------------------------------------------------------- #
# effects (the mixer's fx.py, rendered per stem on the server)
# --------------------------------------------------------------------------- #

def fx_body(**stems):
    return {"settings": {"stems": stems}}


def test_effects_render_only_the_stems_that_have_them(client, wav_bytes):
    sid = upload(client, wav_bytes)
    r = client.post(f"/api/songs/{sid}/fx", json=fx_body(guitar={"drive": 0.6}, vocals={"reverb_db": -12}))
    assert r.status_code == 200, r.text
    urls = r.json()["stems"]
    assert set(urls) == {"guitar", "vocals"}
    guitar, _ = sf.read(io.BytesIO(client.get(urls["guitar"]).content), always_2d=True)
    assert guitar.shape == (int(SR * SECONDS), 2)
    # drive: odd harmonics of the guitar's 330 Hz (no stem plays 990 Hz)
    assert tone_energy(guitar, FREQS["guitar"] * 3) > 0.02 * tone_energy(guitar, FREQS["guitar"])
    vocals, _ = sf.read(io.BytesIO(client.get(urls["vocals"]).content), always_2d=True)
    plain = client.get(f"/api/songs/{sid}/stems/vocals").content
    assert not np.allclose(vocals, sf.read(io.BytesIO(plain), always_2d=True)[0], atol=1e-3)

    # the same settings come from the cache; changing one stem leaves the other's file alone
    assert client.post(f"/api/songs/{sid}/fx", json=fx_body(guitar={"drive": 0.6}, vocals={"reverb_db": -12})).json()["stems"] == urls
    again = client.post(f"/api/songs/{sid}/fx", json=fx_body(guitar={"drive": 0.3}, vocals={"reverb_db": -12})).json()["stems"]
    assert again["vocals"] == urls["vocals"] and again["guitar"] != urls["guitar"]
    assert client.post(f"/api/songs/{sid}/fx", json=fx_body()).json()["stems"] == {}


def test_effect_settings_are_clamped_and_paths_checked(client, wav_bytes, tmp_path):
    from mixpipe.splitter import effects

    s = effects.validate({"stems": {"guitar": {"drive": 5, "bass_db": -40, "reverb_db": 9}}})
    assert s["stems"]["guitar"]["drive"] == 1.0 and s["stems"]["guitar"]["bass_db"] == -12.0
    assert s["stems"]["guitar"]["reverb_db"] == 0.0 and s["send_buses"]["reverb"]["room_size"] > 0
    sid = upload(client, wav_bytes)
    assert client.get(f"/api/songs/{sid}/fx/orig/..%2Fmeta.json").status_code == 404
    assert client.get(f"/api/songs/{sid}/fx/nope/guitar-0123456789abcdef.flac").status_code == 404
    assert client.post(f"/api/songs/{sid}/fx", json={"version": "x070+00", "quality": "fine"}).status_code == 404
    assert client.post(f"/api/songs/{sid}/fx", json={"version": "x070+00", "quality": "best"}).status_code == 400


def test_downloads_include_the_effects(client, wav_bytes):
    sid = upload(client, wav_bytes)
    fxs = {"stems": {"guitar": {"drive": 0.8}}}
    dry = client.post(f"/api/songs/{sid}/stems/guitar/render", json={})
    wet = client.post(f"/api/songs/{sid}/stems/guitar/render", json={"fx": fxs})
    a, _ = sf.read(io.BytesIO(dry.content), always_2d=True)
    b, _ = sf.read(io.BytesIO(wet.content), always_2d=True)
    assert tone_energy(b, FREQS["guitar"] * 3) > 20 * tone_energy(a, FREQS["guitar"] * 3)
    mix = client.post(f"/api/songs/{sid}/render", json={"gains": {}, "fx": fxs})
    m, _ = sf.read(io.BytesIO(mix.content), always_2d=True)
    assert tone_energy(m, FREQS["guitar"] * 3) > 0.01 * tone_energy(m, FREQS["guitar"])


def test_old_effect_renders_are_pruned(tmp_path):
    from mixpipe.splitter import effects

    folder = tmp_path / "fx" / "orig"
    folder.mkdir(parents=True)
    for i in range(effects.KEEP + 5):
        (folder / f"guitar-{i:016x}.flac").write_bytes(b"x")
    effects.prune(tmp_path)
    assert len(list(folder.iterdir())) == effects.KEEP


# --------------------------------------------------------------------------- #
# takes: recordings over the backing, kept like stems
# --------------------------------------------------------------------------- #

def burst_at(seconds: float, at: float, f: float = 990.0) -> np.ndarray:
    """A recording: silence, then a 0.2 s tone at `at` seconds."""
    y = np.zeros(int(SR * seconds), dtype=np.float32)
    a = int(SR * at)
    t = np.arange(int(SR * 0.2)) / SR
    y[a:a + len(t)] = 0.5 * np.sin(2 * np.pi * f * t)
    return np.stack([y, y], axis=1)


def onset(audio: np.ndarray) -> float:
    return int(np.argmax(np.abs(audio[:, 0]) > 0.05)) / SR


def add_take(client, sid, wav_bytes, recording, **fields):
    form = {"start_s": "0", "speed": "1", "semitones": "0", "latency_ms": "0", **{k: str(v) for k, v in fields.items()}}
    return client.post(f"/api/songs/{sid}/takes", data=form,
                       files={"file": ("take.wav", wav_bytes(recording, SR), "audio/wav")})


def test_a_take_is_placed_on_the_song_and_nudged(client, wav_bytes):
    sid = upload(client, wav_bytes)
    r = add_take(client, sid, wav_bytes, burst_at(1.0, 0.3), start_s=0.5, latency_ms=100)
    assert r.status_code == 200, r.text
    assert r.json()["n"] == 1 and client.get(f"/api/songs/{sid}").json()["takes"][0]["name"] == "Take 1"
    take, _ = sf.read(io.BytesIO(client.get(f"/api/songs/{sid}/takes/1").content), always_2d=True)
    assert take.shape == (int(SR * SECONDS), 2)                  # as long as the song, like a stem
    assert onset(take) == pytest.approx(0.5 + 0.3 - 0.1, abs=0.005)   # where it was played

    assert client.patch(f"/api/songs/{sid}/takes/1", json={"ms": 50}).status_code == 200
    take, _ = sf.read(io.BytesIO(client.get(f"/api/songs/{sid}/takes/1").content), always_2d=True)
    assert onset(take) == pytest.approx(0.75, abs=0.005)
    assert client.patch(f"/api/songs/{sid}/takes/7", json={"ms": 5}).status_code == 404


def test_takes_are_in_the_downloads_and_can_be_deleted(client, wav_bytes, tmp_path):
    import zipfile

    sid = upload(client, wav_bytes)
    add_take(client, sid, wav_bytes, burst_at(1.0, 0.2), start_s=0.3)
    silent = {n: 0 for n in STEM_NAMES}
    mix, _ = sf.read(io.BytesIO(client.post(f"/api/songs/{sid}/render", json={"gains": {**silent, "take:1": 1}}).content),
                     always_2d=True)
    assert onset(mix) == pytest.approx(0.5, abs=0.005) and tone_energy(mix, 990) > 0.01
    without, _ = sf.read(io.BytesIO(client.post(f"/api/songs/{sid}/render", json={"gains": silent}).content), always_2d=True)
    assert tone_energy(without, 990) < 1e-4                      # a take is only in when asked for
    assert client.post(f"/api/songs/{sid}/render", json={"gains": {"take:9": 1}}).status_code == 400
    alone = client.post(f"/api/songs/{sid}/stems/take:1/render", json={})
    assert alone.status_code == 200 and "-take:1" not in alone.headers["content-disposition"]
    assert client.post(f"/api/songs/{sid}/stems/take:9/render", json={}).status_code == 404
    with zipfile.ZipFile(io.BytesIO(client.post(f"/api/songs/{sid}/stems.zip", json={}).content)) as zf:
        assert f"{sid}-take1.flac" in zf.namelist()

    assert client.delete(f"/api/songs/{sid}/takes/1").status_code == 200
    assert client.get(f"/api/songs/{sid}").json()["takes"] == []
    assert not any((tmp_path / sid / "takes").glob("1*"))
    assert client.get(f"/api/songs/{sid}/takes/1").status_code == 404
    assert client.post(f"/api/songs/{sid}/takes", data={}).status_code == 400


@needs_rubberband
def test_a_take_recorded_slower_is_kept_at_the_songs_speed(client, wav_bytes):
    sid = upload(client, wav_bytes)
    # played at 50%: 0.6 s into the recording is 0.3 s of the song
    r = add_take(client, sid, wav_bytes, burst_at(1.5, 0.6), start_s=0.2, speed=0.5)
    assert r.status_code == 200, r.text
    take, _ = sf.read(io.BytesIO(client.get(f"/api/songs/{sid}/takes/1").content), always_2d=True)
    assert onset(take) == pytest.approx(0.5, abs=0.03)
    slow, _ = sf.read(io.BytesIO(client.get(f"/api/songs/{sid}/takes/1?version=x050%2B00&quality=fast").content),
                      always_2d=True)
    assert len(slow) == pytest.approx(2 * SR * SECONDS, abs=2) and onset(slow) == pytest.approx(1.0, abs=0.05)
