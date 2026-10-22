"""The hub: the chord finder, the splitter and the mixer on one port (mixpipe.cli ui)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("soundfile")

from test_splitter import SR, fake_separator, song  # noqa: E402


@pytest.fixture
def hub(tmp_path, wav_bytes):
    from fastapi.testclient import TestClient

    from mixpipe.web.hub import create_app

    app = create_app(tmp_path / "chords", tmp_path / "splitter", tmp_path / "mixer",
                     splitter_options={"separator": fake_separator, "run_inline": True},
                     mixer_options={"use_llm": False, "run_inline": True,
                                    "session_options": {"align": False}})
    c = TestClient(app)
    c.wav = wav_bytes(song(), SR)
    return c


def test_home_page_links_every_tool(hub):
    r = hub.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    for path in ('href="/chords/"', 'href="/splitter/"', 'href="/mixer/"'):
        assert path in r.text
    assert "/static/home.js?v=" in r.text and "/static/common.js?v=" in r.text


@pytest.mark.parametrize("tool,script", [("chords", "app.js"), ("splitter", "splitter.js"),
                                         ("mixer", "mixer.js")])
def test_each_tool_is_served_under_its_path(hub, tool, script):
    r = hub.get(f"/{tool}/")
    assert r.status_code == 200
    assert f"/static/{script}?v=" in r.text            # assets come from the hub's one /static
    assert hub.get(f"/static/{script}").status_code == 200
    bare = hub.get(f"/{tool}", follow_redirects=False)
    assert bare.status_code in (301, 307) and bare.headers["location"].endswith(f"/{tool}/")


def test_apis_live_under_each_tool(hub):
    assert hub.get("/chords/api/songs").json() == []
    assert hub.get("/chords/api/openrouter/status").status_code == 200
    assert hub.get("/mixer/api/config").status_code == 200
    assert hub.get("/api/songs").status_code == 404     # no tool owns the root's /api


def test_a_split_song_is_ready_to_mix(hub):
    sid = hub.post("/splitter/api/songs", files={"files": ("song.wav", hub.wav, "audio/wav")}).json()["id"]
    assert hub.get(f"/splitter/api/songs/{sid}/status").json()["state"] == "done"
    listed = hub.get("/mixer/api/splitter-songs").json()
    assert [s["id"] for s in listed] == [sid]
    session = hub.post("/mixer/api/sessions", json={"splitter_song": sid}).json()["id"]
    assert hub.get(f"/mixer/api/sessions/{session}").json()["state"] == "ready"


def test_the_chord_finder_shares_its_song_service(tmp_path):
    from mixpipe.web.server import create_app

    app = create_app(tmp_path)
    assert app.state.songs.song_dir is not None
