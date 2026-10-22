"""AI edits: the ACE-Step client, splicing a repaint back in, the session's
history around an edit, the web API and the CLI's /ai and /cover."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

from mixpipe import acestep
from mixpipe.cli import parse_ai_edit
from mixpipe.io_utils import read_audio, write_audio
from mixpipe.session import MixSession

ROOT = Path(__file__).resolve().parents[1]
SR = 48_000


@pytest.fixture(scope="module")
def stems_dir(tmp_path_factory):
    """The synthetic session, 8 s long."""
    spec = importlib.util.spec_from_file_location("make_demo_session", ROOT / "scripts" / "make_demo_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = tmp_path_factory.mktemp("ai_stems")
    stems, meta = mod.build(seconds=8.0, seed=3)
    for name, audio in stems.items():
        write_audio(str(d / f"{name}.wav"), audio, meta["sr"])
    return d


def fake_editor(gain: float = 0.89, sr: int = SR):
    """Stands in for ACE-Step: the clip back at `gain`, with a tone over the range
    (or over everything for a cover), a few samples short like the real one."""
    calls = []

    def editor(path, prompt, mode="repaint", start_s=0.0, end_s=None, strength=None, seed=None):
        calls.append({"prompt": prompt, "mode": mode, "start_s": start_s, "end_s": end_s,
                      "strength": strength, "seed": seed})
        audio, src_sr = read_audio(path)
        out = audio * gain
        a = int(start_s * src_sr) if mode == "repaint" else 0
        b = len(out) if mode == "cover" or end_s is None else int(end_s * src_sr)
        t = np.arange(b - a) / src_sr
        out[a:b] = 0.3 * np.sin(2 * np.pi * 220 * t)[:, None]
        return out[:-850], src_sr, {"seed": "7", "dit_model": "fake-turbo", "seconds": 1.5}

    editor.calls = calls
    return editor


@pytest.fixture()
def session(stems_dir, tmp_path):
    s = MixSession(str(stems_dir), out_dir=str(tmp_path / "out"), use_llm=False, align=False)
    s.editor = fake_editor()
    s.first_mix()
    return s


# --------------------------------------------------------------------------- #
# splice
# --------------------------------------------------------------------------- #

def test_splice_keeps_the_original_outside_the_range():
    rng = np.random.default_rng(0)
    orig = rng.normal(0, 0.1, (SR * 6, 2))
    gen = orig * 0.89
    gen[2 * SR:4 * SR] = rng.normal(0, 0.1, (2 * SR, 2))
    out = acestep.splice(orig, gen[:-500], SR, 2.0, 4.0)
    assert out.shape == orig.shape
    edge = int(acestep.FADE_S * SR)
    assert np.array_equal(out[:2 * SR - edge], orig[:2 * SR - edge])
    assert np.array_equal(out[4 * SR + edge:], orig[4 * SR + edge:])
    # inside, the new audio, scaled back up by the gain it came back at (the padded
    # tail is at the original's level, which moves the fitted gain a hair)
    assert np.allclose(out[2 * SR + edge:4 * SR - edge], gen[2 * SR + edge:4 * SR - edge] / 0.89, rtol=1e-3)


def test_splice_to_the_end_and_bad_ranges():
    orig = np.full((SR * 3, 2), 0.1)
    gen = np.full((SR * 3, 2), 0.2)                              # came back at twice the level
    gen[2 * SR:] = 0.4
    out = acestep.splice(orig, gen, SR, 2.0, None)
    assert np.allclose(out[:int(1.9 * SR)], 0.1) and np.allclose(out[int(2.1 * SR):], 0.2)
    with pytest.raises(ValueError):
        acestep.splice(orig, gen, SR, 2.0, 1.0)


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #

def test_repaint_is_a_new_version_that_only_changes_the_range(session):
    v1 = session.current
    v2 = session.ai_edit("a rock drum beat", start_s=2.0, end_s=4.0, seed=7)
    assert v2["n"] == 2 and v2["kind"] == "ai" and v2["parent"] == 1 and v2["base"] == 1
    assert v2["plan"] == v1["plan"] and v2["levels"] is None
    assert v2["changes"][0]["text"] == "AI: repainted 0:02-0:04"
    assert v2["edit"] == {"mode": "repaint", "start_s": 2.0, "end_s": 4.0, "strength": None, "seed": "7"}
    assert "fake-turbo" in v2["reply"] and "from v1" in v2["reply"]
    assert session.editor.calls[-1] == {"prompt": "a rock drum beat", "mode": "repaint", "start_s": 2.0,
                                        "end_s": 4.0, "strength": None, "seed": 7}
    a, _ = read_audio(v1["wav"])
    b, _ = read_audio(v2["wav"])
    assert len(a) == len(b)
    assert np.allclose(a[:int(1.9 * SR)], b[:int(1.9 * SR)], atol=1e-6)       # 24-bit files
    assert np.allclose(a[int(4.1 * SR):], b[int(4.1 * SR):], atol=1e-6)
    assert not np.allclose(a[int(2.5 * SR):int(3.5 * SR)], b[int(2.5 * SR):int(3.5 * SR)], atol=1e-3)
    summary = session.summary(v2)
    assert summary["kind"] == "ai" and summary["base"] == 1 and summary["stems"]


def test_mixing_after_an_ai_edit_starts_from_its_base(session):
    session.ai_edit("strings", start_s=1.0, end_s=2.0)
    v3 = session.chat("the bass is too loud")
    assert v3["parent"] == 1 and v3.get("kind") is None
    assert v3["reply"].startswith("(Mixed from v1: the AI edit in v2")
    session.goto(2)
    v4 = session.set_params([{"target": "bass", "param": "mute", "value": True}])
    assert v4["parent"] == 1 and v4["reply"].startswith("(Mixed from v1")


def test_an_ai_edit_of_an_ai_edit_keeps_the_plan_base(session):
    session.chat("the bass is too loud")                        # v2, a plan version
    session.ai_edit("strings", start_s=1.0, end_s=2.0)          # v3 from v2
    v4 = session.ai_edit("choir", start_s=5.0, end_s=6.0)       # v4 from v3
    assert (v4["parent"], v4["base"]) == (3, 2)
    a, _ = read_audio(session.version(3)["wav"])
    b, _ = read_audio(v4["wav"])
    assert np.allclose(a[int(1.2 * SR):int(1.8 * SR)], b[int(1.2 * SR):int(1.8 * SR)], atol=1e-6)


def test_cover_matches_the_loudness_of_what_it_restyled(session):
    from mixpipe.analysis import integrated_lufs

    v2 = session.ai_edit("lo-fi", mode="cover", strength=0.3)
    assert v2["edit"]["strength"] == 0.3 and session.editor.calls[-1]["strength"] == 0.3
    a, _ = read_audio(session.version(1)["wav"])
    b, _ = read_audio(v2["wav"])
    assert abs(integrated_lufs(a.mean(axis=1), SR) - integrated_lufs(b.mean(axis=1), SR)) < 0.5


def test_what_the_model_adds_goes_through_the_limiter(session):
    def loud(path, prompt, mode="repaint", start_s=0.0, end_s=None, strength=None, seed=None):
        audio, sr = read_audio(path)
        out = audio.copy()
        if mode == "repaint":
            out[int(start_s * sr):int(end_s * sr)] = 1.5          # far over full scale
        else:
            out = out * 0.05                                      # quiet: the loudness match lifts it
            out[1000] = 0.9                                       # ... into one big peak
        return out, sr, {"seconds": 1.0}

    session.editor = loud
    ceiling = 10 ** (session.current["plan"]["master"]["ceiling_db"] / 20)
    for kwargs, part in (({"start_s": 2.0, "end_s": 4.0}, slice(int(2.1 * SR), int(3.9 * SR))),
                         ({"mode": "cover"}, slice(None))):
        session.goto(1)
        v = session.ai_edit("x", **kwargs)
        b, _ = read_audio(v["wav"])
        assert np.abs(b[part]).max() <= ceiling + 1e-3      # the original's own peaks are the limiter's


def test_bad_edits_are_refused_before_the_model_is_asked(session):
    for kwargs in ({"prompt": "  "}, {"prompt": "x", "mode": "remix"},
                   {"prompt": "x", "start_s": 5.0, "end_s": 3.0}, {"prompt": "x", "start_s": 60.0},
                   {"prompt": "x", "mode": "cover", "strength": 1.5}):
        with pytest.raises(ValueError):
            session.ai_edit(**kwargs)
    assert session.editor.calls == [] and len(session.versions) == 1


def test_export_writes_an_ai_edit_as_it_is(session, tmp_path):
    v2 = session.ai_edit("strings", start_s=1.0, end_s=2.0)
    path = session.export(str(tmp_path / "ai.wav"))
    a, _ = read_audio(v2["wav"])
    b, _ = read_audio(path)
    assert np.allclose(a, b, atol=1e-6)


def test_the_turn_log_records_the_edit(session):
    session.ai_edit("strings", start_s=1.0, end_s=2.0)
    saved = json.loads((Path(session.out_dir) / "session.json").read_text())
    assert saved["turns"][-1]["kind"] == "ai" and saved["turns"][-1]["base"] == 1
    assert saved["versions"][-1]["edit"]["mode"] == "repaint"


# --------------------------------------------------------------------------- #
# ACE-Step client
# --------------------------------------------------------------------------- #

class Resp:
    def __init__(self, body=None, content=b"", status=200):
        self.body, self.content, self.status_code = body, content, status
        self.ok = status < 400
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self.body is None:
            raise ValueError("no json")
        return self.body

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


def wav_bytes(seconds=1.0):
    buf = io.BytesIO()
    import soundfile as sf

    sf.write(buf, np.zeros((int(SR * seconds), 2), dtype=np.float32), SR, format="WAV")
    return buf.getvalue()


def test_client_submits_polls_and_downloads(monkeypatch, tmp_path):
    src = tmp_path / "v1.wav"
    write_audio(str(src), np.zeros((SR, 2)), SR)
    sent, polls = {}, iter([0, 0, 1])

    def post(url, files=None, data=None, timeout=None, **kw):
        if url.endswith("/release_task"):
            sent.update(data=data, file=files["src_audio"].name)
            return Resp({"data": {"task_id": "t1", "status": "queued"}, "code": 200, "error": None})
        result = [{"file": "/v1/audio?path=x.wav", "seed_value": "9", "dit_model": "turbo"}]
        return Resp({"data": [{"task_id": "t1", "status": next(polls), "result": json.dumps(result)}],
                     "code": 200, "error": None})

    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("requests.get", lambda url, timeout=None: Resp(content=wav_bytes()))
    audio, sr, info = acestep.edit(src, "rock drums", start_s=2.0, end_s=4.0, seed=9, poll_s=0)
    assert sr == SR and audio.shape == (SR, 2)
    assert info["seed"] == "9" and info["dit_model"] == "turbo"
    assert sent["file"] == str(src)
    assert sent["data"]["task_type"] == "repaint" and sent["data"]["repainting_end"] == 4.0
    assert sent["data"]["chunk_mask_mode"] == "explicit" and sent["data"]["seed"] == 9


def test_client_reports_a_failed_task(monkeypatch, tmp_path):
    src = tmp_path / "v1.wav"
    write_audio(str(src), np.zeros((SR, 2)), SR)

    def post(url, **kw):
        if url.endswith("/release_task"):
            return Resp({"data": {"task_id": "t1"}, "code": 200, "error": None})
        return Resp({"data": [{"task_id": "t1", "status": 2, "progress_text": "out of memory"}],
                     "code": 200, "error": None})

    monkeypatch.setattr("requests.post", post)
    with pytest.raises(RuntimeError, match="out of memory"):
        acestep.edit(src, "x", poll_s=0)


def test_status_when_the_server_is_not_running(monkeypatch):
    def refuse(*a, **kw):
        raise ConnectionError("refused")

    monkeypatch.setattr("requests.get", refuse)
    monkeypatch.setenv("ACESTEP_URL", "http://127.0.0.1:1/")
    st = acestep.status()
    assert st["ok"] is False and st["url"] == "http://127.0.0.1:1" and "refused" in st["error"]


# --------------------------------------------------------------------------- #
# web API and CLI
# --------------------------------------------------------------------------- #

def test_ai_edit_api(stems_dir, tmp_path):
    from fastapi.testclient import TestClient

    from mixpipe.web.mixer_server import create_app

    splitter = tmp_path / "splitter"
    (splitter / "song-1" / "stems").mkdir(parents=True)
    for p in Path(stems_dir).glob("*.wav"):
        (splitter / "song-1" / "stems" / p.name).write_bytes(p.read_bytes())
    c = TestClient(create_app(root=tmp_path / "mixer", splitter_root=splitter, use_llm=False,
                              run_inline=True, session_options={"align": False}, editor=fake_editor()))
    assert c.get("/api/ai-edit/status").json()["ok"] is True
    sid = c.post("/api/sessions", json={"splitter_song": "song-1"}).json()["id"]
    res = c.post(f"/api/sessions/{sid}/ai-edit", json={"prompt": "rock drums", "start": 2, "end": 4}).json()
    assert res["turn"]["kind"] == "ai" and res["current"] == 2
    assert res["versions"][-1]["edit"]["end_s"] == 4
    assert c.get(f"/api/sessions/{sid}/audio/2.wav").content[:4] == b"RIFF"
    assert c.post(f"/api/sessions/{sid}/ai-edit", json={"prompt": "x", "start": 5, "end": 1}).status_code == 400
    exp = c.post(f"/api/sessions/{sid}/export", json={"bit_depth": 24})
    assert exp.status_code == 200 and exp.content[:4] == b"RIFF"


def test_ai_edit_api_without_a_server(stems_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mixpipe.web.mixer_server import create_app

    monkeypatch.setattr(acestep, "status", lambda: {"url": "http://x", "ok": False, "model": None, "error": "down"})
    c = TestClient(create_app(root=tmp_path / "mixer", splitter_root=tmp_path / "none", use_llm=False,
                              run_inline=True, session_options={"align": False}))
    files = [("files", (p.name, p.read_bytes(), "audio/wav")) for p in sorted(Path(stems_dir).glob("*.wav"))]
    sid = c.post("/api/sessions", files=files).json()["id"]
    r = c.post(f"/api/sessions/{sid}/ai-edit", json={"prompt": "rock drums"})
    assert r.status_code == 503 and "make acestep" in r.json()["detail"]


@pytest.mark.parametrize("cmd,rest,want", [
    ("/ai", "0:20-0:32 rock drums", {"mode": "repaint", "start_s": 20.0, "end_s": 32.0, "prompt": "rock drums"}),
    ("/ai", "1:05.5- strings", {"mode": "repaint", "start_s": 65.5, "end_s": None, "prompt": "strings"}),
    ("/ai", "rock drums", {"mode": "repaint", "start_s": 0.0, "end_s": None, "prompt": "rock drums"}),
    ("/cover", "0.3 lo-fi tape", {"mode": "cover", "strength": 0.3, "prompt": "lo-fi tape"}),
    ("/cover", "lo-fi tape", {"mode": "cover", "strength": None, "prompt": "lo-fi tape"}),
])
def test_cli_ai_edit_commands(cmd, rest, want):
    assert parse_ai_edit(cmd, rest) == want
