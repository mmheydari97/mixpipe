"""Mix by chat: plan revision, the session (history, cache), the web API and the CLI."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mixpipe.advisor import apply_adjustments, diff_plans, revise_plan, rules_advisor
from mixpipe.analysis import analyze_all
from mixpipe.config import load_config
from mixpipe.io_utils import load_stems, write_audio
from mixpipe.mixdown import mixdown
from mixpipe.session import MixSession, llm_choice

ROOT = Path(__file__).resolve().parents[1]


def _demo_module():
    spec = importlib.util.spec_from_file_location("make_demo_session", ROOT / "scripts" / "make_demo_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def stems_dir(tmp_path_factory):
    """A short version of the synthetic session (8 stems, 8 s)."""
    d = tmp_path_factory.mktemp("chat_stems")
    stems, meta = _demo_module().build(seconds=8.0, seed=3)
    for name, audio in stems.items():
        write_audio(str(d / f"{name}.wav"), audio, meta["sr"])
    return d


@pytest.fixture(scope="module")
def base(stems_dir):
    cfg = load_config(None)
    stems = load_stems(stems_dir, role_table=cfg["roles"])
    plan = rules_advisor(stems, analyze_all(stems), cfg)
    return cfg, stems, plan


def fake_llm(reply: dict | str):
    calls = []

    def llm(system, payload):
        calls.append(json.loads(payload))
        return reply if isinstance(reply, str) else json.dumps(reply)

    llm.calls = calls
    return llm


# --------------------------------------------------------------------------- #
# adjustments
# --------------------------------------------------------------------------- #

def test_gain_adjustment_moves_the_fader_only(base):
    cfg, stems, plan = base
    new, rejected = apply_adjustments(plan, stems, [{"target": "acoustic_gtr", "param": "gain_db", "delta": -3}], cfg)
    assert rejected == []
    assert new["stems"]["acoustic_gtr"]["fader_db"] == pytest.approx(plan["stems"]["acoustic_gtr"]["fader_db"] - 3)
    assert new["stems"]["acoustic_gtr"]["gain_db"] == plan["stems"]["acoustic_gtr"]["gain_db"]
    for name in plan["stems"]:
        if name != "acoustic_gtr":
            assert new["stems"][name] == plan["stems"][name]
    assert new["master"] == plan["master"]
    assert [c["text"] for c in diff_plans(plan, new)] == ["acoustic_gtr -3.0 dB"]


def test_targets_resolve_by_name_role_and_words(base):
    cfg, stems, plan = base
    new, _ = apply_adjustments(plan, stems, [
        {"target": "role:vocal", "param": "reverb_db", "delta": 4},
        {"target": "the acoustic guitar", "param": "pan", "value": -0.5},
        {"target": "all", "param": "eq_add", "value": {"freq": 300, "gain_db": -1, "q": 1}},
    ], cfg)
    for v in ("vocal_lead", "vocal_harmony"):
        assert new["stems"][v]["sends"]["reverb_db"] == pytest.approx(plan["stems"][v]["sends"]["reverb_db"] + 4)
    assert new["stems"]["acoustic_gtr"]["pan"] == -0.5
    assert all(len(new["stems"][n]["eq"]) == min(6, len(plan["stems"][n]["eq"]) + 1) for n in plan["stems"])


def test_hostile_adjustments_are_clamped_or_rejected(base):
    cfg, stems, plan = base
    new, rejected = apply_adjustments(plan, stems, [
        {"target": "bass", "param": "gain_db", "delta": 400},            # step limit, then hard limit
        {"target": "drums", "param": "gain_db", "value": float("nan")},  # unusable: ignored
        {"target": "bass", "param": "os.system", "value": "rm -rf /"},   # not a parameter
        {"target": "tuba", "param": "gain_db", "delta": -3},             # not in the session
        {"target": "master", "param": "target_lufs", "value": 5},        # clamped to the limit
        {"target": "master", "param": "ceiling_db", "value": 3},
        "not even a dict",
    ], cfg)
    assert new["stems"]["bass"]["fader_db"] == 12.0
    assert new["stems"]["drums"] == plan["stems"]["drums"]
    assert new["master"]["target_lufs"] == -6.0 and new["master"]["ceiling_db"] == -0.1
    assert any("os.system" in r for r in rejected) and any("tuba" in r for r in rejected)


def test_more_reverb_on_a_dry_stem_starts_from_a_usable_level(base):
    cfg, stems, plan = base
    assert plan["stems"]["bass"]["sends"]["reverb_db"] <= -60
    new, _ = apply_adjustments(plan, stems, [{"target": "bass", "param": "reverb_db", "delta": 4}], cfg)
    assert -30 < new["stems"]["bass"]["sends"]["reverb_db"] < -10


# --------------------------------------------------------------------------- #
# one turn
# --------------------------------------------------------------------------- #

def test_llm_turn_applies_relative_adjustments_and_reports_the_diff(base):
    cfg, stems, plan = base
    llm = fake_llm({"reply": "Pulling the acoustic down.",
                    "adjustments": [{"target": "acoustic_gtr", "param": "gain_db", "delta": -3}]})
    out = revise_plan("the acoustic guitar is too loud", plan, stems, cfg, llm=llm,
                      stem_levels={"acoustic_gtr": -4.0})
    assert out["source"] == "chat:llm"
    assert out["reply"] == "Pulling the acoustic down."
    assert [c["text"] for c in out["changes"]] == ["acoustic_gtr -3.0 dB"]
    sent = llm.calls[0]
    assert sent["message"] == "the acoustic guitar is too loud"
    assert sent["level_in_mix_db"] == {"acoustic_gtr": -4.0}
    assert "reason" not in json.dumps(sent["plan"])         # the model gets numbers, not our prose


def test_llm_claiming_a_change_it_did_not_make_is_flagged(base):
    cfg, stems, plan = base
    out = revise_plan("make it warmer", plan, stems, cfg,
                      llm=fake_llm({"reply": "Done, I made it much warmer!", "adjustments": []}))
    assert out["changes"] == [] and "Nothing in the mix was changed" in out["reply"]


def test_llm_reply_wrapped_in_prose_is_still_read(base):
    cfg, stems, plan = base
    raw = 'Sure! Here you go:\n```json\n{"reply": "ok", "adjustments": [{"target": "bass", "param": "gain_db", "delta": 2}]}\n```'
    out = revise_plan("bass up", plan, stems, cfg, llm=fake_llm(raw))
    assert out["source"] == "chat:llm" and out["changes"][0]["text"] == "bass +2.0 dB"


def test_malformed_llm_reply_falls_back_to_the_keyword_rules(base):
    cfg, stems, plan = base
    llm = fake_llm("I think the guitar is too loud, so")
    out = revise_plan("the acoustic guitar is too loud", plan, stems, cfg, llm=llm)
    assert len(llm.calls) == 2                               # retried once
    assert out["source"] == "chat:rules" and "llm_error" in out
    assert out["reply"].startswith("(model unavailable")
    assert [c["text"] for c in out["changes"]] == ["acoustic_gtr -3.0 dB"]


def test_keyword_rules_say_what_they_did_not_understand(base):
    cfg, stems, plan = base
    out = revise_plan("make it sound like the eighties", plan, stems, cfg, use_llm=False)
    assert out["changes"] == [] and "Not understood" in out["reply"]
    out = revise_plan("make the harmonica less honky", plan, stems, cfg, use_llm=False)
    texts = [c["text"] for c in out["changes"]]
    assert texts == ["harmonica EQ -2.0 dB at 1.2 kHz"]      # tone, not a level drop


def test_llm_choice_uses_openrouter_when_only_its_key_is_set(monkeypatch):
    cfg = load_config(None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert llm_choice(cfg) == ("openrouter", cfg["advisor"]["openrouter_model"])
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert llm_choice(cfg) is None


# --------------------------------------------------------------------------- #
# the session
# --------------------------------------------------------------------------- #

@pytest.fixture()
def session(stems_dir, tmp_path):
    return MixSession(str(stems_dir), out_dir=str(tmp_path / "s"), use_llm=False, align=False)


def test_acoustic_too_loud_end_to_end(session):
    """The case that motivated the feature: say it, and the rendered mix changes by that much."""
    v1 = session.first_mix()
    v2 = session.chat("the acoustic guitar is too loud")
    assert v2["n"] == 2 and v2["parent"] == 1
    assert Path(v2["wav"]).is_file()
    drop = v2["levels"]["acoustic_gtr"] - v1["levels"]["acoustic_gtr"]
    assert drop == pytest.approx(-3.0, abs=0.5)
    for name in v1["levels"]:
        if name != "acoustic_gtr":
            assert abs(v2["levels"][name] - v1["levels"][name]) < 0.6, name
    # the master still lands on target, so the change is balance, not just level
    assert abs(v2["output"]["lufs"] - v1["output"]["lufs"]) < 0.4
    assert v2["output"]["true_peak_db"] <= v2["plan"]["master"]["ceiling_db"] + 0.05


def test_history_undo_redo_and_branches(session):
    session.first_mix()
    v2 = session.chat("bass louder")
    v3 = session.chat("more reverb on the vocals")
    assert session.undo()["n"] == 2 and session.undo()["n"] == 1 and session.undo()["n"] == 1
    assert session.redo()["n"] == 2 and session.redo()["n"] == 3
    session.goto(2)
    v4 = session.chat("drums quieter")                      # a branch off v2; v3 is kept
    assert v4["parent"] == v2["n"] and len(session.versions) == 4
    assert [v["n"] for v in session.lineage()] == [1, 2, 4]
    assert v4["plan"]["stems"]["vocal_lead"]["sends"] == v2["plan"]["stems"]["vocal_lead"]["sends"]
    assert session.version(3)["n"] == v3["n"]
    noop = session.chat("hmm")
    assert noop["n"] is None and len(session.versions) == 4
    # the conversation keeps replies that changed nothing, so a reloaded page shows them
    assert session.turns[-1]["n"] is None and session.turns[-1]["message"] == "hmm"
    assert [t["n"] for t in session.turns] == [1, 2, 3, 4, None]
    with pytest.raises(KeyError):
        session.goto(99)


def test_cleaned_stem_cache_renders_the_same_audio(session):
    v = session.first_mix()
    warm = session.render(v["plan"])["bus"]
    cold, _, _ = mixdown(session.stems, v["plan"], session.cfg, verbose=False)
    assert np.allclose(warm, cold, atol=1e-6)


def test_direct_fader_edit_goes_through_the_same_path(session):
    session.first_mix()
    v = session.set_params([{"target": "keys_pad", "param": "gain_db", "value": -6.0}])
    assert v["source"] == "manual" and v["plan"]["stems"]["keys_pad"]["fader_db"] == -6.0
    assert session.set_params([{"target": "keys_pad", "param": "gain_db", "value": -6.0}]) is None


def test_long_songs_get_a_preview_slice_and_export_renders_everything(stems_dir, tmp_path, monkeypatch):
    import mixpipe.session as sess

    monkeypatch.setattr(sess, "PREVIEW_OVER_S", 5.0)
    monkeypatch.setattr(sess, "PREVIEW_S", 3.0)
    s = MixSession(str(stems_dir), out_dir=str(tmp_path / "p"), use_llm=False, align=False)
    assert s.preview and s.duration_s == 3.0 and s.stems[0].duration == pytest.approx(3.0, abs=0.01)
    v = s.first_mix()
    assert v["output"]["duration_s"] < 6.0
    out = s.export(str(tmp_path / "full.wav"), bit_depth=16)
    import soundfile as sf

    info = sf.info(out)
    assert info.duration > 7.9 and info.subtype == "PCM_16"


# --------------------------------------------------------------------------- #
# web API
# --------------------------------------------------------------------------- #

def test_mixer_api_flow(stems_dir, tmp_path):
    from fastapi.testclient import TestClient

    from mixpipe.web.mixer_server import create_app

    splitter = tmp_path / "splitter"
    (splitter / "song-1" / "stems").mkdir(parents=True)
    for p in Path(stems_dir).glob("*.wav"):
        (splitter / "song-1" / "stems" / p.name).write_bytes(p.read_bytes())
    llm = fake_llm({"reply": "Acoustic down 3 dB.",
                    "adjustments": [{"target": "acoustic_gtr", "param": "gain_db", "delta": -3}]})
    app = create_app(root=tmp_path / "mixer", splitter_root=splitter, llm=llm, run_inline=True,
                     session_options={"align": False})
    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert [s["id"] for s in c.get("/api/splitter-songs").json()] == ["song-1"]
    assert c.post("/api/sessions", json={"splitter_song": "../etc"}).status_code == 404

    sid = c.post("/api/sessions", json={"splitter_song": "song-1", "llm": True, "notes": "bass up"}).json()["id"]
    st = c.get(f"/api/sessions/{sid}").json()
    assert st["state"] == "ready", st["error"]
    assert st["current"] == 1 and st["llm"]
    off = c.post("/api/sessions", json={"splitter_song": "song-1", "llm": False}).json()["id"]
    assert c.get(f"/api/sessions/{off}").json()["llm"] is None      # the page's checkbox is honoured

    res = c.post(f"/api/sessions/{sid}/chat", json={"message": "the acoustic guitar is too loud"}).json()
    assert res["turn"]["n"] == 2 and res["turn"]["changes"][0]["text"] == "acoustic_gtr -3.0 dB"
    assert res["current"] == 2 and len(res["versions"]) == 2
    wav = c.get(f"/api/sessions/{sid}/audio/2.wav")
    assert wav.status_code == 200 and wav.content[:4] == b"RIFF"

    res = c.post(f"/api/sessions/{sid}/set", json={"adjustments": [{"target": "bass", "param": "mute", "value": True}]}).json()
    assert res["turn"]["n"] == 3 and res["versions"][-1]["stems"]["bass"]["mute"] is True
    assert c.post(f"/api/sessions/{sid}/undo").json()["current"] == 2
    assert c.post(f"/api/sessions/{sid}/goto/1").json()["current"] == 1
    assert c.post(f"/api/sessions/{sid}/goto/42").status_code == 404
    exp = c.post(f"/api/sessions/{sid}/export", json={"bit_depth": 16})
    assert exp.status_code == 200 and exp.content[:4] == b"RIFF"
    assert c.get("/api/sessions/nope").status_code == 404


def test_mixer_api_upload(stems_dir, tmp_path):
    from fastapi.testclient import TestClient

    from mixpipe.web.mixer_server import create_app

    c = TestClient(create_app(root=tmp_path / "mixer", splitter_root=tmp_path / "none", use_llm=False,
                              run_inline=True, session_options={"align": False}))
    files = [("files", (p.name, p.read_bytes(), "audio/wav")) for p in sorted(Path(stems_dir).glob("*.wav"))[:3]]
    sid = c.post("/api/sessions", files=files, data={"notes": "bass louder", "llm": "true"}).json()["id"]
    st = c.get(f"/api/sessions/{sid}").json()
    assert st["state"] == "ready", st["error"]
    assert st["llm"] is None                       # use_llm=False on the server wins over the form
    bad = c.post("/api/sessions", files=[("files", ("evil.sh", b"#!/bin/sh", "text/plain"))])
    assert bad.status_code == 400


# --------------------------------------------------------------------------- #
# live model (opt in: costs a fraction of a cent)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not (os.environ.get("OPENROUTER_API_KEY") and os.environ.get("MIXPIPE_LIVE_TESTS")),
                    reason="set MIXPIPE_LIVE_TESTS=1 (and OPENROUTER_API_KEY) to call the real model")
def test_live_openrouter_turn(base):
    cfg, stems, plan = base
    out = revise_plan("the acoustic guitar is too loud", plan, stems, cfg, provider="openrouter",
                      model=cfg["advisor"]["openrouter_model"])
    assert out["source"].startswith("chat:llm"), out.get("llm_error")
    level = [c for c in out["changes"] if c["target"] == "acoustic_gtr" and c["param"] == "fader_db"]
    assert level and level[0]["text"].startswith("acoustic_gtr -")


# --------------------------------------------------------------------------- #
# silent stems (a separated "bass" of a song that has none)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def stems_with_silence(stems_dir, tmp_path):
    d = tmp_path / "with_silence"
    d.mkdir()
    for p in Path(stems_dir).glob("*.wav"):
        (d / p.name).write_bytes(p.read_bytes())
    rng = np.random.default_rng(0)
    write_audio(str(d / "bass.wav"), (rng.standard_normal(8 * 48_000) * 1e-4).astype(np.float32), 48_000)
    return d


def test_advisors_never_boost_a_silent_stem(stems_with_silence):
    cfg = load_config(None)
    stems = load_stems(stems_with_silence, role_table=cfg["roles"])
    analysis = analyze_all(stems)
    bass = next(a for a in analysis["stems"] if a["name"] == "bass")
    assert bass["silent"] and analysis["summary"]["silent_stems"] == ["bass"]
    assert analysis["summary"]["quietest_stem"] != "bass"
    assert np.isfinite(analysis["summary"]["lufs_spread_db"])

    plan = rules_advisor(stems, analysis, cfg)
    assert plan["stems"]["bass"]["gain_db"] <= 0.0 and plan["stems"]["bass"]["fader_db"] <= 0.0

    from mixpipe.advisor import llm_advisor

    boosted = {"stems": {"bass": {"gain_db": 8.0, "fader_db": 4.0}}, "master": {}}
    import mixpipe.advisor as adv

    orig = adv._call_anthropic
    adv._call_anthropic = lambda payload, model, max_tokens=4000: json.dumps(boosted)
    try:
        plan = llm_advisor(stems, analysis, cfg, provider="anthropic", model="x")
    finally:
        adv._call_anthropic = orig
    assert plan["source"].startswith("llm") and plan["stems"]["bass"]["gain_db"] == 0.0
    assert "not boosted" in plan["stems"]["bass"]["reason"]


def test_session_state_with_a_silent_stem_is_valid_json(stems_with_silence, tmp_path):
    from fastapi.testclient import TestClient

    from mixpipe.web.mixer_server import create_app

    splitter = tmp_path / "sp"
    (splitter / "s1").mkdir(parents=True)
    (splitter / "s1" / "stems").symlink_to(stems_with_silence, target_is_directory=True)
    c = TestClient(create_app(root=tmp_path / "m", splitter_root=splitter, use_llm=False, run_inline=True,
                              session_options={"align": False}))
    sid = c.post("/api/sessions", json={"splitter_song": "s1"}).json()["id"]
    r = c.get(f"/api/sessions/{sid}")
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["state"] == "ready" and st["versions"][0]["levels"]["bass"] is None
    assert "Infinity" not in r.text and "NaN" not in r.text


@pytest.mark.parametrize("message,expected", [
    ("vocals up a little", {"vocal_lead": 1.5, "vocal_harmony": 1.5}),
    ("the bass is way too loud", {"bass": -6.0}),
    ("turn the drums up", {"drums": 3.0}),
    ("the keys are a bit too loud", {"keys_pad": -1.5}),
])
def test_keyword_level_steps_follow_the_wording(base, message, expected):
    cfg, stems, plan = base
    out = revise_plan(message, plan, stems, cfg, use_llm=False)
    moved = {c["target"]: c for c in out["changes"]}
    assert set(moved) == set(expected), out["reply"]
    for name, db in expected.items():
        assert out["plan"]["stems"][name]["fader_db"] - plan["stems"][name]["fader_db"] == pytest.approx(db)


# --------------------------------------------------------------------------- #
# choosing the OpenRouter model
# --------------------------------------------------------------------------- #

@pytest.fixture()
def model_calls(monkeypatch):
    """Record which model every LLM call goes to (no network): the first-mix advisor
    and the chat turns. A message containing "fail" makes the model misbehave."""
    import mixpipe.advisor as adv

    calls = []

    def first_mix(payload, model, max_tokens=None):
        calls.append(("plan", model))
        return "{}"

    def turn(system, payload, provider, model):
        calls.append(("turn", f"{provider}:{model}"))
        if "fail" in json.loads(payload)["message"]:
            return "not json"
        return json.dumps({"reply": "ok", "adjustments": [{"target": "bass", "param": "gain_db", "delta": -1}]})

    monkeypatch.setattr(adv, "_call_openrouter", first_mix)
    monkeypatch.setattr(adv, "_llm_text", turn)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return calls


@pytest.fixture()
def splitter_song(stems_dir, tmp_path):
    root = tmp_path / "splitter"
    (root / "song-1").mkdir(parents=True)
    (root / "song-1" / "stems").symlink_to(stems_dir, target_is_directory=True)
    return root


def test_mixer_picks_and_switches_the_openrouter_model(splitter_song, tmp_path, model_calls):
    from fastapi.testclient import TestClient

    from mixpipe.web.mixer_server import create_app

    c = TestClient(create_app(root=tmp_path / "m", splitter_root=splitter_song, run_inline=True,
                              session_options={"align": False}))
    cfg = c.get("/api/config").json()
    assert cfg["openrouter"] == {"has_key": True, "default_model": load_config(None)["advisor"]["openrouter_model"]}
    assert c.get("/api/openrouter/status").json() == {"has_key": True}

    first = "meta-llama/llama-3.3-70b-instruct:free"
    sid = c.post("/api/sessions", json={"splitter_song": "song-1", "llm": True, "model": first}).json()["id"]
    st = c.get(f"/api/sessions/{sid}").json()
    assert st["state"] == "ready", st["error"]
    assert st["llm"] == {"provider": "openrouter", "model": first}
    assert model_calls == [("plan", first)] and st["versions"][0]["model"] == first

    turn = c.post(f"/api/sessions/{sid}/chat", json={"message": "bass down"}).json()["turn"]
    assert model_calls[-1] == ("turn", f"openrouter:{first}") and turn["model"] == first

    second = "anthropic/claude-haiku-4.5"
    st = c.post(f"/api/sessions/{sid}/model", json={"model": second}).json()
    assert st["llm"] == {"provider": "openrouter", "model": second}
    turn = c.post(f"/api/sessions/{sid}/chat", json={"message": "bass down"}).json()["turn"]
    assert model_calls[-1] == ("turn", f"openrouter:{second}") and turn["model"] == second
    st = c.get(f"/api/sessions/{sid}").json()
    assert [v["model"] for v in st["versions"]] == [first, first, second]    # earlier versions keep theirs
    assert [t["model"] for t in st["turns"]] == [first, first, second]

    # a model that answers with nonsense: the rules take over, and the turn says so
    turn = c.post(f"/api/sessions/{sid}/chat", json={"message": "fail: bass down"}).json()["turn"]
    assert turn["model"] is None and turn["llm_error"] and turn["source"] == "chat:rules"

    assert c.post(f"/api/sessions/{sid}/model", json={"model": "not a model"}).status_code == 400
    st = c.post(f"/api/sessions/{sid}/model", json={"model": None}).json()
    assert st["llm"] is None
    n = len(model_calls)
    turn = c.post(f"/api/sessions/{sid}/chat", json={"message": "bass down a little"}).json()["turn"]
    assert len(model_calls) == n and turn["source"] == "chat:rules"     # no model asked

    off = c.post("/api/sessions", json={"splitter_song": "song-1", "llm": False, "model": first}).json()["id"]
    assert c.get(f"/api/sessions/{off}").json()["llm"] is None              # unticked: the model is ignored


def test_mixer_refuses_a_model_it_cannot_use(splitter_song, stems_dir, tmp_path, model_calls, monkeypatch):
    from fastapi.testclient import TestClient

    from mixpipe.web.mixer_server import create_app

    root = tmp_path / "m"
    c = TestClient(create_app(root=root, splitter_root=splitter_song, run_inline=True,
                              session_options={"align": False}))
    bad = c.post("/api/sessions", json={"splitter_song": "song-1", "model": "../../etc"})
    assert bad.status_code == 400
    monkeypatch.delenv("OPENROUTER_API_KEY")
    r = c.post("/api/sessions", json={"splitter_song": "song-1", "model": "anthropic/claude-haiku-4.5"})
    assert r.status_code == 400 and "OPENROUTER_API_KEY" in r.json()["detail"]
    files = [("files", (p.name, p.read_bytes(), "audio/wav")) for p in sorted(Path(stems_dir).glob("*.wav"))[:2]]
    r = c.post("/api/sessions", files=files, data={"model": "anthropic/claude-haiku-4.5"})
    assert r.status_code == 400 and not any(root.iterdir())                # nothing written for a refused upload
    assert c.get("/api/config").json()["openrouter"]["has_key"] is False

    rules_only = TestClient(create_app(root=tmp_path / "r", splitter_root=splitter_song, use_llm=False,
                                       run_inline=True, session_options={"align": False}))
    assert rules_only.get("/api/config").json()["openrouter"] is None
    sid = rules_only.post("/api/sessions", json={"splitter_song": "song-1"}).json()["id"]
    assert rules_only.post(f"/api/sessions/{sid}/model", json={"model": "anthropic/claude-haiku-4.5"}).status_code == 400
    assert model_calls == []


def test_openrouter_ids_go_to_openrouter(monkeypatch):
    from mixpipe.advisor import resolve_llm

    cfg = load_config(None)
    cfg["advisor"].update(provider="anthropic", model="claude-sonnet-4-6")
    assert resolve_llm(None, "google/gemini-2.5-flash", cfg) == ("openrouter", "google/gemini-2.5-flash")
    assert resolve_llm("openrouter", None, cfg) == ("openrouter", cfg["advisor"]["openrouter_model"])
    assert resolve_llm(None, None, cfg) == ("anthropic", "claude-sonnet-4-6")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    assert llm_choice(cfg, model="google/gemini-2.5-flash") == ("openrouter", "google/gemini-2.5-flash")
    assert llm_choice(cfg) == ("anthropic", "claude-sonnet-4-6")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    # a Claude id is not an OpenRouter id: without the Anthropic key, OpenRouter's default is used
    assert llm_choice(cfg, model="claude-sonnet-4-6") == ("openrouter", cfg["advisor"]["openrouter_model"])
