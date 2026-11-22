"""The mixer's console (lanes + inspector): regions, editing one EQ band, input
trim, the EQ curve, the per-channel renders the page mixes live, the master
render and sessions coming back after a restart.

Run with:  PYTHONPATH=src:tests python -m pytest tests/test_console.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mixpipe.advisor import apply_adjustments, diff_plans
from mixpipe.eq import apply_eq_chain, eq_response
from mixpipe.mixdown import process_stem, region_envelope, region_points
from mixpipe.plan import validate_regions
from mixpipe.session import MixSession

from test_chat import base, stems_dir  # noqa: F401  (fixtures)

SR = 48_000


# ---- regions ---------------------------------------------------------------------


def test_regions_are_validated_sorted_and_never_overlap():
    out = validate_regions([
        {"start_s": 5, "end_s": 8, "gain_db": -6},
        {"start_s": 1, "end_s": 3},                      # a mute by default
        {"start_s": 7, "end_s": 9, "gain_db": -100},     # overlaps: starts where the one before ends
        {"start_s": 4, "end_s": 4},                      # empty: dropped
        "junk",
    ])
    assert [(r["start_s"], r["end_s"], r["gain_db"]) for r in out] == [(1, 3, -60), (5, 8, -6), (8, 9, -60)]


def test_region_envelope_fades_and_follows_offset_and_speed():
    regions = [{"start_s": 1.0, "end_s": 2.0, "gain_db": -60.0, "fade_ms": 100.0}]
    pts = region_points(regions)
    assert pts == [(1.0, 1.0), (1.1, 0.0), (1.9, 0.0), (2.0, 1.0)]
    env = region_envelope(regions, 3 * SR, SR)
    assert env[int(0.5 * SR)] == 1.0 and env[int(1.5 * SR)] == 0.0 and 0.4 < env[int(1.05 * SR)] < 0.6
    # a preview slice starting at 1 s: the mute is at its 0.5 s mark
    assert region_envelope(regions, SR, SR, offset_s=1.0)[int(0.5 * SR)] == 0.0
    # at double speed one second of audio covers two of the song
    assert region_envelope(regions, SR, SR, speed=2.0)[int(0.75 * SR)] == 0.0
    assert region_envelope([], 10, SR) is None


def test_a_region_silences_the_stem_in_the_render(base):
    cfg, stems, plan = base
    name = stems[0].name
    settings = {**plan["stems"][name], "regions": validate_regions([{"start_s": 2, "end_s": 4}])}
    out = process_stem(stems[0], settings, cfg)
    a = out.audio
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
    assert rms(a[int(2.2 * stems[0].sr):int(3.8 * stems[0].sr)]) < 1e-6
    assert rms(a[int(5 * stems[0].sr):int(7 * stems[0].sr)]) > 1e-4


def test_region_set_remove_and_the_change_list(base):
    cfg, stems, plan = base
    name = stems[0].name
    new, rej = apply_adjustments(plan, stems, [
        {"target": name, "param": "region_set", "value": {"start_s": 70, "end_s": 100}}], cfg)
    assert not rej and new["stems"][name]["regions"][0]["gain_db"] == -60
    assert diff_plans(plan, new)[0]["text"] == f"{name} muted 1:10-1:40"
    lower, _ = apply_adjustments(new, stems, [
        {"target": name, "param": "region_set", "value": {"start_s": 80, "end_s": 90, "gain_db": -6}}], cfg)
    assert [(r["start_s"], r["gain_db"]) for r in lower["stems"][name]["regions"]] == [(80, -6)]
    gone, _ = apply_adjustments(lower, stems, [
        {"target": name, "param": "region_remove", "value": {"start_s": 85, "end_s": 86}}], cfg)
    assert gone["stems"][name]["regions"] == []
    assert "back to normal" in diff_plans(lower, gone)[0]["text"]


# ---- EQ bands, trim, the curve -------------------------------------------------------


def test_eq_set_changes_one_band_and_eq_remove_drops_it(base):
    cfg, stems, plan = base
    name = stems[0].name
    p1, _ = apply_adjustments(plan, stems, [{"target": name, "param": "eq_clear"},
                                            {"target": name, "param": "eq_add", "value": {"freq": 300, "gain_db": -3}},
                                            {"target": name, "param": "eq_add", "value": {"freq": 5000, "gain_db": 2}}], cfg)
    p2, rej = apply_adjustments(p1, stems, [{"target": name, "param": "eq_set",
                                             "value": {"index": 0, "freq": 250, "gain_db": -5}}], cfg)
    assert not rej
    assert p2["stems"][name]["eq"][0] == {"freq": 250.0, "gain_db": -5.0, "q": 1.0, "type": "peaking"}
    assert p2["stems"][name]["eq"][1]["freq"] == 5000
    p3, _ = apply_adjustments(p2, stems, [{"target": name, "param": "eq_remove", "value": 0}], cfg)
    assert [b["freq"] for b in p3["stems"][name]["eq"]] == [5000]
    assert any("band removed" in c["text"] for c in diff_plans(p2, p3))
    _, rej = apply_adjustments(p3, stems, [{"target": name, "param": "eq_remove", "value": 7}], cfg)
    assert rej
    p4, _ = apply_adjustments(p3, stems, [{"target": name, "param": "trim_db", "value": -4}], cfg)
    assert p4["stems"][name]["gain_db"] == -4 and p4["stems"][name]["fader_db"] == p3["stems"][name]["fader_db"]
    m, _ = apply_adjustments(p4, stems, [{"target": "master", "param": "eq_set",
                                          "value": {"index": 0, "freq": 100, "gain_db": 1.5, "type": "low_shelf"}},
                                         {"target": "master", "param": "glue_threshold_db", "value": -20}], cfg)
    assert m["master"]["master_eq"][0]["type"] == "low_shelf" and m["master"]["glue_comp"]["threshold_db"] == -20


def test_the_curve_is_what_the_audio_gets():
    bands = [{"freq": 1000.0, "gain_db": 6.0, "q": 1.0, "type": "peaking"}]
    curve = eq_response(80.0, 0.0, bands, SR)
    f = np.array(curve["freq"])
    k = int(np.argmin(np.abs(f - 1000)))
    assert curve["db"][k] == pytest.approx(6.0, abs=0.3)
    assert curve["db"][0] < -20                                  # the high-pass at 80 Hz
    t = np.arange(SR) / SR
    tone = np.sin(2 * np.pi * f[k] * t).astype(np.float32)
    out = apply_eq_chain(tone, 80.0, 0.0, bands, SR)
    gain_db = 20 * np.log10(np.std(out[SR // 2:]) / np.std(tone[SR // 2:]))
    assert gain_db == pytest.approx(curve["db"][k], abs=0.3)


# ---- the session's console ----------------------------------------------------------------


@pytest.fixture()
def session(stems_dir, tmp_path):  # noqa: F811
    s = MixSession(str(stems_dir), out_dir=str(tmp_path / "s"), use_llm=False, align=False)
    s.first_mix()
    return s


def test_console_channels_mix_back_to_the_render(session):
    import soundfile as sf

    con = session.console()
    assert {c["name"] for c in con["channels"]} == {st.name for st in session.stems}
    folder = session.out_dir / "console"
    plan = session.current["plan"]
    # fader + pan on the page's side, the same functions the render uses
    from mixpipe.mixdown import place_stem

    total = None
    for c in con["channels"]:
        audio, sr = sf.read(str(folder / c["file"]), dtype="float32", always_2d=False)
        audio = audio * 10 ** (c["scale_db"] / 20)
        if plan["stems"][c["name"]]["mute"]:
            continue
        placed = place_stem(audio, sr, plan["stems"][c["name"]])
        total = placed if total is None else total + placed[: len(total)]
    from mixpipe.mixdown import mixdown

    stems = session.full_stems()
    bus, _, _ = mixdown(stems, plan, session.cfg, verbose=False, cleaned=session._clean_all(stems, session._full_cleaned or session.cleaned))
    if con["fx"]:
        ret, _ = sf.read(str(folder / con["fx"]["file"]), dtype="float32", always_2d=True)
        total = total + ret[: len(total)] * 10 ** (con["fx"]["scale_db"] / 20)
    # the bus is trimmed to headroom; compare shapes, not level
    n = min(len(total), len(bus))
    a, b = total[:n].ravel(), bus[:n].ravel()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert corr > 0.99


def test_console_is_cached_and_only_a_chain_change_renders_again(session):
    first = {c["name"]: c["file"] for c in session.console()["channels"]}
    name = next(iter(first))
    session.set_params([{"target": name, "param": "gain_db", "delta": -3}])          # a fader: live
    after_fader = {c["name"]: c["file"] for c in session.console()["channels"]}
    assert after_fader == first
    session.set_params([{"target": name, "param": "eq_add", "value": {"freq": 2000, "gain_db": -4}}])
    after_eq = {c["name"]: c["file"] for c in session.console()["channels"]}
    assert after_eq[name] != first[name]
    assert all(after_eq[k] == first[k] for k in first if k != name)
    peaks = session.peaks(points=200)
    assert set(peaks) == set(first) and 150 <= len(peaks[name]) <= 400


def test_master_render_and_restore(session, tmp_path):
    stats = session.render_master()
    assert Path(session.out_dir / stats["file"]).exists()
    assert stats["output"]["lufs"] == pytest.approx(session.current["plan"]["master"]["target_lufs"], abs=1.5)
    assert stats["level_db"] and stats["level_hop_s"] == 0.5
    assert session.render_master() == json.loads((session.out_dir / f"master_v{stats['n']}.json").read_text())
    name = session.stems[0].name
    session.set_params([{"target": name, "param": "region_set", "value": {"start_s": 1, "end_s": 2}}])
    back = MixSession.restore(str(session.out_dir), use_llm=False, align=False)
    assert [v["n"] for v in back.versions] == [v["n"] for v in session.versions]
    assert back.current["n"] == session.current["n"]
    assert back.current["plan"]["stems"][name]["regions"][0]["start_s"] == 1
    assert back.eq_curve(name)["freq"]


def test_console_routes_and_reload(stems_dir, tmp_path):  # noqa: F811
    from fastapi.testclient import TestClient

    from mixpipe.web.mixer_server import create_app

    splitter = tmp_path / "splitter"
    (splitter / "song-1" / "stems").mkdir(parents=True)
    for p in Path(stems_dir).glob("*.wav"):
        (splitter / "song-1" / "stems" / p.name).write_bytes(p.read_bytes())
    mk = lambda: create_app(root=tmp_path / "mixer", splitter_root=splitter, use_llm=False,  # noqa: E731
                            run_inline=True, session_options={"align": False})
    c = TestClient(mk())
    sid = c.post("/api/sessions", json={"splitter_song": "song-1"}).json()["id"]
    con = c.get(f"/api/sessions/{sid}/console").json()
    f = c.get(f"/api/sessions/{sid}/console/files/{con['channels'][0]['file']}")
    assert f.status_code == 200 and f.headers["content-type"] == "audio/flac"
    assert c.get(f"/api/sessions/{sid}/console/files/..%2Fsession.json").status_code == 404
    assert c.get(f"/api/sessions/{sid}/eq-curve?target=master").json()["freq"]
    assert c.get(f"/api/sessions/{sid}/eq-curve?target=nope").status_code == 404
    assert set(c.get(f"/api/sessions/{sid}/peaks").json()) == {ch["name"] for ch in con["channels"]}
    m = c.post(f"/api/sessions/{sid}/master").json()
    assert c.get(f"/api/sessions/{sid}/console/files/{m['file']}").status_code == 200
    # a restarted server lists the session and reopens it on first use
    c2 = TestClient(mk())
    listed = c2.get("/api/sessions").json()
    assert [s["id"] for s in listed] == [sid] and listed[0]["state"] == "stored" and listed[0]["title"] == "song-1"
    assert c2.get(f"/api/sessions/{sid}").json()["state"] == "ready"
    assert c2.delete(f"/api/sessions/{sid}").json()["ok"]
    assert not (tmp_path / "mixer" / sid).exists()
    assert TestClient(mk()).get("/api/sessions").json() == []
