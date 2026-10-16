"""Tests for the chord finder.

Run with:  PYTHONPATH=src python -m pytest tests/test_chords.py -q

Needs requirements-chords.txt (librosa, fastapi). No model downloads and no
network: separation and transcription are replaced with fakes.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("librosa")

from mixpipe import openrouter  # noqa: E402
from mixpipe.chords.align import build_rows, reanchor, word_diff  # noqa: E402
from mixpipe.chords.recognize import recognize  # noqa: E402
from mixpipe.chords.sheet import to_chordpro, to_text  # noqa: E402
from mixpipe.chords.transcribe import normalize_remote, split_lines  # noqa: E402
from mixpipe.io_utils import Stem  # noqa: E402
from mixpipe.pitch import detect_key  # noqa: E402

SR = 44_100
BPM = 120
BEAT = 60.0 / BPM
FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# synthetic music
# --------------------------------------------------------------------------- #

def tone(midi: int, n: int) -> np.ndarray:
    f = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(n) / SR
    return sum((0.6 ** (h - 1)) * np.sin(2 * np.pi * f * h * t) for h in range(1, 6))


def strummed(notes, bass, beats=4) -> np.ndarray:
    """A chord re-struck on every beat, with a separate bass note."""
    n = int(BEAT * SR)
    env = np.exp(-np.arange(n) / (0.35 * SR))
    hit = (sum(tone(m, n) for m in notes) * 0.25 + tone(bass, n) * 0.4) * env
    return np.tile(hit, beats)


PROGRESSION = [
    ("C", [60, 64, 67], 48), ("G", [59, 62, 67], 43), ("Am", [60, 64, 69], 45),
    ("F", [60, 65, 69], 41), ("G7", [59, 62, 65, 67], 43), ("C/E", [60, 64, 67], 52),
    ("Dm", [62, 65, 69], 50), ("Em", [59, 64, 67], 52),
]


@pytest.fixture(scope="module")
def progression_audio() -> np.ndarray:
    y = np.concatenate([strummed(notes, bass) for _, notes, bass in PROGRESSION])
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


@pytest.fixture(scope="module")
def recognized(progression_audio):
    return recognize(progression_audio, SR)


# --------------------------------------------------------------------------- #
# recognition
# --------------------------------------------------------------------------- #

def test_recognizes_progression_top1(recognized):
    labels = [s["candidates"][0]["label"] for s in recognized["segments"]]
    assert labels == [name for name, _, _ in PROGRESSION]


def test_segment_boundaries_within_a_beat(recognized):
    for k, seg in enumerate(recognized["segments"]):
        assert abs(seg["start"] - k * 4 * BEAT) <= BEAT
        assert abs(seg["end"] - (k + 1) * 4 * BEAT) <= BEAT


def test_three_distinct_candidates_with_confidences(recognized):
    for seg in recognized["segments"]:
        labels = [c["label"] for c in seg["candidates"]]
        assert len(labels) == 3 and len(set(labels)) == 3
        confs = [c["conf"] for c in seg["candidates"]]
        assert confs[0] == max(confs)
        assert abs(sum(confs) - 1.0) < 0.01
        assert seg["selected"] == 0 and seg["custom"] is None


def test_key_from_chords(recognized):
    assert recognized["key"] == "C:maj"
    assert recognized["key_label"] == "C major"


def test_detect_key_on_cadence():
    y = np.concatenate([strummed(n, b) for n, b in (
        ([60, 64, 67], 48), ([60, 65, 69], 41), ([59, 62, 67], 43), ([60, 64, 67], 48))])
    assert detect_key(y.astype(np.float32), SR) == "C:maj"


def test_silence_yields_no_segments():
    out = recognize(np.zeros(SR * 2, dtype=np.float32), SR)
    assert out["segments"] == []


# --------------------------------------------------------------------------- #
# alignment + editing
# --------------------------------------------------------------------------- #

def W(text, start, end):
    return {"text": text, "start": start, "end": end}


def chord(cid, label, start, end):
    return {"id": cid, "start": start, "end": end, "beats": 4, "selected": 0, "custom": None,
            "candidates": [{"label": label, "conf": 0.7, "root": 0, "quality": "", "bass": None,
                            "pcs": [0, 4, 7]},
                           {"label": "Am", "conf": 0.2, "root": 9, "quality": "m", "bass": None,
                            "pcs": [9, 0, 4]},
                           {"label": "F", "conf": 0.1, "root": 5, "quality": "", "bass": None,
                            "pcs": [5, 9, 0]}]}


@pytest.fixture
def small_sheet():
    lines = [
        {"words": [W("hello", 4.0, 4.4), W("darkness", 4.5, 5.2), W("my", 5.3, 5.5),
                   W("old", 5.6, 6.0), W("friend", 6.1, 7.0)], "gap_before": False},
        {"words": [W("I've", 12.0, 12.3), W("come", 12.4, 12.8), W("to", 12.9, 13.0),
                   W("talk", 13.1, 13.6)], "gap_before": False},
    ]
    chords = [
        chord("c0", "G", 0.0, 2.0),      # intro, instrumental
        chord("c1", "D", 2.0, 3.8),      # intro, instrumental
        chord("c2", "Em", 3.8, 5.6),     # just before line 1 -> first word
        chord("c3", "C", 5.62, 9.0),     # on "old"
        chord("c4", "G", 9.0, 12.5),     # gap between lines -> instrumental
        chord("c5", "N.C.", 12.5, 13.0),  # hidden
    ]
    return {"title": "Test", "key": "G:maj", "tempo": 120.0, "beat_period": 0.5,
            "lines": lines, "chords": chords}


def test_build_rows_places_chords(small_sheet):
    rows = build_rows(small_sheet["lines"], small_sheet["chords"], 0.5)
    kinds = [r["type"] for r in rows]
    assert kinds == ["instrumental", "lyric", "instrumental", "lyric"]
    assert rows[0]["chords"] == ["c0", "c1"]
    line1 = rows[1]["words"]
    assert line1[0]["chords"] == ["c2"]          # anticipation lands on first word
    assert line1[3]["chords"] == ["c3"]          # "old"
    assert rows[1]["carry"] is None
    assert rows[2]["chords"] == ["c4"]
    # line 2 has no change of its own: G is still sounding, shown as carry
    assert rows[3]["carry"] == "c4"
    assert all(not w["chords"] for w in rows[3]["words"])  # N.C. is hidden


def test_reanchor_keeps_times_of_kept_words(small_sheet):
    text = "hello darkness my dear old friend\n\nI have come to talk"
    lines = reanchor(small_sheet["lines"], text)
    assert [len(ln["words"]) for ln in lines] == [6, 5]
    assert lines[1]["gap_before"] is True
    hello, darkness, my, dear, old = lines[0]["words"][:5]
    assert (hello["start"], hello["end"]) == (4.0, 4.4)
    assert (old["start"], old["end"]) == (5.6, 6.0)
    assert my["end"] <= dear["start"] <= dear["end"] <= old["start"] and dear.get("approx")
    # "I've" -> "I have": replaced span shares the old word's time
    i, have = lines[1]["words"][:2]
    assert i["start"] == 12.0 and abs(have["end"] - 12.3) < 1e-6
    # chords still land on the same words after the edit
    rows = build_rows(lines, small_sheet["chords"], 0.5)
    lyric = [r for r in rows if r["type"] == "lyric"][0]
    assert [w["text"] for w in lyric["words"] if w["chords"]] == ["hello", "old"]


def test_reanchor_deletion_and_line_split(small_sheet):
    lines = reanchor(small_sheet["lines"], "hello darkness\nmy friend\nI've come to talk")
    assert [[w["text"] for w in ln["words"]] for ln in lines] == [
        ["hello", "darkness"], ["my", "friend"], ["I've", "come", "to", "talk"]]
    assert lines[1]["words"][1]["start"] == 6.1


def test_word_diff():
    diff = word_diff("hello darknes my", "hello darkness my")
    assert ["delete", "darknes"] in diff and ["insert", "darkness"] in diff


def test_exports(small_sheet):
    txt = to_text(small_sheet)
    assert "| G | D |" in txt
    lines = txt.splitlines()
    i = lines.index("hello darkness my old friend")
    assert lines[i - 1].startswith("Em") and lines[i - 1].index("C") == len("hello darkness my ")
    cho = to_chordpro(small_sheet)
    assert "{title: Test}" in cho and "{key: G}" in cho
    assert "[Em]hello darkness my [C]old friend" in cho
    assert "{comment: Instrumental}" in cho


def test_custom_label_wins(small_sheet):
    small_sheet["chords"][3]["custom"] = "Cadd9"
    assert "[Cadd9]old" in to_chordpro(small_sheet)
    small_sheet["chords"][3]["custom"] = None
    small_sheet["chords"][3]["selected"] = 1
    assert "[Am]old" in to_chordpro(small_sheet)


# --------------------------------------------------------------------------- #
# transcripts + OpenRouter
# --------------------------------------------------------------------------- #

def test_split_lines_on_gaps():
    seg = [W("a", 0, 0.2), W("b", 0.3, 0.5), W("c", 1.5, 1.7), W("d", 1.8, 2.0)]
    assert [[w["text"] for w in ln] for ln in split_lines([seg])] == [["a", "b"], ["c", "d"]]


def test_normalize_remote_words():
    resp = {"text": "a b c", "words": [{"word": "a", "start": 0, "end": 0.2},
                                        {"word": "b", "start": 0.3, "end": 0.5},
                                        {"word": "c", "start": 2.0, "end": 2.2}]}
    lines, timing = normalize_remote(resp, [], 3.0)
    assert timing == "word" and [len(ln) for ln in lines] == [2, 1]


def test_normalize_remote_segments_only():
    resp = {"segments": [{"start": 1.0, "end": 3.0, "text": "one two"},
                         {"start": 5.0, "end": 6.0, "text": "three"}]}
    lines, timing = normalize_remote(resp, [], 7.0)
    assert timing == "approx"
    assert lines[0][0]["start"] == 1.0 and lines[0][1]["end"] == 3.0
    assert lines[1][0]["start"] == 5.0


def test_normalize_remote_text_only_uses_vocal_activity():
    resp = {"text": "first line here.\nsecond line"}
    regions = [(10.0, 12.0), (20.0, 22.0)]
    lines, timing = normalize_remote(resp, regions, 30.0)
    assert timing == "approx" and len(lines) == 2
    for ln in lines:
        for w in ln:
            assert any(a - 1e-6 <= w["start"] <= b + 1e-6 for a, b in regions)


def test_openrouter_model_normalisation():
    raw = json.loads((FIXTURES / "openrouter_models.json").read_text())["data"]
    by_id = {m["id"]: openrouter.normalize_model(m, "chat") for m in raw}
    assert by_id["thinkingmachines/inkling-small:free"]["is_free"]
    assert by_id["inclusionai/ling-3.1-flash"]["is_free"]       # zero-priced, no suffix
    assert not by_id["openrouter/auto"]["is_free"]              # -1 = variable, not free
    assert by_id["openrouter/auto"]["price_label"] == "variable"
    assert not by_id["google/gemini-2.5-flash"]["is_free"]
    assert "per 1M" in by_id["google/gemini-2.5-flash"]["price_label"]
    stt = openrouter.normalize_model({"id": "openai/whisper-1", "pricing": {"prompt": "0.0001"}},
                                     "transcription")
    assert stt["price_label"] == "$0.0060/min"


# --------------------------------------------------------------------------- #
# web API, end to end with fakes
# --------------------------------------------------------------------------- #

def fake_separator(audio, sr, progress=None):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    silent = np.zeros_like(mono)
    return [Stem("vocals", mono, sr, "vocal"), Stem("other", mono, sr, "other"),
            Stem("bass", silent, sr, "bass"), Stem("drums", silent, sr, "drums")]


def fake_transcriber(vocal, sr, backend="mlx", model=None, language=None):
    words = [W(t, 4.1 + 0.5 * k, 4.4 + 0.5 * k) for k, t in enumerate("la la lah".split())]
    return {"backend": "fake", "model": "fake", "language": "en", "timing": "word",
            "lines": [words]}


def test_api_flow(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    wav = buf.getvalue()

    app = create_app(tmp_path, separator=fake_separator, transcriber=fake_transcriber,
                     run_inline=True)
    client = TestClient(app)

    r = client.post("/api/songs", files=[("files", ("my song.wav", wav, "audio/wav"))],
                    data={"language": "en"})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    assert client.get(f"/api/songs/{sid}/status").json()["state"] == "done"

    song = client.get(f"/api/songs/{sid}").json()
    assert song["key"] == "C:maj"
    assert [c["candidates"][0]["label"] for c in song["chords"]][:3] == ["C", "G", "Am"]
    lyric_rows = [r for r in song["rows"] if r["type"] == "lyric"]
    assert [w["text"] for w in lyric_rows[0]["words"]] == ["la", "la", "lah"]
    assert client.get(f"/api/songs/{sid}/audio").status_code == 200

    cid = song["chords"][1]["id"]
    r = client.patch(f"/api/songs/{sid}/chords/{cid}", json={"selected": 2})
    assert r.json()["selected"] == 2
    r = client.patch(f"/api/songs/{sid}/chords/{cid}", json={"custom": "G6"})
    assert r.json()["custom"] == "G6"
    assert client.patch(f"/api/songs/{sid}/chords/{cid}", json={"selected": 9}).status_code == 400

    r = client.put(f"/api/songs/{sid}/lyrics", json={"text": "la la la"})
    assert [w["text"] for w in r.json()["lines"][0]["words"]] == ["la", "la", "la"]

    cho = client.get(f"/api/songs/{sid}/export?fmt=cho").text
    assert "{key: C}" in cho and "[G6]" in cho

    # uploading the same file again keeps the edited sheet
    client.post("/api/songs", files=[("files", ("my song.wav", wav, "audio/wav"))])
    again = client.get(f"/api/songs/{sid}").json()
    assert next(c for c in again["chords"] if c["id"] == cid)["custom"] == "G6"

    assert [s["id"] for s in client.get("/api/songs").json()] == [sid]
    assert client.get("/api/songs/..%2Fetc").status_code in (400, 404)
    assert client.delete(f"/api/songs/{sid}").status_code == 200
    assert client.get("/api/songs").json() == []


def test_api_stems_upload_routes_vocals(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    def wav_bytes(y):
        b = io.BytesIO()
        sf.write(b, y, SR, format="WAV")
        return b.getvalue()

    seen = {}

    def transcriber(vocal, sr, **kw):
        seen["vocal_rms"] = float(np.sqrt(np.mean(vocal ** 2)))
        return fake_transcriber(vocal, sr, **kw)

    def no_separator(*a, **k):  # stems must not be separated
        raise AssertionError("separator called for a stems upload")

    voice = (0.3 * np.sin(2 * np.pi * 330 * np.arange(len(progression_audio)) / SR)).astype(np.float32)
    app = create_app(tmp_path, separator=no_separator, transcriber=transcriber, run_inline=True)
    client = TestClient(app)
    r = client.post("/api/songs", files=[
        ("files", ("guitar.wav", wav_bytes(progression_audio), "audio/wav")),
        ("files", ("vox.wav", wav_bytes(voice), "audio/wav")),
    ])
    sid = r.json()["id"]
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "done", st
    assert seen["vocal_rms"] > 0.1
    song = client.get(f"/api/songs/{sid}").json()
    assert [c["candidates"][0]["label"] for c in song["chords"]][:2] == ["C", "G"]


def test_hallucinated_tail_line_is_dropped():
    from mixpipe.chords.transcribe import filter_hallucinations

    regions = [(27.95, 29.9)]
    lines = [
        [W("and", 27.48, 28.1), W("the", 28.1, 28.24), W("river", 28.24, 29.7)],
        [W("Thank", 30.0, 30.6)],
    ]
    kept = filter_hallucinations(lines, regions)
    assert [[w["text"] for w in ln] for ln in kept] == [["and", "the", "river"]]


def test_token_priced_transcription_model_is_labelled_per_token():
    m = openrouter.normalize_model(
        {"id": "openai/gpt-4o-mini-transcribe",
         "pricing": {"prompt": "0.00000125", "completion": "0.000005"}}, "transcription")
    assert "per 1M" in m["price_label"]


# --------------------------------------------------------------------------- #
# meter / bars
# --------------------------------------------------------------------------- #

def accented(notes, bass, beats):
    """One bar: beat 1 louder with the bass, the other beats lighter."""
    n = int(BEAT * SR)
    env = np.exp(-np.arange(n) / (0.35 * SR))
    out = []
    for b in range(beats):
        g = 1.6 if b == 0 else 1.0
        out.append((sum(tone(m, n) for m in notes) * 0.25
                    + tone(bass, n) * (0.4 if b == 0 else 0.2)) * env * g)
    return np.concatenate(out)


FOUR_CHORDS = [([60, 64, 67], 48), ([59, 62, 67], 43), ([60, 64, 69], 45), ([60, 65, 69], 41)]


def _first_downbeat(r):
    return r["beat_times"][r["meter"]["downbeat"]]


def test_meter_four_four(recognized):
    assert recognized["meter"]["beats_per_bar"] == 4
    bar = 4 * BEAT
    t = _first_downbeat(recognized)
    assert min(t % bar, bar - t % bar) < 0.1      # downbeats on the chord changes


def test_meter_waltz():
    y = np.concatenate([accented(n, b, 3) for _ in range(3) for n, b in FOUR_CHORDS])
    r = recognize((y / np.abs(y).max() * 0.8).astype(np.float32), SR)
    assert r["meter"]["beats_per_bar"] == 3
    bar = 3 * BEAT
    t = _first_downbeat(r)
    assert min(t % bar, bar - t % bar) < 0.1
    assert [s["candidates"][0]["label"] for s in r["segments"]] == ["C", "G", "Am", "F"] * 3


def test_meter_with_pickup_beat():
    pickup = accented([59, 62, 67], 43, 1)
    y = np.concatenate([pickup] + [accented(n, b, 4) for _ in range(3) for n, b in FOUR_CHORDS])
    r = recognize((y / np.abs(y).max() * 0.8).astype(np.float32), SR)
    assert r["meter"]["beats_per_bar"] == 4
    t = _first_downbeat(r) - BEAT                 # bars start one beat in
    bar = 4 * BEAT
    assert min(t % bar, bar - t % bar) < 0.1


def test_bar_grid_extrapolates():
    from mixpipe.chords.align import BarGrid

    g = BarGrid([0.5 * k for k in range(1, 9)], beats_per_bar=4, downbeat=3, beat_period=0.5)
    assert g.bar_start(0) == 2.0 and g.bar_start(1) == 4.0
    assert g.bar_start(-1) == 0.0                  # before the first tracked beat
    assert g.bar_of(0.0) == -1 and g.bar_of(2.0) == 0 and g.bar_of(3.9) == 0
    assert g.bar_of(6.0) == 2                      # past the last tracked beat


@pytest.fixture
def grid_sheet():
    beat_times = [0.5 * k for k in range(64)]      # 120 bpm, bars every 2 s from t=0
    chords = [
        chord("c0", "G", 0.0, 4.0),     # bars 0-1 (held through bar 1)
        chord("c1", "D", 4.0, 5.0),     # bar 2, first half
        chord("c2", "C", 5.0, 8.5),     # bar 2 second half + bar 3 held
        chord("c3", "Em", 8.5, 14.0),   # line
    ]
    lines = [{"words": [W("one", 8.4, 8.9), W("two", 9.0, 9.9), W("three", 10.05, 11.0)],
              "gap_before": False}]
    return {"title": "Bars", "key": "G:maj", "tempo": 120.0, "beat_period": 0.5,
            "beat_times": beat_times,
            "meter": {"time_signature": "4/4", "beats_per_bar": 4, "downbeat": 0,
                      "detected": None},
            "transpose": 0, "capo": 0, "lines": lines, "chords": chords}


def test_instrumental_rows_in_bars(grid_sheet):
    from mixpipe.chords.align import grid_for

    rows = build_rows(grid_sheet["lines"], grid_sheet["chords"], 0.5, grid=grid_for(grid_sheet))
    intro = rows[0]
    assert intro["type"] == "instrumental"
    assert [(b["chords"], b["hold"]) for b in intro["bars"]] == [
        (["c0"], None), ([], "c0"), (["c1", "c2"], None), ([], "c2")]
    lyric = rows[1]
    assert lyric["words"][0]["chords"] == ["c3"]
    # bar 5 starts at 10.0 s, while "three" (10.05) is about to be sung
    assert [w.get("bar") for w in lyric["words"]] == [None, None, 5]


def test_text_export_shows_bars_and_time(grid_sheet):
    from mixpipe.chords.sheet import to_chordpro, to_text

    txt = to_text(grid_sheet)
    assert "| G | % | D C | % |" in txt
    assert "Time: 4/4" in txt
    cho = to_chordpro(grid_sheet)
    assert "| [G] | % | [D] [C] | % |" in cho and "{time: 4/4}" in cho


def test_changing_meter_regroups_bars(grid_sheet):
    from mixpipe.chords.align import grid_for
    from mixpipe.chords.sheet import apply_settings, to_text

    apply_settings(grid_sheet, time_signature="3/4")  # bars every 1.5 s
    rows = build_rows(grid_sheet["lines"], grid_sheet["chords"], 0.5, grid=grid_for(grid_sheet))
    bars = [(b["chords"], b["hold"]) for r in rows[:2] for b in r["bars"]]
    # mid-bar changes keep the chord held on the downbeat: | G | % | G D | D C | % | % |
    assert bars == [(["c0"], None), ([], "c0"), (["c1"], "c0"), (["c2"], "c1"),
                    ([], "c2"), ([], "c2")]
    assert "| G | % | G D | D C |" in to_text(grid_sheet)
    apply_settings(grid_sheet, shift_downbeat=1)
    assert grid_sheet["meter"]["downbeat"] == 1
    apply_settings(grid_sheet, shift_downbeat=-2)
    assert grid_sheet["meter"]["downbeat"] == 2
    with pytest.raises(ValueError):
        apply_settings(grid_sheet, time_signature="5/4")


# --------------------------------------------------------------------------- #
# transpose / capo
# --------------------------------------------------------------------------- #

def test_transpose_label_spelling():
    from mixpipe.chords.theory import key_names, transpose_label

    d_major = key_names("C:maj", 2)
    assert transpose_label("C/E", 2, d_major) == "D/F#"
    assert transpose_label("Cadd9", 2, d_major) == "Dadd9"
    assert transpose_label("F#m7", -1, key_names("D:maj", -1)) == "Fm7"
    assert transpose_label("Bb", 1, key_names("F:maj", 1)) == "B"
    assert transpose_label("A", 1, key_names("D:maj", 1)) == "Bb"   # Eb major spells flats
    assert transpose_label("N.C.", 3, d_major) == "N.C."
    assert transpose_label("whatever", 3, d_major) == "whatever"


def test_suggest_capo_for_flat_key():
    from mixpipe.chords.theory import suggest_capo

    # Bb Gm Eb F, bar each: capo 3 turns them into G Em C D
    song = [(10, "", 2.0), (7, "m", 2.0), (3, "", 2.0), (5, "", 2.0)]
    assert suggest_capo(song) == {"capo": 3, "easy": 1.0}
    assert suggest_capo(song, current_capo=3) is None           # already there
    # already easy: no capo
    assert suggest_capo([(7, "", 2.0), (2, "", 2.0), (4, "m", 2.0), (0, "", 2.0)]) is None
    # C G Am F: F is a barre, but that's no reason to suggest capo 5
    assert suggest_capo([(0, "", 2.0), (7, "", 2.0), (9, "m", 2.0), (5, "", 2.0)]) is None


def test_view_transpose_and_capo(grid_sheet):
    from mixpipe.chords.sheet import apply_settings, custom_from_display, to_chordpro, view

    apply_settings(grid_sheet, transpose=2)
    v = view(grid_sheet)
    assert [c["display"] for c in v["chords"]] == ["A", "E", "D", "F#m"]
    assert v["display"]["key_label"] == "A major"
    apply_settings(grid_sheet, capo=2)             # transposed up 2, capo 2: G shapes again
    v = view(grid_sheet)
    assert [c["display"] for c in v["chords"]] == ["G", "D", "C", "Em"]
    assert v["display"]["key_label"] == "A major"
    assert v["display"]["shape_key_label"] == "G major"
    assert v["chords"][1]["candidates"][1]["display"] == "Am"   # alternates move too
    cho = to_chordpro(grid_sheet)
    assert "{key: A}" in cho and "{capo: 2}" in cho and "[Em]one" in cho
    # a custom chord typed in shape space is stored at concert pitch
    assert custom_from_display("Cadd9", grid_sheet) == "Cadd9"  # shift 0 overall
    apply_settings(grid_sheet, capo=0)
    assert custom_from_display("Dsus4", grid_sheet) == "Csus4"


def test_api_settings(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    client = TestClient(create_app(tmp_path, separator=fake_separator,
                                   transcriber=fake_transcriber, run_inline=True))
    sid = client.post("/api/songs", files=[("files", ("s.wav", buf.getvalue(), "audio/wav"))]).json()["id"]
    song = client.get(f"/api/songs/{sid}").json()
    assert song["meter"]["beats_per_bar"] == 4 and song["meter"]["detected"]["beats_per_bar"] == 4
    assert song["display"]["time_signature"] == "4/4"
    assert any("bars" in r for r in song["rows"] if r["type"] == "instrumental")

    v = client.patch(f"/api/songs/{sid}/settings", json={"transpose": 2}).json()
    assert [c["display"] for c in v["chords"]][:3] == ["D", "A", "Bm"]
    cid = v["chords"][0]["id"]
    client.patch(f"/api/songs/{sid}/chords/{cid}", json={"custom": "Dadd9"})
    raw = client.get(f"/api/songs/{sid}").json()
    first = next(c for c in raw["chords"] if c["id"] == cid)
    assert first["custom"] == "Cadd9" and first["display"] == "Dadd9"
    assert client.patch(f"/api/songs/{sid}/settings", json={"capo": 40}).status_code == 400
    v = client.patch(f"/api/songs/{sid}/settings", json={"time_signature": "3/4", "shift_downbeat": 1}).json()
    assert v["display"]["time_signature"] == "3/4" and v["meter"]["beats_per_bar"] == 3
    v = client.patch(f"/api/songs/{sid}/settings", json={"reset_meter": True}).json()
    assert v["meter"]["beats_per_bar"] == 4
    assert v["meter"]["downbeat"] == v["meter"]["detected"]["downbeat"]


# --------------------------------------------------------------------------- #
# time signatures beyond 3 vs 4, and modulations
# --------------------------------------------------------------------------- #

D_CHORDS = [([62, 66, 69], 50), ([61, 64, 69], 45), ([62, 66, 71], 47), ([62, 67, 71], 43)]


def _norm(y):
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def test_six_eight_from_ternary_pulse():
    def bar_68(notes, bass, eighth=0.25):
        n = int(eighth * SR)
        env = np.exp(-np.arange(n) / (0.12 * SR))
        out = []
        for e in range(6):
            g = 1.6 if e == 0 else (1.2 if e == 3 else 0.7)
            out.append((sum(tone(m, n) for m in notes) * 0.25
                        + tone(bass, n) * (0.4 if e % 3 == 0 else 0.1)) * env * g)
        return np.concatenate(out)

    r = recognize(_norm(np.concatenate([bar_68(n, b) for _ in range(4) for n, b in FOUR_CHORDS])), SR)
    assert r["meter"]["time_signature"] == "6/8"
    assert r["meter"]["subdivision"] == "ternary"
    assert r["meter"]["beats_per_bar"] == 2           # tracked the dotted quarter


def test_two_four_march():
    def march(notes, bass):
        n = int(BEAT * SR)
        env = np.exp(-np.arange(n) / (0.2 * SR))
        return np.concatenate([tone(bass, n) * 1.2 * env, sum(tone(m, n) for m in notes) * 0.25 * env])

    r = recognize(_norm(np.concatenate([march(n, b) for _ in range(6) for n, b in FOUR_CHORDS])), SR)
    assert r["meter"]["time_signature"] == "2/4"


@pytest.fixture(scope="module")
def modulating():
    y = np.concatenate([accented(n, b, 4) for _ in range(4) for n, b in FOUR_CHORDS]
                       + [accented(n, b, 4) for _ in range(4) for n, b in D_CHORDS])
    return recognize(_norm(y), SR)


def test_modulation_handled_per_chord(modulating):
    r = modulating
    assert r["key"] == "C:maj"                        # overview: the opening key
    assert [k["key"] for k in r["key_regions"]] == ["C:maj", "D:maj"]
    first_d = next(s for s in r["segments"] if s["candidates"][0]["label"] == "D")
    assert r["key_regions"][1]["start"] == first_d["start"]   # change lands on a chord
    labels = [s["candidates"][0]["label"] for s in r["segments"]]
    assert labels == ["C", "G", "Am", "F"] * 4 + ["D", "A", "Bm", "G"] * 4
    assert {s["key"] for s in r["segments"][:16]} == {"C:maj"}
    assert {s["key"] for s in r["segments"][16:]} == {"D:maj"}


def test_no_false_modulation_between_relative_keys():
    am_then_c = ([([60, 64, 69], 45), ([62, 65, 69], 50), ([59, 64, 68], 52), ([60, 64, 69], 45)] * 3
                 + FOUR_CHORDS * 3)                    # Am Dm E Am ... then C G Am F
    r = recognize(_norm(np.concatenate([accented(n, b, 4) for n, b in am_then_c])), SR)
    assert len(r["key_regions"]) == 1


def test_key_change_marked_on_chord_and_exports(modulating):
    from mixpipe.chords.sheet import apply_settings, to_chordpro, to_text, view

    sheet = {"title": "Mod", "key": modulating["key"], "tempo": 120.0, "beat_period": 0.5,
             "beat_times": modulating["beat_times"],
             "meter": {**modulating["meter"], "detected": modulating["meter"]},
             "key_regions": modulating["key_regions"], "transpose": 0, "capo": 0,
             "lines": [], "chords": modulating["segments"]}
    v = view(sheet)
    marked = [(c["display"], c["key_change"]) for c in v["chords"] if c["key_change"]]
    assert marked == [("D", "D major")]
    assert "[Key change: D major]" in to_text(sheet)
    cho = to_chordpro(sheet)
    assert cho.index("{key: C}") < cho.index("{key: D}")
    # transposing moves both the opening key and the change, spelled per key
    apply_settings(sheet, transpose=1)
    v = view(sheet)
    assert v["display"]["key_label"] == "Db major"
    assert [(c["display"], c["key_change"]) for c in v["chords"] if c["key_change"]] == [
        ("Eb", "Eb major")]
    assert "Cm" in [c["display"] for c in v["chords"]]   # Bm in D -> Cm in Eb


def test_time_signature_setting_maps_to_tracked_beats(grid_sheet):
    from mixpipe.chords.sheet import apply_settings

    apply_settings(grid_sheet, time_signature="6/8")
    assert grid_sheet["meter"]["beats_per_bar"] == 2     # 120 bpm pulse = dotted quarters
    grid_sheet["tempo"] = 170.0
    apply_settings(grid_sheet, time_signature="6/8")
    assert grid_sheet["meter"]["beats_per_bar"] == 6     # fast pulse = eighths
    apply_settings(grid_sheet, time_signature="2/4")
    assert grid_sheet["meter"]["beats_per_bar"] == 2


# --------------------------------------------------------------------------- #
# songs from a link (no network: resolve/download are faked)
# --------------------------------------------------------------------------- #

SPOTIFY_PAGE = """<html><head><title>Song | Spotify</title>
<meta property="og:title" content="Don&#x27;t Stop Me Now"/>
<meta property="og:description" content="Queen · Jazz · Song · 1978"/>
<meta name="music:duration" content="209"/>
<meta name="music:musician_description" content="Queen"/>
</head></html>"""


def test_parse_spotify_page():
    from mixpipe.chords.fetch import parse_spotify_page

    assert parse_spotify_page(SPOTIFY_PAGE) == {
        "title": "Don't Stop Me Now", "artist": "Queen", "duration": 209.0}
    no_artist_tag = SPOTIFY_PAGE.replace('name="music:musician_description"', 'name="x"')
    assert parse_spotify_page(no_artist_tag)["artist"] == "Queen"  # from og:description


def test_spotify_match_prefers_same_recording():
    from mixpipe.chords.fetch import pick_match

    track = {"title": "Don't Stop Me Now", "artist": "Queen", "duration": 209.0}
    results = [
        {"id": "a", "title": "Queen - Don't Stop Me Now (Live at Wembley)", "duration": 210},
        {"id": "b", "title": "Queen - Don't Stop Me Now (Official Video)", "duration": 216},
        {"id": "c", "title": "Don't Stop Me Now - 1 hour loop", "duration": 3600},
    ]
    assert pick_match(results, track)["id"] == "b"
    assert pick_match([], track) is None


def test_link_checks():
    from mixpipe.chords.fetch import FetchError, _clean_error, check_url, is_spotify

    for bad in ("", "youtube.com/watch?v=x", "ftp://host/a.mp3", "file:///etc/passwd"):
        with pytest.raises(FetchError):
            check_url(bad)
    assert check_url("  https://youtu.be/x ") == "https://youtu.be/x"
    assert is_spotify("https://open.spotify.com/track/abc")
    assert is_spotify("https://spotify.link/xyz")
    assert not is_spotify("https://notspotify.com/track/abc")
    assert _clean_error(Exception("ERROR: Unsupported URL: https://x")) == \
        "couldn't find any audio at that link"
    assert _clean_error(Exception("ERROR: [youtube] abc: Video unavailable")) == "Video unavailable"


def _fake_link(wav: bytes):
    calls = {"resolve": 0, "download": 0, "fail": False}

    def resolver(url):
        from mixpipe.chords.fetch import FetchError

        calls["resolve"] += 1
        if "playlist" in url:
            raise FetchError("that's a playlist or album; paste a link to a single song")
        return {"url": url, "page_url": "https://www.youtube.com/watch?v=abc",
                "download_url": "https://www.youtube.com/watch?v=abc", "site": "Youtube",
                "title": "Some Band - Some Song", "duration": 16.0, "note": None}

    def downloader(source, dest, progress=None):
        from mixpipe.chords.fetch import FetchError

        calls["download"] += 1
        if calls["fail"]:
            raise FetchError("download failed: Video unavailable")
        progress(0.5, "downloading")
        path = Path(dest) / "audio.wav"
        path.write_bytes(wav)
        return path

    return calls, resolver, downloader


def test_api_link_flow(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    calls, resolver, downloader = _fake_link(buf.getvalue())
    client = TestClient(create_app(tmp_path, separator=fake_separator,
                                   transcriber=fake_transcriber, run_inline=True,
                                   resolver=resolver, downloader=downloader))

    r = client.post("/api/songs/link", json={"url": "https://www.youtube.com/playlist?list=x"})
    assert r.status_code == 400 and "single song" in r.json()["detail"]
    assert client.get("/api/songs").json() == []

    r = client.post("/api/songs/link", json={"url": "https://youtu.be/abc", "language": "en"})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    assert sid.startswith("some-band-some-song-")
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "done" and st["from_url"] is True, st
    song = client.get(f"/api/songs/{sid}").json()
    assert song["title"] == "Some Band - Some Song"
    assert song["source"]["site"] == "Youtube"
    assert [c["candidates"][0]["label"] for c in song["chords"]][:3] == ["C", "G", "Am"]
    assert client.get(f"/api/songs/{sid}/audio").status_code == 200
    meta = json.loads((tmp_path / sid / "meta.json").read_text())
    assert meta["inputs"] == ["audio.wav"]

    # the same link again: same song, edits kept, nothing downloaded twice
    cid = song["chords"][1]["id"]
    client.patch(f"/api/songs/{sid}/chords/{cid}", json={"custom": "G6"})
    r = client.post("/api/songs/link", json={"url": "https://www.youtube.com/watch?v=abc"})
    assert r.json()["id"] == sid and calls["download"] == 1
    again = client.get(f"/api/songs/{sid}").json()
    assert next(c for c in again["chords"] if c["id"] == cid)["custom"] == "G6"

    # uploads are not marked as links
    r = client.post("/api/songs", files=[("files", ("up.wav", buf.getvalue(), "audio/wav"))])
    assert client.get(f"/api/songs/{r.json()['id']}/status").json()["from_url"] is False


def test_api_link_download_failure_then_retry(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    calls, resolver, downloader = _fake_link(buf.getvalue())
    client = TestClient(create_app(tmp_path, separator=fake_separator,
                                   transcriber=fake_transcriber, run_inline=True,
                                   resolver=resolver, downloader=downloader))
    calls["fail"] = True
    sid = client.post("/api/songs/link", json={"url": "https://youtu.be/abc"}).json()["id"]
    st = client.get(f"/api/songs/{sid}/status").json()
    assert st["state"] == "error" and st["stage"] == "download"
    assert st["error"] == "download failed: Video unavailable"

    calls["fail"] = False
    client.post("/api/songs/link", json={"url": "https://youtu.be/abc", "rerun": True})
    assert client.get(f"/api/songs/{sid}/status").json()["state"] == "done"
