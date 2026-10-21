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
    from mixpipe.fetch import parse_spotify_page

    assert parse_spotify_page(SPOTIFY_PAGE) == {
        "title": "Don't Stop Me Now", "artist": "Queen", "duration": 209.0}
    no_artist_tag = SPOTIFY_PAGE.replace('name="music:musician_description"', 'name="x"')
    assert parse_spotify_page(no_artist_tag)["artist"] == "Queen"  # from og:description


def test_spotify_match_prefers_same_recording():
    from mixpipe.fetch import pick_match

    track = {"title": "Don't Stop Me Now", "artist": "Queen", "duration": 209.0}
    results = [
        {"id": "a", "title": "Queen - Don't Stop Me Now (Live at Wembley)", "duration": 210},
        {"id": "b", "title": "Queen - Don't Stop Me Now (Official Video)", "duration": 216},
        {"id": "c", "title": "Don't Stop Me Now - 1 hour loop", "duration": 3600},
    ]
    assert pick_match(results, track)["id"] == "b"
    assert pick_match([], track) is None


def test_link_checks():
    from mixpipe.fetch import FetchError, _clean_error, check_url, is_spotify

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


def test_api_link_flow(tmp_path, progression_audio, fake_link):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    calls, resolver, downloader = fake_link(buf.getvalue())
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


def test_api_link_download_failure_then_retry(tmp_path, progression_audio, fake_link):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    calls, resolver, downloader = fake_link(buf.getvalue())
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


# --------------------------------------------------------------------------- #
# syllables: chords over the syllable they change on
# --------------------------------------------------------------------------- #

from mixpipe.chords import syllables as syl  # noqa: E402


@pytest.mark.parametrize("word,expected", [
    ("amazing", "a-ma-zing"), ("hello", "hel-lo"), ("table", "ta-ble"), ("little", "lit-tle"),
    ("mother", "mo-ther"), ("monster", "mon-ster"), ("winter", "win-ter"), ("people", "peo-ple"),
    ("love", "love"), ("walked", "walked"), ("wanted", "wan-ted"), ("friend", "friend"),
    ("I've", "I've"), ("don't", "don't"), ("lonely", "lone-ly"), ("hallelujah", "hal-le-lu-jah"),
    ("beautiful", "beau-ti-ful"), ("my", "my"), ("baby", "ba-by"), ("believe", "be-lieve"),
])
def test_english_syllables(word, expected):
    assert "-".join(syl.syllables(word, "en")) == expected


def test_other_scripts_and_languages():
    assert syl.syllables("ありがとう") == list("ありがとう")
    assert syl.syllables("きょう") == ["きょ", "う"]            # small kana stay with the one before
    assert syl.syllables("사랑해") == list("사랑해")
    # scripts without written short vowels are not guessed: the word stays whole
    assert syl.syllable_breaks("سلام") == [] and syl.syllables("שלום") == ["שלום"]
    assert syl.syllables("Liebe", "de") == ["Lie", "be"] and syl.syllables("bella", "it") == ["bel", "la"]
    assert syl.syllable_breaks("123") == [] and syl.syllables("") == [] and syl.syllables("a") == ["a"]
    assert syl.syllable_breaks("amazing") == [1, 3]


def test_syllable_spans_share_the_words_time():
    spans = syl.syllable_spans({"text": "amazing", "start": 10.0, "end": 11.5})
    assert [s["text"] for s in spans] == ["a", "ma", "zing"]
    assert [s["c"] for s in spans] == [0, 1, 3]
    assert spans[0]["start"] == 10.0 and spans[-1]["end"] == 11.5
    assert all(a["end"] == b["start"] for a, b in zip(spans, spans[1:]))
    assert spans[2]["end"] - spans[2]["start"] > spans[0]["end"] - spans[0]["start"]   # longer syllable, longer share
    assert syl.syllable_spans({"text": "friend", "start": 1, "end": 2}) == []
    assert syl.syllable_spans({"text": "amazing", "start": 2, "end": 2}) == []


def test_envelope_dip_moves_the_boundary():
    word = {"text": "hello", "start": 2.0, "end": 3.0}
    flat = syl.syllable_spans(word, "en", [200] * 600)
    plain = syl.syllable_spans(word, "en")
    assert flat == plain                                   # nothing to follow: weights decide
    env = [200] * 600
    for i in range(257, 263):                               # the voice dips at ~2.6 s
        env[i] = 20
    spans = syl.syllable_spans(word, "en", env)
    assert abs(spans[1]["start"] - 2.6) < 0.03 and abs(plain[1]["start"] - 2.6) > 0.05
    assert spans[0]["end"] == spans[1]["start"] and spans[1]["end"] == 3.0
    # via the factory: envelope indices are absolute seconds
    fn = syl.make_syllabifier("en", {"hop": 0.01, "v": env})
    assert abs(fn(word)[1]["start"] - 2.6) < 0.03
    assert syl.make_syllabifier("en", {"hop": 0.02, "v": env})(word) == plain   # wrong resolution: ignored


def test_loudness_envelope():
    sr = 8000
    t = np.arange(sr * 2) / sr
    y = (0.5 * np.sin(2 * np.pi * 200 * t) * (t < 1.0)).astype(np.float32)
    env = syl.loudness_envelope(y, sr)
    assert abs(len(env) - 200) <= 2 and all(0 <= v <= 255 for v in env)
    assert min(env[10:90]) > 230 and max(env[110:190]) < 5
    assert syl.loudness_envelope(np.zeros(5, dtype=np.float32), sr) == []


@pytest.fixture
def held_word_sheet():
    lines = [{"words": [W("I", 9.6, 9.8), W("amazing", 10.0, 11.5), W("grace", 11.6, 12.4)],
              "gap_before": False}]
    chords = [chord("a", "G", 9.4, 10.5), chord("b", "D", 10.5, 11.6), chord("c", "Em", 11.6, 13.0)]
    return {"title": "Grace", "key": "G:maj", "tempo": 120.0, "beat_period": 0.5,
            "lines": lines, "chords": chords}


def test_mid_word_chord_sits_over_its_syllable(held_word_sheet):
    rows = build_rows(held_word_sheet["lines"], held_word_sheet["chords"], 0.5,
                      syllabify=syl.make_syllabifier("en"))
    w = rows[0]["words"][1]
    assert rows[0]["type"] == "lyric" and w["text"] == "amazing" and w["chords"] == ["b"]
    assert [s["text"] for s in w["syl"]] == ["a", "ma", "zing"]
    assert w["at"] == {"b": 1}                          # the change at 10.5 is on "ma"
    assert "syl" not in rows[0]["words"][0] and "syl" not in rows[0]["words"][2]   # nothing to place
    plain = build_rows(held_word_sheet["lines"], held_word_sheet["chords"], 0.5)
    assert "syl" not in plain[0]["words"][1]


def test_exports_put_the_chord_before_the_syllable(held_word_sheet):
    cho = to_chordpro(held_word_sheet)
    assert "[G]I a[D]mazing [Em]grace" in cho
    txt = to_text(held_word_sheet).splitlines()
    i = txt.index("I amazing grace")
    assert txt[i - 1].index("D") == txt[i].index("amazing") + 1       # over the "m" of a-ma-zing
    assert txt[i - 1].split() == ["G", "D", "Em"]


def test_one_syllable_words_keep_chords_at_the_edges():
    lines = [{"words": [W("friend", 6.0, 7.0)], "gap_before": False}]
    chords = [chord("a", "G", 5.8, 6.6), chord("b", "C", 6.6, 9.0)]
    cho = to_chordpro({"title": "T", "key": "G:maj", "tempo": 120.0, "beat_period": 0.5,
                       "lines": lines, "chords": chords})
    # the late change goes after the word rather than splitting the letters
    assert "[G]friend[C]" in cho
    assert "fr[" not in cho and "frie[" not in cho


def test_view_does_not_serve_the_envelope(held_word_sheet):
    from mixpipe.chords.sheet import view

    held_word_sheet["vocal_env"] = {"hop": 0.01, "v": [10] * 100}
    sheet = {**held_word_sheet, "version": 3, "id": "x", "melody": None,
             "meter": {"time_signature": "4/4", "beats_per_bar": 4, "downbeat": 0, "detected": None},
             "transpose": 0, "capo": 0}
    out = view(sheet)
    assert "vocal_env" not in out
    lyric = next(r for r in out["rows"] if r["type"] == "lyric")
    assert lyric["words"][1]["syl"][1]["text"] == "ma"


# --------------------------------------------------------------------------- #
# re-timing text that came without word times
# --------------------------------------------------------------------------- #

from mixpipe.chords import retime as rt  # noqa: E402


def sung_syllables(onsets, dur=0.35, sr=16_000, total=None):
    """A 'vocal' that starts abruptly at each onset time and decays."""
    total = total or (max(onsets) + 1.0)
    y = np.zeros(int(total * sr), dtype=np.float32)
    for k, t in enumerate(onsets):
        i, n = int(t * sr), int(dur * sr)
        tt = np.arange(n) / sr
        f = 180 * 2 ** ((k % 5) / 12)
        y[i:i + n] += (0.4 * np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.12)).astype(np.float32)[: len(y) - i]
    return y


TRUE_STARTS = [1.00, 1.62, 2.10, 2.95, 3.40, 4.30, 4.85, 5.60]


def even_lines(words, t0, t1):
    """What a text-only model gives: words spread evenly over the stretch."""
    step = (t1 - t0) / len(words)
    return [[{"text": w, "start": round(t0 + i * step, 3), "end": round(t0 + (i + 0.9) * step, 3)}
             for i, w in enumerate(words)]]


def test_snap_to_onsets_follows_the_voice():
    vocal = sung_syllables(TRUE_STARTS, total=7.0)
    lines = even_lines("one two three four five six seven eight".split(), 1.0, 6.0)
    before = [w["start"] for w in lines[0]]
    times, strength = rt._onsets(vocal, 16_000)
    assert len(times) >= 7
    snapped = rt.snap_to_onsets([{"words": lines[0], "gap_before": False}], times, strength)
    after = [w["start"] for w in snapped[0]["words"]]
    err_before = np.mean(np.abs(np.array(before) - TRUE_STARTS))
    err_after = np.mean(np.abs(np.array(after) - TRUE_STARTS))
    assert err_after < 0.03 < err_before
    assert after == sorted(after) and all(b - a >= rt.SNAP_MIN_GAP_S - 1e-6 for a, b in zip(after, after[1:]))
    assert all(w["end"] > w["start"] for w in snapped[0]["words"])
    assert snapped[0]["words"][0]["text"] == "one"
    assert rt.snap_to_onsets(snapped, np.zeros(0), np.zeros(0)) == snapped         # no onsets: unchanged


def test_snap_leaves_a_word_with_no_onset_nearby():
    lines = even_lines(["far", "away"], 10.0, 12.0)
    times, strength = np.array([2.0, 3.0]), np.array([1.0, 1.0])
    out = rt.snap_to_onsets([{"words": lines[0], "gap_before": False}], times, strength)
    assert [w["start"] for w in out[0]["words"]] == [10.0, 11.0]


def test_match_fraction():
    timed = [[W("hello", 1, 1.4), W("darkness", 1.5, 2.2), W("my", 2.3, 2.5), W("cold", 2.6, 3.0)]]
    assert rt.match_fraction(timed, "hello darkness my old friend") == pytest.approx(0.6)
    assert rt.match_fraction(timed, "") == 0.0 and rt.match_fraction([], "a b") == 0.0
    assert rt.match_fraction(timed, "Hello, DARKNESS!") == 1.0          # case and punctuation don't matter


def approx_transcript(words):
    return {"backend": "openrouter", "model": "m", "language": "en", "timing": "approx",
            "lines": even_lines(words, 1.0, 6.0)}


def test_retime_with_whisper_anchors():
    words = "hello darkness my old friend".split()
    heard = [[W("hello", 1.05, 1.40), W("darkness", 1.62, 2.60), W("my", 3.10, 3.30),
              W("cold", 3.40, 3.90), W("friend", 4.30, 5.40)]]       # Whisper misheard "old"
    out = rt.retime(approx_transcript(words), np.zeros(16_000, dtype=np.float32), 16_000,
                    whisper=lambda v, sr, lang: heard)
    assert out["timing"] == "aligned" and out["retimed"] == "whisper"
    got = {w["text"]: w for w in out["lines"][0]}
    assert [w["text"] for w in out["lines"][0]] == words              # the model's text, not Whisper's
    assert (got["hello"]["start"], got["darkness"]["start"], got["friend"]["start"]) == (1.05, 1.62, 4.3)
    assert got["my"]["end"] == 3.3
    assert 3.3 <= got["old"]["start"] < got["old"]["end"] <= 4.3 and got["old"].get("approx")


def test_retime_falls_back_to_onsets():
    words = "one two three four five six seven eight".split()
    vocal = sung_syllables(TRUE_STARTS, total=7.0)
    tr = {"backend": "openrouter", "model": "m", "language": "en", "timing": "approx",
          "lines": even_lines(words, 1.0, 6.0)}
    for whisper in (lambda v, sr, lang: None,                                       # not installed
                    lambda v, sr, lang: [[W("completely", 1, 2), W("different", 2, 3)]],  # too few words match
                    lambda v, sr, lang: (_ for _ in ()).throw(RuntimeError("model missing"))):
        out = rt.retime(tr, vocal, 16_000, whisper=whisper)
        assert out["timing"] == "snapped" and out["retimed"] == "onsets"
        starts = [w["start"] for w in out["lines"][0]]
        assert np.mean(np.abs(np.array(starts) - TRUE_STARTS)) < 0.03
    # nothing to go on: silence and no Whisper
    silent = rt.retime(tr, np.zeros(16_000 * 7, dtype=np.float32), 16_000, whisper=lambda *a: None)
    assert silent["timing"] == "approx" and silent["lines"] == tr["lines"]
    assert rt.retime({"lines": [], "timing": "none"}, vocal, 16_000)["timing"] == "none"


def approx_transcriber(vocal, sr, backend="mlx", model=None, language=None):
    return {"backend": "openrouter", "model": "some/model", "language": "en", "timing": "approx",
            "lines": even_lines("la la lah".split(), 4.0, 6.0)}


def test_pipeline_retimes_and_records_the_models_own_capability(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    seen = []

    def retimer(transcript, vocal, sr):
        seen.append(transcript["timing"])
        return {**transcript, "timing": "aligned", "retimed": "whisper"}

    client = TestClient(create_app(tmp_path, separator=fake_separator, transcriber=approx_transcriber,
                                   retimer=retimer, run_inline=True))
    data = {"language": "en", "stt": "openrouter", "stt_model": "some/model"}
    sid = client.post("/api/songs", files=[("files", ("a.wav", buf.getvalue(), "audio/wav"))],
                      data=data).json()["id"]
    song = client.get(f"/api/songs/{sid}").json()
    assert seen == ["approx"]
    assert song["transcript"]["timing"] == "aligned" and song["transcript"]["retimed"] == "whisper"
    # the model picker still learns that this model returns no word times
    caps = json.loads((tmp_path / "openrouter_caps.json").read_text())
    assert caps == {"some/model": "approx"}
    # the vocal's loudness envelope is stored with the sheet, for syllables
    stored = json.loads((tmp_path / sid / "sheet.json").read_text())
    assert stored["vocal_env"]["hop"] == 0.01 and len(stored["vocal_env"]["v"]) > 100

    # opting out leaves the estimated times alone
    client2 = TestClient(create_app(tmp_path / "b", separator=fake_separator, transcriber=approx_transcriber,
                                    retimer=retimer, run_inline=True))
    sid2 = client2.post("/api/songs", files=[("files", ("a.wav", buf.getvalue(), "audio/wav"))],
                        data={**data, "retime": "false"}).json()["id"]
    assert client2.get(f"/api/songs/{sid2}").json()["transcript"]["timing"] == "approx"
    assert seen == ["approx"]


# --------------------------------------------------------------------------- #
# other instruments and tunings
# --------------------------------------------------------------------------- #

def test_instrument_profiles():
    from mixpipe.chords.theory import INSTRUMENTS, instrument_choices, instrument_profile

    assert set(INSTRUMENTS) >= {"guitar", "ukulele", "ukulele_d", "baritone_ukulele",
                                "guitar_half_down", "guitar_whole_down"}
    assert [c["id"] for c in instrument_choices()][0] == "guitar"
    assert instrument_profile("nonsense") is INSTRUMENTS["guitar"]
    assert instrument_profile(None)["tuning"] == 0
    # D tuning sounds a whole step above the C-tuned shapes
    assert INSTRUMENTS["ukulele_d"]["tuning"] == 2 and INSTRUMENTS["guitar_half_down"]["tuning"] == -1
    # the ukulele's open chords are not the guitar's: E major and F# minor are barres there
    uke, gtr = INSTRUMENTS["ukulele"]["easy"], INSTRUMENTS["guitar"]["easy"]
    assert (5, "") in uke and (5, "") not in gtr         # F is open on a ukulele (2010)
    assert (4, "") not in uke and (4, "") in gtr         # E is open on a guitar (022100)


def test_capo_suggestion_depends_on_the_instrument():
    from mixpipe.chords.theory import easy_fraction, suggest_capo

    # E, E7, A, B7: all open on a guitar, barres on a ukulele
    song = [(4, "", 2.0), (4, "7", 2.0), (9, "", 2.0), (11, "7", 2.0)]
    assert easy_fraction(song, 0, 0, "guitar") > easy_fraction(song, 0, 0, "ukulele")
    assert easy_fraction(song, 0, 0, "guitar") == 1.0
    gtr = suggest_capo(song, instrument="guitar")
    uke = suggest_capo(song, instrument="ukulele")
    assert gtr is None                                   # nothing to improve
    assert uke is not None and uke["capo"] != 0 and uke["easy"] > 0.75
    # F C Dm Bb: friendly on a ukulele already
    f_song = [(5, "", 2.0), (0, "", 2.0), (2, "m", 2.0), (10, "", 2.0)]
    assert easy_fraction(f_song, 0, 0, "ukulele") >= 0.75
    # a D-tuned ukulele plays the same shapes a whole step higher, so the shapes it needs
    # for a song in C are the C-tuned ones for a song in B-flat
    assert easy_fraction([(0, "", 2), (7, "", 2), (9, "m", 2), (5, "", 2)], 0, 0, "ukulele_d") \
        == easy_fraction([(10, "", 2), (5, "", 2), (7, "m", 2), (3, "", 2)], 0, 0, "ukulele")


def test_instrument_changes_the_shapes_shown_and_exported(grid_sheet):
    from mixpipe.chords.sheet import apply_settings, to_chordpro, to_text, view

    apply_settings(grid_sheet, instrument="ukulele_d")
    v = view(grid_sheet)
    # G D C Em sound a whole step above the C-tuned shapes: F C Bb Dm
    assert [c["display"] for c in v["chords"]] == ["F", "C", "Bb", "Dm"]
    assert v["display"]["instrument"] == "ukulele_d" and "D tuning" in v["display"]["instrument_label"]
    assert any(i["id"] == "ukulele" for i in v["display"]["instruments"])
    assert v["display"]["shape_key_label"] == "F major" and v["display"]["key_label"] == "G major"
    cho = to_chordpro(grid_sheet)
    assert "{comment: Instrument: Ukulele, D tuning (ADF♯B)}" in cho and "[Dm]one" in cho
    txt = to_text(grid_sheet)
    assert "Instrument: Ukulele, D tuning" in txt and "Play F shapes" in txt
    apply_settings(grid_sheet, instrument="guitar_half_down", capo=0)
    assert [c["display"] for c in view(grid_sheet)["chords"]] == ["G#", "D#", "C#", "Fm"] or \
        [c["display"] for c in view(grid_sheet)["chords"]] == ["Ab", "Eb", "Db", "Fm"]
    with pytest.raises(ValueError):
        apply_settings(grid_sheet, instrument="banjo")
    apply_settings(grid_sheet, instrument="guitar")
    assert "Instrument:" not in to_text(grid_sheet) and "Play" not in to_text(grid_sheet)


def test_api_instrument_setting(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    client = TestClient(create_app(tmp_path, separator=fake_separator, transcriber=fake_transcriber,
                                   run_inline=True))
    sid = client.post("/api/songs", files=[("files", ("a.wav", buf.getvalue(), "audio/wav"))],
                      data={"language": "en"}).json()["id"]
    v = client.patch(f"/api/songs/{sid}/settings", json={"instrument": "ukulele"}).json()
    assert v["display"]["instrument"] == "ukulele"
    assert client.patch(f"/api/songs/{sid}/settings", json={"instrument": "lute"}).status_code == 400
    assert client.get(f"/api/songs/{sid}").json()["display"]["instrument"] == "ukulele"   # persisted


# --------------------------------------------------------------------------- #
# the job queue
# --------------------------------------------------------------------------- #

import threading  # noqa: E402
import time  # noqa: E402


def _wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_queue_runs_one_at_a_time_by_default_and_many_when_asked():
    from mixpipe.web.common import JobQueue

    def run_batch(workers):
        q = JobQueue(workers)
        gate = threading.Event()
        live, peak, lock = [0], [0], threading.Lock()

        def job():
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            gate.wait(2)
            with lock:
                live[0] -= 1

        futures = [q.submit(f"s{i}", job) for i in range(3)]
        _wait_for(lambda: live[0] >= min(workers, 3))
        time.sleep(0.05)
        gate.set()
        for f in futures:
            f.result(timeout=5)
        return peak[0]

    assert run_batch(1) == 1
    assert run_batch(2) == 2


def test_queue_position_and_cancelling():
    from mixpipe.web.common import JobQueue

    q = JobQueue(1)
    gate, started, done, dropped = threading.Event(), threading.Event(), [], []

    def slow():
        started.set()
        gate.wait(5)
        done.append("a")

    q.submit("a", slow)
    started.wait(2)
    q.submit("b", lambda: done.append("b"), on_cancel=lambda: dropped.append("b"))
    q.submit("c", lambda: done.append("c"))
    assert (q.ahead("a"), q.ahead("b"), q.ahead("c")) == (None, 1, 2)     # a is running
    assert q.cancel("b") == "queued" and q.ahead("c") == 1
    assert q.cancel("nothing") is None
    assert q.cancel("a") == "running" and q.is_cancelled("a")
    gate.set()
    assert _wait_for(lambda: "c" in done)
    assert done == ["a", "c"] and dropped == ["b"]                         # b never ran
    assert not q.is_cancelled("a")                                          # cleared once it ended


def test_run_job_reports_cancelled_state(tmp_path):
    from mixpipe import songstore

    d = tmp_path / "s"
    d.mkdir()
    flag = {"stop": False}

    def work(progress):
        progress("separate", 0.0, "start")
        flag["stop"] = True
        progress("separate", 0.5, "still going")        # raises JobCancelled
        return "unreachable"

    assert songstore.run_job(d, work, "align", cancelled=lambda: flag["stop"]) is None
    st = songstore.get_status(d)
    assert st["state"] == "cancelled" and st["stage"] == "separate"
    assert songstore.run_job(d, lambda p: "ok", "align", cancelled=lambda: True) is None   # before it starts
    assert songstore.get_status(d)["state"] == "cancelled"
    assert songstore.run_job(d, lambda p: "ok", "align") == "ok"


def _blocking_app(tmp_path, gate, started):
    from mixpipe.web.server import create_app

    def separator(audio, sr, progress=None):
        started.set()
        while not gate.is_set():
            time.sleep(0.01)
            if progress:
                progress(0.1)                          # a progress report is where a cancel lands
        return fake_separator(audio, sr, progress)

    return create_app(tmp_path, separator=separator, transcriber=fake_transcriber)


def test_api_queue_position_and_cancel(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    buf1, buf2 = io.BytesIO(), io.BytesIO()
    sf.write(buf1, progression_audio, SR, format="WAV")
    sf.write(buf2, progression_audio[: SR * 20], SR, format="WAV")
    gate, started = threading.Event(), threading.Event()
    client = TestClient(_blocking_app(tmp_path, gate, started))
    try:
        first = client.post("/api/songs", files=[("files", ("one.wav", buf1.getvalue(), "audio/wav"))],
                            data={"language": "en"}).json()["id"]
        assert started.wait(5)
        second = client.post("/api/songs", files=[("files", ("two.wav", buf2.getvalue(), "audio/wav"))],
                             data={"language": "en"}).json()["id"]
        st = client.get(f"/api/songs/{second}/status").json()
        assert st["state"] == "queued" and st["ahead"] == 1
        assert client.get(f"/api/songs/{first}/status").json().get("ahead") is None

        r = client.post(f"/api/songs/{second}/cancel").json()
        assert r["cancelled"] == "queued"
        assert client.get(f"/api/songs/{second}/status").json()["state"] == "cancelled"

        # cancelling the running job stops it at its next progress report
        assert client.post(f"/api/songs/{first}/cancel").json()["cancelled"] == "running"
        assert _wait_for(lambda: client.get(f"/api/songs/{first}/status").json()["state"] == "cancelled")
        assert client.post(f"/api/songs/{first}/cancel").json()["cancelled"] is None
        assert client.post("/api/songs/nope/cancel").status_code == 404

        # and it can be started again once the separator is free
        gate.set()
        client.post("/api/songs", files=[("files", ("one.wav", buf1.getvalue(), "audio/wav"))],
                    data={"language": "en", "rerun": "true"})
        assert _wait_for(lambda: client.get(f"/api/songs/{first}/status").json()["state"] == "done", 20)
    finally:
        gate.set()


# --------------------------------------------------------------------------- #
# between the beats: refined change times, pushes, meter sections
# --------------------------------------------------------------------------- #

from mixpipe.chords import recognize as rec  # noqa: E402

CHORD_NOTES = {"C": ([60, 64, 67], 48), "G": ([59, 62, 67], 43), "Am": ([60, 64, 69], 45),
               "F": ([60, 65, 69], 41), "Dm": ([62, 65, 69], 50), "Em": ([59, 64, 67], 52),
               "D": ([62, 66, 69], 50), "A": ([57, 61, 64], 45), "Bm": ([59, 62, 66], 47)}


def render_strikes(strikes, total):
    """strikes: [(time_s, chord, length_s)], each a plucked chord with its bass note."""
    y = np.zeros(int(total * SR) + SR)
    for t, name, length in strikes:
        notes, bass = CHORD_NOTES[name]
        n = int(length * SR)
        env = np.exp(-np.arange(n) / (0.35 * SR))
        hit = (sum(tone(m, n) for m in notes) * 0.25 + tone(bass, n) * 0.4) * env
        i = int(t * SR)
        y[i:i + n] += hit[: max(0, len(y) - i)][:n]
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def pushed_song(push_beats, progression=("C", "G", "Am", "F") * 3):
    """A chord per 4/4 bar, but each arrives `push_beats` early; eighth-note strums."""
    ev = []
    for bar, name in enumerate(progression):
        start = max(0.0, (bar * 4 - push_beats) * BEAT)
        ev.append((start, name, 4 * BEAT))
        ev += [(start + k * BEAT / 2, name, BEAT / 2 + 0.1) for k in range(1, 8)]
    return render_strikes(ev, len(progression) * 4 * BEAT)


def test_metric_beat_and_beat_position():
    bt = np.array([1.0, 1.5, 2.0, 2.5])
    assert rec._beat_position(1.75, bt, 0.5) == pytest.approx(1.5)
    assert rec._beat_position(0.75, bt, 0.5) == pytest.approx(-0.5)         # before the first beat
    assert rec._beat_position(3.0, bt, 0.5) == pytest.approx(4.0)           # after the last
    assert rec.metric_beat(2.02) == 2 and rec.metric_beat(2.3) == 2          # on or just after: this beat
    assert rec.metric_beat(2.5) == 3 and rec.metric_beat(2.9) == 3           # a half beat early: the next
    assert rec.metric_beat(1.97) == 2                                        # just early: the nearest


def test_pushed_chords_are_timed_between_beats_and_bars_follow_the_downbeat():
    r = recognize(pushed_song(0.5), SR)
    labels = [s["candidates"][0]["label"] for s in r["segments"]]
    assert labels[:4] == ["C", "G", "Am", "F"] or labels[:4] == ["C", "G", "Am", "Fmaj7"]
    true_starts = [max(0.0, (bar * 4 - 0.5) * BEAT) for bar in range(12)]
    got = [s["start"] for s in r["segments"]]
    assert len(got) >= 10
    near = [abs(g - t) for g, t in zip(got, true_starts)]
    assert np.median(near) < 0.06                      # the changes are found between the beats
    # the bar lines belong at the beat *after* each push, i.e. t = 2.0, 4.0 ...
    m = r["meter"]
    bt = np.array(r["beat_times"])
    downbeat_times = bt[m["downbeat"]::m["beats_per_bar"]]
    assert m["beats_per_bar"] == 4
    assert min(abs(downbeat_times - 4 * BEAT)) < 0.08


def test_unpushed_song_keeps_its_downbeat_on_the_chords():
    r = recognize(pushed_song(0.0), SR)
    bt = np.array(r["beat_times"])
    downbeat_times = bt[r["meter"]["downbeat"]::r["meter"]["beats_per_bar"]]
    assert min(abs(downbeat_times - 4 * BEAT)) < 0.08
    assert abs(r["segments"][1]["start"] - 4 * BEAT) < 0.06


def meter_change_song():
    ev, t = [], 0.0
    for name in ("C", "G", "Am", "F", "C", "G", "Am", "F"):
        for _ in range(4):
            ev.append((t, name, BEAT))
            t += BEAT
    for name in ("C", "F", "G", "C", "Am", "F", "G", "C"):
        for _ in range(3):
            ev.append((t, name, BEAT))
            t += BEAT
    return render_strikes(ev, t + 0.5), 8 * 4 * BEAT


def test_meter_sections_found_in_a_song_that_changes_meter():
    y, change_time = meter_change_song()
    r = recognize(y, SR)
    secs = r["meter"]["sections"]
    assert [s["beats_per_bar"] for s in secs] == [4, 3]
    bt = np.array(r["beat_times"])
    assert abs(bt[secs[1]["beat"]] - change_time) < 0.1            # the 3/4 starts at bar 9
    assert r["meter"]["time_signature"] == "4/4"                    # the longer... both are 8 bars: first wins
    starts = [s["start"] for s in r["segments"]]
    assert abs(starts[8] - change_time) < 0.1 and abs(starts[9] - (change_time + 3 * BEAT)) < 0.1


def test_constant_meters_get_no_sections():
    for beats, pattern in ((4, [1, 2]), (3, [1, 1, 2]), (4, [2])):
        rng = np.random.default_rng(beats)
        ev, t, chord, left = [], 0.0, "C", 0
        for _ in range(20):
            if left == 0:
                chord, left = rng.choice(list(CHORD_NOTES)), rng.choice(pattern)
            left -= 1
            for _ in range(beats):
                ev.append((t, chord, BEAT))
                t += BEAT
        r = recognize(render_strikes(ev, t + 0.5), SR)
        assert r["meter"]["sections"] == []


def test_meter_sections_unit():
    acc = np.ones(60)
    # 8 bars of 4, then 8 bars of 3
    four, three = [4 * k for k in range(8)], [32 + 3 * k for k in range(8)]
    secs = rec.meter_sections(four + three, acc, 56, 4)
    assert [(s["beat"], s["beats_per_bar"]) for s in secs] == [(0, 4), (32, 3)]
    # one meter all the way: nothing to report
    assert rec.meter_sections([4 * k for k in range(14)], acc, 56, 4) == []
    # too little evidence
    assert rec.meter_sections([0, 4, 8], acc, 12, 4) == []
    assert rec.meter_sections(four + three, acc, 56, 6) == []        # 6 is not a section meter
    # a pickup: the first downbeat is before the first tracked beat
    secs = rec.meter_sections([3 + 4 * k for k in range(8)] + [35 + 3 * k for k in range(8)], acc, 60, 4)
    assert secs[0]["beat"] == -1 and secs[1]["beat"] == 35      # bars at 3, 7 ... 31, then 4 more to 35


def test_bar_grid_with_sections():
    from mixpipe.chords.align import BarGrid

    g = BarGrid([0.5 * k for k in range(60)], sections=[{"beat": 0, "beats_per_bar": 4},
                                                       {"beat": 8, "beats_per_bar": 3}])
    assert [g.bar_start(k) for k in range(5)] == [0.0, 2.0, 4.0, 5.5, 7.0]       # 4,4 beats then 3s
    assert g.bar_of(1.9) == 0 and g.bar_of(2.0) == 1 and g.bar_of(4.0) == 2 and g.bar_of(5.4) == 2
    assert g.bar_of(5.5) == 3 and g.bar_of(7.0) == 4
    assert g.bar_start(-1) == -2.0
    assert g.section_changes() == [{"bar": 2, "time": 4.0, "beats_per_bar": 3}]
    # round trip
    for k in range(10):
        assert g.bar_of(g.bar_start(k) + 0.01) == k
    # a short last bar before the next section counts as a bar
    h = BarGrid([0.5 * k for k in range(40)], sections=[{"beat": 0, "beats_per_bar": 4},
                                                       {"beat": 6, "beats_per_bar": 4}])
    assert h.bars_before == [0, 2] and h.bar_start(1) == 2.0 and h.bar_start(2) == 3.0
    plain = BarGrid([0.5 * k for k in range(20)], beats_per_bar=3, downbeat=1)
    assert plain.bar_start(0) == 0.5 and plain.bar_start(1) == 2.0 and plain.section_changes() == []


def test_chord_bar_credits_a_push_to_the_next_bar():
    from mixpipe.chords.align import BarGrid

    g = BarGrid([0.5 * k for k in range(40)], beats_per_bar=4)        # bars every 2 s
    assert g.chord_bar(3.75) == 2 and g.bar_of(3.75) == 1             # an 8th early: written in bar 2
    assert g.chord_bar(3.5) == 1                                      # a whole beat early: not a push
    assert g.chord_bar(4.0) == 2 and g.chord_bar(4.1) == 2 and g.chord_bar(3.95) == 2


def test_instrumental_bars_place_pushed_chords_in_the_next_bar():
    chords = [chord("a", "G", 0.0, 3.75), chord("b", "D", 3.75, 7.75), chord("c", "C", 7.75, 12.0)]
    sheet = {"title": "Push", "key": "G:maj", "tempo": 120.0, "beat_period": 0.5,
             "beat_times": [0.5 * k for k in range(40)],
             "meter": {"time_signature": "4/4", "beats_per_bar": 4, "downbeat": 0, "detected": None},
             "lines": [], "chords": chords}
    from mixpipe.chords.align import grid_for

    rows = build_rows([], chords, 0.5, grid=grid_for(sheet))
    bars = rows[0]["bars"]
    assert [b["chords"] for b in bars] == [["a"], [], ["b"], [], ["c"]] or \
        [b["chords"] for b in bars][:3] == [["a"], [], ["b"]]
    pushed = next(b for b in bars if b["chords"] == ["b"])
    assert pushed["hold"] is None                 # it arrives on the bar, so nothing is "held" into it


def test_sections_in_the_sheet(grid_sheet):
    from mixpipe.chords.sheet import apply_settings, to_chordpro, to_text, view

    grid_sheet["meter"]["sections"] = [{"beat": 0, "beats_per_bar": 4}, {"beat": 8, "beats_per_bar": 3}]
    grid_sheet["meter"]["detected"] = {"time_signature": "4/4", "beats_per_bar": 4, "downbeat": 0,
                                       "sections": list(grid_sheet["meter"]["sections"])}
    v = view(grid_sheet)
    assert v["display"]["meter_changes"] == [{"bar": 3, "time_signature": "3/4", "time": 4.0}]
    assert "{time: 3/4}" in to_chordpro(grid_sheet) and "[Time: 3/4]" in to_text(grid_sheet)
    # a hand-picked time signature replaces the detected changes; reset brings them back
    apply_settings(grid_sheet, time_signature="3/4")
    assert grid_sheet["meter"]["sections"] == [] and view(grid_sheet)["display"]["meter_changes"] == []
    assert "[Time: 3/4]" not in to_text(grid_sheet).replace("Time: 3/4   ", "")
    apply_settings(grid_sheet, reset_meter=True)
    assert len(grid_sheet["meter"]["sections"]) == 2
    # nudging the bar lines slides every section
    apply_settings(grid_sheet, shift_downbeat=1)
    assert [s["beat"] for s in grid_sheet["meter"]["sections"]] == [1, 9]


# --------------------------------------------------------------------------- #
# tempo drift, free time, key sensitivity
# --------------------------------------------------------------------------- #

def drifting_song(periods, progression=("C", "G", "Am", "F")):
    """One chord per 4 beats; the beat period follows `periods` (seconds)."""
    bt = np.concatenate([[0.2], 0.2 + np.cumsum(periods)])         # bt[i] is the time of beat i
    prog = list(progression) * (len(periods) // (4 * len(progression)) + 1)
    ev = [(t, prog[i // 4], bt[i + 1] - t) for i, t in enumerate(bt[:-1])]
    return render_strikes(ev, bt[-1] + 0.5), bt


def test_beat_sync():
    onset = np.zeros(400)
    onset[::20] = 1.0                                             # pulses every 20 frames
    on_grid = np.arange(0, 400, 20)
    assert rec.beat_sync(onset, on_grid) == 1.0
    off_grid = on_grid + 10                                       # half a period off
    assert rec.beat_sync(onset, off_grid) < 0.2
    assert rec.beat_sync(np.zeros(400), on_grid) == 1.0           # no onsets: nothing to judge
    assert rec.beat_sync(onset, on_grid[:3]) == 1.0               # too few beats to judge
    dense = np.arange(0, 400, 5)                                  # a faster grid with extra beats
    assert rec.beat_sync(onset, dense) < 0.4                      # beats with no onset count against it


@pytest.mark.parametrize("name,periods", [
    ("accelerando", np.linspace(0.75, 0.30, 40)),
    ("ritardando", np.linspace(0.40, 0.90, 40)),
    ("wobble", 0.5 * (1 + 0.25 * np.sin(np.linspace(0, 4 * np.pi, 60)))),
])
def test_chord_changes_are_found_through_tempo_drift(name, periods):
    y, bt = drifting_song(periods)
    r = recognize(y, SR)
    starts = np.array([s["start"] for s in r["segments"]])
    true = [bt[4 * k] for k in range(1, len(periods) // 4)]       # a new chord every 4 beats
    err = np.array([np.min(np.abs(starts - t)) for t in true])
    assert err.mean() < 0.08 and err.max() < 0.25


def test_free_time_has_boundaries_but_no_grid(monkeypatch):
    y = pushed_song(0.0)
    real = rec.choose_beats
    monkeypatch.setattr(rec, "choose_beats", lambda yy, onset, beats, tempo, n: (
        rec._pseudo_beats(onset, n), 0.0, True))
    r = recognize(y, SR)
    assert r["beat_times"] == [] and r["tempo"] == 0.0
    assert r["meter"]["free_time"] is True and r["meter"]["sections"] == []
    labels = [s["candidates"][0]["label"] for s in r["segments"]]
    assert labels[:3] == ["C", "G", "Am"]                         # the chords themselves still come out
    monkeypatch.setattr(rec, "choose_beats", real)


def test_choose_beats_prefers_a_better_tracker_and_can_give_up(monkeypatch):
    y = pushed_song(0.0)
    onset = np.ones(2000)
    beats = np.arange(10, 1000, 20)
    # a grid that keeps up with the music is returned untouched
    monkeypatch.setattr(rec, "beat_sync", lambda o, b: 0.97)
    out = rec.choose_beats(y, onset, beats, 120.0, 2000)
    assert out[0] is beats and out[2] is False
    # nothing keeps up: segment boundaries from onsets, no tempo
    monkeypatch.setattr(rec, "beat_sync", lambda o, b: 0.2)
    peaks = np.zeros(2000)
    peaks[::40] = 5.0
    b, tempo, free = rec.choose_beats(y, peaks, beats, 120.0, 2000)
    assert free is True and tempo == 0.0 and len(b) >= 4
    assert np.all(np.diff(b) >= int(round(rec.MIN_PSEUDO_BEAT_S * rec.SR / rec.HOP)))


def test_key_sensitivity_modes():
    def song(mod_bars):
        seq = ["C", "G", "Am", "F"] * 3 + ["D", "A", "Bm", "G"][:mod_bars] + ["C", "G", "Am", "F"] * 3
        ev, t = [], 0.0
        for name in seq:
            for _ in range(4):
                ev.append((t, name, BEAT))
                t += BEAT
        return render_strikes(ev, t + 0.5)

    y = song(3)
    normal = recognize(y, SR)
    sensitive = recognize(y, SR, key_sensitivity="sensitive")
    assert len(normal["key_regions"]) == 1 and normal["key"] == "C:maj"
    keys = [k["key"] for k in sensitive["key_regions"]]
    assert len(keys) == 3 and keys[0] == keys[-1] == "C:maj" and keys[1] != "C:maj"
    # a song that stays put is left alone in both modes
    ev, t = [], 0.0
    for name in ["C", "G", "Am", "F", "Dm", "G", "C", "Em", "Am", "F", "G", "C"] * 3:
        for _ in range(4):
            ev.append((t, name, BEAT))
            t += BEAT
    steady = render_strikes(ev, t + 0.5)
    assert len(recognize(steady, SR, key_sensitivity="sensitive")["key_regions"]) == 1
    assert recognize(y, SR, key_sensitivity="bogus")["key_regions"] == normal["key_regions"]


def test_api_key_changes_option_reaches_the_recognizer(tmp_path, progression_audio):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from mixpipe.web.server import parse_options, create_app

    assert parse_options({})["key_changes"] == "normal"
    assert parse_options({"key_changes": "sensitive"})["key_changes"] == "sensitive"
    assert parse_options({"key_changes": "wild"})["key_changes"] == "normal"
    buf = io.BytesIO()
    sf.write(buf, progression_audio, SR, format="WAV")
    client = TestClient(create_app(tmp_path, separator=fake_separator, transcriber=fake_transcriber,
                                   run_inline=True))
    sid = client.post("/api/songs", files=[("files", ("a.wav", buf.getvalue(), "audio/wav"))],
                      data={"language": "en", "key_changes": "sensitive"}).json()["id"]
    assert json.loads((tmp_path / sid / "chords.json").read_text())["key_changes"] == "sensitive"
