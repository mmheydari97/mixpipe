"""Tests for the whole-lyric LLM revision (chords/revise.py) and its pipeline / API hooks.

Run with:  PYTHONPATH=src python -m pytest tests/test_revise.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("librosa")

from mixpipe.chords.revise import (PROMPT, _clean, revise_lines, revise_text,  # noqa: E402
                                   similarity)

SR = 44_100


def W(text, start, end):
    return {"text": text, "start": start, "end": end}


def LINES():
    return [{"words": [W("I", 1.0, 1.2), W("scream", 1.2, 1.6), W("of", 1.6, 1.8), W("you", 1.8, 2.2)],
             "gap_before": False},
            {"words": [W("every", 3.0, 3.4), W("night", 3.4, 4.0)], "gap_before": False}]


def fake_chat(reply, seen=None):
    def chat(messages, model, max_tokens, reasoning_effort=None):
        if seen is not None:
            seen.append({"messages": messages, "model": model, "max_tokens": max_tokens,
                         "reasoning_effort": reasoning_effort})
        return reply
    return chat


def test_clean_strips_code_fences():
    assert _clean("```text\nI dream of you\nevery night\n```") == "I dream of you\nevery night"
    assert _clean("  I dream of you  ") == "I dream of you"


def test_revise_text_sends_the_whole_lyric_with_title_and_language():
    seen = []
    out = revise_text("I scream of you\nevery night", "a/model", title="Dreams", language="en",
                      chat=fake_chat("I dream of you\nevery night", seen))
    assert out == "I dream of you\nevery night"
    msgs = seen[0]["messages"]
    assert msgs[0]["content"] == PROMPT and seen[0]["model"] == "a/model"
    assert msgs[1]["content"].startswith("Song: Dreams\nLanguage: en\n\nTranscript:\nI scream of you")
    # no context: just the lyric
    revise_text("x y", "a/model", chat=fake_chat("x y", seen))
    assert seen[1]["messages"][1]["content"] == "x y"
    with pytest.raises(RuntimeError):
        revise_text("x", "a/model", chat=fake_chat("  "))
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[0]["max_tokens"] == 6000 + 2 * len("I scream of you\nevery night")


def test_a_cut_off_reply_is_retried_with_a_bigger_budget():
    from mixpipe.openrouter import CutOff

    budgets = []

    def thinker(limit):
        def chat(messages, model, max_tokens, reasoning_effort=None):
            budgets.append(max_tokens)
            if max_tokens < limit:
                raise CutOff("cut off")
            return "I dream of you"
        return chat
    assert revise_text("I scream of you", "a/model", chat=thinker(10_000)) == "I dream of you"
    first = 6000 + 2 * len("I scream of you")
    assert budgets == [first, 2 * first]
    budgets.clear()
    with pytest.raises(CutOff, match="reasoning model"):
        revise_text("I scream of you", "a/model", chat=thinker(10 ** 9))
    assert budgets == [first, 2 * first]


def test_revise_lines_keeps_word_timing():
    lines, info = revise_lines(LINES(), "a/model", chat=fake_chat("I dream of you\nevery night"))
    assert info["applied"] and info["changed"] == 1 and info["source"] == "I scream of you\nevery night"
    first = lines[0]["words"]
    assert [w["text"] for w in first] == ["I", "dream", "of", "you"]
    assert (first[1]["start"], first[1]["end"]) == (1.2, 1.6)        # the fixed word takes its slot
    assert (first[3]["start"], first[3]["end"]) == (1.8, 2.2)        # the others keep theirs
    assert lines[1]["words"][1] == {"text": "night", "start": 3.4, "end": 4.0}


def test_revise_lines_rejects_a_rewrite_and_reuses_the_cache():
    original = LINES()
    lines, info = revise_lines(original, "a/model",
                               chat=fake_chat("Totally different song\nwith other words here"))
    assert not info["applied"] and "rewrote" in info["reason"] and lines is original
    assert similarity("a b c", "a b c") == 1.0 and similarity("a b", "c d") == 0.0

    cached = {"model": "a/model", "source": "I scream of you\nevery night", "text": "I dream of you\nevery night"}

    def boom(*a, **k):
        raise AssertionError("asked again")
    lines, info = revise_lines(LINES(), "a/model", chat=boom, cached=cached)
    assert info["applied"] and lines[0]["words"][1]["text"] == "dream"
    _, info = revise_lines(LINES(), "a/model", chat=fake_chat("I scream of you\nevery night"))
    assert not info["applied"] and info["reason"] == "no changes needed"


# --------------------------------------------------------------------------- #
# pipeline + API
# --------------------------------------------------------------------------- #

def _client(tmp_path, reviser):
    from fastapi.testclient import TestClient

    from mixpipe.web.server import create_app
    from test_chords import fake_separator, fake_transcriber

    return TestClient(create_app(tmp_path, separator=fake_separator, transcriber=fake_transcriber,
                                 melody_extractor=lambda *a, **k: {"notes": []},
                                 reviser=reviser, run_inline=True))


@pytest.fixture
def song_wav(wav_bytes):
    t = np.arange(SR * 8) / SR
    return wav_bytes((0.3 * np.sin(2 * np.pi * 262 * t)).astype(np.float32), SR)


def test_pipeline_revises_the_lyrics_when_asked(tmp_path, song_wav):
    seen = []
    client = _client(tmp_path, fake_chat("la la love", seen))
    files = [("files", ("my song.wav", song_wav, "audio/wav"))]
    sid = client.post("/api/songs", files=files,
                      data={"language": "en", "revise_model": "a/model"}).json()["id"]
    song = client.get(f"/api/songs/{sid}").json()
    words = [w["text"] for ln in song["lines"] for w in ln["words"]]
    assert words == ["la", "la", "love"]
    assert song["transcript"]["revised"] == {"model": "a/model", "applied": True, "changed": 1, "reason": None}
    assert "Song: my song" in seen[0]["messages"][1]["content"]
    stored = json.loads((tmp_path / sid / "revision.json").read_text())
    assert stored["source"] == "la la lah" and stored["text"] == "la la love"

    # without the option nothing is asked
    sid2 = client.post("/api/songs", files=[("files", ("other.wav", song_wav, "audio/wav"))],
                       data={"language": "en"}).json()["id"]
    assert client.get(f"/api/songs/{sid2}").json()["transcript"]["revised"] is None
    assert len(seen) == 1
    assert client.post("/api/songs", files=files,
                       data={"revise_model": "not a model"}).status_code == 400


def test_a_failed_revision_keeps_the_lyrics_as_heard(tmp_path, song_wav):
    def broken(*a, **k):
        raise RuntimeError("rate limited")
    client = _client(tmp_path, broken)
    sid = client.post("/api/songs", files=[("files", ("s.wav", song_wav, "audio/wav"))],
                      data={"language": "en", "revise_model": "a/model"}).json()["id"]
    song = client.get(f"/api/songs/{sid}").json()
    assert client.get(f"/api/songs/{sid}/status").json()["state"] == "done"
    assert [w["text"] for ln in song["lines"] for w in ln["words"]] == ["la", "la", "lah"]
    rev = song["transcript"]["revised"]
    assert not rev["applied"] and "rate limited" in rev["reason"]


def test_revise_endpoint_suggests_without_saving(tmp_path, song_wav):
    client = _client(tmp_path, fake_chat("la la love"))
    sid = client.post("/api/songs", files=[("files", ("s.wav", song_wav, "audio/wav"))],
                      data={"language": "en"}).json()["id"]
    res = client.post(f"/api/songs/{sid}/revise", json={"model": "a/model", "text": "la la lah"}).json()
    assert res["text"] == "la la love" and ["delete", "lah"] in res["diff"]
    assert [w["text"] for w in client.get(f"/api/songs/{sid}").json()["lines"][0]["words"]][-1] == "lah"
    # the editor's text wins over the saved lyrics; empty text is refused
    assert client.post(f"/api/songs/{sid}/revise", json={"model": "a/model", "text": " "}).status_code == 400
    assert client.post(f"/api/songs/{sid}/revise", json={"model": "a/model"}).json()["old"] == "la la lah"
