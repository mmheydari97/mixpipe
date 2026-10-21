"""The OpenRouter catalogue shared by the chord finder and the mixer: listing,
searching, model ids, and the /api/openrouter routes both web apps serve."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mixpipe import openrouter
from mixpipe.web.common import add_openrouter_routes

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def catalogue(monkeypatch):
    """list_models() over the fixture, plus a :batch twin that chat/completions refuses."""
    raw = json.loads((FIXTURES / "openrouter_models.json").read_text())["data"]
    batch = {**next(m for m in raw if m["id"] == "google/gemini-2.5-flash"), "id": "google/gemini-2.5-flash:batch"}

    class Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": self.data}

    calls = []
    monkeypatch.setattr("requests.get", lambda url, params=None, timeout=None: calls.append(params) or Resp(raw + [batch]))
    monkeypatch.setattr(openrouter, "_model_cache", {})
    return calls


def test_batch_variants_are_not_offered(catalogue):
    ids = [m["id"] for m in openrouter.list_models("chat")]
    assert "google/gemini-2.5-flash" in ids and not any(i.endswith(":batch") for i in ids)


def test_models_that_also_return_audio_or_images_are_offered(catalogue):
    by_id = {m["id"]: m for m in openrouter.list_models("chat")}
    assert by_id["openai/gpt-audio-mini"]["output_modalities"] == ["text", "audio"]
    assert by_id["openrouter/auto"]["output_modalities"] == ["text", "image"]
    assert by_id["google/gemini-2.5-flash"]["output_modalities"] == ["text"]


def test_search_ranks_like_the_web_picker(catalogue):
    models = openrouter.list_models("chat")
    assert openrouter.search_models(models, "gemini")[0]["id"] == "google/gemini-2.5-flash"
    free_first = openrouter.search_models(models, "free")
    assert free_first[0]["is_free"] and not free_first[-1]["is_free"]
    assert openrouter.search_models(models, "gemini nonsense-word") == []
    assert len(openrouter.search_models(models, "")) == len(models)


@pytest.mark.parametrize("model,ok", [
    ("anthropic/claude-haiku-5.5", True),
    ("thinkingmachines/inkling-small:free", True),
    ("~anthropic/claude-haiku-latest", True),       # a family alias
    ("openrouter/auto", True),
    ("", False), (None, False), ("claude", False), ("../../etc/passwd", False),
    ("a b/c", False), ("vendor/model\nX-Header: 1", False), ("x/" + "y" * 300, False),
])
def test_model_ids(model, ok):
    assert openrouter.valid_model_id(model) is ok


def test_shared_routes(catalogue, monkeypatch):
    app = FastAPI()
    add_openrouter_routes(app, decorate=lambda kind, ms: [{**m, "seen_as": kind} for m in ms])
    c = TestClient(app)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert c.get("/api/openrouter/status").json() == {"has_key": True}
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert c.get("/api/openrouter/status").json() == {"has_key": False}
    models = c.get("/api/openrouter/models?kind=chat").json()
    assert models and all(m["seen_as"] == "chat" for m in models)
    assert c.get("/api/openrouter/models?kind=images").status_code == 400

    def offline(kind, refresh=False):
        raise ConnectionError("no network")

    monkeypatch.setattr(openrouter, "list_models", offline)
    r = c.get("/api/openrouter/models")
    assert r.status_code == 502 and "no network" in r.json()["detail"]


def test_chord_finder_still_adds_word_timing(catalogue, tmp_path):
    from mixpipe.web.server import create_app

    c = TestClient(create_app(tmp_path))
    models = c.get("/api/openrouter/models?kind=transcription").json()
    assert models and all("word_timing" in m for m in models)
    assert all("word_timing" not in m for m in c.get("/api/openrouter/models?kind=chat").json())
