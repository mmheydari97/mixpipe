"""Thin OpenRouter client shared by the mix advisor and the chord finder.

Three calls:

  list_models(kind)  - public model catalogue, normalised for the UI picker.
                       kind="chat" or "transcription". No API key needed.
                       search_models() ranks it the way the web picker does.
  chat(...)          - chat completion, returns the reply text.
  transcribe(...)    - speech-to-text via /audio/transcriptions.

The key comes from OPENROUTER_API_KEY, same as the advisor always used.
"""

from __future__ import annotations

import base64
import os
import re
import time

API = "https://openrouter.ai/api/v1"

# vendor/model, optionally :variant ("anthropic/claude-haiku-5.5", "meta-llama/x:free"),
# or a "~vendor/family-latest" alias that follows the newest model of a family
MODEL_ID = re.compile(r"^~?[A-Za-z0-9][\w.\-]*/[\w.\-:]+$")
_CACHE_TTL_S = 3600.0
_model_cache: dict[str, tuple[float, list[dict]]] = {}


def api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or None


def _require_key() -> str:
    key = api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_require_key()}",
        "Content-Type": "application/json",
        "X-Title": "mixpipe",
    }


# --------------------------------------------------------------------------- #
# model catalogue
# --------------------------------------------------------------------------- #


def _price(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_free(model: dict) -> bool:
    """`:free` variants, or zero prompt *and* completion price.

    Router entries like `openrouter/auto` report -1 (variable), which is not free.
    """
    if model.get("id", "").endswith(":free"):
        return True
    pricing = model.get("pricing") or {}
    prompt = _price(pricing.get("prompt"))
    completion = _price(pricing.get("completion"))
    if prompt is None:
        return False
    return prompt == 0 and (completion is None or completion == 0)


def normalize_model(model: dict, kind: str) -> dict:
    """Reduce a raw /models entry to what the picker shows and ranks on."""
    pricing = model.get("pricing") or {}
    prompt = _price(pricing.get("prompt"))
    completion = _price(pricing.get("completion"))
    free = is_free(model)

    if free:
        label, sort_price = "free", 0.0
    elif prompt is None or prompt < 0:
        label, sort_price = "variable", float("inf")
    elif kind == "transcription" and not completion:
        # audio-billed STT models: prompt price is per second of audio
        per_min = prompt * 60.0
        label, sort_price = f"${per_min:.4f}/min", per_min
    else:
        per_m_in = prompt * 1e6
        per_m_out = (completion or 0.0) * 1e6
        label = f"${per_m_in:.2f} in / ${per_m_out:.2f} out per 1M"
        sort_price = per_m_in + per_m_out

    arch = model.get("architecture") or {}
    return {
        "id": model.get("id", ""),
        "name": model.get("name", model.get("id", "")),
        "description": (model.get("description") or "")[:300],
        "is_free": free,
        "price_label": label,
        "sort_price": sort_price if sort_price != float("inf") else 1e12,
        "context_length": model.get("context_length"),
        "input_modalities": arch.get("input_modalities") or [],
        "output_modalities": arch.get("output_modalities") or ["text"],
    }


def valid_model_id(model: str | None) -> bool:
    """Shaped like an OpenRouter model id. Says nothing about whether it exists."""
    return bool(model) and len(model) <= 200 and MODEL_ID.match(model) is not None


def search_models(models: list[dict], query: str = "") -> list[dict]:
    """Rank catalogue entries for a search, like the web picker (`rankModels` in
    static/common.js): every word must appear in the id, name or description,
    matches at the start of the id or name count most, then the cheaper model wins.
    The word "free" puts free models first."""
    tokens = query.lower().split()
    want_free = "free" in tokens
    terms = [t for t in tokens if t != "free"]
    scored = []
    for m in models:
        mid, name, desc = m["id"].lower(), (m.get("name") or "").lower(), (m.get("description") or "").lower()
        tail = mid.split("/")[-1]
        score = 0
        for t in terms:
            s = (30 if tail.startswith(t) or mid.startswith(t) else 25 if name.startswith(t)
                 else 20 if t in mid else 15 if t in name else 4 if t in desc else 0)
            if not s:
                break
            score += s
        else:
            scored.append((m, score))
    scored.sort(key=lambda ms: (-int(ms[0]["is_free"]) if want_free else 0, -ms[1],
                                ms[0]["sort_price"], ms[0]["name"]))
    return [m for m, _ in scored]


def list_models(kind: str = "chat", refresh: bool = False) -> list[dict]:
    """Fetch and normalise the catalogue. Cached in memory for an hour."""
    if kind not in ("chat", "transcription"):
        raise ValueError(f"unknown model kind: {kind}")
    now = time.monotonic()
    cached = _model_cache.get(kind)
    if cached and not refresh and now - cached[0] < _CACHE_TTL_S:
        return cached[1]

    import requests

    params = {"output_modalities": "transcription"} if kind == "transcription" else {}
    resp = requests.get(f"{API}/models", params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json().get("data", [])
    # ":batch" variants only work through the batch API; chat/completions answers 404
    raw = [m for m in raw if not str(m.get("id", "")).endswith(":batch")]
    if kind == "chat":
        # anything that can answer in text; models that also return audio or images
        # (gpt-audio, gpt-5-image, Lyria) stay, and the picker tags them
        raw = [
            m for m in raw
            if "text" in ((m.get("architecture") or {}).get("output_modalities") or ["text"])
        ]
    models = [normalize_model(m, kind) for m in raw]
    _model_cache[kind] = (now, models)
    return models


# --------------------------------------------------------------------------- #
# calls
# --------------------------------------------------------------------------- #


class CutOff(RuntimeError):
    """The reply hit max_tokens before it was finished (often a reasoning
    model that spent the budget thinking)."""


def chat(
    messages: list[dict],
    model: str,
    max_tokens: int = 4000,
    timeout: float = 120.0,
    reasoning_effort: str | None = None,
) -> str:
    """Chat completion. Returns the assistant's reply text.

    A reasoning model's thinking counts against `max_tokens`, so a long think can
    cut the reply off mid-JSON; `reasoning_effort` ("low", "medium", "high") keeps
    it in check (models without reasoning ignore it). A reply that was cut off
    raises instead of returning half an answer."""
    import requests

    body = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    resp = requests.post(f"{API}/chat/completions", headers=_headers(), json=body, timeout=timeout)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise CutOff(f"the reply from {model} was cut off at {max_tokens} tokens")
    return choice["message"]["content"] or ""


def transcribe(
    audio_bytes: bytes,
    audio_format: str,
    model: str,
    language: str | None = None,
    timeout: float = 600.0,
) -> dict:
    """Speech-to-text. Returns the raw response JSON.

    Asks for verbose_json with word timestamps. Only OpenAI-compatible upstreams
    honour that; others return plain `text`, which the caller has to cope with.
    """
    import requests

    body = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": audio_format,
        },
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
        "temperature": 0,
    }
    if language:
        body["language"] = language

    resp = requests.post(
        f"{API}/audio/transcriptions", headers=_headers(), json=body, timeout=timeout
    )
    if resp.status_code == 400:
        # some providers reject the combined granularity list or verbose_json
        body["timestamp_granularities"] = ["word"]
        resp = requests.post(
            f"{API}/audio/transcriptions", headers=_headers(), json=body, timeout=timeout
        )
    if resp.status_code == 400:
        body.pop("timestamp_granularities")
        body["response_format"] = "json"
        resp = requests.post(
            f"{API}/audio/transcriptions", headers=_headers(), json=body, timeout=timeout
        )
    if not resp.ok:
        raise RuntimeError(
            f"OpenRouter transcription failed ({resp.status_code}): {resp.text[:300]}"
        )
    return resp.json()
