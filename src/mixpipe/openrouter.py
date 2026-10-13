"""Thin OpenRouter client shared by the mix advisor and the chord finder.

Three calls:

  list_models(kind)  - public model catalogue, normalised for the UI picker.
                       kind="chat" or "transcription". No API key needed.
  chat(...)          - chat completion, returns the reply text.
  transcribe(...)    - speech-to-text via /audio/transcriptions.

The key comes from OPENROUTER_API_KEY, same as the advisor always used.
"""

from __future__ import annotations

import base64
import os
import time

API = "https://openrouter.ai/api/v1"

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
    }


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
    if kind == "chat":
        # text-in/text-out only: drops music, image and speech generators
        raw = [
            m for m in raw
            if ((m.get("architecture") or {}).get("output_modalities") or ["text"]) == ["text"]
        ]
    models = [normalize_model(m, kind) for m in raw]
    _model_cache[kind] = (now, models)
    return models


# --------------------------------------------------------------------------- #
# calls
# --------------------------------------------------------------------------- #


def chat(
    messages: list[dict],
    model: str,
    max_tokens: int = 4000,
    timeout: float = 120.0,
) -> str:
    """Chat completion. Returns the assistant's reply text."""
    import requests

    resp = requests.post(
        f"{API}/chat/completions",
        headers=_headers(),
        json={"model": model, "max_tokens": max_tokens, "messages": messages},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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
