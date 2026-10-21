"""Client for a local ACE-Step 1.5 server, the model behind AI edits.

Two calls:

  status()          - is the server up, and which model it has loaded.
  edit(path, ...)   - send a clip and a prompt, get the edited audio back.
                      mode="repaint" rewrites start_s..end_s and keeps the rest;
                      mode="cover" re-styles the whole clip (`strength` 0..1,
                      higher stays closer to the original).

`splice()` puts a repaint back into the original, so everything outside the
range is the original's samples, not the model's copy of them.

The server runs on this machine (ACESTEP_URL, default http://127.0.0.1:8001), so
the audio goes no further than localhost. From an ACE-Step 1.5 checkout:

    ACESTEP_LM_BACKEND=mlx ACESTEP_INIT_LLM=false uv run acestep-api --port 8001
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_URL = "http://127.0.0.1:8001"
MODES = ("repaint", "cover")
FADE_S = 0.05          # crossfade at each edge of a repaint


def base_url() -> str:
    return (os.environ.get("ACESTEP_URL") or DEFAULT_URL).rstrip("/")


def _data(resp):
    """The `data` of ACE-Step's {data, code, error} wrapper, or an error that says why not."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not resp.ok or body.get("error") or body.get("code", 200) != 200:
        raise RuntimeError(f"ACE-Step {resp.status_code}: {body.get('error') or resp.text[:300]}")
    return body.get("data")


def status(timeout: float = 3.0) -> dict:
    """{"url", "ok", "model", "error"}. Never raises."""
    import requests

    url = base_url()
    try:
        d = _data(requests.get(f"{url}/health", timeout=timeout)) or {}
        return {"url": url, "ok": d.get("status") == "ok", "model": d.get("loaded_model"), "error": None}
    except Exception as exc:  # noqa: BLE001 - not running is an answer, not a crash
        return {"url": url, "ok": False, "model": None, "error": f"{type(exc).__name__}: {exc}"}


def edit(
    src_path: str | os.PathLike,
    prompt: str,
    mode: str = "repaint",
    start_s: float = 0.0,
    end_s: float | None = None,
    strength: float | None = None,
    seed: int | None = None,
    timeout: float = 900.0,
    poll_s: float = 2.0,
) -> tuple[np.ndarray, int, dict]:
    """Edit the audio at `src_path`. Returns (audio, sr, info) with info
    {"seed", "dit_model", "seconds"}. `end_s=None` repaints to the end."""
    import requests

    from .io_utils import read_audio

    if mode not in MODES:
        raise ValueError(f"unknown edit mode: {mode}")
    url = base_url()
    fields = {
        "task_type": mode, "prompt": prompt, "audio_format": "wav", "batch_size": 1,
        "thinking": "false",
    }
    if mode == "repaint":
        fields.update(repainting_start=start_s, repainting_end=-1 if end_s is None else end_s,
                      chunk_mask_mode="explicit")
    if mode == "cover" and strength is not None:
        fields["audio_cover_strength"] = strength
    if seed is not None:
        fields.update(use_random_seed="false", seed=int(seed))

    t0 = time.time()
    with open(src_path, "rb") as f:
        task = _data(requests.post(f"{url}/release_task", files={"src_audio": f}, data=fields, timeout=60))
    task_id = task["task_id"]
    while True:
        items = _data(requests.post(f"{url}/query_result", json={"task_id_list": [task_id]}, timeout=30))
        item = items[0] if isinstance(items, list) else items
        if item.get("status") == 1:
            break
        if item.get("status") == 2:
            raise RuntimeError(f"ACE-Step failed: {item.get('progress_text') or item.get('result') or 'no reason given'}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"ACE-Step took longer than {timeout:.0f} s")
        time.sleep(poll_s)

    result = item["result"]
    result = json.loads(result) if isinstance(result, str) else result
    first = result[0] if isinstance(result, list) else result
    audio_resp = requests.get(f"{url}{first['file']}", timeout=120)
    audio_resp.raise_for_status()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "edit.wav"
        out.write_bytes(audio_resp.content)
        audio, sr = read_audio(out)
    info = {"seed": first.get("seed_value"), "dit_model": first.get("dit_model"),
            "seconds": round(time.time() - t0, 1)}
    return audio, sr, info


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def splice(original: np.ndarray, generated: np.ndarray, sr: int, start_s: float,
           end_s: float | None, fade_s: float = FADE_S,
           finish: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """The original with start_s..end_s taken from `generated`.

    ACE-Step returns the untouched part too, but at its own level (about 1 dB
    down), so the generated audio is first scaled to the original by the
    least-squares gain over the part that should be identical. `finish` (the
    session's limiter) then runs over it, so only the new part is touched."""
    a, b = _stereo(original), _stereo(generated)
    n = len(a)
    if len(b) < n:                                     # a few samples short at the end
        b = np.concatenate([b, a[len(b):]])
    b = b[:n]
    s = max(0, int(round(start_s * sr)))
    e = n if end_s is None else min(n, int(round(end_s * sr)))
    if e <= s:
        raise ValueError("the repaint range is empty")

    outside = np.ones(n, dtype=bool)
    outside[s:e] = False
    den = float(np.sum(a[outside] ** 2))
    if outside.sum() >= sr and den > 0:
        g = float(np.sum(a[outside] * b[outside])) / den
        if g > 1e-3:
            b = b / g
    if finish is not None:
        b = finish(b)

    # a smooth ramp centred on each edge that lies inside the clip; both sides
    # are the same music there, so the gains add to one
    t = np.arange(n)
    f = max(1.0, fade_s * sr)
    rise = np.clip((t - (s - f / 2)) / f, 0, 1) if s > 0 else np.ones(n)
    fall = np.clip(((e + f / 2) - t) / f, 0, 1) if e < n else np.ones(n)
    w = (np.sin(np.pi / 2 * np.minimum(rise, fall)) ** 2)[:, None]
    return (1 - w) * a + w * b
