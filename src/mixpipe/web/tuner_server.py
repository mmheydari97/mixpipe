"""Guitar tuner: the page streams the microphone here, this says which string
is ringing and how far off it is.

    python -m mixpipe.cli tuner-ui          # http://127.0.0.1:8768

GET /api/tunings   every tuning the page's two switches make (theory.guitar_tuning)
WS  /ws            text frames set {sr, whole_down, drop6, a4, lock, wait} (the first
                   one must carry sr; wait: start on the next pluck); binary frames are
                   float32 mono chunks; each chunk is answered with one reading from
                   tuner.StringTracker: silent | pending | note | tuned (locked in).
"""

from __future__ import annotations

import json

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..chords.theory import guitar_tuning, guitar_tunings
from ..tuner import StringTracker
from .common import add_page

MAX_CHUNK = 16384        # samples; a page sends 2048


def _strings(cfg: dict) -> list[int]:
    t = guitar_tuning(bool(cfg.get("whole_down")), bool(cfg.get("drop6")))
    return [s["midi"] for s in t["strings"]]


def _a4(value) -> float:
    try:
        return min(460.0, max(420.0, float(value)))
    except (TypeError, ValueError):
        return 440.0


def create_app() -> FastAPI:
    app = FastAPI(title="mixpipe tuner")

    @app.get("/api/tunings")
    def tunings():
        return guitar_tunings()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        tracker: StringTracker | None = None
        try:
            while True:
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    try:
                        cfg = json.loads(msg["text"])
                    except ValueError:
                        continue
                    if tracker is None:
                        sr = int(cfg.get("sr") or 0)
                        if not 8000 <= sr <= 192000:
                            await sock.send_json({"state": "error", "detail": "send sr first"})
                            continue
                        tracker = StringTracker(sr, _strings(cfg), _a4(cfg.get("a4", 440)),
                                                cfg.get("lock"))
                    else:
                        tracker.configure(
                            strings=_strings(cfg) if {"whole_down", "drop6"} & cfg.keys() else None,
                            a4=_a4(cfg["a4"]) if "a4" in cfg else None,
                            lock=cfg["lock"] if "lock" in cfg else "keep",
                            wait=bool(cfg.get("wait")))
                elif msg.get("bytes") is not None and tracker is not None:
                    data = msg["bytes"]
                    chunk = np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4")[:MAX_CHUNK]
                    await sock.send_json(tracker.feed(chunk))
        except WebSocketDisconnect:
            pass

    add_page(app, "tuner.html")
    return app
