"""All three web apps on one port, with a home page to pick one.

    python -m mixpipe.cli ui                # http://127.0.0.1:8760

The chord finder, the splitter and the mixer are mounted unchanged under
/chords, /splitter and /mixer; each page finds its own API from its path
(BASE in static/common.js), so the same pages also work standalone on their
own ports (chords-ui, split-ui, mix-ui). The splitter's library is the mixer's
"From the splitter" list, as it is when they run separately.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from .common import mount_static, page_html

TOOLS = ("chords", "splitter", "mixer")


def create_app(
    chords_root: str | Path = "out/chords",
    splitter_root: str | Path = "out/splitter",
    mixer_root: str | Path = "out/mixer",
    workers: int = 1,
    chords_options: dict | None = None,
    splitter_options: dict | None = None,
    mixer_options: dict | None = None,
) -> FastAPI:
    """Build the hub. `*_options` go to each app's create_app (tests pass fakes)."""
    from . import mixer_server, server, splitter_server

    app = FastAPI(title="mixpipe")
    chords = server.create_app(chords_root, workers=workers, **(chords_options or {}))
    # the splitter's key, beats and chords come from the chord finder: same store, same queue
    splitter = splitter_server.create_app(splitter_root, workers=workers, chords_root=chords_root,
                                          chords_service=chords.state.songs, **(splitter_options or {}))
    mixer = mixer_server.create_app(mixer_root, splitter_root=splitter_root, **(mixer_options or {}))

    @app.get("/")
    def home():
        return page_html("home.html")

    mount_static(app)               # the mounted pages ask for /static/..., served once here
    app.mount("/chords", chords)
    app.mount("/splitter", splitter)
    app.mount("/mixer", mixer)
    return app
