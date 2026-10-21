"""mixpipe - a prototype multitrack mix and master pipeline."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

SAMPLE_RATE = 48_000


def _load_dotenv() -> None:
    """Load KEY=value pairs from the repo-root .env (or $MIXPIPE_ENV_FILE).

    Real environment variables win, so `OPENROUTER_API_KEY=... chords-ui` still works.
    Every entry point (CLI, chords-ui, splitter-ui) imports this package, so they
    all see the same values.
    """
    path = Path(os.environ.get("MIXPIPE_ENV_FILE") or Path(__file__).resolve().parents[2] / ".env")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


_load_dotenv()
