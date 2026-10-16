"""Fixtures shared by the song tool tests (chord finder, splitter)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def wav_bytes():
    """wav_bytes(audio, sr) -> the audio as WAV file bytes."""
    def encode(audio, sr: int) -> bytes:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()
    return encode


@pytest.fixture
def fake_link():
    """fake_link(wav) -> (calls, resolver, downloader): stand-ins for
    fetch.resolve / fetch.download that never touch the network. A URL with
    "playlist" in it is refused; set calls["fail"] to make downloads fail."""
    def make(wav: bytes):
        from mixpipe.fetch import FetchError

        calls = {"resolve": 0, "download": 0, "fail": False}

        def resolver(url):
            calls["resolve"] += 1
            if "playlist" in url:
                raise FetchError("that's a playlist or album; paste a link to a single song")
            return {"url": url, "page_url": "https://www.youtube.com/watch?v=abc",
                    "download_url": "https://www.youtube.com/watch?v=abc", "site": "Youtube",
                    "title": "Some Band - Some Song", "duration": 16.0, "note": None}

        def downloader(source, dest, progress=None):
            calls["download"] += 1
            if calls["fail"]:
                raise FetchError("download failed: Video unavailable")
            if progress:
                progress(0.5, "downloading")
            path = Path(dest) / "audio.wav"
            path.write_bytes(wav)
            return path

        return calls, resolver, downloader
    return make
