"""Fetch a song from a link instead of an upload.

  resolve(url)   -> source dict: what will be downloaded, title, duration. Fast,
                    runs before the song is created so bad links fail right away.
  download(...)  -> the audio file, saved into the song's inputs/ folder.

Anything yt-dlp understands works: YouTube, SoundCloud, Bandcamp, Vimeo, a
direct link to an .mp3/.wav, and ~1800 other sites. Spotify streams are DRM
protected, so for a Spotify track we read its title/artist/length from the
public page and download the closest-length match from YouTube instead.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

MAX_SECONDS = 30 * 60   # longer than any song; stops a 10-hour mix by accident
MAX_BYTES = 500 * 1024 * 1024
AUDIO_FORMAT = "bestaudio[ext=m4a]/bestaudio/best"  # m4a plays in every browser
# Spotify serves a JS-only web player (no track meta tags) to full browser UAs
UA = "Mozilla/5.0 (compatible; mixpipe-chords)"
# words that mark a different recording than the studio track
_VERSION_WORDS = ("live", "cover", "karaoke", "remix", "instrumental", "acoustic",
                  "sped up", "slowed", "8d", "reverb", "nightcore", "lyrics video")

# yt-dlp's extractor keys, spelled the way the sites spell themselves
SITE_NAMES = {"Youtube": "YouTube", "YoutubeTab": "YouTube", "Soundcloud": "SoundCloud",
              "BandcampTrack": "Bandcamp", "Bandcamp": "Bandcamp"}

YDL_HINT = "downloading from links needs yt-dlp: pip install -r requirements-chords.txt"


class FetchError(ValueError):
    """A link we can't use; the message is shown to the user as is."""


class _Silent:
    """yt-dlp prints errors to stderr even when quiet; we report them ourselves."""

    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def _ydl(**opts):
    try:
        import yt_dlp
    except ImportError as exc:
        raise FetchError(YDL_HINT) from exc
    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noprogress": True,
                             "noplaylist": True, "extract_flat": "in_playlist",
                             "logger": _Silent(), **opts})


def _clean_error(exc: Exception) -> str:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
    msg = re.sub(r"^ERROR:\s*", "", msg)
    msg = re.sub(r"^\[[\w:]+\]\s*[\w-]+:\s*", "", msg)  # "[youtube] abc123: "
    if msg.startswith("Unsupported URL"):
        return "couldn't find any audio at that link"
    if "DRM" in msg:
        return "that site's audio is DRM protected and can't be downloaded"
    if "Failed to resolve" in msg or "Unable to download webpage" in msg:
        host = re.search(r"host='([^']+)'", msg)
        return f"couldn't reach {host.group(1) if host else 'that site'}"
    return msg.split("\n")[0][:300]


def check_url(url: str) -> str:
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise FetchError("paste a full link starting with http:// or https://")
    return url


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _check_length(duration: float | None, what: str) -> None:
    if duration and duration > MAX_SECONDS:
        raise FetchError(f"{what} is {_fmt_time(duration)} long; the chord finder takes "
                         f"songs up to {MAX_SECONDS // 60} minutes")


def song_key(source: dict) -> str:
    """Stable id for a link, so pasting the same song again finds the same entry."""
    return hashlib.sha1(source["page_url"].encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Spotify
# --------------------------------------------------------------------------- #

_SPOTIFY_HOSTS = ("open.spotify.com", "spotify.link", "spotify.app.link", "play.spotify.com")


def is_spotify(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in _SPOTIFY_HOSTS)


def _http_get(url: str) -> tuple[str, str]:
    """(final url after redirects, body)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.geturl(), resp.read(2_000_000).decode("utf-8", errors="replace")


def _meta(page: str, name: str) -> str | None:
    for tag in re.findall(r"<meta\b[^>]*>", page):
        if re.search(rf'(?:property|name)="{re.escape(name)}"', tag):
            m = re.search(r'content="([^"]*)"', tag)
            if m:
                return html.unescape(m.group(1)).strip()
    return None


def parse_spotify_page(page: str) -> dict:
    """{title, artist, duration} from a Spotify track page's meta tags."""
    title = _meta(page, "og:title")
    artist = _meta(page, "music:musician_description")
    if not artist:  # "Rick Astley · Whenever You Need Somebody · Song · 1987"
        artist = (_meta(page, "og:description") or "").split(" · ")[0] or None
    duration = _meta(page, "music:duration")
    if not title:
        raise FetchError("couldn't read the track name from that Spotify page")
    return {"title": title, "artist": artist,
            "duration": float(duration) if duration and duration.isdigit() else None}


def pick_match(candidates: list[dict], track: dict) -> dict | None:
    """The search result most likely to be the same recording as the Spotify track:
    closest length, and no 'live'/'cover'/... unless the track title has it too."""
    wanted = track.get("duration")
    own = track["title"].lower()

    def score(c: dict) -> float:
        s = 0.0
        if wanted and c.get("duration"):
            s += abs(c["duration"] - wanted)
        elif wanted:
            s += 60
        name = (c.get("title") or "").lower()
        s += sum(30 for w in _VERSION_WORDS if w in name and w not in own)
        return s

    usable = [c for c in candidates if c.get("url") or c.get("id")]
    return min(usable, key=score) if usable else None


def _spotify_track(url: str) -> tuple[str, dict]:
    """(clean track url, {title, artist, duration})."""
    try:
        if urlparse(url).netloc.lower() != "open.spotify.com":  # short link: follow it
            url, _ = _http_get(url)
        # "?si=..." links also get the JS-only player, so drop the query
        path = re.sub(r"^/intl-[\w-]+", "", urlparse(url).path)
        clean = f"https://open.spotify.com{path}"
        if "/track/" not in path:
            kind = next((k for k in ("album", "playlist", "artist", "episode", "show")
                         if f"/{k}/" in path), "page")
            raise FetchError(f"that's a Spotify {kind}; paste a link to a single track")
        _, page = _http_get(clean)
    except FetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"couldn't open that Spotify link: {exc}") from exc
    try:
        return clean, parse_spotify_page(page)
    except FetchError:
        pass
    # page layout changed? oEmbed still gives the title (no artist or length)
    try:
        _, body = _http_get(f"https://open.spotify.com/oembed?url={clean}")
        title = json.loads(body).get("title")
    except Exception:  # noqa: BLE001
        title = None
    if not title:
        raise FetchError("couldn't read the track name from that Spotify page")
    return clean, {"title": title, "artist": None, "duration": None}


def _resolve_spotify(url: str) -> dict:
    clean, track = _spotify_track(url)
    _check_length(track["duration"], "that track")
    query = f"{track['artist']} - {track['title']}" if track["artist"] else track["title"]
    try:
        with _ydl() as ydl:
            found = ydl.extract_info(f"ytsearch6:{query}", download=False)
    except FetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"YouTube search failed: {_clean_error(exc)}") from exc
    match = pick_match(list(found.get("entries") or []), track)
    if match is None:
        raise FetchError(f"couldn't find \"{query}\" on YouTube")
    yt_url = match.get("url") or f"https://www.youtube.com/watch?v={match['id']}"
    note = f"Spotify doesn't allow downloads, so this uses the YouTube upload " \
           f"\"{match.get('title')}\""
    if track["duration"] and match.get("duration"):
        note += f" ({_fmt_time(match['duration'])}; Spotify says " \
                f"{_fmt_time(track['duration'])})"
    return {
        "url": url, "page_url": clean, "download_url": yt_url, "site": "Spotify",
        "title": f"{track['artist']} - {track['title']}" if track["artist"] else track["title"],
        "duration": track["duration"], "note": note + ".",
    }


# --------------------------------------------------------------------------- #
# everything else
# --------------------------------------------------------------------------- #


def resolve(url: str) -> dict:
    """Look up a link without downloading it. Raises FetchError with a message
    fit for the user if it can't be used."""
    url = check_url(url)
    if is_spotify(url):
        return _resolve_spotify(url)
    try:
        with _ydl() as ydl:
            info = ydl.extract_info(url, download=False)
    except FetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FetchError(_clean_error(exc)) from exc
    if info.get("_type") == "playlist":
        raise FetchError("that's a playlist or album; paste a link to a single song")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise FetchError("that's a live stream; paste a link to a recording")
    _check_length(info.get("duration"), "that recording")
    title = info.get("title") or info.get("id") or "song"
    if info.get("track") and (info.get("artist") or info.get("creator")):
        title = f"{info.get('artist') or info.get('creator')} - {info['track']}"
    page = info.get("webpage_url") or url
    site = info.get("extractor_key") or "Generic"
    if site == "Generic":  # a direct file or an embed: name the host instead
        site = urlparse(page).netloc.removeprefix("www.") or "link"
    return {
        "url": url, "page_url": page, "download_url": page,
        "site": SITE_NAMES.get(site, site), "title": title,
        "duration": info.get("duration"), "note": None,
    }


def download(source: dict, dest: Path,
             progress: Callable[[float, str], None] | None = None) -> Path:
    """Download the source's audio into dest/ and return the file."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda frac, msg: None)

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            if total:
                report(min(done / total, 0.99), f"downloading ({done / 1e6:.1f} of "
                                                f"{total / 1e6:.1f} MB)")
            else:
                report(0.5, f"downloading ({done / 1e6:.1f} MB)")

    try:
        with _ydl(format=AUDIO_FORMAT, outtmpl=str(dest / "audio.%(ext)s"),
                  max_filesize=MAX_BYTES, progress_hooks=[hook],
                  extract_flat=False) as ydl:
            info = ydl.extract_info(source["download_url"], download=True)
    except FetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"download failed: {_clean_error(exc)}") from exc
    files = [Path(d["filepath"]) for d in info.get("requested_downloads") or []
             if d.get("filepath")]
    if not files or not files[0].exists():
        raise FetchError("download finished but no audio file was saved "
                         "(it may be over the 500 MB limit)")
    return files[0]
