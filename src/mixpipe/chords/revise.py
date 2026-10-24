"""Semantic revision of transcribed lyrics with an LLM (OpenRouter).

Speech recognition on singing hears words one stretch at a time, so a line can
come out as sound-alike nonsense. This pass hands the model the *whole* lyric
(plus the song's title and language when known) so it can fix misheard words
from the song's context: what the song is about, its rhymes, and choruses that
were heard differently each time they came round.

Only the text goes to the model, never the audio. The revised text is put back
on the timed words with align.reanchor, so words that survive keep their times.

Used by the analysis pipeline (the "Revise lyrics" option on a new song) and by
the lyric editor's "Revise with LLM" button; both go through revise_text.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable

from .align import lines_to_text, reanchor, word_diff

PROMPT = """\
You revise a machine transcript of sung lyrics. Speech recognition on singing \
often mishears words, so some lines come out as sound-alike phrases that make \
no sense in the song.

Read the whole lyric first and work out the song: its language, what it is \
about, its rhymes and its repeated sections. Then fix:
- words and phrases that make no sense in context: replace them with what was \
most likely sung, preferring words that sound like what was heard;
- repeated sections (choruses, refrains) that were heard differently each \
time: make them match the clearest version;
- spelling, script and obvious punctuation for the language.

If you recognise the song, use what you know of it to fix misheard words, but \
never add or swap in lines that the transcript does not contain.

Keep everything that already makes sense exactly as it is. Keep the original \
language and never translate. Keep exactly the same lines in the same order, \
one lyric line per row, and keep blank rows (verse breaks) where they are. Do \
not merge, split, add or remove lines. No titles, section labels, notes or \
commentary. Return only the lyrics."""

MIN_SIMILARITY = 0.35   # an unreviewed revision sharing fewer words than this is a rewrite: drop it
BASE_TOKENS = 6000      # reply budget: room for a reasoning model's thinking...
TOKENS_PER_CHAR = 2     # ...plus the lyric itself (non-Latin scripts take ~1 token a character)
MAX_TOKENS = 32000      # a reply cut off is retried once with twice the budget, up to this

_FENCE = re.compile(r"^```[\w-]*\s*\n(.*?)\n?```\s*$", re.S)


def _clean(reply: str) -> str:
    """The lyrics in a reply: code fences and surrounding blank space removed."""
    text = (reply or "").strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    return text.strip("`").strip()


def similarity(old: str, new: str) -> float:
    """Share of words the two texts have in common, in order (0..1)."""
    a, b = old.lower().split(), new.lower().split()
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def changed_words(old: str, new: str) -> int:
    """Words removed or replaced by the revision."""
    return sum(len(text.split()) for op, text in word_diff(old, new) if op == "delete")


def revise_text(text: str, model: str, title: str | None = None, language: str | None = None,
                chat: Callable | None = None) -> str:
    """The lyric `text` (one line per row) revised by `model`. `chat(messages,
    model=, max_tokens=, reasoning_effort=)` defaults to openrouter.chat.

    Reasoning models are asked to think briefly; a reply that still runs out of
    tokens is retried once with twice the budget. Raises on API errors, on an
    empty reply, and (openrouter.CutOff) when even the retry is cut off."""
    from ..openrouter import CutOff

    if chat is None:
        from .. import openrouter
        chat = openrouter.chat
    context = []
    if title:
        context.append(f"Song: {title}")
    if language and language not in ("auto", "none"):
        context.append(f"Language: {language}")
    user = ("\n".join(context) + "\n\nTranscript:\n" if context else "") + text
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": user}]
    budget = min(MAX_TOKENS, BASE_TOKENS + TOKENS_PER_CHAR * len(text))
    try:
        reply = chat(messages, model=model, max_tokens=budget, reasoning_effort="low")
    except CutOff:
        retry = min(MAX_TOKENS, 2 * budget)
        try:
            reply = chat(messages, model=model, max_tokens=retry, reasoning_effort="low")
        except CutOff as exc:
            raise CutOff(f"{model} used up {retry} tokens without finishing the lyrics. It is "
                         "probably a reasoning model that thinks at length: pick one that "
                         "doesn't (e.g. a Gemini Flash, Llama or Mistral model)") from exc
    revised = _clean(reply)
    if not revised:
        raise RuntimeError(f"{model} returned no lyrics")
    return revised


def revise_lines(lines: list[dict], model: str, title: str | None = None,
                 language: str | None = None, chat: Callable | None = None,
                 cached: dict | None = None) -> tuple[list[dict], dict]:
    """Revise timed lyric lines without a human looking first (the pipeline).

    Returns (lines, info). info = {"model", "source", "text", "changed",
    "applied", "reason"}; a revision is not applied when it shares less than
    MIN_SIMILARITY of its words with the transcript (the model rewrote the song
    rather than correcting it). `cached` is an earlier info for the same model
    and source text, reused instead of asking again. Raises on API errors.
    """
    source = lines_to_text(lines)
    if cached and cached.get("model") == model and cached.get("source") == source:
        new = cached["text"]
    else:
        new = revise_text(source, model, title, language, chat)
    info = {"model": model, "source": source, "text": new,
            "changed": changed_words(source, new), "applied": False, "reason": None}
    if similarity(source, new) < MIN_SIMILARITY:
        info["reason"] = "the model rewrote the lyrics instead of correcting them"
        return lines, info
    if info["changed"] == 0 and new.split() == source.split():
        info["reason"] = "no changes needed"
        return lines, info
    info["applied"] = True
    return reanchor(lines, new), info
