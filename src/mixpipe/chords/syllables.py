"""Split lyric words into syllables, and time the syllables inside a word.

Whisper gives a start and end for each *word*. A chord change on the second
syllable of a long word ("a-MA-zing") therefore used to show up on the whole
word. This module finds where the syllables are, so the chord can sit over the
right one.

  syllables(word, language)         -> ["a", "ma", "zing"]
  syllable_spans(word, ...)         -> [{"text", "start", "end", "c"}, ...]

Text splitting is rule-based, not a dictionary:

  - Latin script: vowel groups (a e i o u y, accented ones too) make the
    syllables; the consonants between two groups are shared out by whether they
    can start a word ("be-gin", "win-ter", "a-stray", "ta-ble"); English also
    drops a silent final "e" and "-ed", so "love" and "walked" are one syllable.
  - Chinese, Japanese and Korean: one syllable per character (small kana and the
    long-vowel mark stay with the kana before them).
  - Persian, Arabic and other scripts whose short vowels aren't written: no
    split is attempted (an empty result), because guessing would put chords in
    wrong places.

Timing: syllables share the word's duration in proportion to their length
(vowel groups weigh most). Given the vocal's loudness envelope, each boundary is
then moved to the deepest dip nearby, which is where one sung syllable ends and
the next begins.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

_VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœ")
_CODA_ONLY = {"ck"}              # stays with the syllable before it (pock-et)
_ONSETS = {  # clusters that can start a syllable (besides single consonants)
    "ch", "sh", "th", "ph", "wh", "qu", "wr",
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl", "sm",
    "sn", "sp", "st", "sw", "tr", "tw", "str", "spl", "spr", "scr", "thr", "shr", "chr", "sch",
    "pf", "tz", "kr", "kl",
}
_CJK = re.compile(r"[぀-ヿㇰ-ㇿ㐀-䶿一-鿿가-힯豈-﫿]")
_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮー")
_NO_SPLIT = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿֐-׿]")   # Arabic / Persian / Hebrew


def _is_vowel(ch: str, prev: str, nxt: str) -> bool:
    c = ch.lower()
    if c == "y":                      # "yes" and "you": consonant; "my" and "baby": vowel
        return bool(prev) and prev.isalpha()
    return c in _VOWELS


def _cluster_split(cluster: str) -> int:
    """Index inside the consonants between two vowel groups where the next
    syllable starts: doubled consonants split between ("lit-tle"), "ck" stays
    behind ("pock-et"), otherwise the longest tail that can begin a word goes
    forward (maximal onset: "ta-ble", "mo-ther", "mon-ster", "win-ter")."""
    n = len(cluster)
    if n <= 1:
        return 0
    if cluster in _CODA_ONLY:
        return n
    if cluster[0] == cluster[1]:
        return 1
    for take in range(min(n, 3), 0, -1):
        if take == 1 or cluster[n - take:] in _ONSETS:
            return n - take
    return n - 1


def _latin_breaks(word: str, language: str | None) -> list[int]:
    """Character indices (in `word`) where a new syllable starts."""
    letters = [(i, ch) for i, ch in enumerate(word) if ch.isalpha()]
    if len(letters) < 2:
        return []
    text = "".join(ch for _, ch in letters)
    low = text.lower()
    n = len(low)
    english = (language or "en").split("-")[0].lower() == "en"

    flags = []
    for k, ch in enumerate(low):
        flags.append(_is_vowel(ch, low[k - 1] if k else "", low[k + 1] if k + 1 < n else ""))
    # vowel groups = nuclei
    nuclei: list[list[int]] = []
    k = 0
    while k < n:
        if flags[k]:
            j = k
            while j + 1 < n and flags[j + 1]:
                j += 1
            nuclei.append([k, j])
            k = j + 1
        else:
            k += 1
    if not nuclei:
        return []

    if english and len(nuclei) > 1:
        last = nuclei[-1]
        # silent final e: "love", "time" ("table"/"little" keep theirs: a consonant + le)
        if last[0] == last[1] == n - 1 and low[-1] == "e" and n > 2 and not flags[n - 2] \
                and not (low[-2] == "l" and n > 3 and not flags[n - 3]):
            nuclei.pop()
        # "-ed" is silent unless it follows t or d; "-es" likewise after a sibilant
        elif low.endswith("ed") and last == [n - 2, n - 2] and n > 3 and low[n - 3] not in "td":
            nuclei.pop()
        elif low.endswith("es") and last == [n - 2, n - 2] and n > 3 and low[n - 3] not in "sxzc":
            if not (low[n - 3] in "gh" and n > 3):
                nuclei.pop()
    if english and len(nuclei) > 2:
        # silent "e" before a suffix: lone-ly, care-less, hope-ful, move-ment
        for k in range(1, len(nuclei) - 1):
            a, b = nuclei[k]
            if (a == b and low[a] == "e" and a >= 2 and not flags[a - 1] and flags[a - 2]
                    and low[a + 1:].startswith(("ly", "ness", "ful", "less", "ment", "some", "ty"))):
                nuclei.pop(k)
                break
    if len(nuclei) < 2:
        return []

    cuts = []
    for a, b in zip(nuclei[:-1], nuclei[1:]):
        gap = low[a[1] + 1: b[0]]
        cut_at = a[1] + 1 + _cluster_split(gap) if gap else a[1] + 1
        cuts.append(cut_at)
    return [letters[c][0] for c in cuts if c < len(letters)]


def syllable_breaks(word: str, language: str | None = None) -> list[int]:
    """Character indices where each syllable after the first begins.
    [] for one syllable, or when the script can't be split reliably."""
    if not word or _NO_SPLIT.search(word):
        return []
    if _CJK.search(word):
        breaks = []
        for i, ch in enumerate(word):
            if i and _CJK.match(ch) and ch not in _SMALL_KANA:
                breaks.append(i)
        return breaks
    if not any(ch.isalpha() for ch in word):
        return []
    norm = unicodedata.normalize("NFC", word)
    if len(norm) != len(word):
        word = norm
    return _latin_breaks(word, language)


def syllables(word: str, language: str | None = None) -> list[str]:
    cuts = [0] + syllable_breaks(word, language) + [len(word)]
    return [word[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if word[a:b]]


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #

ENV_HOP = 0.01          # seconds per envelope value
ENV_DYNAMIC_DB = 40.0   # the stored envelope spans this many dB below the loud level


def loudness_envelope(vocal: np.ndarray, sr: int) -> list[int]:
    """The vocal's loudness over time as 0-255 values, one per ENV_HOP.

    20 ms RMS in dB, relative to the loud (95th percentile) level and mapped
    linearly from -ENV_DYNAMIC_DB..0 dB onto 0..255; silence is 0."""
    mono = vocal if vocal.ndim == 1 else vocal.mean(axis=1)
    hop, win = max(1, int(ENV_HOP * sr)), max(2, int(0.02 * sr))
    n = max(0, (len(mono) - win) // hop + 1)
    if n == 0:
        return []
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    db = 20.0 * np.log10(np.sqrt((mono[idx].astype(np.float64) ** 2).mean(axis=1)) + 1e-9)
    loud = float(np.percentile(db, 95))
    scaled = np.clip((db - (loud - ENV_DYNAMIC_DB)) / ENV_DYNAMIC_DB, 0.0, 1.0) * 255.0
    return [int(round(v)) for v in scaled]


def _weights(parts: list[str]) -> np.ndarray:
    """Relative lengths of syllables: a vowel nucleus counts most, consonants add a little."""
    out = []
    for p in parts:
        vowels = sum(1 for k, ch in enumerate(p.lower()) if _is_vowel(ch, p[k - 1] if k else "", ""))
        out.append(1.0 + 0.25 * max(0, len(p) - max(vowels, 1)) + 0.15 * max(0, vowels - 1))
    return np.array(out, dtype=float)


def _refine(bounds: np.ndarray, env: np.ndarray, hop: float, t0: float, t1: float) -> np.ndarray:
    """Move each interior boundary to the deepest dip of the envelope within
    +/- 35 % of the neighbouring syllable lengths; leave it if the envelope is flat."""
    out = bounds.copy()
    smooth = np.convolve(env, np.ones(3) / 3.0, mode="same")
    for i in range(1, len(bounds) - 1):
        left, right = out[i - 1], bounds[i + 1]
        reach = 0.35 * min(bounds[i] - left, right - bounds[i])
        a = max(left + 0.04, bounds[i] - reach)
        b = min(right - 0.04, bounds[i] + reach)
        ia, ib = int(round(a / hop)), int(round(b / hop))
        if ib <= ia or ia < 0 or ib >= len(smooth):
            continue
        window = smooth[ia: ib + 1]
        around = smooth[max(0, ia - int(0.1 / hop)): min(len(smooth), ib + int(0.1 / hop) + 1)]
        peak = float(around.max()) if around.size else 0.0
        if peak <= 1e-6:
            continue
        k = int(np.argmin(window))
        if (peak - window[k]) / peak >= 0.12:           # a real dip, not a held vowel
            out[i] = (ia + k) * hop
    return out


def syllable_spans(word: dict, language: str | None = None,
                   envelope: list[int] | None = None) -> list[dict]:
    """[{"text", "start", "end", "c"}] for the syllables of a timed word
    {"text", "start", "end"}; "c" is the character offset of the syllable in the
    word. [] when the word is a single syllable or can't be split."""
    text = word["text"]
    cuts = syllable_breaks(text, language)
    if not cuts:
        return []
    starts = [0] + cuts
    parts = [text[a:b] for a, b in zip(starts, cuts + [len(text)])]
    t0, t1 = float(word["start"]), float(word["end"])
    if t1 <= t0:
        return []
    w = _weights(parts)
    bounds = t0 + (t1 - t0) * np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
    if envelope:
        env = np.asarray(envelope, dtype=float)
        bounds = _refine(bounds, env, ENV_HOP, t0, t1)
    return [{"text": p, "start": round(float(bounds[i]), 3), "end": round(float(bounds[i + 1]), 3),
             "c": starts[i]} for i, p in enumerate(parts)]


def make_syllabifier(language: str | None = None, envelope_obj: dict | None = None):
    """word -> syllable_spans(word), bound to a song's language and (optional)
    stored loudness envelope {"hop", "v": [...]}; what `align.build_rows` takes."""
    env = envelope_obj.get("v") if isinstance(envelope_obj, dict) else None
    if env and abs(float(envelope_obj.get("hop", ENV_HOP)) - ENV_HOP) > 1e-9:
        env = None                                   # stored at another resolution: ignore

    def fn(word: dict) -> list[dict]:
        if env and word["end"] > word["start"]:
            lo = max(0, int(word["start"] / ENV_HOP) - 3)
            hi = int(word["end"] / ENV_HOP) + 4
            local = env[lo:hi]
            shifted = {**word, "start": word["start"] - lo * ENV_HOP, "end": word["end"] - lo * ENV_HOP}
            spans = syllable_spans(shifted, language, local)
            return [{**sp, "start": round(sp["start"] + lo * ENV_HOP, 3),
                     "end": round(sp["end"] + lo * ENV_HOP, 3)} for sp in spans]
        return syllable_spans(word, language)

    return fn
