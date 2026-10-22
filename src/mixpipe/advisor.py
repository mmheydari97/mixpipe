"""Advisors: turn analysis numbers into a mix plan.

Two implementations, same output schema:

  rules_advisor  - deterministic, no network, no API key. Start here.
  llm_advisor    - sends the analysis JSON (numbers only, never audio) to Claude
                   or an OpenRouter model and asks for a mix plan back.

The LLM never sees samples. It sees measurements. That's the whole trick: an LLM
is bad at DSP and decent at "the harmonica is 6 dB louder than everything else
and its energy is concentrated at 2 kHz where the lead guitar also sits, so pull
it down and notch it".
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Callable

from .analysis import BANDS, compare_balance
from .config import DEFAULT_CONFIG, treatment_for
from .fx import SYNC, tone_bands
from .io_utils import Stem
from .plan import SEND_OFF_DB, validate_plan, validate_send_buses

SYSTEM_PROMPT = """\
You are a mixing engineer. You will receive JSON measurements for each stem of a \
multitrack recording, plus optional notes from the musician.

Return ONLY a JSON object, no markdown fences, no prose before or after, matching \
this schema exactly:

{
  "notes": "one or two sentences on your overall approach",
  "stems": {
    "<exact stem name from the input>": {
      "gain_db": number,
      "hpf_hz": number,
      "lpf_hz": number,
      "eq": [{"freq": number, "gain_db": number, "q": number, "type": "peaking"}],
      "comp": {"threshold_db": number, "ratio": number, "attack_ms": number,
               "release_ms": number, "makeup_db": number},
      "pan": number,
      "sends": {"reverb_db": number, "delay_db": number},
      "deess": {"freq_hz": number, "max_reduction_db": number},
      "transient": {"attack": number, "sustain": number},
      "reason": "short justification tied to the measurements"
    }
  },
  "master": {
    "target_lufs": number,
    "ceiling_db": number,
    "master_eq": [{"freq": number, "gain_db": number, "q": number}],
    "glue_comp": {"threshold_db": number, "ratio": number, "attack_ms": number,
                  "release_ms": number, "makeup_db": number}
  }
}

Rules:
- Use the exact stem names from the input as keys.
- pan is -1 (hard left) to +1 (hard right). Keep bass and lead vocal near 0.
- Prefer cuts over boosts. Max +/-6 dB per EQ band, at most 4 bands per stem.
- band_energy_db is normalized so the loudest band of that stem reads 0.
- If two stems have high energy in the same band, carve one and leave the other.
- lufs tells you relative loudness. Aim for stems within about 6 dB of each other
  before the bus, with the melodic lead 2-3 dB above the accompaniment.
- Low crest_db (under 8) means the take is already squashed: use a gentle ratio.
- Low snr_db (under 20) means a noisy take: high-pass more aggressively.
- "silent": true (lufs below -60 or -Infinity) means the stem holds no music, only \
bleed or noise: leave its gain at 0 or below, never boost it.
- sends are levels in dB (-60 = off) into one shared reverb and one shared delay, \
so the dry close-miked stems sit in the same space: roughly -14 to -22 dB for a \
vocal or lead, none for bass. deess only matters for vocals (0 = off, up to ~8 dB). \
transient attack/sustain are -1..+1 (0 = leave alone).
- Never write a number outside the plausible range for its parameter.
"""

CRITIQUE_PROMPT = """\
You are a mixing and mastering engineer reviewing a finished mix. You will receive \
JSON measurements of the OUTPUT (loudness, true peak, crest factor, spectral \
balance, stereo picture), the mix plan that produced it, the measurements of each \
input stem, optionally measurements of a REFERENCE track the musician wants to \
sound like (plus balance_error: output minus reference per band, in dB, level \
removed), and optional notes from the musician.

Judge the result against the reference and the notes, say what is wrong in plain \
terms, and return a REVISED plan that fixes it. Return ONLY a JSON object, no \
markdown fences, no prose around it:

{
  "critique": "two to five sentences: what is off, in terms of the measurements",
  "plan": { ...the complete revised plan, exactly the same schema as the plan you were given... }
}

Rules:
- Change as little as possible: only parameters that address a problem you named. \
Copy everything else through unchanged.
- balance_error > 0 means the output has too much in that band, < 0 too little. \
Fix tonal balance with master_eq first (max +/-3 dB per round), then with stem EQ \
only if one stem clearly owns that band.
- true_peak_db must stay at or below the plan's ceiling_db; a crest_db drop of \
more than 6 dB from the stems means the limiter is working too hard: lower \
master target_lufs or the stem levels feeding the bus.
- lr_correlation below 0.2 or side_to_mid_db above -3 means the stereo image is \
too wide for mono playback: pull in pans and reverb sends.
- Never write a number outside the plausible range for its parameter.
"""


# --------------------------------------------------------------------------- #
# rule-based
# --------------------------------------------------------------------------- #


def rules_advisor(
    stems: list[Stem], analysis: dict, cfg: dict | None = None, notes: str = ""
) -> dict:
    """Deterministic plan: role defaults, adjusted by the measurements.

    Adjustments applied:
      - level match every stem toward its role's target_lufs
      - extra high-pass on noisy stems
      - softer ratio on already-squashed takes
      - spread same-role stems across the stereo field instead of stacking them
      - de-boost stems that are the loudest in a contested band
    """
    cfg = cfg or DEFAULT_CONFIG
    by_name = {a["name"]: a for a in analysis["stems"]}
    plan_stems: dict[str, dict] = {}

    # Count roles so duplicates can be panned apart. Stereo stems already have
    # their own image (pan is only a balance control there), so only mono ones count.
    role_counts: dict[str, int] = {}
    for s in stems:
        if s.channels == 1:
            role_counts[s.role] = role_counts.get(s.role, 0) + 1
    role_seen: dict[str, int] = {}

    # the louder of several vocal takes is taken as the lead
    vocals = [s.name for s in stems if s.role == "vocal"]
    lead_vocal = (max(vocals, key=lambda n: by_name.get(n, {}).get("lufs", -99.0))
                  if len(vocals) > 1 else None)

    backing_seen = 0
    for stem in stems:
        t = treatment_for(stem.role, cfg)
        a = by_name.get(stem.name, {})
        reasons = []

        # 1. Level match to the role target (not for a stem with nothing in it).
        measured = a.get("lufs", -18.0)
        target = t.get("target_lufs", -18.0)
        gain = float(target - measured) if not a.get("silent") else 0.0
        gain = max(-18.0, min(12.0, gain))
        # a second vocal take is a harmony or double: it sits under the lead, not level with it
        if stem.role == "vocal" and stem.name != lead_vocal and lead_vocal is not None:
            gain = max(-18.0, gain - _BACKING_VOCAL_DB)
            reasons.append(f"backing vocal, {_BACKING_VOCAL_DB:.0f} dB under the lead")
        t["gain_db"] = round(t["gain_db"] + gain, 2)
        reasons.append(f"{measured:.1f} LUFS measured, {gain:+.1f} dB to hit {target}")

        # 2. Noisy take: push the high-pass up to clear rumble and handling noise.
        snr = a.get("snr_db", 40.0)
        # Only for instruments with nothing below ~120 Hz to lose: a bass or kick "noise
        # floor" is usually just its sustain, and cutting there removes the low end.
        if snr < 20.0 and stem.role in _HPF_RAISE_ROLES:
            t["hpf_hz"] = max(t["hpf_hz"], 120.0)
            reasons.append(f"SNR {snr:.0f} dB is low, high-pass raised")

        # 3. Already-squashed take: don't compress it further.
        crest = a.get("crest_db", 14.0)
        if crest < 8.0:
            t["comp"]["ratio"] = min(t["comp"]["ratio"], 1.8)
            t["comp"]["threshold_db"] = min(t["comp"]["threshold_db"] + 4.0, -6.0)
            reasons.append(f"crest {crest:.1f} dB, already dense, gentle ratio")
        elif crest > 20.0:
            t["comp"]["ratio"] = min(t["comp"]["ratio"] + 1.0, 6.0)
            reasons.append(f"crest {crest:.1f} dB is spiky, firmer ratio")

        # 4. Clipped take: leave headroom and warn.
        if a.get("clipping_ratio", 0.0) > 0.0:
            t["gain_db"] = min(t["gain_db"], -3.0)
            reasons.append("input is clipped, trimmed and flagged, re-record if you can")

        # 5. Brightness correction from the spectral centroid.
        centroid = a.get("spectral_centroid_hz", 0.0)
        if centroid and centroid < 700.0:
            t["eq"].append({"freq": 3000.0, "gain_db": 2.0, "q": 0.8, "type": "peaking"})
            reasons.append(f"centroid {centroid:.0f} Hz is dark, added presence")
        elif centroid > 4500.0:
            t["eq"].append({"freq": 400.0, "gain_db": 1.5, "q": 0.8, "type": "peaking"})
            reasons.append(f"centroid {centroid:.0f} Hz is thin, added body")

        # 6. Pan duplicates of the same role apart.
        idx = role_seen.get(stem.role, 0)
        role_seen[stem.role] = idx + 1
        if stem.channels == 1 and role_counts.get(stem.role, 0) > 1:
            spread = 0.6
            n = role_counts[stem.role]
            t["pan"] = round(-spread + 2 * spread * idx / max(1, n - 1), 2)
            reasons.append(f"one of {n} {stem.role} stems, panned to spread them")
            if stem.role == "vocal":     # the lead stays in the middle, the backings go either side
                if stem.name == lead_vocal:
                    t["pan"] = 0.0
                    reasons[-1] = "lead vocal kept in the centre"
                else:
                    t["pan"] = round(-0.5 if backing_seen % 2 == 0 else 0.5, 2)
                    backing_seen += 1
                    reasons[-1] = "backing vocal panned to the side"

        t["reason"] = "; ".join(reasons)
        plan_stems[stem.name] = t

    note_log = _apply_notes(plan_stems, stems, notes)

    plan = {
        "source": "rules",
        "notes": (
            "Rule-based plan from role defaults plus measured loudness, SNR, crest "
            "factor and spectral centroid."
            + (f" Musician notes: {notes}" if notes else "")
            + (f" Acted on: {'; '.join(note_log['applied'])}." if note_log["applied"] else "")
            + (f" Not understood: {'; '.join(note_log['ignored'])}." if note_log["ignored"] else "")
        ),
        "stems": plan_stems,
        "master": {
            "target_lufs": cfg["master"]["target_lufs"],
            "ceiling_db": cfg["master"]["ceiling_db"],
            "master_hpf_hz": cfg["master"]["master_hpf_hz"],
            "master_eq": cfg["master"]["master_eq"],
            "glue_comp": cfg["master"]["glue_comp"],
        },
    }
    note_log["finish"](plan, cfg)
    return guard_silent(validate_plan(plan, stems, cfg), analysis)


# --------------------------------------------------------------------------- #
# musician notes for the rules advisor
# --------------------------------------------------------------------------- #

# role -> phrases that mean it (longest phrases are matched first, so "lead guitar"
# is not mistaken for "lead vocal")
_NOTE_ROLES = {
    "vocal": ("lead vocal", "vocals", "vocal", "voice", "vox", "singer", "singing"),
    "harmonica": ("harmonica", "harp", "mouth organ"),
    "acoustic": ("acoustic guitar", "acoustic", "nylon"),
    "electric_lead": ("lead guitar", "guitar solo", "solo"),
    "electric": ("electric guitar", "electric", "rhythm guitar", "rhythm"),
    "bass": ("bass guitar", "bass"),
    "keys": ("keyboard", "keys", "piano", "synth", "pad", "seaboard"),
    "drums": ("drums", "drum", "percussion", "kick", "snare", "cymbals"),
}
_NOTE_UP = ("louder", "turn up", "bring up", "boost", "more", "forward", "up front", "up",
            "can't hear", "cannot hear", "can not hear", "too quiet", "too soft", "buried",
            "lost", "hidden", "needs to be heard", "push", "prominent", "present")
_NOTE_DOWN = ("quieter", "turn down", "pull back", "tuck", "softer", "less", "too loud",
              "too much", "overpowering", "drowning", "back a bit", "lower", "down",
              "sits too far", "dominates", "dominating", "stepping on")
_NOTE_STEP_DB = 3.0
# how big a level move is: "a little" half the step, "way too" twice it (the chat prompt
# asks the model for the same 1.5 / 3 / 6 dB)
_NOTE_SMALL = ("a little", "a bit", "slightly", "a touch", "a tad", "just a")
_NOTE_BIG = ("way too", "much too", "far too", "a lot", "much more", "much less", "way more", "way less")
_BACKING_VOCAL_DB = 4.0
_HPF_RAISE_ROLES = {"vocal", "acoustic", "electric", "electric_lead", "harmonica"}


def _has(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text) is not None


_DRIVE_WORDS = ("gritty", "grit", "dirty", "distorted", "distortion", "crunch", "crunchy",
                "overdrive", "overdriven", "saturate", "saturated", "saturation", "fuzz", "fuzzy")
_CHORUS_WORDS = ("chorus", "shimmer", "shimmery", "lush", "doubled")
_TONE = re.compile(r"\b(more|less|cut the|boost the|cut|boost)\s+(bass|low end|lows|treble|highs|top end|top)\b")
_BIG_ROOM = ("bigger room", "bigger space", "larger room", "bigger reverb", "hall", "cathedral")
_SMALL_ROOM = ("smaller room", "smaller space", "tighter room", "smaller reverb")
_NOTE_SYNC = re.compile(r"\b(dotted\s+|triplet\s+)?(1/4|1/8|1/16|quarter|eighth|sixteenth)(?:[- ]note)?"
                        r"(\s+dotted|\s+triplet)?\s+(?:echo|delay)")


def _effect_notes(clause: str, targets: list[str], plan_stems: dict, bus_edits: list,
                  tone: bool = False) -> str | None:
    """Drive, chorus, bass/treble, room size and echo words in one clause.
    Stem effects change `plan_stems` now; reverb/echo bus changes go to
    `bus_edits` for the finish step. `tone`: the clause's "more bass" / "less
    treble" is about the named stems' sound (not the bass stem's level).
    Returns what was done, or None."""
    small = any(_has(clause, w) for w in _NOTE_SMALL)
    big = any(_has(clause, w) for w in _NOTE_BIG + ("much", "lot"))
    scale = 0.5 if small else 2.0 if big else 1.0
    less = (any(_has(clause, w) for w in ("less", "cleaner", "clean", "no", "remove", "without", "too"))
            and not _has(clause, "too clean"))
    done = []
    if targets and any(_has(clause, w) for w in _DRIVE_WORDS + ("cleaner", "clean")):
        for n in targets:
            d = plan_stems[n].setdefault("drive", {})
            d["amount"] = round(max(0.0, min(1.0, d.get("amount", 0.0) + (-0.25 if less else 0.25) * scale)), 2)
        done.append(f"drive {'down' if less else 'up'} on {', '.join(targets)}")
    if targets and any(_has(clause, w) for w in _CHORUS_WORDS):
        for n in targets:
            c = plan_stems[n].setdefault("chorus", {})
            c["mix"] = round(max(0.0, min(1.0, c.get("mix", 0.0) + (-0.35 if less else 0.35) * scale)), 2)
        done.append(f"chorus {'down' if less else 'up'} on {', '.join(targets)}")
    m = _TONE.search(clause)
    if tone and targets and m:
        db = (-3.0 if m.group(1) in ("less", "cut", "cut the") else 3.0) * scale
        low = m.group(2) in ("bass", "low end", "lows")
        for n in targets:
            plan_stems[n]["eq"] = (list(plan_stems[n].get("eq", []))
                                   + tone_bands(db if low else 0.0, 0.0 if low else db))[-6:]
        done.append(f"{'bass' if low else 'treble'} {db:+.1f} dB on {', '.join(targets)}")
    if any(_has(clause, w) for w in _BIG_ROOM + _SMALL_ROOM):
        sign = -1.0 if any(_has(clause, w) for w in _SMALL_ROOM) else 1.0
        bus_edits.append(("reverb", "room_size", sign * 0.2 * scale))
        bus_edits.append(("reverb", "decay_s", sign * 0.5 * scale))
        done.append(f"{'smaller' if sign < 0 else 'bigger'} room")
    sync = _NOTE_SYNC.search(clause)
    if sync:
        value = {"quarter": "1/4", "eighth": "1/8", "sixteenth": "1/16"}.get(sync.group(2), sync.group(2))
        value += {"dotted": "d", "triplet": "t"}.get((sync.group(1) or sync.group(3) or "").strip(), "")
        if value in SYNC:
            bus_edits.append(("delay", "sync", value))
            for n in targets:                    # echo needs a send to the delay bus
                sends = plan_stems[n].setdefault("sends", {})
                if sends.get("delay_db", SEND_OFF_DB) <= SEND_OFF_DB:
                    sends["delay_db"] = -18.0
            done.append(f"{value} note echo" + (f" on {', '.join(targets)}" if targets else ""))
    elif any(_has(clause, w) for w in ("longer echo", "longer delay", "shorter echo", "shorter delay")):
        sign = -1.0 if _has(clause, "shorter echo") or _has(clause, "shorter delay") else 1.0
        bus_edits.append(("delay", "delay_ms", sign * 100.0 * scale))
        done.append(f"{'shorter' if sign < 0 else 'longer'} echo")
    if any(_has(clause, w) for w in ("more repeats", "fewer repeats", "less repeats", "more feedback",
                                      "less feedback")):
        sign = 1.0 if _has(clause, "more repeats") or _has(clause, "more feedback") else -1.0
        bus_edits.append(("delay", "feedback", sign * 0.15 * scale))
        done.append(f"{'more' if sign > 0 else 'fewer'} echo repeats")
    return "; ".join(done) or None


_TEMPO_WORDS = ("slower", "faster", "slow down", "slow it down", "speed up", "speed it up",
                "tempo", "bpm", "half time", "half speed", "double time", "double speed",
                "original speed", "normal speed")
_KEY_WORDS = ("semitone", "semitones", "half step", "half-step", "whole step", "whole tone",
              "fret", "frets", "key", "transpose", "pitch")
_KEY_UP = ("up", "higher", "raise", "sharp")
_KEY_DOWN = ("down", "lower", "drop", "flat")


def _tempo_key(clause: str, tempo_bpm: float | None) -> tuple[str, str, float] | None:
    """A clause about the song's tempo or key, as (what, how, amount):
    ("speed", "set" | "add", percent) or ("key", "set" | "add", semitones).
    None when the clause is about something else; ("speed", "bpm", n) for a BPM
    target the song's tempo isn't known for."""
    percent = re.search(r"\d\s*(?:%|percent)", clause) is not None
    if any(_has(clause, w) for w in _TEMPO_WORDS) or percent and any(
            _has(clause, w) for w in ("play", "speed", "tempo")):
        if any(_has(clause, w) for w in ("original speed", "original tempo", "normal speed",
                                          "normal tempo", "back to 100")):
            return "speed", "set", 100.0
        if _has(clause, "half time") or _has(clause, "half speed"):
            return "speed", "set", 50.0
        if _has(clause, "double time") or _has(clause, "double speed"):
            return "speed", "set", 200.0
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*bpm", clause)
        if m:
            target = float(m.group(1))
            return ("speed", "set", target / tempo_bpm * 100.0) if tempo_bpm else ("speed", "bpm", target)
        m = re.search(r"(\d{1,3})\s*(?:%|percent)", clause)
        slower = any(_has(clause, w) for w in ("slower", "slow down", "slow it down"))
        sign = -1.0 if slower else 1.0
        if m and (slower or any(_has(clause, w) for w in ("faster", "speed up", "speed it up"))):
            return "speed", "add", sign * float(m.group(1))
        if m:
            return "speed", "set", float(m.group(1))
        step = 5.0 if any(_has(clause, w) for w in _NOTE_SMALL) else \
            20.0 if any(_has(clause, w) for w in _NOTE_BIG + ("much", "lot")) else 10.0
        return "speed", "add", sign * step
    if any(_has(clause, w) for w in _KEY_WORDS) or re.search(r"[+-]?\d{1,2}\s*st\b", clause):
        if _has(clause, "original key"):
            return "key", "set", 0.0
        sign = -1.0 if any(_has(clause, w) for w in _KEY_DOWN) else 1.0
        m = re.search(r"([+-]?\d{1,2})\s*(?:semitones?|half[- ]steps?|frets?|st\b)", clause)
        if m:
            n = float(m.group(1))
            return "key", "add", n if m.group(1)[0] in "+-" else sign * n
        if _has(clause, "whole step") or _has(clause, "whole tone") or _has(clause, "a tone") \
                or _has(clause, "two frets"):
            return "key", "add", sign * 2.0
        if any(_has(clause, w) for w in _KEY_UP + _KEY_DOWN):
            return "key", "add", sign
    return None


def _named_stems(text: str, plan_stems: dict, stems: list[Stem]) -> list[str]:
    """The stems a clause names: by their own name, or by an instrument word."""
    targets: list[str] = []
    rest = text
    for name in sorted(plan_stems, key=len, reverse=True):
        if name.lower() in rest:
            targets.append(name)
            rest = rest.replace(name.lower(), " ")
    for role, words in _NOTE_ROLES.items():
        for w in sorted(words, key=len, reverse=True):
            if _has(rest, w):
                targets += [n for n, st in plan_stems.items()
                            if st.get("role", None) == role or _stem_role(stems, n) == role]
                rest = re.sub(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", " ", rest)
                break
    return list(dict.fromkeys(targets))


def _apply_notes(plan_stems: dict, stems: list[Stem], notes: str,
                 tempo_bpm: float | None = None) -> dict:
    """Act on plain-English notes with keyword rules.

    Handles, clause by clause: a stem or instrument ("vocals", "the harmonica",
    a stem's own name) with a direction ("louder", "too quiet", "less", "up",
    "can't hear" ...) moves its fader by 3 dB (1.5 for "a little", 6 for "way too");
    tempo and key ("a little slower", "at 90 bpm", "up a semitone") are read
    first, so "speed up the drums" changes the tempo, not the drums; "more/less reverb|space|echo|wet|dry"
    moves its sends (or all sends when no instrument is named); and mood words for
    the whole mix: bright/harsh/sizzly, dull/dark/muffled, muddy/boomy, thin, loud,
    quiet; and effects (fx.py): drive ("gritty guitar"), chorus, bass/treble on
    a stem ("more bass on the guitar"), the room ("bigger room") and the echo
    ("1/8 note echo on the vocals", "more repeats"). Anything not understood is
    listed in the plan notes instead of being silently dropped. Returns
    {"applied": [...], "ignored": [...], "finish": callable(plan, cfg) that
    makes the master and send-bus changes}.
    """
    log = {"applied": [], "ignored": [], "finish": lambda plan, cfg=None: None}
    if not notes.strip():
        return log

    master_eq: list[dict] = []
    master_target = [0.0]
    tempo_key: list[tuple[str, str, float]] = []
    bus_edits: list[tuple[str, str, object]] = []

    for raw in re.split(r"[.;,!?\n]|\bbut\b|\band\b", notes.lower()):
        clause = raw.strip()
        if not clause:
            continue
        tk = _tempo_key(clause, tempo_bpm)
        if tk and tk[1] == "bpm":
            log["ignored"].append(f"{clause} (the song's tempo isn't known, so say it in %)")
            continue
        if tk:
            tempo_key.append(tk)
            what, how, amount = tk
            unit = "%" if what == "speed" else " st"
            change = f"by {amount:+.0f}{unit}" if how == "add" else f"to {amount:.0f}{unit}"
            log["applied"].append(f"'{clause}' -> {what} {change}")
            continue
        direction = 0
        if any(_has(clause, p) for p in _NOTE_DOWN) and not any(
                _has(clause, p) for p in ("not loud enough", "not enough")):
            direction = -1
        if any(_has(clause, p) for p in _NOTE_UP):
            direction = 1 if direction == 0 or any(
                _has(clause, p) for p in ("can't hear", "cannot hear", "too quiet", "buried")
            ) else direction

        # which stems does the clause name? "more bass on the guitar" is about the
        # guitar's low end, but "more bass" on its own is still the bass stem
        tone_of = _named_stems(_TONE.sub(" ", clause), plan_stems, stems) if _TONE.search(clause) else []
        targets = tone_of or _named_stems(clause, plan_stems, stems)

        effects = _effect_notes(clause, targets, plan_stems, bus_edits, tone=bool(tone_of))
        if effects:
            log["applied"].append(f"'{clause}' -> {effects}")
            continue

        tone = None
        if any(_has(clause, w) for w in ("bright", "harsh", "sizzly", "sibilant", "piercing", "shrill")):
            tone = ("high_shelf", 8000.0, -1.5)
        elif any(_has(clause, w) for w in ("dull", "dark", "muffled", "veiled", "lifeless")):
            tone = ("high_shelf", 8000.0, 1.5)
        elif any(_has(clause, w) for w in ("muddy", "boomy", "boxy", "cloudy", "wooly")):
            tone = ("peaking", 250.0, -1.5)
        elif any(_has(clause, w) for w in ("thin", "weak", "no body", "lacks body")):
            tone = ("low_shelf", 150.0, 1.5)
        elif any(_has(clause, w) for w in ("honky", "honk", "nasal", "nasally", "honking")):
            tone = ("peaking", 1200.0, -2.0)
        space_up = any(_has(clause, w) for w in ("reverb", "space", "echo", "wet", "room", "ambience"))
        dry = _has(clause, "dry") or _has(clause, "drier")
        done = False

        if (space_up or dry) and (direction != 0 or dry):
            delta = -6.0 if (dry or direction < 0) else 4.0
            for n in (targets or list(plan_stems)):
                for key in ("reverb_db", "delay_db"):
                    cur = plan_stems[n].setdefault("sends", {}).get(key, SEND_OFF_DB)
                    if cur > SEND_OFF_DB or delta > 0 and key == "reverb_db":
                        plan_stems[n]["sends"][key] = max(SEND_OFF_DB, (cur if cur > SEND_OFF_DB else -22.0) + delta)
                plan_stems[n]["reason"] = (plan_stems[n].get("reason", "") + f"; notes: '{clause}'").strip("; ")
            log["applied"].append(f"'{clause}' -> {'drier' if delta < 0 else 'more space'} for "
                                  f"{', '.join(targets) if targets else 'everything'}")
            done = True
        elif targets and direction != 0 and tone is None:   # "less honky" is EQ, not level
            step = _NOTE_STEP_DB * (0.5 if any(_has(clause, w) for w in _NOTE_SMALL)
                                    else 2.0 if any(_has(clause, w) for w in _NOTE_BIG) else 1.0)
            for n in targets:
                plan_stems[n]["fader_db"] = round(plan_stems[n].get("fader_db", 0.0) + direction * step, 2)
                plan_stems[n]["reason"] = (plan_stems[n].get("reason", "") + f"; notes: '{clause}'").strip("; ")
            log["applied"].append(f"'{clause}' -> {direction * step:+.1f} dB on {', '.join(targets)}")
            done = True

        if tone:
            band = {"type": tone[0], "freq": tone[1], "gain_db": tone[2], "q": 0.7}
            if targets:       # "the bass is boomy": fix that instrument, not the whole mix
                for n in targets:
                    plan_stems[n].setdefault("eq", []).append(dict(band))
                    plan_stems[n]["reason"] = (plan_stems[n].get("reason", "")
                                               + f"; notes: '{clause}'").strip("; ")
                where = f"{', '.join(targets)} {tone[0].replace('_', ' ')}"
            else:
                master_eq.append(band)
                where = f"master {tone[0].replace('_', ' ')}"
            log["applied"].append(f"'{clause}' -> {where} {tone[2]:+.1f} dB at {tone[1]:.0f} Hz")
            done = True
        if not targets and not done:
            if _has(clause, "loud") or _has(clause, "louder") or _has(clause, "hotter"):
                master_target[0] += 1.5
                log["applied"].append(f"'{clause}' -> master target +1.5 LUFS")
                done = True
            elif _has(clause, "quiet") or _has(clause, "quieter") or _has(clause, "softer overall"):
                master_target[0] -= 1.5
                log["applied"].append(f"'{clause}' -> master target -1.5 LUFS")
                done = True
        if not done:
            log["ignored"].append(clause)

    def finish(plan: dict, cfg: dict | None = None) -> None:
        master = plan.setdefault("master", {})
        if bus_edits:
            buses = plan["send_buses"] = validate_send_buses(plan.get("send_buses"), cfg or DEFAULT_CONFIG)
            for bus, key, change in bus_edits:
                if key == "sync":
                    buses[bus]["sync"] = change
                else:
                    buses[bus][key] = buses[bus].get(key, 0.0) + change
                    if key == "delay_ms":             # a set time replaces a note value
                        buses[bus]["sync"] = ""
        if master_eq:
            master["master_eq"] = list(master.get("master_eq", [])) + master_eq
        if master_target[0]:
            master["target_lufs"] = master.get("target_lufs", -14.0) + master_target[0]
        for what, how, amount in tempo_key:
            if what == "speed":
                pct = amount if how == "set" else float(master.get("speed", 1.0)) * 100 + amount
                master["speed"] = pct / 100.0
            else:
                master["semitones"] = amount if how == "set" else master.get("semitones", 0) + amount

    log["finish"] = finish
    return log


def guard_silent(plan: dict, analysis: dict | None) -> dict:
    """Keep an advisor from boosting stems that hold no music (`analysis.is_silent`).

    Level matching a near-empty stem to its role target asks for unlimited gain,
    which would only lift separation bleed and noise. Their trim and fader are
    capped at 0 dB; muting them stays the musician's call. Edits `plan` in place."""
    if not analysis:
        return plan
    for a in analysis.get("stems", []):
        entry = plan.get("stems", {}).get(a["name"])
        if entry is None or not a.get("silent"):
            continue
        if entry.get("gain_db", 0.0) > 0.0 or entry.get("fader_db", 0.0) > 0.0:
            entry["gain_db"] = min(entry.get("gain_db", 0.0), 0.0)
            entry["fader_db"] = min(entry.get("fader_db", 0.0), 0.0)
            note = "almost silent (bleed or noise only), not boosted"
            entry["reason"] = f"{entry.get('reason', '')}; {note}".strip("; ")
    return plan


def _stem_role(stems: list[Stem], name: str) -> str | None:
    for s in stems:
        if s.name == name:
            return s.role
    return None


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #


def resolve_llm(provider: str | None, model: str | None, cfg: dict) -> tuple[str, str]:
    """(provider, model) for an LLM call. An id with a slash ("google/gemini-2.5-flash")
    is an OpenRouter id, so it goes to OpenRouter even when the configured provider is
    Anthropic; no model means the configured one (`advisor.openrouter_model` on
    OpenRouter when the configured model is a bare Anthropic id)."""
    a = cfg["advisor"]
    if not provider:
        provider = "openrouter" if model and "/" in model else a["provider"]
    if not model:
        model = a["model"]
        if provider == "openrouter" and "/" not in model:
            model = a.get("openrouter_model", "anthropic/claude-haiku-5.5")
    return provider, model


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _parse_json(raw: str):
    """The JSON object in a model's reply: fences stripped, and when the model
    wrapped it in prose anyway, the outermost {...}."""
    text = _strip_fences(raw or "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise
        return json.loads(text[a:b + 1])


def _call_anthropic(payload: str, model: str, max_tokens: int = 4000) -> str:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install anthropic to use the LLM advisor") from exc

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": payload}],
    )
    return "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )


# reasoning models spend part of max_tokens thinking; the plans themselves are ~3k tokens
_LLM_MAX_TOKENS = 16000


def _call_openrouter(payload: str, model: str, max_tokens: int = _LLM_MAX_TOKENS) -> str:
    from . import openrouter

    return openrouter.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        model=model,
        max_tokens=max_tokens,
        reasoning_effort="low",
    )


def llm_advisor(
    stems: list[Stem],
    analysis: dict,
    cfg: dict | None = None,
    notes: str = "",
    provider: str | None = None,
    model: str | None = None,
    masking: list[dict] | None = None,
) -> dict:
    """Ask an LLM for a mix plan, then clamp whatever comes back.

    Falls back to the rule-based plan on any failure (no key, network error,
    unparseable response), because a broken advisor shouldn't mean no mix.
    """
    cfg = cfg or DEFAULT_CONFIG
    provider, model = resolve_llm(provider, model, cfg)

    payload = json.dumps(
        {
            "analysis": analysis,
            "masking_report": masking or [],
            "musician_notes": notes,
            "role_defaults": {
                s.name: treatment_for(s.role, cfg) for s in stems
            },
        },
        indent=2,
    )

    try:
        if provider == "openrouter":
            raw = _call_openrouter(payload, model)
        else:
            raw = _call_anthropic(payload, model)
        parsed = _parse_json(raw)
        parsed["source"] = f"llm:{provider}:{model}"
        return guard_silent(validate_plan(parsed, stems, cfg), analysis)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all fallback
        print(f"[advisor] LLM advisor failed ({type(exc).__name__}: {exc}).")
        print("[advisor] Falling back to the rule-based plan.")
        fallback = rules_advisor(stems, analysis, cfg, notes)
        fallback["source"] = "rules (llm fallback)"
        return fallback


def get_advisor(mode: str):
    """Dispatch by name, for the CLI."""
    if mode == "llm":
        return llm_advisor
    if mode == "rules":
        return rules_advisor
    if mode == "none":
        return lambda stems, analysis, cfg=None, notes="": validate_plan(
            {"source": "defaults", "notes": "Role defaults only."}, stems, cfg
        )
    raise ValueError(f"unknown advisor mode: {mode}")


# --------------------------------------------------------------------------- #
# critique: look at the result, revise the plan
# --------------------------------------------------------------------------- #

# centre frequency and filter type used to correct each analysis band
_BAND_FIX = {
    "sub": ("low_shelf", 50.0), "low": ("peaking", 120.0), "low_mid": ("peaking", 350.0),
    "mid": ("peaking", 1100.0), "high_mid": ("peaking", 3500.0), "high": ("peaking", 8000.0),
    "air": ("high_shelf", 12000.0),
}
_CRITIQUE_TOLERANCE_DB = 2.0     # a band within this of the reference is left alone
_CRITIQUE_STEP = 0.6             # fraction of the error corrected per round
_CRITIQUE_MAX_MOVE_DB = 3.0


def balance_error(output_analysis: dict, reference_analysis: dict | None) -> dict | None:
    """Per-band tonal error vs a reference, or None without one (`compare_balance`)."""
    if not reference_analysis:
        return None
    return compare_balance(output_analysis, reference_analysis)


def rules_critique(
    output_analysis: dict, plan: dict, reference_analysis: dict | None = None,
    master_report: dict | None = None,
) -> tuple[list[str], dict]:
    """Deterministic review: (findings, revised plan). Changes only what it names.

    Looks at: true peak against the ceiling (lowers the ceiling by the overshoot),
    limiter squash (crest lost over 6 dB lowers the loudness target 1.5 dB), phase
    trouble in mono (reported), and, with a reference, the tonal balance of every
    band: a band more than 2 dB off gets 60 % of the error corrected (at most 3 dB)
    by a master EQ band that replaces any earlier correction at that frequency, so
    repeated rounds converge instead of stacking.
    """
    revised = copy.deepcopy(plan)
    master = revised.setdefault("master", {})
    findings: list[str] = []

    over = output_analysis.get("true_peak_db", -99.0) - master.get("ceiling_db", -1.0)
    if over > 0.05:
        master["ceiling_db"] = round(master.get("ceiling_db", -1.0) - over - 0.1, 2)
        findings.append(f"true peak {output_analysis['true_peak_db']:+.2f} dBTP is over the "
                        f"{plan['master'].get('ceiling_db', -1.0):+.1f} dB ceiling: ceiling lowered")

    lost = (master_report or {}).get("crest_lost_db", 0.0)
    if lost > 6.0:
        master["target_lufs"] = round(master.get("target_lufs", -14.0) - 1.5, 2)
        findings.append(f"the limiter removed {lost:.1f} dB of crest: loudness target lowered 1.5 LUFS")

    corr = output_analysis.get("lr_correlation")
    if corr is not None and corr < 0.0:
        findings.append(f"left/right correlation is {corr:+.2f}: parts of the mix cancel in mono; "
                        "check reverb sends and panning")

    err = balance_error(output_analysis, reference_analysis)
    if err:
        eq = [b for b in master.get("master_eq", [])
              if not any(abs(b.get("freq", 0) / f - 1.0) < 0.12 for _, f in _BAND_FIX.values())]
        # earlier corrections at these frequencies are replaced, but remember them
        previous = {name: sum(b.get("gain_db", 0.0) for b in master.get("master_eq", [])
                              if abs(b.get("freq", 0) / f - 1.0) < 0.12)
                    for name, (_, f) in _BAND_FIX.items()}
        for band, dev in err.items():
            kind, freq = _BAND_FIX[band]
            if abs(dev) <= _CRITIQUE_TOLERANCE_DB:
                if previous[band]:
                    eq.append({"type": kind, "freq": freq, "gain_db": previous[band], "q": 0.8})
                continue
            move = -_CRITIQUE_STEP * dev
            gain = max(-_CRITIQUE_MAX_MOVE_DB, min(_CRITIQUE_MAX_MOVE_DB, previous[band] + move))
            eq.append({"type": kind, "freq": freq, "gain_db": round(gain, 2), "q": 0.8})
            findings.append(f"{band} is {dev:+.1f} dB {'above' if dev > 0 else 'below'} the "
                            f"reference: master EQ {gain:+.1f} dB at {freq:.0f} Hz")
        master["master_eq"] = eq
    return findings, revised


def _llm_text(system: str, payload: str, provider: str, model: str) -> str:
    if provider == "openrouter":
        from . import openrouter

        return openrouter.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": payload}],
            model=model, max_tokens=_LLM_MAX_TOKENS, reasoning_effort="low",
        )
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install anthropic to use the LLM critique") from exc
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    resp = anthropic.Anthropic(api_key=key).messages.create(
        model=model, max_tokens=_LLM_MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": payload}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def critique_mix(
    mix_analysis: dict,
    cfg: dict | None = None,
    plan: dict | None = None,
    stems: list[Stem] | None = None,
    reference_analysis: dict | None = None,
    stem_analysis: dict | None = None,
    master_report: dict | None = None,
    masking: list[dict] | None = None,
    notes: str = "",
    use_llm: bool = False,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Second pass: look at the *output* and suggest a revised plan.

    `mix_analysis` is `analysis.analyze_master` of the rendered mix. With a `plan`
    and the `stems` (needed to validate a plan), returns a revised plan; without,
    only the critique. `reference_analysis` (the same measurements of a track the
    musician wants to sound like) is what gives the critique something to
    compare against; without one only objective problems are caught.

    `use_llm=False` runs `rules_critique`. `use_llm=True` sends the output,
    reference, balance error, plan and stem measurements (numbers only) to the
    model with CRITIQUE_PROMPT, validates and clamps whatever plan comes back,
    and falls back to the rules critique on any failure.

    Returns {"critique": str, "plan": revised plan | None, "changed": bool,
    "findings": [...], "balance_error": {...} | None, "source": str}.
    """
    cfg = cfg or DEFAULT_CONFIG
    err = balance_error(mix_analysis, reference_analysis)
    result = {"critique": "", "plan": None, "changed": False, "findings": [],
              "balance_error": err, "source": "rules"}

    if use_llm and plan is not None and stems is not None:
        provider, model = resolve_llm(provider, model, cfg)
        payload = json.dumps(
            {"output": mix_analysis, "reference": reference_analysis, "balance_error": err,
             "plan": plan, "stem_analysis": stem_analysis, "masking_report": masking or [],
             "master_report": master_report, "musician_notes": notes},
            indent=2,
        )
        try:
            parsed = _parse_json(_llm_text(CRITIQUE_PROMPT, payload, provider, model))
            revised = guard_silent(validate_plan(parsed["plan"], stems, cfg), stem_analysis)
            revised["source"] = f"critique:llm:{provider}:{model}"
            result.update(critique=str(parsed.get("critique", ""))[:2000], plan=revised,
                          source=revised["source"])
            result["changed"] = _plan_core(revised) != _plan_core(plan)
            return result
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all fallback
            print(f"[critique] LLM critique failed ({type(exc).__name__}: {exc}); using rules.")

    findings, revised = rules_critique(mix_analysis, plan or {"master": {}}, reference_analysis,
                                       master_report)
    result["findings"] = findings
    result["critique"] = "; ".join(findings) if findings else "no problems found by the rules"
    if plan is not None:
        if stems is not None:
            revised = validate_plan(revised, stems, cfg)
        revised["source"] = "critique:rules"
        result["source"] = "critique:rules"
        result["plan"] = revised
        result["changed"] = _plan_core(revised) != _plan_core(plan)
    return result


def _plan_core(plan: dict) -> str:
    """The plan without its free-text fields, for 'did anything change' checks."""
    stripped = {k: v for k, v in plan.items() if k not in ("notes", "source")}
    stripped["stems"] = {n: {k: v for k, v in e.items() if k != "reason"}
                         for n, e in plan.get("stems", {}).items()}
    return json.dumps(stripped, sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# chat: revise a plan from one message
# --------------------------------------------------------------------------- #

REVISE_PROMPT = """\
You are the mixing engineer in a live session. The musician has heard the current \
mix and tells you, in their own words, what to change. You receive JSON: the \
message, the last few turns, the current mix plan (per stem, plus master), \
measurements of the rendered output, each stem's level in the mix relative to \
the whole (level_in_mix_db) and the stem measurements. You never hear audio.

Answer with ONLY a JSON object, no markdown fences, no prose around it:

{
  "reply": "one or two plain sentences to the musician: what you are changing and why",
  "adjustments": [
    {"target": "<stem name> | role:<role> | all | master", "param": "<param>", "delta": number}
    or {"target": ..., "param": ..., "value": ...}
  ]
}

Stem params (target = a stem name, role:vocal etc., or all):
  gain_db (delta, dB: the stem's fader, after its compressor), pan (value -1..+1 or delta), mute (value true/false),
  hpf_hz / lpf_hz (value, Hz, 0 = off),
  eq_add (value {"freq", "gain_db", "q", "type": peaking|low_shelf|high_shelf}), eq_clear (value true),
  comp_ratio (delta or value), comp_threshold_db (delta or value),
  deess_db (value 0..10: de-esser depth), reverb_db / delay_db (delta, dB; send level, -60 = off),
  transient_attack / transient_sustain (delta or value, -1..+1),
  drive (delta or value 0..1: saturation; 0 clean, 0.2 warm, 0.6+ distorted),
  chorus (delta or value 0..1: chorus amount, 0 = off),
  bass_db / treble_db (delta, dB: a low shelf at 120 Hz / a high shelf at 6 kHz on that stem)
Master params (target = master):
  target_lufs (delta or value), ceiling_db (value), eq_add, eq_clear, glue_ratio (delta or value),
  reverb_return_db / delay_return_db (delta), reverb_decay_s (delta or value, seconds),
  reverb_size (delta or value 0..1: the room's size), reverb_pre_delay_ms (delta or value),
  delay_ms (delta or value: echo time; frees it from the tempo), delay_sync (value \
"1/4" | "1/4d" | "1/4t" | "1/8" | "1/8d" | "1/8t" | "1/16", or "" for a free time: echo locked \
to the tempo), delay_feedback (delta or value 0..0.9: how many repeats). An echo is only \
heard on stems that send to it (delay_db).
  speed_pct (delta or value: the whole song's tempo in percent of the recording, 100 = as \
recorded, 90 = 10% slower; the key stays), semitones (delta or value: whole semitones the \
whole song moves, +2 = a whole step up; the tempo stays)

Rules:
- Change only what the message asks for. Do not "improve" anything else.
- Relative words map to modest steps: "a little" 1.5 dB, plain "too loud" 3 dB, "way too" 6 dB.
- "Too loud" for a stem is a gain cut on that stem, not a master change.
- Tempo: "a little slower/faster" 5%, plain 10%, "much" 20%. For "at N BPM" use \
speed_pct = N / tempo.bpm_as_recorded * 100 (when it is known). Key: "a semitone" / \
"half step" 1, "whole step" / "a tone" 2.
- Prefer one clear move over several small ones. Tone words: bright/harsh -> cut 3-8 kHz, \
muddy -> cut 200-400 Hz, thin -> boost 100-200 Hz, boxy -> cut 400-800 Hz.
- If the message names something that is not in the session, or is only a question, \
return no adjustments and say so in the reply.
- Use stem names exactly as given.
"""

# param -> (where it lives, how to clamp the size of one step)
_STEM_PARAMS = {
    "gain_db", "pan", "mute", "hpf_hz", "lpf_hz", "eq_add", "eq_clear", "comp_ratio",
    "comp_threshold_db", "deess_db", "reverb_db", "delay_db", "transient_attack",
    "transient_sustain", "drive", "chorus", "bass_db", "treble_db",
}
_MASTER_PARAMS = {
    "target_lufs", "ceiling_db", "eq_add", "eq_clear", "glue_ratio", "reverb_return_db",
    "delay_return_db", "reverb_decay_s", "speed_pct", "semitones", "reverb_size",
    "reverb_pre_delay_ms", "delay_ms", "delay_sync", "delay_feedback",
}
_MAX_STEP = {"gain_db": 12.0, "pan": 2.0, "comp_ratio": 6.0, "comp_threshold_db": 18.0,
             "deess_db": 10.0, "reverb_db": 24.0, "delay_db": 24.0, "transient_attack": 2.0,
             "transient_sustain": 2.0, "target_lufs": 6.0, "glue_ratio": 3.0,
             "reverb_return_db": 18.0, "delay_return_db": 18.0, "reverb_decay_s": 4.0,
             "speed_pct": 50.0, "semitones": 12.0, "drive": 1.0, "chorus": 1.0, "bass_db": 12.0,
             "treble_db": 12.0, "reverb_size": 1.0, "reverb_pre_delay_ms": 100.0, "delay_ms": 500.0,
             "delay_feedback": 0.9}


def _resolve_targets(target: str, plan: dict) -> list[str] | None:
    """Stem names a target means; None for the master; [] for nothing known."""
    t = str(target or "").strip()
    if t.lower() == "master":
        return None
    names = list(plan["stems"])
    if t.lower() in ("all", "*", "everything"):
        return names
    if t.lower().startswith("role:"):
        role = t[5:].strip().lower()
        return [n for n in names if plan["stems"][n].get("role") == role]
    for n in names:
        if n.lower() == t.lower():
            return [n]
    # "acoustic guitar" -> the acoustic stems, the way the keyword parser reads it
    low = t.lower()
    for role, words in _NOTE_ROLES.items():
        if any(_has(low, w) for w in words):
            return [n for n in names if plan["stems"][n].get("role") == role]
    return []


def _num(adj: dict, current: float, param: str) -> float | None:
    """New value from an adjustment's delta or value; None if neither is usable."""
    step = _MAX_STEP.get(param, 1e9)
    try:
        if "delta" in adj:
            d = float(adj["delta"])
            if d != d:
                return None
            return current + max(-step, min(step, d))
        if "value" in adj:
            v = float(adj["value"])
            return None if v != v else v
    except (TypeError, ValueError):
        return None
    return None


def _band(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    try:
        return {"freq": float(value.get("freq", 1000.0)), "gain_db": float(value.get("gain_db", 0.0)),
                "q": float(value.get("q", 1.0)),
                "type": value.get("type", "peaking") if value.get("type") in
                ("peaking", "low_shelf", "high_shelf") else "peaking"}
    except (TypeError, ValueError):
        return None


def apply_adjustments(plan: dict, stems: list[Stem], adjustments: list,
                      cfg: dict | None = None) -> tuple[dict, list[str]]:
    """Apply chat adjustments to a copy of `plan`, then validate (clamp) it.

    Only whitelisted params are touched; deltas are limited per step, and
    `validate_plan` enforces the hard limits afterwards, so a model asking for
    +40 dB gets the limit, not +40. Returns (new plan, [rejected adjustment notes])."""
    cfg = cfg or DEFAULT_CONFIG
    new = copy.deepcopy(plan)
    rejected: list[str] = []
    for adj in adjustments if isinstance(adjustments, list) else []:
        if not isinstance(adj, dict):
            continue
        param = str(adj.get("param", ""))
        targets = _resolve_targets(adj.get("target", ""), new)
        if targets is None:
            if param not in _MASTER_PARAMS or not _apply_master(new, param, adj):
                rejected.append(f"master {param}")
            continue
        if param not in _STEM_PARAMS or not targets:
            rejected.append(f"{adj.get('target')} {param}")
            continue
        for n in targets:
            if not _apply_stem(new["stems"][n], param, adj):
                rejected.append(f"{n} {param}")
    return validate_plan(new, stems, cfg), rejected


def _apply_stem(st: dict, param: str, adj: dict) -> bool:
    if param == "mute":
        st["mute"] = bool(adj.get("value", True))
        return True
    if param == "eq_add":
        b = _band(adj.get("value"))
        if b is None:
            return False
        st["eq"] = (list(st.get("eq", [])) + [b])[-6:]
        return True
    if param == "eq_clear":
        st["eq"] = []
        return True
    if param in ("bass_db", "treble_db"):     # one shelf band (fx.tone_bands), like eq_add
        try:
            db = float(adj["delta"] if "delta" in adj else adj.get("value"))
        except (TypeError, ValueError, KeyError):
            return False
        db = max(-_MAX_STEP[param], min(_MAX_STEP[param], db))
        bands = tone_bands(db, 0.0) if param == "bass_db" else tone_bands(0.0, db)
        if not bands:
            return False
        st["eq"] = (list(st.get("eq", [])) + bands)[-6:]
        return True
    slot = {"gain_db": (st, "fader_db"), "pan": (st, "pan"), "hpf_hz": (st, "hpf_hz"),
            "lpf_hz": (st, "lpf_hz"),
            "comp_ratio": (st.setdefault("comp", {}), "ratio"),
            "comp_threshold_db": (st.setdefault("comp", {}), "threshold_db"),
            "deess_db": (st.setdefault("deess", {}), "max_reduction_db"),
            "reverb_db": (st.setdefault("sends", {}), "reverb_db"),
            "delay_db": (st.setdefault("sends", {}), "delay_db"),
            "transient_attack": (st.setdefault("transient", {}), "attack"),
            "transient_sustain": (st.setdefault("transient", {}), "sustain"),
            "drive": (st.setdefault("drive", {}), "amount"),
            "chorus": (st.setdefault("chorus", {}), "mix")}[param]
    holder, key = slot
    current = float(holder.get(key, SEND_OFF_DB if param in ("reverb_db", "delay_db") else 0.0))
    if param in ("reverb_db", "delay_db") and current <= SEND_OFF_DB and "delta" in adj:
        try:
            if float(adj["delta"]) > 0:      # "more reverb" on a dry stem starts from a usable level
                current = -24.0
        except (TypeError, ValueError):
            return False
    value = _num(adj, current, param)
    if value is None:
        return False
    holder[key] = value
    return True


def _apply_master(plan: dict, param: str, adj: dict) -> bool:
    m = plan.setdefault("master", {})
    if param == "eq_add":
        b = _band(adj.get("value"))
        if b is None:
            return False
        m["master_eq"] = (list(m.get("master_eq", [])) + [b])[-6:]
        return True
    if param == "eq_clear":
        m["master_eq"] = []
        return True
    if param == "speed_pct":              # said in percent, kept as a ratio
        pct = _num(adj, float(m.get("speed", 1.0)) * 100.0, param)
        if pct is None:
            return False
        m["speed"] = pct / 100.0
        return True
    buses = plan.setdefault("send_buses", {})
    if param == "delay_sync":
        value = str(adj.get("value") or "")
        if value and value not in SYNC:
            return False
        buses.setdefault("delay", {})["sync"] = value
        return True
    if param == "delay_ms":                   # a set echo time replaces a note value
        buses.setdefault("delay", {})["sync"] = ""
    slot = {"target_lufs": (m, "target_lufs"), "ceiling_db": (m, "ceiling_db"),
            "glue_ratio": (m.setdefault("glue_comp", {}), "ratio"),
            "reverb_return_db": (buses.setdefault("reverb", {}), "return_db"),
            "delay_return_db": (buses.setdefault("delay", {}), "return_db"),
            "reverb_decay_s": (buses.setdefault("reverb", {}), "decay_s"),
            "semitones": (m, "semitones"),
            "reverb_size": (buses.setdefault("reverb", {}), "room_size"),
            "reverb_pre_delay_ms": (buses.setdefault("reverb", {}), "pre_delay_ms"),
            "delay_ms": (buses.setdefault("delay", {}), "delay_ms"),
            "delay_feedback": (buses.setdefault("delay", {}), "feedback")}[param]
    holder, key = slot
    value = _num(adj, float(holder.get(key, 0.0)), param)
    if value is None:
        return False
    holder[key] = value
    return True


def _fmt_eq(b: dict) -> str:
    kind = {"low_shelf": "low shelf", "high_shelf": "high shelf"}.get(b.get("type"), "")
    f = b["freq"]
    where = f"{f / 1000:.1f} kHz" if f >= 1000 else f"{f:.0f} Hz"
    return f"{b['gain_db']:+.1f} dB {kind + ' ' if kind else ''}at {where}"


def diff_plans(old: dict, new: dict) -> list[dict]:
    """What changed between two plans, as chips: [{"target", "text", "param"}].

    The chat shows these, not the model's description of them, so what the
    musician reads is what will be rendered."""
    chips: list[dict] = []

    def add(target, param, text):
        chips.append({"target": target, "param": param, "text": text})

    for name, b in new.get("stems", {}).items():
        a = old.get("stems", {}).get(name)
        if a is None:
            continue
        for key, label, unit in (("fader_db", "level", "dB"), ("gain_db", "input trim", "dB"), ("pan", "pan", ""),
                                 ("hpf_hz", "high-pass", "Hz"), ("lpf_hz", "low-pass", "Hz")):
            if abs(a.get(key, 0) - b.get(key, 0)) > 1e-6:
                if key == "fader_db":
                    add(name, key, f"{name} {b[key] - a[key]:+.1f} dB")
                elif key == "gain_db":
                    add(name, key, f"{name} input trim {b[key] - a[key]:+.1f} dB")
                elif key == "pan":
                    side = lambda p: "C" if abs(p) < 0.01 else f"{abs(p) * 100:.0f}{'L' if p < 0 else 'R'}"
                    add(name, key, f"{name} pan {side(a[key])} -> {side(b[key])}")
                else:
                    add(name, key, f"{name} {label} {a[key]:.0f} -> {b[key]:.0f} {unit}".rstrip())
        if a.get("mute") != b.get("mute"):
            add(name, "mute", f"{name} {'muted' if b.get('mute') else 'unmuted'}")
        ea, eb = a.get("eq", []), b.get("eq", [])
        if ea != eb:
            added = [x for x in eb if x not in ea]
            if not eb:
                add(name, "eq", f"{name} EQ cleared")
            for x in added:
                add(name, "eq", f"{name} EQ {_fmt_eq(x)}")
        for grp, key, label, unit in (("comp", "ratio", "ratio", ":1"), ("comp", "threshold_db", "comp threshold", " dB"),
                                      ("deess", "max_reduction_db", "de-ess", " dB"),
                                      ("sends", "reverb_db", "reverb send", " dB"),
                                      ("sends", "delay_db", "delay send", " dB"),
                                      ("transient", "attack", "attack", ""), ("transient", "sustain", "sustain", ""),
                                      ("drive", "amount", "drive", ""), ("chorus", "mix", "chorus", "")):
            va, vb = a.get(grp, {}).get(key), b.get(grp, {}).get(key)
            if grp in ("drive", "chorus"):      # older plans have no effects: that's 0, off
                va, vb = va or 0.0, vb or 0.0
            if va is not None and vb is not None and abs(va - vb) > 1e-6:
                fmt = (lambda v: "off" if v <= SEND_OFF_DB else f"{v:.0f}{unit}") if grp == "sends" else \
                      (lambda v: "off" if v <= 1e-3 else f"{v:.2f}") if grp in ("drive", "chorus") else \
                      (lambda v: f"{v:.1f}{unit}")
                add(name, f"{grp}.{key}", f"{name} {label} {fmt(va)} -> {fmt(vb)}")

    ma, mb = old.get("master", {}), new.get("master", {})
    for key, label, unit in (("target_lufs", "loudness", " LUFS"), ("ceiling_db", "ceiling", " dB")):
        if abs(ma.get(key, 0) - mb.get(key, 0)) > 1e-6:
            add("master", key, f"master {label} {ma[key]:.1f} -> {mb[key]:.1f}{unit}")
    if ma.get("master_eq") != mb.get("master_eq"):
        added = [x for x in mb.get("master_eq", []) if x not in ma.get("master_eq", [])]
        if not mb.get("master_eq"):
            add("master", "eq", "master EQ cleared")
        for x in added:
            add("master", "eq", f"master EQ {_fmt_eq(x)}")
    sa, sb = ma.get("speed", 1.0), mb.get("speed", 1.0)
    if abs(sa - sb) > 1e-6:
        add("master", "speed", f"speed {sa * 100:.0f}% -> {sb * 100:.0f}%")
    ka, kb = int(ma.get("semitones", 0)), int(mb.get("semitones", 0))
    if ka != kb:
        add("master", "semitones", f"key {ka:+d} -> {kb:+d} st")
    ga, gb = ma.get("glue_comp", {}).get("ratio"), mb.get("glue_comp", {}).get("ratio")
    if ga is not None and gb is not None and abs(ga - gb) > 1e-6:
        add("master", "glue", f"master glue {ga:.1f}:1 -> {gb:.1f}:1")
    for bus, key, label, unit in (("reverb", "return_db", "reverb return", " dB"),
                                  ("delay", "return_db", "delay return", " dB"),
                                  ("reverb", "decay_s", "reverb decay", " s"),
                                  ("reverb", "room_size", "reverb size", ""),
                                  ("reverb", "pre_delay_ms", "reverb pre-delay", " ms"),
                                  ("delay", "delay_ms", "echo time", " ms"),
                                  ("delay", "feedback", "echo repeats", "")):
        va = old.get("send_buses", {}).get(bus, {}).get(key)
        vb = new.get("send_buses", {}).get(bus, {}).get(key)
        if va is not None and vb is not None and abs(va - vb) > 1e-6:
            fmt = (lambda v: f"{v:.0f}{unit}") if unit == " ms" else (lambda v: f"{v:.2f}") if not unit \
                else (lambda v: f"{v:.1f}{unit}")
            add("master", f"{bus}.{key}", f"{label} {fmt(va)} -> {fmt(vb)}")
    sa = old.get("send_buses", {}).get("delay", {}).get("sync") or ""
    sb = new.get("send_buses", {}).get("delay", {}).get("sync") or ""
    if sa != sb:
        add("master", "delay.sync", f"echo time {sa or 'free'} -> {sb + ' note' if sb else 'free'}")
    return chips


def _tempo_facts(plan: dict, tempo_bpm: float | None) -> dict:
    """What the model needs for "slower", "at 90 BPM" or "up a tone"."""
    m = plan.get("master", {})
    speed = float(m.get("speed", 1.0))
    bpm = round(float(tempo_bpm), 1) if tempo_bpm else None
    return {"bpm_as_recorded": bpm, "speed_pct": round(speed * 100, 1),
            "bpm_now": round(bpm * speed, 1) if bpm else None,
            "semitones": int(m.get("semitones", 0))}


def _plan_for_chat(plan: dict) -> dict:
    """The plan without what the model doesn't need to see (reasons, notes)."""
    return {
        "stems": {n: {k: v for k, v in e.items() if k != "reason"} for n, e in plan["stems"].items()},
        "master": plan.get("master", {}),
        "send_buses": plan.get("send_buses", {}),
    }


def revise_plan(
    message: str,
    plan: dict,
    stems: list[Stem],
    cfg: dict | None = None,
    history: list[dict] | None = None,
    output_analysis: dict | None = None,
    stem_levels: dict | None = None,
    stem_analysis: dict | None = None,
    use_llm: bool = True,
    provider: str | None = None,
    model: str | None = None,
    llm: Callable[[str, str], str] | None = None,
    tempo_bpm: float | None = None,
) -> dict:
    """One chat turn: the musician's message in, a revised plan out.

    With `use_llm`, the model gets the message, recent turns, the plan and the
    measurements (numbers and text only, never audio) and answers with relative
    adjustments (`REVISE_PROMPT`); `apply_adjustments` applies and clamps them.
    On any failure (no key, network, malformed JSON) or with `use_llm=False`,
    the keyword parser `_apply_notes` handles the message instead.
    `llm(system, payload) -> text` can be injected (tests). `tempo_bpm` is the
    song's measured tempo as recorded, so "at 90 BPM" can be turned into a speed.

    Returns {"plan", "reply", "changes": diff chips, "source", "rejected",
    "not_understood"}. "changes" is computed from the two plans, so it is
    exactly what will be rendered; the reply says so when nothing changed.
    """
    cfg = cfg or DEFAULT_CONFIG
    message = (message or "").strip()[:2000]
    result = {"plan": plan, "reply": "", "changes": [], "source": "rules",
              "rejected": [], "not_understood": []}
    if not message:
        result["reply"] = "Say what you'd like changed."
        return result

    if use_llm:
        provider, model = resolve_llm(provider, model, cfg)
        payload = json.dumps({
            "message": message,
            "recent_turns": [{k: h.get(k) for k in ("message", "reply", "changes")}
                             for h in (history or [])[-6:]],
            "plan": _plan_for_chat(plan),
            "output": output_analysis,
            "level_in_mix_db": stem_levels,
            "stem_analysis": stem_analysis,
            "tempo": _tempo_facts(plan, tempo_bpm),
        }, default=str)
        try:
            parsed = None
            for attempt in range(2):          # one retry: an empty or prose-only reply is usually a fluke
                raw = llm(REVISE_PROMPT, payload) if llm else _llm_text(REVISE_PROMPT, payload, provider, model)
                try:
                    parsed = _parse_json(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("the model did not return a JSON object")
                    break
                except ValueError:            # JSONDecodeError is a ValueError
                    if attempt:
                        raise
            new, rejected = apply_adjustments(plan, stems, parsed.get("adjustments") or [], cfg)
            new["source"] = f"chat:llm:{provider}:{model}" if not llm else "chat:llm"
            result.update(plan=new, source=new["source"], rejected=rejected,
                          reply=str(parsed.get("reply", "")).strip()[:600])
            result["changes"] = diff_plans(plan, new)
            if not result["changes"] and result["reply"]:
                result["reply"] += " (Nothing in the mix was changed.)"
            return result
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all fallback
            result["llm_error"] = f"{type(exc).__name__}: {exc}"[:300]

    new = copy.deepcopy(plan)
    log = _apply_notes(new["stems"], stems, message, tempo_bpm=tempo_bpm)
    log["finish"](new, cfg)
    new = validate_plan(new, stems, cfg)
    new["source"] = "chat:rules"
    result.update(plan=new, source="chat:rules", not_understood=log["ignored"])
    result["changes"] = diff_plans(plan, new)
    prefix = "(model unavailable, used the keyword rules) " if use_llm else ""
    if result["changes"]:
        result["reply"] = prefix + "Done: " + "; ".join(c["text"] for c in result["changes"]) + "."
    else:
        result["reply"] = prefix + "I didn't find anything to change in that."
    if log["ignored"]:
        result["reply"] += " Not understood: " + "; ".join(f"'{c}'" for c in log["ignored"]) + "."
    return result
