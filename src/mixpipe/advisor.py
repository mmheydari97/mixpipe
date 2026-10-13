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

import json
import os

from .config import DEFAULT_CONFIG, treatment_for
from .io_utils import Stem
from .plan import validate_plan

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

    # Count roles so duplicates can be panned apart.
    role_counts: dict[str, int] = {}
    for s in stems:
        role_counts[s.role] = role_counts.get(s.role, 0) + 1
    role_seen: dict[str, int] = {}

    for stem in stems:
        t = treatment_for(stem.role, cfg)
        a = by_name.get(stem.name, {})
        reasons = []

        # 1. Level match to the role target.
        measured = a.get("lufs", -18.0)
        target = t.get("target_lufs", -18.0)
        gain = float(target - measured)
        gain = max(-18.0, min(12.0, gain))
        t["gain_db"] = round(t["gain_db"] + gain, 2)
        reasons.append(f"{measured:.1f} LUFS measured, {gain:+.1f} dB to hit {target}")

        # 2. Noisy take: push the high-pass up to clear rumble and handling noise.
        snr = a.get("snr_db", 40.0)
        if snr < 20.0:
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
        if role_counts[stem.role] > 1:
            spread = 0.6
            n = role_counts[stem.role]
            t["pan"] = round(-spread + 2 * spread * idx / max(1, n - 1), 2)
            reasons.append(f"one of {n} {stem.role} stems, panned to spread them")

        t["reason"] = "; ".join(reasons)
        plan_stems[stem.name] = t

    plan = {
        "source": "rules",
        "notes": (
            "Rule-based plan from role defaults plus measured loudness, SNR, crest "
            "factor and spectral centroid."
            + (f" Musician notes (not acted on by the rules advisor): {notes}" if notes else "")
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
    return validate_plan(plan, stems, cfg)


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


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


def _call_openrouter(payload: str, model: str, max_tokens: int = 4000) -> str:
    from . import openrouter

    return openrouter.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        model=model,
        max_tokens=max_tokens,
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
    a_cfg = cfg["advisor"]
    provider = provider or a_cfg["provider"]
    model = model or a_cfg["model"]

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
        parsed = json.loads(_strip_fences(raw))
        parsed["source"] = f"llm:{provider}:{model}"
        return validate_plan(parsed, stems, cfg)
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


def critique_mix(mix_analysis: dict, cfg: dict | None = None) -> str:
    """Second LLM pass: listen to the result and suggest revisions.

    # TODO: feed the *output* analysis back to the model along with the plan that
    # produced it, and have it return a revised plan. This is the iteration loop
    # that would make the tool actually useful, and it needs a reference-track
    # comparison to have an opinion worth anything.
    """
    pass  # TODO
    return ""
