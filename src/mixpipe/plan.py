"""The mix plan: the contract between the advisor and the processing chain.

A plan is plain JSON, so it can be written by the rule-based advisor, by an LLM,
or by hand in a text editor. Everything is validated and clamped before it
reaches the DSP, because an LLM that returns `"gain_db": 400` should not be able
to blow up your monitors.

Schema:

{
  "notes": "free text, what this plan is trying to do",
  "stems": {
    "<stem name>": {
      "gain_db": float,
      "hpf_hz": float,
      "lpf_hz": float,
      "eq": [{"freq": float, "gain_db": float, "q": float, "type": str}],
      "comp": {"threshold_db": f, "ratio": f, "attack_ms": f,
               "release_ms": f, "makeup_db": f},
      "pan": float,
      "mute": bool,
      "reason": "why these moves"
    }
  },
  "master": {
    "target_lufs": float,
    "ceiling_db": float,
    "master_eq": [...],
    "glue_comp": {...}
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_CONFIG, treatment_for
from .io_utils import Stem

# Hard safety rails. The advisor can't exceed these no matter what it returns.
LIMITS = {
    "gain_db": (-24.0, 12.0),
    "hpf_hz": (0.0, 1000.0),
    "lpf_hz": (0.0, 22000.0),
    "eq_freq": (20.0, 20000.0),
    "eq_gain_db": (-12.0, 9.0),
    "eq_q": (0.2, 10.0),
    "threshold_db": (-60.0, 0.0),
    "ratio": (1.0, 20.0),
    "attack_ms": (0.1, 200.0),
    "release_ms": (5.0, 2000.0),
    "makeup_db": (-12.0, 12.0),
    "pan": (-1.0, 1.0),
    "target_lufs": (-30.0, -6.0),
    "ceiling_db": (-6.0, -0.1),
}

MAX_EQ_BANDS = 6


def _clamp(value, key: str, default=0.0) -> float:
    lo, hi = LIMITS[key]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if v != v:  # NaN
        return float(default)
    return max(lo, min(hi, v))


def validate_stem_entry(entry: dict, fallback: dict) -> dict:
    """Clamp one stem's settings, filling gaps from the role treatment."""
    entry = entry or {}
    out = {
        "gain_db": _clamp(entry.get("gain_db", fallback["gain_db"]), "gain_db"),
        "hpf_hz": _clamp(entry.get("hpf_hz", fallback["hpf_hz"]), "hpf_hz"),
        "lpf_hz": _clamp(entry.get("lpf_hz", fallback["lpf_hz"]), "lpf_hz"),
        "pan": _clamp(entry.get("pan", fallback["pan"]), "pan"),
        "mute": bool(entry.get("mute", False)),
        "reason": str(entry.get("reason", ""))[:500],
        "eq": [],
        "comp": {},
    }

    bands = entry.get("eq", fallback.get("eq", []))
    if not isinstance(bands, list):
        bands = []
    for band in bands[:MAX_EQ_BANDS]:
        if not isinstance(band, dict):
            continue
        kind = band.get("type", "peaking")
        if kind not in ("peaking", "low_shelf", "high_shelf"):
            kind = "peaking"
        out["eq"].append(
            {
                "freq": _clamp(band.get("freq", 1000.0), "eq_freq", 1000.0),
                "gain_db": _clamp(band.get("gain_db", 0.0), "eq_gain_db"),
                "q": _clamp(band.get("q", 1.0), "eq_q", 1.0),
                "type": kind,
            }
        )

    comp_in = entry.get("comp", fallback.get("comp", {})) or {}
    fb_comp = fallback.get("comp", {})
    out["comp"] = {
        "threshold_db": _clamp(
            comp_in.get("threshold_db", fb_comp.get("threshold_db", -18.0)),
            "threshold_db", -18.0,
        ),
        "ratio": _clamp(comp_in.get("ratio", fb_comp.get("ratio", 2.0)), "ratio", 2.0),
        "attack_ms": _clamp(
            comp_in.get("attack_ms", fb_comp.get("attack_ms", 10.0)), "attack_ms", 10.0
        ),
        "release_ms": _clamp(
            comp_in.get("release_ms", fb_comp.get("release_ms", 150.0)),
            "release_ms", 150.0,
        ),
        "makeup_db": _clamp(
            comp_in.get("makeup_db", fb_comp.get("makeup_db", 0.0)), "makeup_db"
        ),
    }
    return out


def validate_plan(plan: dict, stems: list[Stem], cfg: dict | None = None) -> dict:
    """Clamp a whole plan and guarantee an entry exists for every stem."""
    cfg = cfg or DEFAULT_CONFIG
    plan = plan or {}
    m_cfg = cfg["master"]
    raw_master = plan.get("master", {}) or {}

    out = {
        "notes": str(plan.get("notes", ""))[:2000],
        "source": plan.get("source", "unknown"),
        "stems": {},
        "master": {
            "target_lufs": _clamp(
                raw_master.get("target_lufs", m_cfg["target_lufs"]),
                "target_lufs", -14.0,
            ),
            "ceiling_db": _clamp(
                raw_master.get("ceiling_db", m_cfg["ceiling_db"]), "ceiling_db", -1.0
            ),
            "master_hpf_hz": _clamp(
                raw_master.get("master_hpf_hz", m_cfg["master_hpf_hz"]),
                "hpf_hz", 25.0,
            ),
            "master_eq": [],
            "glue_comp": {},
        },
    }

    for band in (raw_master.get("master_eq", m_cfg["master_eq"]) or [])[:MAX_EQ_BANDS]:
        if isinstance(band, dict):
            out["master"]["master_eq"].append(
                {
                    "freq": _clamp(band.get("freq", 1000.0), "eq_freq", 1000.0),
                    "gain_db": _clamp(band.get("gain_db", 0.0), "eq_gain_db"),
                    "q": _clamp(band.get("q", 0.7), "eq_q", 0.7),
                    "type": band.get("type", "peaking"),
                }
            )

    glue_in = raw_master.get("glue_comp", m_cfg["glue_comp"]) or {}
    out["master"]["glue_comp"] = {
        "threshold_db": _clamp(glue_in.get("threshold_db", -12.0), "threshold_db", -12.0),
        "ratio": _clamp(glue_in.get("ratio", 1.6), "ratio", 1.6),
        "attack_ms": _clamp(glue_in.get("attack_ms", 30.0), "attack_ms", 30.0),
        "release_ms": _clamp(glue_in.get("release_ms", 300.0), "release_ms", 300.0),
        "makeup_db": _clamp(glue_in.get("makeup_db", 0.0), "makeup_db"),
    }

    raw_stems = plan.get("stems", {}) or {}
    for stem in stems:
        fallback = treatment_for(stem.role, cfg)
        entry = raw_stems.get(stem.name)
        if entry is None:
            # Try a case-insensitive match; LLMs rename things.
            for key, val in raw_stems.items():
                if str(key).lower() == stem.name.lower():
                    entry = val
                    break
        out["stems"][stem.name] = validate_stem_entry(entry, fallback)
        out["stems"][stem.name]["role"] = stem.role

    return out


def save_plan(plan: dict, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2))
    return str(path)


def load_plan(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def describe_plan(plan: dict) -> str:
    """Human-readable summary, printed by the CLI."""
    lines = [f"mix plan (source: {plan.get('source', '?')})"]
    if plan.get("notes"):
        lines.append(f"  notes: {plan['notes']}")
    for name, s in plan.get("stems", {}).items():
        eq = ", ".join(
            f"{b['freq']:.0f}Hz {b['gain_db']:+.1f}dB Q{b['q']:.1f}" for b in s["eq"]
        )
        lines.append(
            f"  {name} [{s.get('role', '?')}]"
            f" gain {s['gain_db']:+.1f}dB"
            f" pan {s['pan']:+.2f}"
            f" hpf {s['hpf_hz']:.0f}Hz"
            f" comp {s['comp']['ratio']:.1f}:1 @ {s['comp']['threshold_db']:.0f}dB"
            + (f"\n      eq: {eq}" if eq else "")
            + (f"\n      why: {s['reason']}" if s.get("reason") else "")
        )
    m = plan.get("master", {})
    lines.append(
        f"  master: target {m.get('target_lufs')} LUFS, "
        f"ceiling {m.get('ceiling_db')} dB"
    )
    return "\n".join(lines)
