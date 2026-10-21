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
      "gain_db": float,          # input trim, before the chain (level matching)
      "fader_db": float,         # level after the chain: what "too loud" moves
      "hpf_hz": float,
      "lpf_hz": float,
      "eq": [{"freq": float, "gain_db": float, "q": float, "type": str}],
      "comp": {"threshold_db": f, "ratio": f, "attack_ms": f,
               "release_ms": f, "makeup_db": f},
      "pan": float,
      "sends": {"reverb_db": float, "delay_db": float},   # -60 = off
      "deess": {"freq_hz": float, "max_reduction_db": float},   # 0 = off
      "transient": {"attack": float, "sustain": float},         # -1 .. +1
      "mute": bool,
      "reason": "why these moves"
    }
  },
  "send_buses": {                       # optional, defaults come from the config
    "reverb": {"room_size", "decay_s", "damping", "pre_delay_ms", "width", "return_db"},
    "delay": {"delay_ms", "feedback", "damping", "return_db"}
  },
  "master": {
    "target_lufs": float,
    "ceiling_db": float,
    "master_eq": [...],
    "glue_comp": {...},
    "multiband": {"enabled": bool, "crossovers": [Hz, ...], "bands": [{comp}, ...]}
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
    "fader_db": (-40.0, 12.0),
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
    "send_db": (-60.0, 0.0),
    "return_db": (-40.0, 6.0),
    "room_size": (0.0, 1.0),
    "decay_s": (0.1, 8.0),
    "damping": (0.0, 0.95),
    "pre_delay_ms": (0.0, 200.0),
    "width": (0.0, 1.0),
    "delay_ms": (10.0, 1000.0),
    "feedback": (0.0, 0.9),
    "deess_hz": (3000.0, 12000.0),
    "deess_db": (0.0, 18.0),
    "transient": (-1.0, 1.0),
    "crossover_hz": (40.0, 16000.0),
}

# A send at or below this is treated as off.
SEND_OFF_DB = -60.0

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


def _validate_comp(raw: dict | None, fallback: dict, default_ratio: float = 2.0) -> dict:
    """Clamp a compressor settings dict, filling gaps from `fallback`."""
    raw = raw if isinstance(raw, dict) else {}
    return {
        "threshold_db": _clamp(
            raw.get("threshold_db", fallback.get("threshold_db", -18.0)),
            "threshold_db", -18.0,
        ),
        "ratio": _clamp(raw.get("ratio", fallback.get("ratio", default_ratio)),
                        "ratio", default_ratio),
        "attack_ms": _clamp(raw.get("attack_ms", fallback.get("attack_ms", 10.0)),
                            "attack_ms", 10.0),
        "release_ms": _clamp(raw.get("release_ms", fallback.get("release_ms", 150.0)),
                             "release_ms", 150.0),
        "makeup_db": _clamp(raw.get("makeup_db", fallback.get("makeup_db", 0.0)),
                            "makeup_db"),
    }


def validate_stem_entry(entry: dict, fallback: dict) -> dict:
    """Clamp one stem's settings, filling gaps from the role treatment."""
    entry = entry or {}
    out = {
        "gain_db": _clamp(entry.get("gain_db", fallback["gain_db"]), "gain_db"),
        "fader_db": _clamp(entry.get("fader_db", 0.0), "fader_db"),
        "hpf_hz": _clamp(entry.get("hpf_hz", fallback["hpf_hz"]), "hpf_hz"),
        "lpf_hz": _clamp(entry.get("lpf_hz", fallback["lpf_hz"]), "lpf_hz"),
        "pan": _clamp(entry.get("pan", fallback["pan"]), "pan"),
        "mute": bool(entry.get("mute", False)),
        "reason": str(entry.get("reason", ""))[:500],
        "eq": [],
        "comp": {},
        "sends": {},
    }

    fb_deess = fallback.get("deess", {})
    deess_in = entry.get("deess", fb_deess)
    deess_in = deess_in if isinstance(deess_in, dict) else {}
    out["deess"] = {
        "freq_hz": _clamp(deess_in.get("freq_hz", fb_deess.get("freq_hz", 6500.0)),
                          "deess_hz", 6500.0),
        "max_reduction_db": _clamp(
            deess_in.get("max_reduction_db", fb_deess.get("max_reduction_db", 0.0)),
            "deess_db", 0.0),
    }
    fb_tr = fallback.get("transient", {})
    tr_in = entry.get("transient", fb_tr)
    tr_in = tr_in if isinstance(tr_in, dict) else {}
    out["transient"] = {
        k: _clamp(tr_in.get(k, fb_tr.get(k, 0.0)), "transient") for k in ("attack", "sustain")
    }

    sends_in = entry.get("sends", fallback.get("sends", {}))
    sends_in = sends_in if isinstance(sends_in, dict) else {}
    fb_sends = fallback.get("sends", {})
    for key in ("reverb_db", "delay_db"):
        out["sends"][key] = _clamp(
            sends_in.get(key, fb_sends.get(key, SEND_OFF_DB)), "send_db", SEND_OFF_DB
        )

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

    out["comp"] = _validate_comp(
        entry.get("comp", fallback.get("comp", {})), fallback.get("comp", {})
    )
    return out


def validate_multiband(raw: dict | None, base: dict) -> dict:
    """Clamp the master multiband compressor; falls back to `base` per field.

    Crossovers are sorted and de-duplicated; the number of bands is forced to
    len(crossovers) + 1, padding with a transparent (ratio 1) band if short.
    """
    raw = raw if isinstance(raw, dict) else {}
    crossovers = raw.get("crossovers", base.get("crossovers", [150.0, 4000.0]))
    if not isinstance(crossovers, list):
        crossovers = base.get("crossovers", [150.0, 4000.0])
    xo = sorted({round(_clamp(f, "crossover_hz", 1000.0), 1) for f in crossovers[:5]})
    raw_bands = raw.get("bands", base.get("bands", []))
    raw_bands = raw_bands if isinstance(raw_bands, list) else []
    bands = [
        _validate_comp(b, {"ratio": 1.0, "threshold_db": -18.0, "attack_ms": 20.0,
                           "release_ms": 150.0}, default_ratio=1.0)
        for b in (raw_bands + [{}] * len(xo))[: len(xo) + 1]
    ]
    return {
        "enabled": bool(raw.get("enabled", base.get("enabled", False))),
        "crossovers": xo,
        "bands": bands,
    }


def validate_send_buses(raw: dict | None, cfg: dict) -> dict:
    """Clamp the shared reverb/delay settings, filling gaps from cfg["sends"]."""
    raw = raw if isinstance(raw, dict) else {}
    base = cfg.get("sends", {})
    specs = {   # every parameter's clamp range is LIMITS[<its own name>]
        "reverb": ("room_size", "decay_s", "damping", "pre_delay_ms", "width", "return_db"),
        "delay": ("delay_ms", "feedback", "damping", "return_db"),
    }
    out: dict = {}
    for bus, keys in specs.items():
        given = raw.get(bus, {}) if isinstance(raw.get(bus, {}), dict) else {}
        default = base.get(bus, {})
        out[bus] = {
            k: _clamp(given.get(k, default.get(k, 0.0)), k, default.get(k, 0.0))
            for k in keys
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

    out["master"]["glue_comp"] = _validate_comp(
        raw_master.get("glue_comp", m_cfg["glue_comp"]),
        {"threshold_db": -12.0, "ratio": 1.6, "attack_ms": 30.0, "release_ms": 300.0},
    )
    out["master"]["multiband"] = validate_multiband(
        raw_master.get("multiband", m_cfg.get("multiband")), m_cfg.get("multiband") or {}
    )

    out["send_buses"] = validate_send_buses(plan.get("send_buses"), cfg)

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


def _describe_sends(sends: dict) -> str:
    parts = [
        f"{name} {sends[key]:+.0f}dB"
        for name, key in (("reverb", "reverb_db"), ("delay", "delay_db"))
        if sends.get(key, SEND_OFF_DB) > SEND_OFF_DB
    ]
    return ", ".join(parts)


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
            + (f" fader {s['fader_db']:+.1f}dB" if s.get("fader_db") else "")
            + f" pan {s['pan']:+.2f}"
            f" hpf {s['hpf_hz']:.0f}Hz"
            f" comp {s['comp']['ratio']:.1f}:1 @ {s['comp']['threshold_db']:.0f}dB"
            + (f"\n      eq: {eq}" if eq else "")
            + (f"\n      sends: {_describe_sends(s['sends'])}"
               if _describe_sends(s.get("sends", {})) else "")
            + (f"\n      why: {s['reason']}" if s.get("reason") else "")
        )
    m = plan.get("master", {})
    lines.append(
        f"  master: target {m.get('target_lufs')} LUFS, "
        f"ceiling {m.get('ceiling_db')} dB"
    )
    return "\n".join(lines)
