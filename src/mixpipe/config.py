"""Built-in defaults, plus optional YAML override.

Config resolution order: built-in DEFAULT_CONFIG  <-  config/default.yaml
<-  --config path on the CLI. Later wins, merged shallowly per top-level key.
"""

from __future__ import annotations

import copy
from pathlib import Path

try:
    import yaml

    HAVE_YAML = True
except ImportError:  # pragma: no cover
    yaml = None
    HAVE_YAML = False


# Filename substring -> role. Longest match wins (see io_utils.infer_role).
ROLES: dict[str, str] = {
    "acoustic": "acoustic",
    "ac_gtr": "acoustic",
    "nylon": "acoustic",
    "electric_lead": "electric_lead",
    "lead": "electric_lead",
    "solo": "electric_lead",
    "electric": "electric",
    "rhythm": "electric",
    "harmonica": "harmonica",
    "harp": "harmonica",
    "seaboard_bass": "bass",
    "bass": "bass",
    "sub": "bass",
    "seaboard": "keys",
    "keys": "keys",
    "pad": "keys",
    "synth": "keys",
    "piano": "keys",
    "vox": "vocal",
    "vocal": "vocal",
    "voice": "vocal",
    "memo": "vocal",
    "drum": "drums",
    "perc": "drums",
    "click": "drums",
    # the six Demucs stems from the splitter (vocals / guitar / drums / bass / piano / other)
    "guitar": "electric",
    "other": "keys",
}


# Per-role starting treatment. These are conservative "get it in the ballpark"
# moves, not taste. The advisor adjusts them from the analysis.
#
#   gain_db     : static trim applied before everything
#   hpf_hz      : high-pass to clear low-end mud (0 = off)
#   lpf_hz      : low-pass (0 = off)
#   eq          : list of peaking bands {freq, gain_db, q}
#   comp        : {threshold_db, ratio, attack_ms, release_ms, makeup_db}
#   pan         : -1 hard left .. 0 centre .. +1 hard right
#   target_lufs : desired stem loudness before the bus (advisory)
#   sends       : level (dB, -60 = off) of this stem into the shared reverb and
#                 delay buses; the buses are set up under DEFAULT_CONFIG["sends"]
#   deess       : split-band sibilance control {freq_hz, max_reduction_db}; 0 = off
#   transient   : attack / sustain shaping, each -1 (softer) .. +1 (punchier)
ROLE_TREATMENTS: dict[str, dict] = {
    "acoustic": {
        "gain_db": 0.0,
        "hpf_hz": 80.0,
        "lpf_hz": 0.0,
        "eq": [
            {"freq": 220.0, "gain_db": -2.5, "q": 0.9},   # boxiness
            {"freq": 3500.0, "gain_db": 1.5, "q": 0.8},    # pick definition
            {"freq": 9000.0, "gain_db": 1.0, "q": 0.7},    # air
        ],
        "comp": {"threshold_db": -20.0, "ratio": 2.5, "attack_ms": 15.0,
                 "release_ms": 180.0, "makeup_db": 2.0},
        "pan": -0.35,
        "target_lufs": -18.0,
        "sends": {"reverb_db": -18.0, "delay_db": -60.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0},
        "transient": {"attack": 0.2, "sustain": 0.0},   # a little pick definition
    },
    "electric": {
        "gain_db": 0.0,
        "hpf_hz": 100.0,
        "lpf_hz": 12000.0,
        "eq": [
            {"freq": 400.0, "gain_db": -2.0, "q": 1.0},
            {"freq": 2500.0, "gain_db": 1.5, "q": 0.9},
        ],
        "comp": {"threshold_db": -18.0, "ratio": 3.0, "attack_ms": 10.0,
                 "release_ms": 150.0, "makeup_db": 2.0},
        "pan": 0.35,
        "target_lufs": -18.0,
        "sends": {"reverb_db": -20.0, "delay_db": -60.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0},
        "transient": {"attack": 0.0, "sustain": 0.0},
    },
    "electric_lead": {
        "gain_db": 0.0,
        "hpf_hz": 120.0,
        "lpf_hz": 11000.0,
        "eq": [
            {"freq": 800.0, "gain_db": -1.5, "q": 1.0},
            {"freq": 3000.0, "gain_db": 2.5, "q": 0.8},
        ],
        "comp": {"threshold_db": -16.0, "ratio": 3.5, "attack_ms": 8.0,
                 "release_ms": 120.0, "makeup_db": 2.5},
        "pan": 0.15,
        "target_lufs": -16.0,
        "sends": {"reverb_db": -16.0, "delay_db": -22.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0},
        "transient": {"attack": 0.0, "sustain": 0.0},
    },
    "harmonica": {
        # Harmonica lives right where vocals and lead guitar do, and it has a
        # nasty 1-3k honk. Carve, don't boost.
        "gain_db": -1.0,
        "hpf_hz": 150.0,
        "lpf_hz": 10000.0,
        "eq": [
            {"freq": 1800.0, "gain_db": -3.0, "q": 1.4},   # honk
            {"freq": 500.0, "gain_db": 1.5, "q": 0.8},     # body
            {"freq": 6000.0, "gain_db": -1.5, "q": 0.9},   # reed harshness
        ],
        "comp": {"threshold_db": -18.0, "ratio": 4.0, "attack_ms": 5.0,
                 "release_ms": 100.0, "makeup_db": 3.0},
        "pan": 0.0,
        "target_lufs": -18.0,
        "sends": {"reverb_db": -16.0, "delay_db": -24.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0},
        "transient": {"attack": 0.0, "sustain": 0.0},
    },
    "keys": {
        "gain_db": -2.0,
        "hpf_hz": 60.0,
        "lpf_hz": 0.0,
        "eq": [{"freq": 300.0, "gain_db": -1.5, "q": 0.8}],
        "comp": {"threshold_db": -22.0, "ratio": 2.0, "attack_ms": 25.0,
                 "release_ms": 250.0, "makeup_db": 1.0},
        "pan": 0.0,
        "target_lufs": -20.0,
        "sends": {"reverb_db": -20.0, "delay_db": -60.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0},
        "transient": {"attack": 0.0, "sustain": 0.0},
    },
    "bass": {
        "gain_db": 0.0,
        "hpf_hz": 30.0,
        "lpf_hz": 6000.0,
        "eq": [
            {"freq": 80.0, "gain_db": 1.5, "q": 0.7},
            {"freq": 700.0, "gain_db": 1.0, "q": 1.0},     # audibility on phones
        ],
        "comp": {"threshold_db": -18.0, "ratio": 4.0, "attack_ms": 20.0,
                 "release_ms": 120.0, "makeup_db": 2.5},
        "pan": 0.0,
        "target_lufs": -17.0,
        "sends": {"reverb_db": -60.0, "delay_db": -60.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0},
        "transient": {"attack": 0.0, "sustain": 0.0},
    },
    "vocal": {
        "gain_db": 0.0,
        "hpf_hz": 90.0,
        "lpf_hz": 0.0,
        "eq": [
            {"freq": 250.0, "gain_db": -1.5, "q": 0.9},
            {"freq": 3200.0, "gain_db": 2.0, "q": 0.8},
        ],
        "comp": {"threshold_db": -16.0, "ratio": 3.5, "attack_ms": 8.0,
                 "release_ms": 120.0, "makeup_db": 3.0},
        "pan": 0.0,
        "target_lufs": -15.0,
        "sends": {"reverb_db": -14.0, "delay_db": -20.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 6.0},
        "transient": {"attack": 0.0, "sustain": 0.0},
    },
    "drums": {
        "gain_db": 0.0,
        "hpf_hz": 35.0,
        "lpf_hz": 0.0,
        "eq": [{"freq": 400.0, "gain_db": -2.0, "q": 1.0}],
        "comp": {"threshold_db": -14.0, "ratio": 3.0, "attack_ms": 12.0,
                 "release_ms": 100.0, "makeup_db": 2.0},
        "pan": 0.0,
        "target_lufs": -14.0,
        "sends": {"reverb_db": -22.0, "delay_db": -60.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0},
        "transient": {"attack": 0.0, "sustain": 0.0},
    },
    "default": {
        "gain_db": 0.0,
        "hpf_hz": 60.0,
        "lpf_hz": 0.0,
        "eq": [],
        "comp": {"threshold_db": -20.0, "ratio": 2.0, "attack_ms": 15.0,
                 "release_ms": 180.0, "makeup_db": 1.0},
        "pan": 0.0,
        "target_lufs": -18.0,
        "sends": {"reverb_db": -20.0, "delay_db": -60.0},
        "deess": {"freq_hz": 6500.0, "max_reduction_db": 0.0},
        "transient": {"attack": 0.0, "sustain": 0.0},
    },
}


DEFAULT_CONFIG: dict = {
    "roles": ROLES,
    "treatments": ROLE_TREATMENTS,
    "cleanup": {
        "remove_dc": True,
        "dehum": "auto",           # auto | 50 | 60 | false. Notches mains hum only when found
        "declick": False,          # median-outlier click repair; off until you need it
        "declick_threshold": 8.0,  # robust sigmas; lower catches quieter clicks
        "declick_max_ms": 2.0,     # runs longer than this are treated as music, not clicks
        "denoise": True,
        "denoise_strength": 0.6,   # 0 = off, 1 = aggressive
        "noise_profile_ms": 400,   # leading silence used as the noise sample
        "gate_enabled": True,
        "gate_threshold_db": -48.0,
        "gate_hold_ms": 80.0,
        "gate_release_ms": 120.0,
        "fade_ms": 5.0,            # de-click edges
    },
    "master": {
        "target_lufs": -14.0,      # streaming-ish
        "ceiling_db": -1.0,        # limiter output ceiling
        "glue_comp": {"threshold_db": -12.0, "ratio": 1.6, "attack_ms": 30.0,
                      "release_ms": 300.0, "makeup_db": 0.0},
        "master_eq": [
            {"freq": 120.0, "gain_db": 0.5, "q": 0.7},
            {"freq": 10000.0, "gain_db": 1.0, "q": 0.6},
        ],
        "master_hpf_hz": 25.0,
        # Three-band compression ahead of the glue comp. Off by default: the
        # single-band glue comp is enough for most mixes; turn this on when the
        # low end pumps the whole bus or the top end gets harsh when it's loud.
        "multiband": {
            "enabled": False,
            "crossovers": [150.0, 4000.0],
            "bands": [
                {"threshold_db": -18.0, "ratio": 2.0, "attack_ms": 30.0,
                 "release_ms": 180.0, "makeup_db": 0.0},
                {"threshold_db": -16.0, "ratio": 1.8, "attack_ms": 15.0,
                 "release_ms": 120.0, "makeup_db": 0.0},
                {"threshold_db": -20.0, "ratio": 1.8, "attack_ms": 5.0,
                 "release_ms": 80.0, "makeup_db": 0.0},
            ],
        },
        "limiter_lookahead_ms": 5.0,
        "true_peak": True,         # limiter watches 4x-oversampled (inter-sample) peaks
        "reference_strength": 0.8, # how far a --reference pulls the tonal balance
        "limiter_release_ms": 60.0,
    },
    "mixdown": {
        "headroom_db": -6.0,       # pre-master bus peak target
        "bus_comp_enabled": True,
        # stereo files stay stereo (pan becomes balance); false folds every stem to mono.
        # A "stereo" file whose two sides are the same is folded either way.
        "keep_stereo": True,
    },
    # Shared effect buses. Every stem feeds them at its own send level (see the
    # role treatments); each bus runs once on the summed sends and its return is
    # added to the dry bus. A mix of close-miked dry sources needs this shared
    # space to sound like one performance.
    "sends": {
        "enabled": True,
        "reverb": {"room_size": 0.5, "decay_s": 1.4, "damping": 0.45,
                   "pre_delay_ms": 18.0, "width": 1.0, "return_db": -3.0},
        "delay": {"delay_ms": 120.0, "feedback": 0.25, "damping": 0.35,
                  "return_db": -6.0},
    },
    "advisor": {
        "mode": "rules",           # rules | llm | none
        "provider": "anthropic",   # anthropic | openrouter
        "model": "claude-sonnet-4-6",
        # used for chat turns when only OPENROUTER_API_KEY is set (fast and cheap)
        "openrouter_model": "anthropic/claude-haiku-5.5",
        "max_gain_db": 12.0,       # plan clamps
        "max_eq_gain_db": 9.0,
    },
}


def load_config(path: str | Path | None = None) -> dict:
    """Return DEFAULT_CONFIG, shallow-merged with a YAML file if given."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    candidates = []
    if path:
        candidates.append(Path(path))
    else:
        default_yaml = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
        if default_yaml.exists():
            candidates.append(default_yaml)

    for c in candidates:
        if not c.exists():
            raise FileNotFoundError(f"config not found: {c}")
        if not HAVE_YAML:
            # Not fatal: the built-in defaults are a complete config.
            print(f"[config] PyYAML missing, ignoring {c}")
            continue
        with open(c) as fh:
            user = yaml.safe_load(fh) or {}
        for key, val in user.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


def treatment_for(role: str, cfg: dict | None = None) -> dict:
    """Per-role treatment dict, deep-copied so callers can mutate freely."""
    cfg = cfg or DEFAULT_CONFIG
    treatments = cfg["treatments"]
    return copy.deepcopy(treatments.get(role, treatments["default"]))
