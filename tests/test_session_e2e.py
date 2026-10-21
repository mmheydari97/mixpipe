"""The whole mixer on a multitrack session: every feature on, every flaw checked.

Uses the same synthetic session and checks as `scripts/verify_mix.py`
(8 stems: two vocals, acoustic with clicks, electric lead with hum, harmonica,
a bass recorded late, drums, keys), shortened to 20 s.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mixpipe.io_utils import write_audio
from mixpipe.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    demo, verify = _load("make_demo_session"), _load("verify_mix")
    d = tmp_path_factory.mktemp("session")
    stems_dir = d / "stems"
    stems_dir.mkdir()
    stems, meta = demo.build(seconds=20.0, seed=11)
    for name, audio in stems.items():
        write_audio(str(stems_dir / f"{name}.wav"), audio, meta["sr"])
    cfg = verify.feature_config()
    cfg_path = d / "cfg.json"
    cfg_path.write_text(json.dumps(cfg, default=str))
    ref = verify.reference_from_stems(str(stems_dir), d / "ref.wav", 0.0, None)
    result = run(str(stems_dir), out_path=str(d / "out" / "mix.wav"), config_path=str(cfg_path),
                 notes="vocals a little louder, the harmonica is honky", align=True, reference=ref,
                 iterations=2, bit_depth=16, noise_shaping=True, verbose=False)
    checks = verify.generic_checks(result, cfg, 20.0, len(stems), 16)
    checks += verify.synthetic_checks(str(stems_dir), cfg, result["plan"], meta)
    return result, {c["name"]: c for c in checks}


@pytest.mark.parametrize("name", [
    "true peak under the ceiling", "loudness on target", "no clipping", "mono-safe",
    "16-bit grid", "reverb tail", "critique does not worsen",
    "dehum", "declick", "alignment", "de-esser",
])
def test_feature(mixed, name):
    _, checks = mixed
    assert name in checks, f"check {name!r} did not run"
    assert checks[name]["ok"], checks[name]["detail"]


def test_plan_reads_like_a_mix(mixed):
    """Taste rules that used to go wrong on multitrack material."""
    result, _ = mixed
    stems = result["plan"]["stems"]
    lead, harmony = stems["vocal_lead"], stems["vocal_harmony"]
    assert lead["pan"] == 0.0 and abs(harmony["pan"]) >= 0.4            # lead centred, backing to a side
    assert stems["bass"]["hpf_hz"] <= 40 and stems["drums"]["hpf_hz"] <= 40   # low end kept
    assert any(b["freq"] < 2000 and b["gain_db"] < 0 for b in stems["harmonica"]["eq"])   # "honky" acted on
    assert lead["fader_db"] > 0                                         # "vocals a little louder"
