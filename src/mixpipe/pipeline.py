"""End-to-end orchestration: stems in, mastered file out."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .advisor import llm_advisor, rules_advisor
from .analysis import analyze_all, masking_report
from .config import load_config
from .io_utils import load_stems, write_audio
from .master import export_lufs_report, master
from .mixdown import export_stems, mixdown
from .plan import describe_plan, load_plan, save_plan, validate_plan


def run(
    stems_dir: str,
    out_path: str = "out/mix_master.wav",
    config_path: str | None = None,
    advisor_mode: str = "rules",
    plan_path: str | None = None,
    notes: str = "",
    provider: str | None = None,
    model: str | None = None,
    skip_cleanup: bool = False,
    dump_stems: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the whole pipeline. Returns a result dict with every report."""
    t0 = time.time()
    cfg = load_config(config_path)
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load
    if verbose:
        print("[1/5] loading stems")
    stems = load_stems(stems_dir, force_mono=True, role_table=cfg["roles"])
    for s in stems:
        if verbose:
            print(f"  {s.name:<20} {s.role:<14} {s.duration:6.2f}s "
                  f"(source {s.meta['source_sr']} Hz)")

    # 2. analyse
    if verbose:
        print("[2/5] analysing")
    analysis = analyze_all(stems)
    masking = masking_report(stems)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    if verbose:
        sm = analysis["summary"]
        print(f"  loudest: {sm['loudest_stem']}  quietest: {sm['quietest_stem']}  "
              f"spread: {sm['lufs_spread_db']} dB")
        if sm["noisy_stems"]:
            print(f"  noisy takes (SNR < 20 dB): {', '.join(sm['noisy_stems'])}")
        if sm["any_clipped"]:
            print("  WARNING: at least one stem is clipped on input")

    # 3. plan
    if verbose:
        print("[3/5] building mix plan")
    if plan_path:
        plan = validate_plan(load_plan(plan_path), stems, cfg)
        plan["source"] = f"file:{plan_path}"
    elif advisor_mode == "llm":
        plan = llm_advisor(
            stems, analysis, cfg, notes=notes, provider=provider,
            model=model, masking=masking,
        )
    elif advisor_mode == "none":
        plan = validate_plan(
            {"source": "defaults", "notes": "Role defaults only."}, stems, cfg
        )
    else:
        plan = rules_advisor(stems, analysis, cfg, notes=notes)
    save_plan(plan, out_dir / "mix_plan.json")
    if verbose:
        print(describe_plan(plan))

    # 4. mix
    if verbose:
        print("[4/5] mixing")
    bus, processed, mix_report = mixdown(
        stems, plan, cfg, do_cleanup=not skip_cleanup, verbose=verbose
    )
    if dump_stems:
        written = export_stems(processed, str(out_dir / "processed_stems"))
        if verbose:
            print(f"  wrote {len(written)} processed stems for DAW import")

    # 5. master
    if verbose:
        print("[5/5] mastering")
    final, master_report = master(bus, plan, cfg, verbose=verbose)

    write_audio(out_path, final, cfg.get("sample_rate", 48_000))
    premaster_path = str(Path(out_path).with_name(Path(out_path).stem + "_premaster.wav"))
    write_audio(premaster_path, bus, cfg.get("sample_rate", 48_000))

    result = {
        "out_path": out_path,
        "premaster_path": premaster_path,
        "plan_path": str(out_dir / "mix_plan.json"),
        "analysis_path": str(out_dir / "analysis.json"),
        "analysis": analysis,
        "masking": masking,
        "plan": plan,
        "mix_report": mix_report,
        "master_report": master_report,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "run_report.json").write_text(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("analysis", "plan")}, indent=2
        )
    )

    if verbose:
        print()
        print(export_lufs_report(master_report))
        for w in mix_report.get("warnings", []):
            print(f"  note: {w}")
        print()
        print(f"mastered : {out_path}")
        print(f"premaster: {premaster_path}")
        print(f"plan     : {result['plan_path']}")
        print(f"done in {result['elapsed_s']}s")

    return result
