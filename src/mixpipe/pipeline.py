"""End-to-end orchestration: stems in, mastered file out."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .advisor import critique_mix, llm_advisor, rules_advisor
from .analysis import analyze_all, analyze_master, compare_balance, masking_report
from .config import load_config
from .io_utils import load_stems, normalize_shape, read_audio, stems_mono, to_sample_rate, write_audio
from .master import dither, export_lufs_report, master
from .mixdown import export_stems, mixdown
from .plan import describe_plan, load_plan, save_plan, validate_plan
from .timing import align_stems


def _analyze_reference(path: str, sr: int) -> dict:
    audio, file_sr = read_audio(path)
    return analyze_master(to_sample_rate(normalize_shape(audio), file_sr, sr), sr)


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
    align: bool = False,
    reference: str | None = None,
    iterations: int = 0,
    bit_depth: int = 24,
    noise_shaping: bool = False,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict:
    """Run the whole pipeline. Returns a result dict with every report.

    align       line the stems up in time first (`timing.align_stems`)
    reference   path to a track whose tonal balance the master is matched to; it is
                also what the critique loop compares the result against
    iterations  how many times to listen to the result and revise the plan
                (`advisor.critique_mix`; an LLM critique with --advisor llm, the
                rule-based one otherwise). Stops early when nothing changes.
    start_s, duration_s   mix only this excerpt of the stems
    bit_depth   24 (default) or 16, which adds TPDF dither (`noise_shaping` for
                shaped dither) before writing
    """
    t0 = time.time()
    cfg = load_config(config_path)
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load
    if verbose:
        print("[1/5] loading stems")
    stems = load_stems(stems_dir, force_mono=stems_mono(cfg), role_table=cfg["roles"],
                       start_s=start_s, duration_s=duration_s)
    if align:
        stems = align_stems(stems)
    for s in stems:
        if verbose:
            shift = s.meta.get("align_offset_samples")
            print(f"  {s.name:<20} {s.role:<14} {s.duration:6.2f}s "
                  f"(source {s.meta['source_sr']} Hz)"
                  + (f"  aligned {shift / s.sr * 1000:+.1f} ms" if shift else ""))

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

    sr_out = cfg.get("sample_rate", 48_000)

    def render(current_plan: dict):
        bus_, processed_, mix_report_ = mixdown(
            stems, current_plan, cfg, do_cleanup=not skip_cleanup, verbose=verbose
        )
        final_, master_report_ = master(
            bus_, current_plan, cfg, verbose=verbose, reference=reference
        )
        return bus_, processed_, mix_report_, final_, master_report_

    # 4 + 5. mix and master
    if verbose:
        print("[4/5] mixing, [5/5] mastering")
    bus, processed, mix_report, final, master_report = render(plan)
    if dump_stems:
        written = export_stems(processed, str(out_dir / "processed_stems"))
        if verbose:
            print(f"  wrote {len(written)} processed stems for DAW import")

    # 6. optional: listen to the result, revise the plan, render again
    history: list[dict] = []
    if iterations > 0:
        ref_analysis = _analyze_reference(reference, sr_out) if reference else None
        best = None
        for n in range(1, iterations + 1):
            out_analysis = analyze_master(final, sr_out)
            crit = critique_mix(
                out_analysis, cfg, plan=plan, stems=stems, reference_analysis=ref_analysis,
                stem_analysis=analysis, master_report=master_report, masking=masking,
                notes=notes, use_llm=advisor_mode == "llm", provider=provider, model=model,
            )
            err = crit["balance_error"]
            score = float(np.mean(np.abs(list(err.values())))) if err else None
            history.append({"round": n, "critique": crit["critique"], "changed": crit["changed"],
                            "balance_error": err, "balance_score_db": score,
                            "output": out_analysis})
            if score is not None and (best is None or score < best[0]):
                best = (score, final, bus, processed, mix_report, master_report, plan)
            if verbose:
                print(f"[critique {n}/{iterations}] {crit['critique']}"
                      + (f"  (balance error {score:.2f} dB)" if score is not None else ""))
            if not crit["changed"] or crit["plan"] is None:
                break
            plan = crit["plan"]
            save_plan(plan, out_dir / f"mix_plan_round{n}.json")
            bus, processed, mix_report, final, master_report = render(plan)
        if ref_analysis is not None and best is not None:
            last = float(np.mean(np.abs(list(
                compare_balance(analyze_master(final, sr_out), ref_analysis).values()))))
            if best[0] < last:
                _, final, bus, processed, mix_report, master_report, plan = best
                if verbose:
                    print(f"  kept the round with the best balance "
                          f"({best[0]:.2f} dB vs {last:.2f} dB)")
        save_plan(plan, out_dir / "mix_plan.json")

    if bit_depth == 16:
        final_out = dither(final, 16, noise_shaping=noise_shaping)
        subtype = "PCM_16"
    else:
        final_out, subtype = final, "PCM_24"
    write_audio(out_path, final_out, sr_out, subtype=subtype)
    premaster_path = str(Path(out_path).with_name(Path(out_path).stem + "_premaster.wav"))
    write_audio(premaster_path, bus, sr_out)

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
        "iterations": history,
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
