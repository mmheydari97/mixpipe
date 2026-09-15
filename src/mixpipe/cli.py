"""Command line interface.

    python -m mixpipe.cli analyze --stems stems/
    python -m mixpipe.cli plan    --stems stems/ --advisor llm --notes "..."
    python -m mixpipe.cli mix     --stems stems/ --out out/mix_master.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .advisor import llm_advisor, rules_advisor
from .analysis import analyze_all, masking_report
from .config import load_config
from .io_utils import load_stems
from .plan import describe_plan, save_plan
from .pipeline import run


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stems", default="stems/", help="directory of stem audio files")
    p.add_argument("--config", default=None, help="YAML config override")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")


def cmd_analyze(args) -> int:
    cfg = load_config(args.config)
    stems = load_stems(args.stems, force_mono=True, role_table=cfg["roles"])
    analysis = analyze_all(stems)
    analysis["masking_report"] = masking_report(stems)

    out = Path(args.out or "out/analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2))

    if not args.quiet:
        for s in analysis["stems"]:
            print(
                f"{s['name']:<20} {s['role']:<14} "
                f"{s['duration_s']:6.2f}s  "
                f"{s['lufs']:7.1f} LUFS  "
                f"peak {s['peak_db']:+6.1f}  "
                f"crest {s['crest_db']:5.1f}  "
                f"SNR {s['snr_db']:5.1f}  "
                f"centroid {s['spectral_centroid_hz']:7.0f} Hz"
            )
        if analysis["stems"] and analysis["stems"][0]["lufs_estimated"]:
            print("\nnote: LUFS values are RMS approximations "
                  "(pip install pyloudnorm for real metering)")
        for m in analysis["masking_report"]:
            print(f"masking: {m['stems'][0]} vs {m['stems'][1]} "
                  f"-> {', '.join(m['contested_bands'])}")
        print(f"\nwrote {out}")
    return 0


def cmd_plan(args) -> int:
    cfg = load_config(args.config)
    stems = load_stems(args.stems, force_mono=True, role_table=cfg["roles"])
    analysis = analyze_all(stems)

    if args.advisor == "llm":
        plan = llm_advisor(
            stems, analysis, cfg, notes=args.notes,
            provider=args.provider, model=args.model,
            masking=masking_report(stems),
        )
    else:
        plan = rules_advisor(stems, analysis, cfg, notes=args.notes)

    path = save_plan(plan, args.out or "out/mix_plan.json")
    if not args.quiet:
        print(describe_plan(plan))
        print(f"\nwrote {path}")
        print("Edit it by hand and re-run with: mix --plan " + path)
    return 0


def cmd_mix(args) -> int:
    run(
        stems_dir=args.stems,
        out_path=args.out or "out/mix_master.wav",
        config_path=args.config,
        advisor_mode=args.advisor,
        plan_path=args.plan,
        notes=args.notes,
        provider=args.provider,
        model=args.model,
        skip_cleanup=args.skip_cleanup,
        dump_stems=args.dump_stems,
        verbose=not args.quiet,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mixpipe",
        description="Prototype multitrack mix and master pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="measure stems, write analysis.json")
    _add_common(p_an)
    p_an.add_argument("--out", default=None)
    p_an.set_defaults(func=cmd_analyze)

    p_pl = sub.add_parser("plan", help="build a mix plan without rendering")
    _add_common(p_pl)
    p_pl.add_argument("--out", default=None)
    p_pl.add_argument("--advisor", choices=["rules", "llm", "none"], default="rules")
    p_pl.add_argument("--notes", default="", help="plain-English mix notes for the LLM")
    p_pl.add_argument("--provider", choices=["anthropic", "openrouter"], default=None)
    p_pl.add_argument("--model", default=None)
    p_pl.set_defaults(func=cmd_plan)

    p_mx = sub.add_parser("mix", help="full pipeline: mix and master to a WAV")
    _add_common(p_mx)
    p_mx.add_argument("--out", default=None)
    p_mx.add_argument("--advisor", choices=["rules", "llm", "none"], default="rules")
    p_mx.add_argument("--plan", default=None, help="use an existing plan JSON")
    p_mx.add_argument("--notes", default="")
    p_mx.add_argument("--provider", choices=["anthropic", "openrouter"], default=None)
    p_mx.add_argument("--model", default=None)
    p_mx.add_argument("--skip-cleanup", action="store_true",
                      help="bypass denoise and gating (faster, useful for A/B)")
    p_mx.add_argument("--dump-stems", action="store_true",
                      help="also write processed stems for DAW import")
    p_mx.set_defaults(func=cmd_mix)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
