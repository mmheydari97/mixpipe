"""Command line interface.

    python -m mixpipe.cli analyze --stems stems/
    python -m mixpipe.cli plan    --stems stems/ --advisor llm --notes "..."
    python -m mixpipe.cli mix     --stems stems/ --out out/mix_master.wav
    python -m mixpipe.cli chords  song.mp3 [--lang fa] [--stt openrouter --stt-model ID]
    python -m mixpipe.cli chords  https://youtu.be/...   (YouTube, SoundCloud, Spotify, ...)
    python -m mixpipe.cli chords-ui [--port 8765]
    python -m mixpipe.cli split   song.mp3 | https://...   (vocals / guitar / ... stems)
    python -m mixpipe.cli split-ui  [--port 8766]
"""

from __future__ import annotations

import argparse
import json
import re
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


CHORDS_DEPS_HINT = "the chord finder needs extra packages: pip install -r requirements-chords.txt"
SPLIT_DEPS_HINT = "the splitter needs extra packages: pip install -r requirements-chords.txt"


def _add_song(paths: list[str], root: Path, title: str | None, quiet: bool,
              many: bool = True) -> str:
    """Store a song from audio files or one link (songstore); returns its id."""
    from . import songstore

    links = [f for f in paths if re.match(r"^https?://", f)]
    if links:
        from .fetch import FetchError, resolve

        if len(paths) > 1:
            raise ValueError("give either one link or audio files, not both")
        try:
            source = resolve(links[0])
        except FetchError as exc:
            raise ValueError(f"{links[0]}: {exc}") from exc
        if source.get("note") and not quiet:
            print(source["note"])
        return songstore.new_song_from_url(source, root, title=title)
    if not many and len(paths) > 1:
        raise ValueError("give one file: the full song")
    files = []
    for f in paths:
        p = Path(f)
        if not p.is_file():
            raise FileNotFoundError(f"no such file: {p}")
        files.append((p.name, p.read_bytes()))
    return songstore.new_song(files, root, title=title)


def _progress_printer(quiet: bool):
    def progress(stage: str, frac: float, msg: str) -> None:
        if not quiet and frac in (0.0, 0.1, 1.0):
            print(f"[{stage:<10}] {msg}")
    return progress


def _serve(app, name: str, host: str, port: int, no_browser: bool) -> int:
    import uvicorn

    url = f"http://{host}:{port}/"
    print(f"mixpipe {name} on {url}  (Ctrl+C to stop)")
    if not no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def cmd_chords(args) -> int:
    try:
        from .chords import pipeline
        from .chords.harmonica import to_harp_text
        from .chords.sheet import to_chordpro, to_text
    except ImportError as exc:
        raise RuntimeError(f"{CHORDS_DEPS_HINT} ({exc})") from exc

    root = Path(args.out)
    song_id = _add_song(args.files, root, args.title, args.quiet)
    song_dir = root / song_id
    sheet = pipeline.analyze(
        song_dir,
        {"language": args.lang, "stt": args.stt, "stt_model": args.stt_model,
         "separate": not args.no_separate},
        progress=_progress_printer(args.quiet),
    )
    pipeline.set_status(song_dir, "done", "align", 1.0, "done")
    (song_dir / f"{song_id}.cho").write_text(to_chordpro(sheet))
    (song_dir / f"{song_id}.txt").write_text(to_text(sheet))
    (song_dir / f"{song_id}.harp.txt").write_text(to_harp_text(sheet))
    if not args.quiet:
        print()
        print(to_text(sheet))
        print(f"wrote {song_dir}/sheet.json, {song_id}.cho, {song_id}.txt, {song_id}.harp.txt")
        print("open it in the editor with: python -m mixpipe.cli chords-ui")
    return 0


def cmd_chords_ui(args) -> int:
    try:
        from .web.server import create_app
    except ImportError as exc:
        raise RuntimeError(f"{CHORDS_DEPS_HINT} ({exc})") from exc
    return _serve(create_app(args.out), "chords UI", args.host, args.port, args.no_browser)


def cmd_split(args) -> int:
    try:
        from . import songstore
        from .splitter import pipeline
    except ImportError as exc:
        raise RuntimeError(f"{SPLIT_DEPS_HINT} ({exc})") from exc

    root = Path(args.out)
    song_id = _add_song([args.file], root, args.title, args.quiet, many=False)
    song_dir = root / song_id
    pipeline.split(song_dir, progress=_progress_printer(args.quiet))
    songstore.set_status(song_dir, "done", "separate", 1.0, "done")
    if not args.quiet:
        print()
        for name in pipeline.STEM_NAMES:
            print(pipeline.stem_path(song_dir, name))
        print("mix them in the browser with: python -m mixpipe.cli split-ui")
    return 0


def cmd_split_ui(args) -> int:
    try:
        from .web.splitter_server import create_app
    except ImportError as exc:
        raise RuntimeError(f"{SPLIT_DEPS_HINT} ({exc})") from exc
    return _serve(create_app(args.out), "splitter", args.host, args.port, args.no_browser)


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

    p_ch = sub.add_parser("chords", help="find chords + lyrics in a song (headless)")
    p_ch.add_argument("files", nargs="+",
                      help="one full song, several stems (vox.wav, bass.wav, ...), or a "
                           "link (YouTube, SoundCloud, Spotify, a direct audio URL, ...)")
    p_ch.add_argument("--lang", default="auto",
                      help="lyrics language: auto, an ISO code like en/fa, or none")
    p_ch.add_argument("--stt", choices=["mlx", "openrouter"], default="mlx",
                      help="speech-to-text backend (default: local mlx-whisper)")
    p_ch.add_argument("--stt-model", default=None, help="OpenRouter transcription model id")
    p_ch.add_argument("--no-separate", action="store_true",
                      help="skip Demucs separation of a single full-mix file")
    p_ch.add_argument("--title", default=None)
    p_ch.add_argument("--out", default="out/chords", help="song store directory")
    p_ch.add_argument("--quiet", action="store_true")
    p_ch.set_defaults(func=cmd_chords)

    p_ui = sub.add_parser("chords-ui", help="start the chord finder web UI")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.add_argument("--out", default="out/chords", help="song store directory")
    p_ui.add_argument("--no-browser", action="store_true")
    p_ui.set_defaults(func=cmd_chords_ui)

    p_sp = sub.add_parser("split", help="split a song into vocals / guitar / drums / bass / "
                                        "keys / other (headless)")
    p_sp.add_argument("file", help="the full song, or a link (YouTube, SoundCloud, Spotify, ...)")
    p_sp.add_argument("--title", default=None)
    p_sp.add_argument("--out", default="out/splitter", help="song store directory")
    p_sp.add_argument("--quiet", action="store_true")
    p_sp.set_defaults(func=cmd_split)

    p_su = sub.add_parser("split-ui", help="start the stem splitter / practice mixer web UI")
    p_su.add_argument("--host", default="127.0.0.1")
    p_su.add_argument("--port", type=int, default=8766)
    p_su.add_argument("--out", default="out/splitter", help="song store directory")
    p_su.add_argument("--no-browser", action="store_true")
    p_su.set_defaults(func=cmd_split_ui)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
