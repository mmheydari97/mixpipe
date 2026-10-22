"""Command line interface.

    python -m mixpipe.cli analyze --stems stems/
    python -m mixpipe.cli plan    --stems stems/ --advisor llm --notes "..."
    python -m mixpipe.cli mix     --stems stems/ --out out/mix_master.wav
    python -m mixpipe.cli chat    --stems stems/      (mix by talking: "the bass is boomy")
    python -m mixpipe.cli mix-ui  [--port 8767]       (the same in the browser)
    python -m mixpipe.cli chords  song.mp3 [--lang fa] [--stt openrouter --stt-model ID]
    python -m mixpipe.cli chords  https://youtu.be/...   (YouTube, SoundCloud, Spotify, ...)
    python -m mixpipe.cli chords  song.mp3 --recognizer neural   (also asks the BTC model)
    python -m mixpipe.cli fetch-model btc                        (download it ahead of time)
    python -m mixpipe.cli chords-ui [--port 8765] [--workers N]
    python -m mixpipe.cli split   song.mp3 | https://...   (vocals / guitar / ... stems)
    python -m mixpipe.cli split-ui  [--port 8766]
    python -m mixpipe.cli ui        [--port 8760]       (all three, with a home page)
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
from .io_utils import load_stems, stems_mono
from .plan import describe_plan, save_plan
from .pipeline import run


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stems", default="stems/", help="directory of stem audio files")
    p.add_argument("--config", default=None, help="YAML config override")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")


def cmd_analyze(args) -> int:
    cfg = load_config(args.config)
    stems = load_stems(args.stems, force_mono=stems_mono(cfg), role_table=cfg["roles"])
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
    stems = load_stems(args.stems, force_mono=stems_mono(cfg), role_table=cfg["roles"])
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
        align=args.align,
        reference=args.reference,
        iterations=args.iterate,
        bit_depth=args.bit_depth,
        noise_shaping=args.noise_shaping,
        start_s=args.start,
        duration_s=args.duration,
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
         "separate": not args.no_separate, "recognizer": args.recognizer,
         "key_changes": args.key_changes, "retime": not args.no_retime},
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
    return _serve(create_app(args.out, workers=args.workers), "chords UI", args.host, args.port,
                  args.no_browser)


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
    return _serve(create_app(args.out, workers=args.workers), "splitter", args.host, args.port,
                  args.no_browser)


_CLOCK = r"\d+(?::\d+)?(?:\.\d+)?"                  # 65, 65.5, 1:05, 1:05.5
_RANGE = re.compile(rf"^({_CLOCK})-({_CLOCK})?$")      # 0:20-0:32, or 0:20- to the end


def _seconds(clock: str) -> float:
    """"1:05.5" or "65.5" -> 65.5"""
    m, _, sec = clock.rpartition(":")
    return (int(m) * 60 if m else 0) + float(sec)


def parse_ai_edit(cmd: str, rest: str) -> dict:
    """The options of `/ai [start-end] <prompt>` or `/cover [strength] <prompt>`."""
    first, _, tail = rest.strip().partition(" ")
    if cmd == "/ai":
        m = _RANGE.match(first)
        if m:
            return {"mode": "repaint", "start_s": _seconds(m.group(1)),
                    "end_s": _seconds(m.group(2)) if m.group(2) else None, "prompt": tail.strip()}
        return {"mode": "repaint", "start_s": 0.0, "end_s": None, "prompt": rest.strip()}
    try:
        return {"mode": "cover", "strength": float(first), "prompt": tail.strip()}
    except ValueError:
        return {"mode": "cover", "strength": None, "prompt": rest.strip()}


def cmd_chat(args) -> int:
    """Mix by talking: type what's wrong, hear the new version, /undo if worse."""
    from .session import MixSession

    out = Path(args.out)
    s = MixSession(args.stems, out_dir=str(out), config_path=args.config,
                   start_s=args.start, duration_s=args.duration, reference=args.reference,
                   use_llm=False if args.no_llm else None, provider=args.provider, model=args.model)
    print(f"{len(s.stems)} stems: {', '.join(f'{x.name} ({x.role})' for x in s.stems)}")
    if s.preview:
        print(f"working on {s.start_s:.0f}-{s.start_s + s.duration_s:.0f} s of "
              f"{s.full_duration:.0f} s; /export renders the whole song")
    print("chat via " + (f"{s.provider}:{s.model}" if s.use_llm else "keyword rules (no API key)"))
    print("making the first mix ...")

    def show(v: dict) -> None:
        if v.get("reply"):
            print(f"  {v['reply']}")
        for c in v.get("changes") or []:
            print(f"    - {c['text']}")
        if v.get("llm_error"):
            print(f"  (model unavailable: {v['llm_error']})")
        if v.get("n"):
            o = v["output"]
            print(f"  v{v['n']}: {v['wav']}  ({o['lufs']:.1f} LUFS, {o['true_peak_db']:+.1f} dBTP, "
                  f"rendered in {v['render_s']:.1f} s)")

    def model_command(rest: str) -> None:
        from . import openrouter

        arg = rest.strip()
        if not arg:
            print("  " + (f"{s.provider}: {s.model}" if s.use_llm else "keyword rules")
                  + "   (/model <id> to switch, /model rules, /models <words> to search)")
        elif arg in ("rules", "none", "off"):
            s.set_model(None)
            print("  the keyword rules read your messages now")
        elif not openrouter.valid_model_id(arg):
            print(f"  not an OpenRouter model id: {arg} (try /models {arg})")
        elif openrouter.api_key() is None:
            print("  OPENROUTER_API_KEY is not set (see .env_template)")
        else:
            s.set_model(arg)
            print(f"  {arg} answers from your next message on")

    def models_command(rest: str) -> None:
        from . import openrouter

        try:
            found = openrouter.search_models(openrouter.list_models("chat"), rest)
        except Exception as exc:  # noqa: BLE001 - no network: say so, keep the session
            print(f"  couldn't load the OpenRouter catalogue: {exc}")
            return
        for m in found[:15]:
            mark = "*" if s.use_llm and m["id"] == s.model else " "
            print(f" {mark} {m['id']:<48} {m['price_label']}")
        if not found:
            print("  no matching models")
        elif len(found) > 15:
            print(f"  ... {len(found) - 15} more; add words to narrow it down ('free' puts free ones first)")

    def ai_command(cmd: str, rest: str) -> None:
        from . import acestep

        opts = parse_ai_edit(cmd, rest)
        if not opts["prompt"]:
            print("  usage: /ai [start-end] <prompt>   e.g. /ai 0:20-0:32 a driving rock drum beat\n"
                  "         /cover [strength 0-1] <prompt>")
            return
        st = acestep.status()
        if not st["ok"]:
            print(f"  ACE-Step isn't running at {st['url']} (make acestep in another terminal)")
            return
        print(f"  asking ACE-Step ({st['model']}) ...")
        try:
            show(s.ai_edit(opts.pop("prompt"), **opts))
        except ValueError as exc:
            print(f"  {exc}")
        except Exception as exc:  # noqa: BLE001 - keep the session
            print(f"  ACE-Step failed: {exc}")

    show(s.first_mix(args.notes))
    help_line = ("/undo /redo /goto N /plan /levels /model [id] /models [words] "
                 "/ai [start-end] <prompt> /cover [strength] <prompt> /export [path] /quit")
    print(f"type what to change, or {help_line}")
    while True:
        try:
            line = input("mix> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/undo":
            show(s.undo() or {})
        elif cmd == "/redo":
            show(s.redo() or {})
        elif cmd == "/goto":
            try:
                show(s.goto(int(rest)))
            except (ValueError, KeyError):
                print("  no such version")
        elif cmd == "/plan":
            print(describe_plan(s.current["plan"]))
        elif cmd == "/levels" and s.current.get("levels") is None:
            print(f"  v{s.current['n']} is an AI edit: no per-stem levels")
        elif cmd == "/levels":
            for name, lv in s.current["levels"].items():
                st = s.current["plan"]["stems"][name]
                print(f"  {name:<20} {lv:+6.1f} dB in mix   fader {st.get('fader_db', 0.0):+5.1f} dB")
        elif cmd == "/model":
            model_command(rest)
        elif cmd == "/models":
            models_command(rest)
        elif cmd in ("/ai", "/cover"):
            ai_command(cmd, rest)
        elif cmd == "/export":
            path = rest.strip() or str(out / f"mix_v{s.current['n']}_full.wav")
            print(f"  wrote {s.export(path, bit_depth=args.bit_depth)}")
        elif cmd.startswith("/"):
            print(f"  commands: {help_line}")
        else:
            show(s.chat(line))
    print(f"versions are in {out}/")
    return 0


def cmd_mix_ui(args) -> int:
    from .web.mixer_server import create_app

    return _serve(create_app(args.out, splitter_root=args.splitter),
                  "mixer", args.host, args.port, args.no_browser)


def cmd_ui(args) -> int:
    try:
        from .web.hub import create_app
    except ImportError as exc:
        raise RuntimeError(f"{CHORDS_DEPS_HINT} ({exc})") from exc
    return _serve(create_app(args.chords_out, args.splitter_out, args.mixer_out, workers=args.workers),
                  "tools", args.host, args.port, args.no_browser)


def cmd_fetch_model(args) -> int:
    try:
        from .chords import btc
    except ImportError as exc:
        raise RuntimeError(f"{CHORDS_DEPS_HINT} ({exc})") from exc
    try:
        path = btc.ensure_model(progress=print)
        btc.load_model(path)                       # checks it really loads, not just that it downloaded
    except btc.ModelUnavailable as exc:
        raise RuntimeError(str(exc)) from exc
    print(f"BTC chord model ready: {path}")
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
    p_mx.add_argument("--align", action="store_true",
                      help="line the stems up in time first (separate-pass recordings)")
    p_mx.add_argument("--reference", default=None, metavar="TRACK",
                      help="match the master's tonal balance to this track")
    p_mx.add_argument("--iterate", type=int, default=0, metavar="N",
                      help="listen to the result and revise the plan up to N times "
                           "(an LLM critique with --advisor llm)")
    p_mx.add_argument("--bit-depth", type=int, choices=[16, 24], default=24,
                      help="output bit depth; 16 adds dither")
    p_mx.add_argument("--noise-shaping", action="store_true",
                      help="shaped dither for the 16-bit output")
    p_mx.add_argument("--start", type=float, default=0.0, metavar="S",
                      help="mix only an excerpt, starting at S seconds")
    p_mx.add_argument("--duration", type=float, default=None, metavar="D",
                      help="length of the excerpt in seconds")
    p_mx.set_defaults(func=cmd_mix)

    p_ct = sub.add_parser("chat", help="mix by talking: 'the acoustic guitar is too loud'")
    _add_common(p_ct)
    p_ct.add_argument("--out", default="out/chat", help="where the versions go")
    p_ct.add_argument("--notes", default="", help="notes for the first mix")
    p_ct.add_argument("--start", type=float, default=None, metavar="S")
    p_ct.add_argument("--duration", type=float, default=None, metavar="D")
    p_ct.add_argument("--reference", default=None, metavar="TRACK")
    p_ct.add_argument("--no-llm", action="store_true", help="keyword rules only, no model")
    p_ct.add_argument("--provider", choices=["anthropic", "openrouter"], default=None)
    p_ct.add_argument("--model", default=None)
    p_ct.add_argument("--bit-depth", type=int, choices=[16, 24], default=24)
    p_ct.set_defaults(func=cmd_chat)

    p_hub = sub.add_parser("ui", help="the chord finder, splitter and mixer on one port")
    p_hub.add_argument("--host", default="127.0.0.1")
    p_hub.add_argument("--port", type=int, default=8760)
    p_hub.add_argument("--chords-out", default="out/chords", help="the chord finder's song store")
    p_hub.add_argument("--splitter-out", default="out/splitter", help="the splitter's song store")
    p_hub.add_argument("--mixer-out", default="out/mixer", help="mixer session directory")
    p_hub.add_argument("--no-browser", action="store_true")
    p_hub.add_argument("--workers", type=int, default=1,
                       help="songs analysed or split at once, per tool (default 1)")
    p_hub.set_defaults(func=cmd_ui)

    p_mu = sub.add_parser("mix-ui", help="start the mix-by-chat web UI")
    p_mu.add_argument("--host", default="127.0.0.1")
    p_mu.add_argument("--port", type=int, default=8767)
    p_mu.add_argument("--out", default="out/mixer", help="session directory")
    p_mu.add_argument("--splitter", default="out/splitter", help="the splitter's song store")
    p_mu.add_argument("--no-browser", action="store_true")
    p_mu.set_defaults(func=cmd_mix_ui)

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
    p_ch.add_argument("--recognizer", choices=["template", "neural"], default="template",
                      help="chord recognition: chroma templates (default) or also the BTC "
                           "neural model (downloads 12 MB once; see fetch-model)")
    p_ch.add_argument("--key-changes", choices=["normal", "sensitive"], default="normal",
                      help="how short a modulation can be and still be marked")
    p_ch.add_argument("--no-retime", action="store_true",
                      help="keep estimated word times from text-only transcription models")
    p_ch.add_argument("--title", default=None)
    p_ch.add_argument("--out", default="out/chords", help="song store directory")
    p_ch.add_argument("--quiet", action="store_true")
    p_ch.set_defaults(func=cmd_chords)

    p_ui = sub.add_parser("chords-ui", help="start the chord finder web UI")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.add_argument("--out", default="out/chords", help="song store directory")
    p_ui.add_argument("--no-browser", action="store_true")
    p_ui.add_argument("--workers", type=int, default=1,
                      help="songs analysed at once (default 1; more needs the memory for it)")
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
    p_su.add_argument("--workers", type=int, default=1,
                      help="songs split at once (default 1; more needs the memory for it)")
    p_su.set_defaults(func=cmd_split_ui)

    p_fm = sub.add_parser("fetch-model", help="download an optional model once (BTC: neural chords)")
    p_fm.add_argument("model", choices=["btc"])
    p_fm.set_defaults(func=cmd_fetch_model)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
