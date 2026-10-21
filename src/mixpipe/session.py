"""A mix you can talk to: stems loaded once, a plan revised turn by turn.

    s = MixSession("stems/", out_dir="out/session")
    s.first_mix()                              # version 1, from the rules (or LLM) advisor
    s.chat("the acoustic guitar is too loud")  # version 2: acoustic -3 dB, re-rendered
    s.undo()                                   # back to version 1
    s.export("out/final.wav")                  # the current plan over the whole song

What makes a turn fast: cleanup (dehum, declick, denoise, gate) does not depend
on the plan, so every stem is cleaned once and the cache is reused by every
render. Long songs are worked on as a preview slice (the loudest
`preview_s` seconds, usually a chorus); `export` renders the full length.

Every version keeps its plan, what changed (computed from the plan diff), the
measured output and each stem's level in the mix, so the history is a list of
facts, not of what a model said it did. Audio never leaves the machine: a chat
turn sends the message, the plan and measurements to the model, nothing else.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from .advisor import (apply_adjustments, diff_plans, llm_advisor, resolve_llm, revise_plan,
                      rules_advisor)
from .analysis import analyze_all, analyze_master, integrated_lufs, masking_report
from .cleanup import clean_stem
from .config import load_config
from .io_utils import Stem, load_stems, stems_mono, write_audio
from .master import dither, master
from .mixdown import mixdown
from .plan import save_plan, validate_plan
from .timing import align_stems

PREVIEW_S = 45.0          # slice length a long song is worked on in
PREVIEW_OVER_S = 75.0     # songs longer than this get a preview slice


def llm_choice(cfg: dict, provider: str | None = None, model: str | None = None) -> tuple[str, str] | None:
    """(provider, model) to use for chat turns, or None when no key is set.

    An OpenRouter id ("vendor/model") goes to OpenRouter; otherwise the configured
    provider wins when its key is present, and OpenRouter (with
    `advisor.openrouter_model`) is the fallback when only OPENROUTER_API_KEY is set."""
    provider, model = resolve_llm(provider, model, cfg)
    if provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            return provider, model
        provider, model = resolve_llm("openrouter", None, cfg)    # a Claude id means nothing there
    if provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        return None
    return provider, model


def json_safe(obj):
    """Replace inf / NaN (a silent stem's loudness) with None, which JSON can carry."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def loudest_window(stems: list[Stem], seconds: float) -> float:
    """Start (s) of the loudest `seconds` of the summed stems."""
    sr = stems[0].sr
    n = max(s.n_samples for s in stems)
    hop = sr // 2
    power = np.zeros(n // hop + 1)
    for s in stems:
        m = s.mono().astype(np.float64)
        frames = m[: m.size // hop * hop].reshape(-1, hop)
        power[: frames.shape[0]] += (frames ** 2).mean(axis=1)
    k = max(1, int(seconds * sr / hop))
    if power.size <= k:
        return 0.0
    window = np.convolve(power, np.ones(k), mode="valid")
    return float(int(np.argmax(window)) * hop / sr)


class MixSession:
    def __init__(
        self,
        stems_dir: str,
        out_dir: str = "out/session",
        config_path: str | None = None,
        cfg: dict | None = None,
        start_s: float | None = None,
        duration_s: float | None = None,
        preview: bool = True,
        align: bool = True,
        reference: str | None = None,
        use_llm: bool | None = None,
        provider: str | None = None,
        model: str | None = None,
        llm=None,
        verbose: bool = False,
    ):
        self.cfg = cfg or load_config(config_path)
        self.stems_dir = str(stems_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.reference = reference
        self.align = align
        self.verbose = verbose
        self.llm = llm                       # injectable llm(system, payload) -> text
        self.lock = threading.RLock()

        choice = llm_choice(self.cfg, provider, model)
        self.use_llm = (choice is not None or llm is not None) if use_llm is None else bool(use_llm)
        self.provider, self.model = choice if choice else (provider, model)

        full = load_stems(stems_dir, force_mono=stems_mono(self.cfg), role_table=self.cfg["roles"])
        self.full_duration = max(s.duration for s in full)
        if start_s is None and duration_s is None and preview and self.full_duration > PREVIEW_OVER_S:
            start_s, duration_s = loudest_window(full, PREVIEW_S), PREVIEW_S
        self.start_s = float(start_s or 0.0)
        self.duration_s = duration_s
        self.preview = duration_s is not None or self.start_s > 0
        self._full = full
        self.stems = self._slice(full)
        if align:
            self.stems = align_stems(self.stems)
        self.analysis = analyze_all(self.stems)
        self.masking = masking_report(self.stems)
        self.cleaned: dict[str, Stem] = {}
        self.versions: list[dict] = []
        self.turns: list[dict] = []          # the conversation, including replies that changed nothing
        self.pos = -1

    # ------------------------------------------------------------------ #
    def _slice(self, stems: list[Stem]) -> list[Stem]:
        if not self.start_s and self.duration_s is None:
            return stems
        out = []
        for s in stems:
            a = int(self.start_s * s.sr)
            b = s.n_samples if self.duration_s is None else a + int(self.duration_s * s.sr)
            out.append(s.copy_with(s.audio[a:b]))
        return out

    def _clean_all(self, stems: list[Stem], cache: dict[str, Stem]) -> dict[str, Stem]:
        for s in stems:
            if s.name not in cache:
                cache[s.name] = clean_stem(s, self.cfg)
        return cache

    def render(self, plan: dict, stems: list[Stem] | None = None,
               cache: dict[str, Stem] | None = None) -> dict:
        """Mix and master `plan`. Returns the audio and its measurements."""
        stems = stems or self.stems
        cache = self._clean_all(stems, self.cleaned if cache is None else cache)
        t0 = time.time()
        bus, processed, mix_report = mixdown(stems, plan, self.cfg, verbose=False, cleaned=cache)
        final, master_report = master(bus, plan, self.cfg, verbose=False, reference=self.reference)
        sr = self.cfg.get("sample_rate", 48_000)
        bus_lufs = integrated_lufs(bus.mean(axis=1), sr)
        levels = {}
        for p in processed:
            mono = p.audio.mean(axis=1) if p.audio.ndim == 2 else p.audio
            lv = integrated_lufs(mono, sr) - bus_lufs
            levels[p.name] = round(float(lv), 2) if np.isfinite(lv) else None   # None: silent here
        return {"audio": final, "bus": bus, "master_report": master_report, "mix_report": mix_report,
                "output": analyze_master(final, sr), "levels": levels,
                "render_s": round(time.time() - t0, 2)}

    def _add_version(self, plan: dict, message: str, reply: str, source: str,
                     extra: dict | None = None) -> dict:
        prev = self.current
        r = self.render(plan)
        n = len(self.versions) + 1
        wav = self.out_dir / f"mix_v{n}.wav"
        write_audio(str(wav), r["audio"], self.cfg.get("sample_rate", 48_000))
        save_plan(plan, self.out_dir / f"mix_plan_v{n}.json")
        v = {
            "n": n, "message": message, "reply": reply, "source": source,
            "changes": diff_plans(prev["plan"], plan) if prev else [],
            "plan": plan, "output": r["output"], "levels": r["levels"],
            "master_report": r["master_report"], "wav": str(wav), "render_s": r["render_s"],
            "parent": prev["n"] if prev else None, "time": time.time(),
            **(extra or {}),
        }
        # nothing is ever dropped: a turn after an undo starts a branch from the
        # version being listened to (its "parent"), and undo walks back up parents
        self.versions.append(v)
        self.pos = len(self.versions) - 1
        self._log_turn(v)
        return v

    _TURN_KEYS = ("n", "message", "reply", "source", "model", "changes", "rejected",
                  "not_understood", "llm_error", "render_s")

    def _log_turn(self, entry: dict) -> None:
        self.turns.append(json_safe({k: entry.get(k) for k in self._TURN_KEYS}))
        self._save_index()

    def _save_index(self) -> None:
        idx = [{k: v for k, v in ver.items() if k not in ("plan", "master_report")} for ver in self.versions]
        (self.out_dir / "session.json").write_text(json.dumps(
            {"stems_dir": self.stems_dir, "start_s": self.start_s, "duration_s": self.duration_s,
              "position": self.pos, "versions": idx, "turns": self.turns}, indent=2, default=str))

    # ------------------------------------------------------------------ #
    @property
    def current(self) -> dict | None:
        return self.versions[self.pos] if 0 <= self.pos < len(self.versions) else None

    def set_model(self, model: str | None, provider: str = "openrouter") -> None:
        """Answer the following turns with `model` (an OpenRouter id by default);
        None switches to the keyword rules. Versions already made keep theirs."""
        with self.lock:
            if model is None:
                self.use_llm = False
            else:
                self.provider, self.model, self.use_llm = provider, model, True

    def _answered_by(self, source: str) -> str | None:
        """The model behind a plan's source, or None when the rules made it
        (also when the model failed and the rules took over)."""
        return self.model if source.startswith(("llm", "chat:llm")) else None

    def first_mix(self, notes: str = "") -> dict:
        with self.lock:
            if self.use_llm and self.provider:
                plan = llm_advisor(self.stems, self.analysis, self.cfg, notes=notes,
                                   provider=self.provider, model=self.model, masking=self.masking)
            else:
                plan = rules_advisor(self.stems, self.analysis, self.cfg, notes=notes)
            source = plan.get("source", "rules")
            return self._add_version(plan, notes or "(first mix)", "First mix from the "
                                     + ("LLM advisor." if source.startswith("llm")
                                        else "rule-based advisor."), source,
                                     {"model": self._answered_by(source)})

    def chat(self, message: str) -> dict:
        """One turn. Returns the new version (or, when nothing changed, a reply-only
        entry that is not added to the history)."""
        with self.lock:
            cur = self.current or self.first_mix()
            turn = revise_plan(
                message, cur["plan"], self.stems, self.cfg, history=self.lineage(),
                output_analysis=cur["output"], stem_levels=cur["levels"],
                stem_analysis=self.analysis, use_llm=self.use_llm, provider=self.provider,
                model=self.model, llm=self.llm,
            )
            extra = {"rejected": turn["rejected"], "not_understood": turn["not_understood"],
                     "llm_error": turn.get("llm_error"), "model": self._answered_by(turn["source"])}
            if not turn["changes"]:
                reply_only = {"n": None, "message": message, "reply": turn["reply"],
                              "source": turn["source"], "changes": [], **extra}
                self._log_turn(reply_only)
                return reply_only
            return self._add_version(turn["plan"], message, turn["reply"], turn["source"], extra)

    def set_params(self, adjustments: list[dict], label: str = "manual edit") -> dict | None:
        """Direct edits (faders, mute) through the same validated path as chat."""
        with self.lock:
            cur = self.current or self.first_mix()
            plan, rejected = apply_adjustments(cur["plan"], self.stems, adjustments, self.cfg)
            plan["source"] = "manual"
            changes = diff_plans(cur["plan"], plan)
            if not changes:
                return None
            reply = "Set: " + "; ".join(c["text"] for c in changes) + "."
            return self._add_version(plan, label, reply, "manual", {"rejected": rejected})

    def undo(self) -> dict | None:
        """Back to the version this one was made from."""
        with self.lock:
            cur = self.current
            if cur and cur.get("parent") is not None:
                return self.goto(cur["parent"])
            return cur

    def redo(self) -> dict | None:
        """Forward to the newest version made from this one."""
        with self.lock:
            cur = self.current
            kids = [v for v in self.versions if cur and v.get("parent") == cur["n"]]
            return self.goto(kids[-1]["n"]) if kids else cur

    def goto(self, n: int) -> dict | None:
        with self.lock:
            for i, v in enumerate(self.versions):
                if v["n"] == n:
                    self.pos = i
                    self._save_index()
                    return v
            raise KeyError(f"no version {n}")

    def lineage(self) -> list[dict]:
        """The current version and the ones it was made from, oldest first."""
        chain, v = [], self.current
        while v is not None:
            chain.append(v)
            v = self.version(v["parent"]) if v.get("parent") is not None else None
        return chain[::-1]

    def version(self, n: int) -> dict:
        for v in self.versions:
            if v["n"] == n:
                return v
        raise KeyError(f"no version {n}")

    def export(self, path: str, bit_depth: int = 24, full: bool = True) -> str:
        """Write the current plan to `path`, over the whole song unless `full=False`."""
        with self.lock:
            cur = self.current or self.first_mix()
            if full and self.preview:
                stems = align_stems(self._full) if self.align else self._full
                plan = validate_plan(cur["plan"], stems, self.cfg)
                if not hasattr(self, "_full_cleaned"):
                    self._full_cleaned: dict[str, Stem] = {}
                audio = self.render(plan, stems, self._full_cleaned)["audio"]
            else:
                audio = self.render(cur["plan"])["audio"]
            sr = self.cfg.get("sample_rate", 48_000)
            if bit_depth == 16:
                write_audio(path, dither(audio, 16, noise_shaping=True), sr, subtype="PCM_16")
            else:
                write_audio(path, audio, sr, subtype="PCM_24")
            return path

    def summary(self, v: dict | None = None) -> dict:
        """A version without its audio-sized parts, for the web page."""
        v = v or self.current
        if v is None:
            return {}
        out = {k: v.get(k) for k in ("n", "message", "reply", "source", "model", "changes", "levels",
                                     "render_s", "parent", "rejected", "not_understood", "llm_error")}
        o = v.get("output") or {}
        out["output"] = {k: o.get(k) for k in ("lufs", "true_peak_db", "crest_db", "lr_correlation")}
        out["stems"] = {n: {k: e.get(k) for k in ("role", "gain_db", "fader_db", "pan", "mute", "sends")}
                        for n, e in v["plan"]["stems"].items()} if "plan" in v else None
        out["master"] = {k: v["plan"]["master"].get(k) for k in ("target_lufs", "ceiling_db")} \
            if "plan" in v else None
        return json_safe(out)
