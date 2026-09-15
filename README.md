# mixpipe

A prototype multi-track mixing + mastering pipeline. Drop raw stems in `stems/`,
run one command, get a mixed and mastered stereo WAV in `out/`.

Built for a scratch setup of: acoustic guitar, electric guitar, harmonica, and a
Seaboard (MIDI-rendered audio).

## Design principle

The DSP does the audio work. The LLM (Claude via the Anthropic API, or any model
via OpenRouter) is an **advisor and orchestrator** — it reads *numeric analysis*
of your stems (LUFS, spectral centroid, crest factor, band energies) and returns
a JSON mix plan: gains, EQ moves, compression settings, pan positions.

No LLM ever touches raw samples. That's deliberate — text models can't do
sample-level DSP, but they're good at picking parameters from measurements.

```
stems/*.wav
    |
    v
[io]  load -> mono/stereo normalize -> resample to 48k
    |
    v
[analysis]  per-stem measurements  ------> [llm_advisor]  -> mix_plan.json
    |                                              |
    v                                              v
[cleanup]   dc offset, noise reduction, gate  <----'  (params from plan)
[pitch]     pitch correction            # TODO
[timing]    alignment / quantize        # TODO
[eq]        biquad filter chain
[dynamics]  compressor
[panning]   constant-power pan -> stereo
    |
    v
[mixdown]  sum buses, bus compression, headroom check
    |
    v
[master]   master EQ -> glue comp -> limiter -> loudness target
    |
    v
out/mix_master.wav
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only `numpy` and `scipy` are strictly required. `soundfile` is optional (there's
a stdlib `wave` fallback for 16/24/32-bit WAV). `pyloudnorm`, `noisereduce`, and
`anthropic` unlock better loudness metering, better denoise, and the LLM advisor.

## Usage

```bash
# 1. Make some test stems so you can hear the pipeline work before recording
python scripts/make_test_stems.py

# 2. Analyse only — prints measurements per stem, writes out/analysis.json
python -m mixpipe.cli analyze --stems stems/

# 3. Mix + master with the rule-based plan (no LLM, no API key needed)
python -m mixpipe.cli mix --stems stems/ --out out/mix_master.wav

# 4. Same, but ask an LLM to write the mix plan from the analysis
export ANTHROPIC_API_KEY=sk-...
python -m mixpipe.cli mix --stems stems/ --advisor llm \
    --notes "harmonica is thin and too loud, acoustic sounds muddy"

# 5. Reuse / hand-edit a plan
python -m mixpipe.cli mix --stems stems/ --plan out/mix_plan.json
```

Stem role detection is filename-based: name files like `acoustic_gtr.wav`,
`electric_lead.wav`, `harmonica.wav`, `seaboard_pad.wav`, `vox.wav`. See
`config/default.yaml` for the role table.

## Status

Implemented: IO, analysis, DC/noise cleanup, gate, biquad EQ chain, compressor,
pan, mixdown, master chain (EQ, glue comp, true-ish peak limiter, loudness
normalization), rule-based advisor, LLM advisor, CLI.

Stubbed with `# TODO` (raise/pass, safe to run past):
- `pitch.py` — pitch correction. Needs a real pitch tracker + PSOLA/WORLD.
- `timing.py` — transient alignment and quantization.
- `separation.py` — Demucs stem separation for already-mixed input.
- `render_midi.py` — render Seaboard MIDI via a soundfont.
- `reverb.py` — convolution reverb send bus.

## Caveats

- The limiter is a simple lookahead peak limiter, not an ISP-accurate true-peak
  limiter. Don't master a release with it.
- Loudness targeting without `pyloudnorm` falls back to an RMS approximation,
  which is not LUFS.
- The LLM advisor's plan is unvalidated taste. `mix_plan.json` is clamped to
  sane ranges in `plan.py`, but a bad plan still sounds bad.
