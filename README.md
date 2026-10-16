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

## Chord finder

Upload a song (a full mix, or several stems) or paste a link (YouTube,
SoundCloud, Spotify, a direct audio URL, …) and get an editable chord sheet:
lyrics transcribed locally, chords placed on the word where they change, and a
dropdown on every chord with the 2nd and 3rd candidates plus a "hear it" button.

```bash
# Python 3.12 venv (torch, mlx-whisper, librosa have no wheels for 3.14 betas)
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-chords.txt

PYTHONPATH=src python -m mixpipe.cli chords-ui                  # web UI on http://127.0.0.1:8765
PYTHONPATH=src python -m mixpipe.cli chords song.m4a --lang fa  # headless: sheet.json + .cho + .txt
PYTHONPATH=src python -m mixpipe.cli chords "https://youtu.be/..."  # or a link
```

Full guide (install, using the page, CLI flags, troubleshooting):
[`src/mixpipe/chords/README.md`](src/mixpipe/chords/README.md).

How it works: a single full mix is split with Demucs, so the vocal stem feeds
speech-to-text and bass + other (no drums, no vocals) feed chord detection. With
several files, names decide the routing (`vox.wav` -> lyrics, `bass.wav` -> bass
line, the rest -> chords). Chords come from beat-synchronous CQT chroma scored
against harmonic-aware templates (maj, m, 7, maj7, m7, sus2, sus4, dim, aug,
slash chords from the bass), smoothed with an HMM and biased toward the detected
key. Each change keeps three ranked candidates.

Speech-to-text runs locally with `mlx-whisper` (large-v3-turbo, ~1.6 GB download
on first use). OpenRouter is an optional backend: pick any transcription model
in the UI (needs `OPENROUTER_API_KEY`). Only OpenAI-compatible upstreams return
word timestamps; for the rest, timings are estimated and the sheet says so. The
"Polish lyrics" button sends the lyric *text* to an OpenRouter chat model (type
`free` in the model search to put free models first) and shows a diff first.

Editing lyrics never loses chords: chords are anchored to time, kept words keep
their timestamps, and chords re-land on whatever words are sung at that moment.

Bars: the time signature (2/4, 3/4, 4/4 or 6/8) and where bar 1 starts are
inferred from where chord changes and accents fall and whether beats split in
twos or threes, and the chord smoothing prefers changes on downbeats. Instrumental passages are shown as bars (`| G | % | G D |`) and lyric
lines get a faint tick where each bar starts. Time signature and bar lines can be
corrected by hand.

Keys: the overview shows the key the song starts in. If it modulates, the
chords after the change are judged and spelled in the new key, and the first
chord of the new key carries a small "-> D major" marker (also in the exports). Transpose and capo only change what's displayed and exported
(chords are stored at concert pitch, and "hear" always plays the recording's
pitch); the UI suggests a capo when one would put clearly more of the song on
open guitar shapes.
Songs are stored in `out/chords/<song>/`.

## Status

Implemented: IO, analysis, DC/noise cleanup, gate, biquad EQ chain, compressor,
pan, mixdown, master chain (EQ, glue comp, true-ish peak limiter, loudness
normalization), rule-based advisor, LLM advisor, CLI.

Also implemented: chord finder (`mixpipe.chords`, `mixpipe.web`), Demucs
separation (`separation.py`), key detection (`pitch.detect_key`).

Stubbed with `# TODO` (raise/pass, safe to run past):
- `pitch.py` — pitch correction. Needs a real pitch tracker + PSOLA/WORLD.
- `timing.py` — transient alignment and quantization.
- `render_midi.py` — render Seaboard MIDI via a soundfont.
- `reverb.py` — convolution reverb send bus.

## Caveats

- The limiter is a simple lookahead peak limiter, not an ISP-accurate true-peak
  limiter. Don't master a release with it.
- Loudness targeting without `pyloudnorm` falls back to an RMS approximation,
  which is not LUFS.
- The LLM advisor's plan is unvalidated taste. `mix_plan.json` is clamped to
  sane ranges in `plan.py`, but a bad plan still sounds bad.
