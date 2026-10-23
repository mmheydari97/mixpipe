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
[timing]    align stems in time (--align)
[analysis]  per-stem measurements  ------> [advisor] -> mix_plan.json
    |                                          |        (rules, or an LLM; reads --notes)
    v                                          v
[cleanup]   dc offset, hum notch, clicks, noise reduction, gate  <--'  (params from plan)
[eq]        biquad filter chain
[dynamics]  transient shaper -> compressor
[eq]        de-esser (split-band dynamic EQ)
[panning]   constant-power pan (mono stems) or balance (stereo stems) -> fader
    |
    v
[mixdown]   sum + shared reverb / delay send buses, headroom check
    |
    v
[master]    master EQ -> reference match -> multiband -> glue comp
            -> loudness target -> true-peak limiter -> dither (16-bit)
    |
    v
out/mix_master.wav  --->  [critique] measure the result, revise the plan, render again (--iterate)
                    --->  [chat]     "the acoustic guitar is too loud" -> revised plan -> new version
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only `numpy` and `scipy` are strictly required. `soundfile` is optional (there's
a stdlib `wave` fallback for 16/24/32-bit WAV). `pyloudnorm`, `noisereduce`, and
`anthropic` unlock better loudness metering, better denoise, and the LLM advisor.
`numba` makes the compressor, limiter, gate and dither loops ~100x faster (they
run as plain Python without it, and it is installed with the chord finder's
requirements anyway). `rubberband` (`brew install rubberband`) gives cleaner pitch
shifting, `fluidsynth` (`brew install fluid-synth`) renders MIDI.

## One page for everything

```bash
make ui         # the chord finder, the splitter and the mixer on http://127.0.0.1:8760
make studio     # the same, plus the ACE-Step server for the mixer's AI edits
```

The home page links to the three tools; each runs under its own path (`/chords/`,
`/splitter/`, `/mixer/`) with a switcher in its top bar. They share one library:
a song you split is in the mixer's "From the splitter" list, and a split song has a
"Mix these stems" link. Each tool also still runs on its own port (`make chords-ui`
on 8765, `split-ui` on 8766, `mix-ui` on 8767), but settings a page remembers
(models, engine, per-song mixes) are per address, so they don't carry over between
the single-tool ports and the home page.

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

# 6. Take notes in plain English; the rules advisor acts on them too
python -m mixpipe.cli mix --stems stems/ \
    --notes "can't hear the harmonica, vocals too loud, more reverb on the guitar, a bit dull"

# 7. Separate-pass takes? line them up first. Have a track you like? match its balance
python -m mixpipe.cli mix --stems stems/ --align --reference ~/Music/favourite.wav

# 8. Let it listen to its own result and fix it (up to 3 rounds; an LLM critique with --advisor llm)
python -m mixpipe.cli mix --stems stems/ --reference ~/Music/favourite.wav --iterate 3

# 9. 16-bit output for a CD or older players (TPDF dither; --noise-shaping for shaped dither)
python -m mixpipe.cli mix --stems stems/ --bit-depth 16

# 10. Mix a song the splitter separated, or just a slice of it while you experiment
python -m mixpipe.cli mix --stems out/splitter/<song id>/stems --start 60 --duration 45
```

Stem role detection is filename-based: name files like `acoustic_gtr.wav`,
`electric_lead.wav`, `harmonica.wav`, `seaboard_pad.wav`, `vox.wav`. See
`config/default.yaml` for the role table. The splitter's stems are covered too
(`guitar` is treated as electric guitar, `other` as keys). Stereo files stay stereo,
where pan works as a balance control (`mixdown.keep_stereo: false` folds them to
mono); a "stereo" file whose two sides are identical is treated as mono.

## Mix by chat

After the first mix, say what you hear and the mix changes:

```bash
make mix-ui                                   # web page on http://127.0.0.1:8767
make studio                                   # every tool on :8760, with ACE-Step for AI edits
make chat STEMS=stems/                        # the same in the terminal
make chat STEMS=out/splitter/<song id>/stems
```

```
mix> the acoustic guitar is too loud
  Pulling the acoustic guitar fader down 3 dB.
    - acoustic_gtr -3.0 dB
  v2: out/chat/mix_v2.wav  (-14.1 LUFS, -1.2 dBTP, rendered in 2.8 s)
mix> more reverb on the vocals
mix> /undo
```

How a turn works (`session.MixSession`, `advisor.revise_plan`):

- The model gets your message, the last few turns, the current plan and
  measurements of the rendered mix, including each stem's level in the mix.
  It never gets audio. It answers with **relative adjustments** ("acoustic_gtr
  gain_db -3"), not a new plan. The code applies them to the current plan and
  clamps them (`validate_plan`), so nothing you didn't mention can drift.
- What you are shown as changed (the chips: `acoustic_gtr -3.0 dB`) is computed
  from the difference between the two plans, not taken from the model's text. If
  the model says it did something and didn't, the reply says nothing changed.
- Level changes move the stem's **fader**, which comes after its compressor
  (`fader_db`), so "3 dB quieter" is 3 dB quieter in the mix. The input trim
  (`gain_db`) is left to the advisors' level matching.
- No API key, no network, or a reply that isn't valid JSON: the keyword parser
  that `--notes` uses takes over (instrument + direction, space, tone words), and
  says what it didn't understand.
- **Which model answers** is yours to pick. The page has a searchable OpenRouter
  picker when you start a mix (type `free` to put free models first; the last pick
  is remembered), and the gear button in the chat card switches it between turns.
  In the terminal, `/model` shows it, `/model <id>` switches, `/model rules` turns the
  model off, and `/models <words>` searches the catalogue. Each version records the
  model that made it. With nothing picked, the default is `advisor.openrouter_model`
  (`anthropic/claude-haiku-5.5`) when only `OPENROUTER_API_KEY` is set, otherwise the
  configured provider. The picker lists every model that can answer in text;
  ones that also return audio or images (`gpt-audio`, `gpt-5-image`, Lyria) carry
  a `+audio out` / `+image out` badge, and only their text is used. The catalogue
  is fetched again after an hour. Models that only work through OpenRouter's batch
  API (`:batch`) are left out of the picker. A turn takes the model's answer time plus
  2-4 s of rendering; a small free model may answer in a form the mixer can't use,
  and that message then goes to the keyword rules.
- Each turn is a new version; nothing is thrown away. Undo goes back to the version
  a turn was made from, and a new turn after an undo starts a branch. The web page
  has an A/B player (loudness-matched by default, so the louder version can't win
  just by being louder), faders and mutes that go through the same path, and
  Export. CLI commands: `/undo /redo /goto N /plan /levels /model [id] /models [words]
  /export [path] /quit`.
- **Tempo and key**: "a little slower" (5%), "slower" (10%), "much faster" (20%),
  "play it at 80%", "at 90 bpm" (the session measures the song's tempo), "up a
  semitone", "a whole step lower", "original key". The stems are stretched with
  Rubber Band before the mix (`stretch.py`, the code the splitter uses), with
  formants kept on vocals when the key moves, and each stem keeps its loudness so the
  balance doesn't shift. The first change at a new tempo takes a few seconds; turns
  at the same tempo reuse it. Reverb and delay are added after the stretch, so tails
  stay natural. The page shows Tempo (with BPM) and Key, and the A/B player compares
  two versions at the same point of the song even when their speeds differ.
- **Effects** (`fx.py`, the same code the splitter's effect knobs use): drive
  ("make the guitar gritty", "the lead is too distorted"), chorus ("a little chorus
  on the keys"), bass/treble on one stem ("more bass on the acoustic guitar"; a bare
  "more bass" is still the bass stem's level), the shared room ("bigger room") and
  echo ("1/8 note echo on the vocals" locks the echo to the song's tempo and follows
  it when you slow the song down; "longer echo", "more repeats"). Drive and chorus
  sit in each stem's chain after the de-esser, before its fader; reverb and echo are
  the shared send buses.
- Cleanup does not depend on the plan, so stems are cleaned once per session.
  Songs longer than 75 s are worked on as their loudest 45 s; Export renders the
  whole song with the current plan (measured on a 3:18 song with 6 stems: 39 s the
  first time, 18 s after that, since the full-length stems are then cleaned).

### AI edits (ACE-Step)

Chat changes the mix; an **AI edit** changes the music. A local
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) server rewrites part of
the version you are listening to from a prompt, and the result is the next version:

```bash
make install-acestep        # once: clones ../ACE-Step-1.5 and installs it (its own venv, ~11 GB)
make studio                 # ACE-Step + the home page with every tool; Ctrl+C stops both
make acestep                # or just the server, on http://127.0.0.1:8001
```

`make studio` (`scripts/studio.sh`) starts ACE-Step, waits until it answers,
then starts `make ui`. An ACE-Step that's already running is used
and left running. If ACE-Step isn't installed or fails to start, the mixer still
starts without AI edits, and the script says why (log: `out/logs/acestep.log`).
`ACESTEP_DIR` and `ACESTEP_PORT` override where it lives and listens.

```
mix> /ai 0:20-0:32 a driving rock drum beat with crash cymbals
  ACE-Step repainted 0:20-0:32 of v1 (acestep-v15-turbo, 37 s). ...
mix> /cover 0.7 lo-fi, tape saturation, dusty drums
```

- **Repaint** rewrites a time range and keeps everything else. The model returns
  the whole clip, but outside the range MixPipe keeps the original samples
  (`acestep.splice`, with 50 ms crossfades), so only the range changes.
  **Cover** re-styles the whole clip; a higher strength stays closer to it,
  a lower one reinterprets it more freely. What the model adds goes through the
  master's limiter at the plan's ceiling, since it skipped the master chain.
- The prompt describes the music you want in that stretch (style, instruments,
  mood), not an instruction like "make the drums punchier". ACE-Step generates
  new audio there, so voices and tones can drift from the original.
- An AI edit is audio, not a plan, so it is a branch of the history. A chat turn
  or a fader move while you listen to it is made from the plan it came from
  (shown as "Mixed from vN"), and doesn't include the edit. Export writes the
  edited audio as it is. In a preview session that's the 45 s preview; start the
  session over the whole song to edit and export all of it.
- The audio goes to the ACE-Step server on this machine (`ACESTEP_URL`), nowhere
  else. On an M4 Pro (24 GB) with the turbo model, repainting part of a 47 s clip
  took 21-37 s, plus about 2 minutes to load the model on the first edit.
- In the web page there is one chat box. The gear button in its header switches what
  your messages do: **Mix changes**, answered by the OpenRouter model you pick there or by
  the keyword rules, or **AI edit**, done by local ACE-Step. In AI edit, a row above the
  message box sets the range (with "← playhead" buttons) or the re-style strength, and the
  message is the prompt. Both kinds of turn share one history. In the terminal it's
  `/ai [start-end] <prompt>` and `/cover [strength] <prompt>`.

## Checking a mix

```bash
make demo-session          # stems_demo/: 8 stems with hum, clicks, sibilance, a late bass
make verify-mix            # every feature on, then measured: writes out/verify/<name>/report.md
make verify-mix STEMS=out/splitter/<song id>/stems ARGS="--start 60 --duration 45"
make verify-mix ARGS="--advisor llm --provider openrouter --model anthropic/claude-haiku-5.5"
```

`scripts/verify_mix.py` runs alignment, declick, multiband, the send buses,
reference matching (against the sum of the stems, which for separated stems is the
original song), two critique rounds and 16-bit shaped dither. It then checks true
peak, loudness, clipping, mono compatibility, the 16-bit grid, the reverb tail,
and that the critique did not make the balance worse. On the synthetic session it
also measures each flaw before and after: hum, clicks, the bass offset and
sibilance. `tests/test_session_e2e.py` runs the same checks on a 20 s version.

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
PYTHONPATH=src python -m mixpipe.cli chords song.m4a --recognizer neural  # also ask the BTC model
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
in the UI (needs `OPENROUTER_API_KEY`: `cp .env_template .env` and fill it in; `.env` is git-ignored and loaded automatically). Only OpenAI-compatible upstreams return
word timestamps; for the rest, timings are estimated and the sheet says so. Lyric
revision sends the lyric *text* (never audio) to an OpenRouter chat model, which reads the
whole song and fixes misheard lines (type `free` in the model search to put free models
first). Tick *Revise the lyrics with an LLM* on a new song to run it after transcription,
or use *Revise with LLM…* in the lyric editor to see a diff first.

Beyond that: chord changes are timed between the beats (so a chord that arrives early on
the "and of 4" gets its bar line right), songs that change time signature get meter
sections, tempo drift is followed, chords sit over the right *syllable* of a held word,
text-only transcription models get their word times fixed locally, and the shapes can be
shown for ukulele and other tunings. The Harmonica tab writes tab for a 12-hole chromatic
or a 10-hole diatonic (with bends) in any key and suggests the diatonic key that needs the
fewest bends. Details: [`src/mixpipe/chords/README.md`](src/mixpipe/chords/README.md).

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

## Stem splitter

The reverse of the mix pipeline: give it one song (a file or a link) and it
splits it into vocals, guitar, drums, bass, keys and other (Demucs
`htdemucs_6s`). A practice mixer in the browser has a volume / mute / solo
control per channel, 50%–300% speed without changing the key, key changes in
semitone (one-fret) steps, an A–B loop, and downloads of the mix you hear (e.g.
the song without guitar), of any stem, or of all stems in a zip, as MP3, WAV or FLAC.
It also has per-channel effects (bass, treble, drive, chorus, reverb, echo; the mixer's
own `fx.py`), the song's key and tempo, a metronome and count-in on its beats, its
chords and lyrics following playback (all three from the chord finder, run on the
splitter's stems), and recording yourself over it as extra channels. Speed and key changes are made with Rubber Band
on the server (*Accurate*), or by the browser for speed alone (*Fast*).

```bash
PYTHONPATH=src python -m mixpipe.cli split-ui                 # web UI on http://127.0.0.1:8766
PYTHONPATH=src python -m mixpipe.cli split song.m4a           # headless: six stereo FLAC stems
PYTHONPATH=src python -m mixpipe.cli split "https://youtu.be/..."
```

Same install as the chord finder. It's a separate app (own port, own
`out/splitter/` folder), sharing song storage (`songstore.py`), link downloads
(`fetch.py`) and the web plumbing (`web/common.py`, `static/common.*`) with it.
Full guide: [`src/mixpipe/splitter/README.md`](src/mixpipe/splitter/README.md).

## What's in the mix pipeline

| Stage | Module | What it does |
|---|---|---|
| Alignment | `timing.align_stems` | cross-correlates onset envelopes (so a bass and a drum part line up on their rhythm) and shifts by whole samples; sample-exact when two stems carry the same source. Stems it isn't sure about stay put |
| Tempo / quantize | `timing.estimate_tempo_bpm`, `quantize_onsets` | tempo from the onset autocorrelation; notes moved toward a grid with a pitch-preserving time warp |
| Cleanup | `cleanup` | DC, **mains hum** (auto-detects 50 vs 60 Hz and notches the series, only when it finds it), **clicks** (`declick: true` in the config), spectral denoise, gate |
| EQ | `eq` | biquad chain, **de-esser** and **dynamic EQ** built on phase-coherent Linkwitz-Riley crossovers |
| Dynamics | `dynamics` | compressor, **transient shaper** (attack / sustain), **multiband** compressor, lookahead limiter with a **true-peak** mode |
| Space | `reverb`, `mixdown.add_send_buses` | Freeverb-style reverb, convolution reverb (`convolve_ir`), feedback delay. Every stem feeds one **shared** reverb and delay at its own send level, so the close-miked sources sit in the same room |
| Stereo | `panning` | constant-power pan, mid/side width, mono-safe pseudo-stereo (`haas_spread`: all-pass decorrelation, the side cancels exactly in mono) |
| Pitch | `pitch` | YIN tracker, scale-snapping **pitch correction** for monophonic parts (retune speed keeps vibrato, formants stay put), semitone shifting of anything |
| Master | `master` | EQ, **reference matching** (average-spectrum difference applied as a linear-phase FIR), multiband, glue comp, loudness target, true-peak limiter, **dither** |
| Advisor | `advisor` | rule-based or LLM plan; both act on `--notes`; **`critique_mix`** measures the finished mix, compares it with a reference, and returns a revised plan. Stems with no music in them (a separated bass of a song without one) are never boosted |
| Chat | `session`, `advisor.revise_plan` | a mix you talk to: relative adjustments over a whitelist, clamped, diffed into change chips; versions with undo; a cleaned-stem cache so a turn only re-renders |
| MIDI | `render_midi` | FluidSynth rendering; `render_mpe` restores per-note bend range and instruments so a Seaboard capture keeps its glides |

Every number in a plan is clamped to a sane range in `plan.py` before it reaches
the DSP, whoever wrote it (the rules, an LLM, you in a text editor). Per-stem
plan fields beyond gain / EQ / compression / pan: `fader_db` (level after the
chain), `sends` (dB, -60 = off), `deess`, `transient`. Plan-wide: `send_buses` (room size, decay, delay time...)
and `master.multiband`. Defaults live in `config.py`; override them in
`config/default.yaml`.

## Also in this repo

The chord finder (`mixpipe.chords`, `mixpipe.web`), the stem splitter / practice
mixer (`mixpipe.splitter`, `web/splitter_server.py`), Demucs separation
(`separation.py`) and key detection (`pitch.detect_key`) are separate apps that
share the install; see their sections above.

## Caveats

- The limiter is true-peak aware (4x oversampled, checked after limiting), but
  it is still a straightforward lookahead design: fine for demos and releases
  you will check by ear, not a replacement for a mastering engineer.
- Loudness targeting without `pyloudnorm` falls back to an RMS approximation,
  which is not LUFS.
- The LLM advisor's plan is unvalidated taste. `mix_plan.json` is clamped to
  sane ranges in `plan.py`, but a bad plan still sounds bad. The rules advisor's
  `--notes` handling is keyword-based: it tells you which notes it acted on and
  which it did not understand.
- Chat changes what the plan can express: levels, pan, EQ, compression, de-essing,
  sends, mutes, the master's loudness, ceiling, EQ and glue, the reverb and delay
  buses (room size, pre-delay, echo time or note value, repeats), drive and chorus
  per stem, and the whole song's tempo and key. "Make it sound like the eighties" has no parameter; the model will say so
  rather than guess. It judges from measurements, not by listening.
- Pitch correction is for one note at a time (voice, harmonica, a lead line).
  Chords need `shift_semitones`, which moves everything together.
- `render_mpe` needs a SoundFont whose instrument responds to pitch bend; slide
  and pressure only change what FluidSynth's default modulators map them to.
