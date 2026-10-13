# Chord finder

Upload a song and get an editable chord sheet:

- lyrics transcribed **locally** (Whisper on your Mac), editable in the page
- chords placed on the word where they change, each with a dropdown holding the
  2nd and 3rd candidates, a "hear it" button and a free-text option
- the song's opening key and time signature (2/4, 3/4, 4/4, 6/8) at the top; if
  the song changes key, the change is marked on the chord where it happens
- transpose, capo (with a suggestion for open guitar shapes), bar lines, and
  export to ChordPro or plain text

Works with a full song (vocals + instruments), a single instrumental track, or
several stems uploaded together.

All commands below are run from the **repository root** (the folder with
`Makefile` in it).

---

## 1. Requirements

| What | Why |
|---|---|
| macOS on Apple Silicon (M1-M4) | the default speech-to-text, `mlx-whisper`, runs on the Apple GPU. Other machines: see [Troubleshooting](#7-troubleshooting) |
| Python **3.12** | torch, mlx and librosa don't ship wheels for 3.14 betas |
| `ffmpeg` | decodes mp3 / m4a / aac uploads: `brew install ffmpeg` |
| `uv` (recommended) | creates the 3.12 environment: `brew install uv` or see https://docs.astral.sh/uv/ |
| ~3 GB free disk | torch + models |

The first analysis downloads two models and caches them for later runs:
Whisper large-v3-turbo (~1.6 GB, from Hugging Face) and Demucs `htdemucs`
(~80 MB). After that everything runs offline.

## 2. Install (once)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-chords.txt
```

(`make install-chords` does the same `pip install` inside an activated venv.)

Check it worked:

```bash
PYTHONPATH=src python -c "import librosa, demucs.api, mlx_whisper, fastapi; print('ok')"
```

> Every new terminal: `source .venv/bin/activate` first.
> The package isn't pip-installed, so plain `python -m mixpipe.cli` needs
> `PYTHONPATH=src` in front (the `make` targets add it for you).

## 3. Run the web UI

```bash
make chords-ui
# or
PYTHONPATH=src python -m mixpipe.cli chords-ui
```

Your browser opens **http://127.0.0.1:8765**. Stop the server with `Ctrl+C`.

| Flag | Default | |
|---|---|---|
| `--port` | `8765` | use another port if that one is busy |
| `--host` | `127.0.0.1` | keep it local; `0.0.0.0` exposes it to your network |
| `--out` | `out/chords` | where songs are stored |
| `--no-browser` | off | don't open a browser tab |

### Using the page

**Add a song** (top card):

1. Drop audio on the box or click to choose. Formats: wav, flac, aiff, ogg,
   mp3, m4a, aac, opus, webm.
   - **One file** is treated as a full song. Vocals are separated
     automatically and used for the lyrics; bass + other instruments are used
     for the chords.
   - **Several files** are treated as stems. Name them so the roles are
     obvious: `vox.wav` / `vocal.wav` feed the lyrics, `bass.wav` helps find
     slash chords, everything else feeds the chords, `drums.wav` is ignored.
2. **Lyrics language**: Auto-detect, a specific language (better accuracy,
   Persian and Arabic display right-to-left), or *No lyrics (instrumental)*.
3. **Transcription**: *Local Whisper* (free, private, default) or *OpenRouter*
   (see [section 5](#5-optional-openrouter)).
4. Keep **Separate vocals and drums first** ticked for full songs.
5. Press **Analyze** and watch the stages: Decode → Separate → Lyrics →
   Chords → Align. Time grows with song length; a 40-second test song took
   about 25 s on an M4 Pro (after the first-run downloads).

**On the chord sheet:**

- **Click a chord** to open its menu: the three candidates with confidence
  bars, **▶ hear** (plays that part of the song with the chord synthesised on
  top, so you can hear which one fits), and a box to type any other chord
  (e.g. `Cadd9`, `D/F#`). Empty + *Set* goes back to the detected chord.
  A dot on a chord means you picked an alternative; ✎ means you typed one.
- **Click a word** to jump the player there. While playing, the current chord
  and word are highlighted and the footer shows *Now* / *Next*.
- **Edit lyrics**: one line per row, blank line between verses. Chords stay
  at their times and re-land on the right words.
- **Polish lyrics…**: lets an OpenRouter chat model fix misheard words. You see
  the changes before anything is saved (needs an API key).
- **Transpose − / +**, **Capo**: change what's displayed and exported. When a
  capo would put clearly more of the song on open shapes, a *Try capo N* link
  appears. "▶ hear" always plays at the recording's real pitch.
- **Time**: 2/4, 3/4, 4/4, 6/8 (detected automatically; the badge says
  *detected* or *manual*).
- **Bar lines ◀ 1 beat / 1 beat ▶**: if the bar lines sit a beat before or
  after where you feel "1", nudge them. *reset* goes back to what was
  detected. *Show bar lines* hides the ticks in the lyrics.
- A **→ D major** tag on a chord means the song changes key there.
- **ChordPro** / **Text** download the sheet with your current choices.

Everything you change is saved immediately.

## 4. Headless (no browser)

```bash
make chords FILE="path/to/song.m4a" LANG_CODE=en
# or
PYTHONPATH=src python -m mixpipe.cli chords path/to/song.m4a --lang en
# stems: list several files
PYTHONPATH=src python -m mixpipe.cli chords vox.wav guitar.wav bass.wav
```

| Flag | Default | |
|---|---|---|
| `--lang` | `auto` | ISO code (`en`, `fa`, `es`, …) or `none` for instrumentals |
| `--stt` | `mlx` | `mlx` = local Whisper, `openrouter` = cloud |
| `--stt-model` | | OpenRouter model id, required with `--stt openrouter` (e.g. `openai/whisper-large-v3-turbo`) |
| `--no-separate` | off | skip Demucs on a single full-mix file |
| `--title` | file name | song title |
| `--out` | `out/chords` | song store directory |
| `--quiet` | off | no progress output |

It prints the chord sheet and writes `sheet.json`, `<song-id>.cho` and
`<song-id>.txt` into the song's folder. Songs analysed this way also show up
in the web UI.

## 5. Optional: OpenRouter

Only needed for cloud transcription or *Polish lyrics*. Set the key in the
same terminal **before** starting the UI:

```bash
export OPENROUTER_API_KEY=sk-or-...
make chords-ui
```

In the model search box, typing **free** puts free models first. At the time
of writing there are no free *transcription* models on OpenRouter (local
Whisper is the free option), but there are free chat models for *Polish
lyrics*. Only OpenAI-compatible transcription models (e.g. `openai/whisper-*`)
return word timings; with others the sheet is marked *word timing
approximate*. Polish lyrics sends only the lyric text, never audio.

## 6. Where songs are stored

```
out/chords/<song-id>/
  meta.json         title, upload date, input file names
  inputs/           your uploaded files, untouched
  mix.flac          playback audio (only for multi-file or non-browser formats)
  separated/        vocals / bass / other / drums from Demucs
  transcript.json   cached lyrics
  chords.json       cached chord analysis
  sheet.json        the sheet with all your edits
  status.json       progress of the last analysis
```

- Separation and transcription are cached, so re-runs are quick.
- Uploading the same file again opens the existing sheet and keeps your edits.
- Delete a song with **Delete** in the song list (click twice), or remove its
  folder.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `pip` can't find wheels for torch / mlx / numba | the venv isn't Python 3.12: `python --version`, then recreate it as in step 2 |
| `No module named mixpipe` | run from the repo root with `PYTHONPATH=src`, or use the `make` targets |
| `can't decode this format without ffmpeg` | `brew install ffmpeg` |
| first transcription seems stuck | it's downloading Whisper (~1.6 GB) once |
| `You are sending unauthenticated requests to the HF Hub` | harmless; setting `HF_TOKEN` only speeds up the download |
| `address already in use` | another server is on 8765: `... chords-ui --port 8766` |
| *OPENROUTER_API_KEY isn't set* in the page | export the key, then **restart** the UI |
| lyrics include words that were never sung | pick the language explicitly instead of Auto-detect, or fix them with *Edit lyrics* |
| not on Apple Silicon | `mlx-whisper` won't install: delete its line from `requirements-chords.txt`, install as in step 2, then `pip install faster-whisper`. Transcription then runs on the CPU (slower; untested on other machines) |

## 8. Tests

```bash
make test
# or just the chord finder
PYTHONPATH=src python -m pytest tests/test_chords.py -q
```

No models are downloaded and no network is needed: separation and
transcription are faked in the tests.

Known limitations (accuracy, meter, key changes) are listed in
[`TODO.md`](../../../TODO.md#chord-finder).
