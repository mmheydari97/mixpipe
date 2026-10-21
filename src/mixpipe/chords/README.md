# Chord finder

Upload a song, or paste a link to one (YouTube, SoundCloud, Spotify, …), and
get an editable chord sheet:

- lyrics transcribed **locally** (Whisper on your Mac), editable in the page
- chords placed on the word where they change, each with a dropdown holding the
  2nd and 3rd candidates, a "hear it" button and a free-text option
- the song's opening key and time signature (2/4, 3/4, 4/4, 6/8) at the top; if
  the song changes key, the change is marked on the chord where it happens
- transpose, capo (with a suggestion for open guitar shapes), bar lines, and
  export to ChordPro or plain text
- a **Harmonica** tab: the sung melody as tab for a 12-hole chromatic or a 10-hole
  diatonic harmonica in any key, with hole, blow or draw, and button or bends for
  every note, each editable like the chords (split, merge and add notes too)
- chord shapes for **guitar, ukulele, baritone ukulele** and guitars tuned down, with
  a capo suggestion for each
- optional **neural chord recognition** (the BTC transformer) combined with the
  standard recognizer

Works with a full song (vocals + instruments), a single instrumental track,
several stems uploaded together, or a link.

To mute the vocals or the guitar and play along instead, see the
[stem splitter](../splitter/README.md), a separate app with the same install.

All commands below are run from the **repository root** (the folder with
`Makefile` in it).

---

## 1. Requirements

| What | Why |
|---|---|
| macOS on Apple Silicon (M1-M4) | the default speech-to-text, `mlx-whisper`, runs on the Apple GPU. Other machines: see [Troubleshooting](#7-troubleshooting) |
| Python **3.12** | torch, mlx and librosa don't ship wheels for 3.14 betas |
| `ffmpeg` | decodes mp3 / m4a / aac uploads: `brew install ffmpeg` |
| `deno` (optional) | recommended by yt-dlp for YouTube links: `brew install deno`. Without it YouTube still worked in testing but yt-dlp warns that some formats may be missing |
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

(`make install-chords` does the same. The `make` targets use `.venv` automatically,
so they work without activating it; pass extra flags with `ARGS`, e.g.
`make chords-ui ARGS="--port 8766"`.)

Check it worked:

```bash
PYTHONPATH=src python -c "import librosa, demucs.api, mlx_whisper, fastapi, yt_dlp; print('ok')"
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

   **Or paste a link** in the box under it (files and a link are either/or):
   - **YouTube, SoundCloud, Bandcamp, Vimeo** and the other sites
     [yt-dlp supports](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).
   - **A direct link** to an audio file (`https://…/song.mp3`).
   - **Spotify track links.** Spotify audio is DRM protected and can't be
     downloaded, so the track's title, artist and length are read from the
     Spotify page and the YouTube upload closest in length is used instead.
     The song page says which upload it picked; if it's the wrong version,
     paste a YouTube link yourself.
   - One song per link: playlists, albums and live streams are refused, and
     so is anything over 30 minutes.
   - Downloading may be against a site's terms of service. Use it for music
     you're allowed to download.
2. **Lyrics language**: Auto-detect, a specific language (better accuracy,
   Persian and Arabic display right-to-left), or *No lyrics (instrumental)*.
3. **Transcription**: *Local Whisper* (free, private, default) or *OpenRouter*
   (see [section 5](#5-optional-openrouter)).
4. Keep **Separate vocals and drums first** ticked for full songs.
5. Press **Analyze** and watch the stages: (Download →) Decode → Separate →
   Lyrics → Chords → Melody → Align. Time grows with song length; a 40-second test song took
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

### Harmonica tab

Switch with the **Chords | Harmonica** tabs above the sheet (the link
`#/song/<id>/harmonica` opens it directly). Pick the harmonica in the **Harp**
group: a **12-hole chromatic** (solo tuning, C4 to C♯7 on a C harp) or a **10-hole
diatonic** (Richter), and its **key**. Any of the 12 keys works; a G harp sits a
fourth below a C harp, a D♭ harp a semitone above.

```
chromatic  hole   1  2  3  4 | 5  6  7  8 | 9 10 11 12
(C harp)   blow   C  E  G  C | C  E  G  C | C  E  G  C
           draw   D  F  A  B | D  F  A  B | D  F  A  B      button in = one semitone up

diatonic   hole   1  2  3  4  5  6  7  8  9  10
(C harp)   blow   C  E  G  C  E  G  C  E  G  C
           draw   D  G  B  D  F  A  B  D  F  A              bends fill in the missing notes
```

- Each note shows its name over its tab: **`3↑`** = hole 3 blow, **`3↓`** = hole
  3 draw, **`●`** = button pressed in (`3↓●` = B♭ on the chromatic). On the diatonic,
  primes mark a bend: **`2↓′`** half step, **`2↓″`** whole step (that is F4 on a C
  harp), **`3↓‴`** a step and a half. Overblows and overdraws are not modelled, so a
  few semitones (E♭ in the low octave of a C harp, for example) are unplayable
  and shown with a red dashed border.
- Under the tabs, **Diatonic harps that suit this melody** ranks the 12 keys by how
  much of the melody they play without bending (and says which position that
  makes: 1st straight, 2nd cross harp, 3rd slant...). Click one to switch.
- Notes sit over the lyric words like chords do. A held word can carry several
  notes, over the right syllable, and the gaps between lines become rows of notes
  in bars.
- **Click a note** to open its menu:
  - the three candidates, each with its tab, confidence and **▶ hear** (plays
    that bit of the song with the note on top)
  - **Play as**, for notes with more than one fingering (F = `2↓` or `2↑●`, G on
    the diatonic = `2↓` or `3↑`). By default the tab uses no bend and no button when
    it can, then the hole nearest the previous note
  - a box for typing any note: `F#`, `Bb`, `C5`. Without an octave, the nearest
    one is used
  - **✂ Split** the note in two (at the playhead if it is inside the note, else in
    the middle) when the tracker held two repeated notes as one, and **⇥ Merge
    with next** to join a note that was cut in two
  - **✕ Not a note** for anything that isn't really part of the melody
    (restore it from the same menu)
- **+ Note** adds a note the tracker missed, starting at the playhead.
- **Octave − / +**: the melody is moved by whole octaves so as much of it as
  possible fits the harmonica (picked automatically from the sung notes).
  Change it if you'd rather play higher or lower; *automatic* goes back.
  **Transpose** applies here too; capo doesn't.
- **Fast notes** (tick it before *Find melody*) keeps grace notes and quick runs
  down to about 60 ms. It can add the odd false note.
- Where nobody sings, the notes are a guess from the instruments (a salience tracker
  that follows the top line over the chords), shown faded. Expect to fix those, or
  remove them.
- **Find melody** appears for songs analysed before the harmonica tab existed.
  It takes a few seconds per minute of audio and keeps your chord and lyric
  edits. **Re-detect melody** (click twice) finds the notes again and replaces
  your note edits on this tab; the harp you picked stays.
- **Harmonica tab** downloads a plain-text version (`<song-id>.harp.txt`).

### Other options on the page

- **Instrument** (Chords tab): guitar, guitar tuned down a half or whole step,
  ukulele (GCEA or D tuning), baritone ukulele. The chord shapes shown, the capo
  and its suggestion follow it; the sounding chords don't change.
- **Chord recognition** (new-song form): *Standard* matches chroma against chord
  templates. *Neural* also runs BTC (a pretrained transformer, MIT licence, by Park
  et al.) and combines both. The first time it downloads 12 MB into
  `~/.cache/mixpipe/models/btc` (or `$MIXPIPE_MODEL_DIR`) after checking a pinned
  SHA-256; `python -m mixpipe.cli fetch-model btc` does that ahead of time. If it can't
  be used the standard recognizer runs and the page says so.
- **Key changes** (new-song form): *Normal* marks modulations of about 8+ bars,
  *Sensitive* also 3-4 bar excursions (their key is less certain).
- **Meter changes**: if the song moves between 2, 3 and 4 beats per bar, the badge row
  says where ("3/4 from bar 9") and the bar lines follow. Setting the time signature by hand
  replaces the detected changes; *reset* restores them.
- **Cancel** a queued or running analysis from its progress panel; the queue shows how
  many jobs are ahead. `chords-ui --workers 2` runs two songs at once if you have the memory.

## 4. Headless (no browser)

```bash
make chords FILE="path/to/song.m4a" LANG_CODE=en
# or
PYTHONPATH=src python -m mixpipe.cli chords path/to/song.m4a --lang en
# stems: list several files
PYTHONPATH=src python -m mixpipe.cli chords vox.wav guitar.wav bass.wav
# a link instead of a file (quote it: links often contain & or ?)
make chords FILE="https://www.youtube.com/watch?v=..." LANG_CODE=en
PYTHONPATH=src python -m mixpipe.cli chords "https://open.spotify.com/track/..."
```

| Flag | Default | |
|---|---|---|
| `--lang` | `auto` | ISO code (`en`, `fa`, `es`, …) or `none` for instrumentals |
| `--stt` | `mlx` | `mlx` = local Whisper, `openrouter` = cloud |
| `--stt-model` | | OpenRouter model id, required with `--stt openrouter` (e.g. `openai/whisper-large-v3-turbo`) |
| `--no-separate` | off | skip Demucs on a single full-mix file |
| `--recognizer` | `template` | `neural` also asks the BTC model (12 MB, downloaded once) |
| `--key-changes` | `normal` | `sensitive` marks shorter modulations |
| `--no-retime` | off | keep estimated word times from text-only OpenRouter models |
| `--title` | file name / link title | song title |
| `--out` | `out/chords` | song store directory |
| `--quiet` | off | no progress output |

It prints the chord sheet and writes `sheet.json`, `<song-id>.cho`,
`<song-id>.txt` and the harmonica tab `<song-id>.harp.txt` into the song's
folder. Songs analysed this way also show up in the web UI.

## 5. Optional: OpenRouter

Only needed for cloud transcription or *Polish lyrics*. Set the key in the
same terminal **before** starting the UI:

```bash
export OPENROUTER_API_KEY=sk-or-...      # or put it in .env (copy .env_template); both UIs read it
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
  meta.json         title, upload date, input file names, and the link it came from
  inputs/           your uploaded files, or the downloaded audio.m4a, untouched
  mix.flac          playback audio (only for multi-file or non-browser formats)
  separated/        vocals / bass / other / drums from Demucs
  transcript.json   cached lyrics
  chords.json       cached chord analysis
  melody.json       cached melody notes (harmonica tab)
  sheet.json        the sheet with all your edits
  status.json       progress of the last analysis
```

- Separation and transcription are cached, so re-runs are quick.
- Uploading the same file (or pasting the same link) again opens the existing
  sheet and keeps your edits.
- Delete a song with **Delete** in the song list (click twice), or remove its
  folder.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `pip` can't find wheels for torch / mlx / numba | the venv isn't Python 3.12: `python --version`, then recreate it as in step 2 |
| `No module named mixpipe` | run from the repo root with `PYTHONPATH=src`, or use the `make` targets |
| `can't decode this format without ffmpeg` | `brew install ffmpeg` |
| `downloading from links needs yt-dlp` | `uv pip install -r requirements-chords.txt` again |
| a YouTube link fails with *Sign in to confirm you're not a bot* or a format error | update yt-dlp (`uv pip install -U yt-dlp`; YouTube changes often), install `deno`, or download the audio yourself and upload the file |
| a Spotify link picked the wrong version (live, cover, …) | paste the YouTube link of the version you want instead |
| first transcription seems stuck | it's downloading Whisper (~1.6 GB) once |
| `You are sending unauthenticated requests to the HF Hub` | harmless; setting `HF_TOKEN` only speeds up the download |
| `address already in use` | another server is on 8765: `... chords-ui --port 8766` |
| *OPENROUTER_API_KEY isn't set* in the page | put the key in `.env` (copy `.env_template`) or export it, then **restart** the UI |
| lyrics include words that were never sung | pick the language explicitly instead of Auto-detect, or fix them with *Edit lyrics* |
| not on Apple Silicon | `mlx-whisper` won't install: delete its line from `requirements-chords.txt`, install as in step 2, then `pip install faster-whisper`. Transcription then runs on the CPU (slower; untested on other machines) |

## 8. Tests

```bash
make test
# or just the chord finder
PYTHONPATH=src python -m pytest tests/test_chords.py -q
```

No models are downloaded and no network is needed: separation, transcription and
the neural model are faked in the tests (`tests/test_neural.py` uses random weights and
a `file://` download).

Known limitations (accuracy, meter, key changes, melody) are listed in
[`TODO.md`](../../../TODO.md#chord-finder).
