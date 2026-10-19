# Stem splitter

Give it a song (a file or a link) and it splits it into six channels:
**Vocals, Guitar, Drums, Bass, Keys, Other**. A page in your browser then lets
you rebalance them while you practise:

- a volume slider, **Mute** and **Solo** for every channel, plus one-click
  presets: *No vocals*, *No guitar*, *Guitar only*, *Vocals only*
- **change speed** from 50% (half speed) to 300% (3× faster) without changing the key
- **change key** in semitone (one-fret) steps: +1 = ♯, +2 = ♯♯, −1 = ♭, up to an octave either way
- **Fast** or **Accurate** stretching: the browser's instant one, or Rubber Band's much cleaner one
- an **A–B loop** to repeat a section
- **download** what you hear (e.g. the song without guitar) as MP3 or WAV,
  at the speed and key you picked

It's the reverse of MixPipe's mix/master pipeline, and a separate app from
the [chord finder](../chords/README.md) (its own port and song folder). Both
share the same install and code for song storage, links and the web page.

All commands below are run from the **repository root**.

---

## 1. Requirements and install

The same as the chord finder: see
[chord finder §1–2](../chords/README.md#1-requirements). If the chord finder
already works, there's nothing to install.

Short version (Apple Silicon Mac, Python 3.12, `brew install ffmpeg uv`):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-chords.txt
```

The first split downloads the 6-stem Demucs model (`htdemucs_6s`) once and
caches it.

## 2. Run the web UI

```bash
make split-ui
# or
PYTHONPATH=src python -m mixpipe.cli split-ui
```

Your browser opens **http://127.0.0.1:8766**. Stop it with `Ctrl+C`. The chord
finder (`make chords-ui`, port 8765) can run at the same time.

| Flag | Default | |
|---|---|---|
| `--port` | `8766` | use another port if that one is busy |
| `--host` | `127.0.0.1` | keep it local; `0.0.0.0` exposes it to your network |
| `--out` | `out/splitter` | where songs are stored |
| `--no-browser` | off | don't open a browser tab |

### Using the page

**Add a song:** drop **one** audio file (the full song) or paste a link
(YouTube, SoundCloud, Spotify, Bandcamp, a direct `.mp3` link, …; same rules as
the [chord finder](../chords/README.md#using-the-page)), then press **Split**.
Stages: (Download →) Decode → Separate. A 3.5-minute song took about 20 s on an
M4 Pro, download included.

**The mixer:**

- **Channels:** drag a slider to set a channel's volume. **M** mutes it, **S**
  solos it (with any solo on, only soloed channels play). Silent channels are
  greyed out.
- **Presets:** *Everything* resets all channels; *No vocals* for singing
  along; *No guitar* for playing along; *Guitar only* / *Vocals only* to
  learn a part.
- **Speed:** drag the slider from 50% to 300%, or use **−** / **+** (or the
  `-` / `+` keys); both move in 10% steps. *reset* goes back to 100%. The key
  stays the same.
- **Key:** drag the slider or use **♭** / **♯** (or the `,` / `.` keys) to
  move the song one semitone at a time, the same as one guitar fret: **+2 ♯♯**
  sounds like a capo on fret 2, **−1 ♭** like tuning down a half step. The
  speed stays the same, and vocals keep their natural tone (no chipmunk effect).
  A new key takes a few seconds to prepare (about 5 s for a 4–5 minute song).
  Until then the readout shows the new key dimmed, with *applying 45% · hearing
  the original key* beside it, then *✓ applied* once you hear it.
- **Stretch: Fast or Accurate.** How the speed change is made:
  - **Fast:** the browser stretches the audio itself. Instant, but away from
    100% it smears and warbles (the "noise" you may hear).
  - **Accurate** (the default): the server redoes the stems at that speed with
    [Rubber Band](https://breakfastquay.com/rubberband/). The browser's version
    plays meanwhile. A quick version replaces it after about 5 s, then the
    clean one after about 25 s (for a 4–5 minute song). The note next to the
    switch says which one you're hearing (*✓ … by Rubber Band* = the clean one).
    A setting you've used recently is ready at once.

  The browser can't change the key, so a key change always comes from Rubber
  Band. With *Fast*, the song is shifted at 100% and the browser adds the speed
  on top. Your choice is remembered for all songs.
- **Loop:** play to where the section starts and press **Set A**, then
  **Set B** where it ends. The loop switches on and shows as a band on the
  position bar. Untick *Loop A–B* to play through; *clear* removes it.
- **Download this mix:** MP3 (256 kbps) or WAV with your current volumes,
  mutes, solos, speed and key. The file name says what changed, e.g.
  `song-no-guitar-70pct-up2st.mp3`. Downloads always use Rubber Band's clean
  engine, even with *Fast* selected. If you're already hearing the clean
  version, the download is instant. Otherwise it takes about 5 s per minute of song.
- **Keys:** `Space` play/pause, `←` / `→` jump 5 s, `[` / `]` set A / B,
  `L` toggle the loop, `-` / `+` speed ±10%, `,` / `.` key ♭ / ♯.

Your mixer settings (volumes, speed, key, loop) are remembered per song in this
browser.

### What to expect from the separation

The splitting is done by Demucs' 6-stem model. It's good, but not perfect:

- **Vocals, drums and bass** separate cleanly.
- **Electric guitar** usually separates well.
- **Acoustic guitar** partly lands in *Other*. For *No guitar*, also turn
  *Other* down if you still hear it.
- **Keys** (piano) is the model's weakest channel. Keyboards and synths often
  end up in *Other*.
- Every channel has a little leakage from the others, which you'll notice when
  you solo it.

## 3. Headless (no browser)

```bash
make split FILE="path/to/song.m4a"
make split FILE="https://www.youtube.com/watch?v=..."   # quote links
# or
PYTHONPATH=src python -m mixpipe.cli split path/to/song.m4a
```

| Flag | Default | |
|---|---|---|
| `--title` | file name / link title | song title |
| `--out` | `out/splitter` | song store directory |
| `--quiet` | off | no progress output |

It prints the paths of the six stem files. The song also shows up in the web UI.

## 4. Where songs are stored

```
out/splitter/<song-id>/
  meta.json      title, date, input file name, and the link it came from
  inputs/        your uploaded file, or the downloaded audio.m4a
  stems/         vocals / guitar / drums / bass / piano / other .flac (stereo, lossless)
  stems.json     stem list, duration, model
  status.json    progress of the last split
  versions/      stems at other speeds/keys, e.g. x070+02/ = 70%, 2 semitones up
```

- About 150 MB per 4-minute song: the stems are lossless so downloads don't
  lose quality. Versions add up to three more sets of stems (only the three
  most recently used are kept; slower ones are bigger). You can delete
  `versions/` at any time: it's remade when needed.
- Adding the same file (or link) again reuses the stems.
- Delete a song with **Delete** in the song list (click twice), or remove its
  folder.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `source separation needs Demucs` / `No module named ...` | install as in step 1, inside the 3.12 venv |
| first split seems stuck | it's downloading the 6-stem model once |
| `the splitter takes one file` | drop just the full song; stems from a DAW can be mixed in your DAW |
| no sound after pressing play | click the page once (browsers only start audio after a click) and check the master volume next to the position bar |
| *Download* fails with *ffmpeg* | `brew install ffmpeg` (needed for MP3) |
| *couldn't change the key: … needs Rubber Band* | `brew install rubberband` (Homebrew's ffmpeg normally installs it too) |
| speed changes sound smeary or warbly | switch *Stretch* to **Accurate** and wait for *✓ … by Rubber Band* |
| a link fails | see the link rows in the [chord finder troubleshooting](../chords/README.md#7-troubleshooting) |
| `address already in use` | `make split-ui ARGS="--port 8767"` |
| the page behaves like an older version | reload it; the server now marks scripts so browsers always fetch the current ones |

## 6. Tests

```bash
make test
# or just the splitter
PYTHONPATH=src python -m pytest tests/test_splitter.py -q
```

No models are downloaded and no network is needed: separation and link
downloads are faked. The speed/key tests run the real Rubber Band on
2-second test tones; they're skipped if neither `rubberband` nor `ffmpeg` is installed.
