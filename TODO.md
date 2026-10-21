# Outstanding work

Three apps share this repository. Items are grouped by the app they belong to.

| App | What it is | Section |
|---|---|---|
| **Mixer** | stems in, mixed and mastered WAV out (`mixpipe.cli mix`), or by chat (`mix-ui`, `chat`) | [Mixer](#mixer) |
| **Chord finder** | song to chord sheet, plus the **harmonica tab** (`chords-ui`) | [Chord finder](#chord-finder), [Harmonica](#harmonica) |
| **Stem splitter** | song to six stems, practice mixer (`split-ui`) | [Stem splitter](#stem-splitter) |

Shared code (`songstore.py`, `fetch.py`, `openrouter.py`, `web/common.py`) is listed under the app that runs into the limit first.

Every `# TODO` stub that used to be in the mixer has been implemented and tested.
`grep -rn "TODO" src` finds none.

---

## Mixer

Nothing is stubbed any more. What is left are limits of what exists:

- **Pitch correction is for one note at a time.** It tracks f0 and re-spaces
  pitch-synchronous grains (TD-PSOLA), which keeps a voice's formants. It cannot
  retune chords; use `pitch.shift_semitones`, which moves everything together.
  The built-in phase vocoder behind that is good, not Rubber Band good; with the
  `rubberband` binary installed it is used automatically.
- **`render_midi`** needs the `fluidsynth` binary and a SoundFont. MPE is handled
  by restoring each channel's instrument and +/-48 semitone bend range; slide and
  pressure only do what FluidSynth's default modulators map them to. It was tested
  against a fake `fluidsynth`, not a real one (not installed here).
- **The limiter** is true-peak aware (4x oversampled, checked after limiting) but
  still a plain lookahead design. Fine for demos, not a mastering-grade limiter.
- **Without `pyloudnorm`**, "LUFS" values are RMS approximations with a fixed offset.
- **`declick`** is off by default (`cleanup.declick: true` to use it). It only touches
  runs shorter than 2 ms, so a click longer than that is left alone on purpose.
- **Dehum** needs two clear harmonics in the lowest four to act; a hum with a
  single visible line is treated as a note and left alone.
- **`--notes` in the rules advisor** is keyword-based (instrument + direction, space,
  tone words). It reports what it did not understand; anything subtler needs
  `--advisor llm`.
- **The critique loop** (`--iterate`) can only fix what it can measure: tonal
  balance against a reference, true peak, limiter squash, mono compatibility. Without
  a `--reference` it does not judge taste.
- **Multiband compression** exists but is off by default (`master.multiband.enabled`).
- Time alignment assumes the stems share the same material or rhythm. Stems with
  no onsets, or an unclear correlation (under 6 sigma), are never moved. Among
  correlation peaks nearly as tall as the best one, the smallest shift wins, because
  rhythmic parts also line up a whole beat away. An offset is found to within about
  3 ms between different instruments (37 ms found as 40 ms); it is exact to the
  sample only when two stems carry the same source.
- **Mix by chat** (`mix-ui`, `chat`) changes what the plan can express, judged from
  measurements, never by listening. Asked for something with no parameter ("like
  the eighties"), the model says so; the keyword fallback lists what it didn't
  understand. Sessions live in the server's memory: restarting `mix-ui` keeps the
  rendered versions on disk (`out/mixer/<id>/versions/`, with `session.json`) but
  doesn't reload them as sessions.
- **Stereo stems** keep their image; pan on them is a balance control, and there is no
  per-stem width setting in the plan yet (`panning.width` exists).
- **Separated stems** are not raw takes: the rules advisor's "noisy take" SNR reading
  is really their dynamic range, so its high-pass raise is limited to top-line
  instruments (never bass, drums or keys). Stems with no music in them (a "bass" stem
  of a song without bass) are marked `silent` and never boosted by an advisor.
- **Verification** (`scripts/verify_mix.py`, `tests/test_session_e2e.py`) measures what
  can be measured: true peak, loudness, mono compatibility, the 16-bit grid, the reverb
  tail, the critique's balance error, and on the synthetic session the hum, clicks,
  bass offset and sibilance before and after. Whether a mix sounds good is still
  something to judge by ear.

## Chord finder

- **Accuracy is not measured here.** There is no labelled recording in the repo to
  benchmark on. The standard recognizer (chroma templates + HMM) is documented at
  roughly 65-75% top-1 on real pop; the optional **neural** option adds the
  pretrained BTC transformer and blends its probabilities with the templates
  (`--recognizer neural`, ~12 MB download once). It works on real music (checked on the
  BTC authors' example track) but whether the blend beats the templates on your
  songs is something to judge by ear; BTC has no inversion labels, so slash chords
  still come from the bass analysis.
- **2/4 vs 4/4** is musically ambiguous (4/4 wins ties); the Time selector overrides it.
- **Meter changes** are found among 2, 3 and 4 tracked beats per bar (a 5/4 song comes
  out as alternating 3 and 2). 6/8 songs get no section search. A hand-picked time
  signature replaces detected sections; "reset" brings them back. Sections can't be
  edited individually.
- **Pushes** (chords arriving a half beat early) are handled at half-beat precision;
  a sixteenth-note anticipation counts as on the beat.
- **Key changes** need ~8 bars to be noticed ("normal") or 3-4 ("sensitive", opt-in).
  The key read from a short excursion is ambiguous (D A Bm G fits D, A and G major).
  Moves between a major key and its relative minor are never counted.
- **Free time.** If the beat tracker cannot follow the music even after trying a looser
  and a windowed tracker, the chords are still found from the harmony alone, but
  there is no beat grid: no BPM, no bar lines.
- **Syllables** come from rules (vowel groups, English silent e and -ed, per-character for
  Chinese / Japanese / Korean). Persian and Arabic words are not split. Boundaries
  follow the vocal's loudness dips when the vocal stem was available.
- **Text-only transcription models:** word times are re-derived locally (matched against a
  local Whisper pass, else snapped to sung onsets). Without a local Whisper and
  with a vocal that has no clear onsets they stay estimated and are flagged "approx".
- **Capo / shape tables** are hand-made for guitar, ukulele (C and D tuning), baritone
  ukulele and guitars tuned down. No banjo, mandolin, open tunings, or custom shapes.
- **Queue:** one job at a time by default (`--workers N` for more). A running job can be
  cancelled but stops at its next progress report, so a long model call finishes first.

## Harmonica

- **Instruments modelled:** 12-hole chromatic (solo tuning) and 10-hole Richter diatonic,
  both in all 12 keys. Not modelled: overblows and overdraws (so a few semitones, such
  as E-flat in the low octave of a C harp, are unplayable on the diatonic), other
  tunings (country, Paddy Richter, natural minor), tremolo and octave harps, low
  and high variants beyond the key's octave.
- **Melody** is pYIN on the Demucs vocal stem, with a second range-clamped pass when it
  wanders. Breathy, harmonised or heavily processed vocals still give jittery notes,
  and an octave error is only fixed when the neighbouring notes make it obvious.
  "Fast notes" keeps grace notes down to ~60 ms but can add false notes.
- **Where nobody sings,** the notes come from a harmonic-salience tracker on the
  instrument stem (shown faded). It follows a lead over chords well in a clean mix;
  a loud residual bass or a dense arrangement degrades it.
- Notes can be re-picked, retyped, removed, split, merged and added. The tab is a
  fingering for the melody only, not an arrangement with chords.

## Stem splitter

Nothing tracked. Its limits are the separation model's (Demucs `htdemucs_6s`: acoustic
guitar often lands in *Other*, keys is the weakest stem) and are described in
[`src/mixpipe/splitter/README.md`](src/mixpipe/splitter/README.md#what-to-expect-from-the-separation).
It shares the job queue (`--workers`, cancel) with the chord finder.
