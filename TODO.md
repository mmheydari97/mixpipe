# Outstanding work

Every stub below is safe to call - it returns its input unchanged or an empty
result, so the pipeline runs end to end today. Grep for `# TODO` to find them.

## High value, low effort

- **`master.match_reference()`** - match the tonal balance of a commercial
  reference track. Biggest single quality jump available. Easiest path is to
  shell out to `matchering` rather than implement it.
- **`reverb.slap_delay()`** and **`reverb.algorithmic_reverb()`** - four dry
  close-miked sources need shared space or the mix sounds pasted together.
  Freeverb topology is a couple of hours of work.
- **`cleanup.dehum()`** - notch cascade at 50/60 Hz harmonics. Trivial with
  `scipy.signal.iirnotch`, and single-coil electric guitar will need it.

## High value, real effort

- **`timing.estimate_offset_samples()` / `align_stems()`** - correlate onset
  envelopes, apply integer-sample shifts. Needed if takes were recorded in
  separate passes rather than against a click.
- **`eq.dynamic_eq()`** and **`eq.deesser()`** - split-band compression via a
  Linkwitz-Riley crossover. Unlocks a real de-esser and resonance control.
  The static notch currently in the vocal role treatment is a placeholder.
- **`advisor.critique_mix()`** - second LLM pass: analyse the *output*, compare
  against the plan that produced it and a reference track, return a revised
  plan. This is the iteration loop that would make the tool genuinely useful.
- **`dynamics.multiband_compress()`** - for the master bus.
- **`dynamics.transient_shaper()`** - two envelope followers at different time
  constants; their difference isolates transients.

## Hard, consider not doing yourself

- **`pitch.py`** (all but `detect_key`) - pitch tracking plus formant-preserving shifting.
  pYIN for detection, WORLD or PSOLA for shifting. Doing this badly sounds worse
  than not doing it. `pyworld` / `pyrubberband` are the pragmatic wrappers.
- **`render_midi.py`** - FluidSynth wrapper, plus MPE-aware rendering for the
  Seaboard. Practical workaround: bounce audio out of Equator and skip MIDI.
- **`cleanup.declick()`** - median-filter outlier detection is easy to write and
  hard to tune without false-positiving on pick attacks.
- **`panning.haas_spread()`** - works but comb-filters in mono. Wants an
  all-pass decorrelator instead.

## Known limitations of what IS implemented

- The limiter is sample-peak, not true-peak (no oversampling). Inter-sample
  peaks can exceed the ceiling after lossy encoding. Keep the ceiling at -1 dB.
- Without `pyloudnorm`, "LUFS" values are RMS approximations with a fixed
  offset. Fine for relative balancing, not a real loudness measurement.
- `analysis.masking_report()` is a crude band-overlap check with no temporal
  weighting, so it flags pairs that never actually play at the same time.
- `_smooth_gain` and the limiter's release loop are per-sample Python loops.
  Fine for a few minutes of audio, slow for an album. Vectorize or move to
  numba/Cython if it becomes a problem.
- The rule-based advisor does nothing with `--notes`; only the LLM advisor reads
  them.

## Chord finder

- **Accuracy.** Template + HMM recognisers get roughly 65-75% top-1 on real pop
  recordings, less on the extended qualities and on dense or heavily distorted
  arrangements. The candidate dropdown covers a lot of it; a trained model (BTC,
  or a CRNN on CQT) behind the same `recognize()` contract is the real fix.
- **Time signature** (2/4, 3/4, 4/4, 6/8) is a best guess from where chord
  changes and accents fall, plus whether beats split in 2s or 3s. 2/4 vs 4/4 is
  inherently ambiguous (4/4 wins ties). One time signature per song.
- **Bar lines can sit a beat off** when a song's chords habitually change just
  before "1" (a push). The UI's "Bar lines ◀ 1 beat / 1 beat ▶" fixes it.
- **Key changes** are tracked per chord over ~8-bar windows, so a modulation
  shorter than ~4 bars (a borrowed chord, a quick tonicisation) is deliberately
  ignored, and relative major/minor moves are not counted as key changes.
- **Tempo drift / rubato.** librosa's tracker assumes a roughly steady tempo; free
  time falls back to fixed 0.5 s frames.
- **Lyric timing from Whisper** is word-level but not syllable-level, so a chord
  change in the middle of a long held word shows on that word, not the syllable.
- **OpenRouter transcription** only returns word timing from OpenAI-compatible
  upstreams; others are flagged "timing approximate". A local forced-alignment
  pass (wav2vec2) would fix positions for any text source.
- **Capo suggestions are guitar-only** and use a fixed list of open shapes
  (F and Bm count half). No ukulele / alternate tunings.
- **One job at a time.** The worker is a single thread; fine for a personal tool.
