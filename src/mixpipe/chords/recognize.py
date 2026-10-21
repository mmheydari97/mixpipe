"""Chord recognition: beat-synchronous chroma -> template scores -> HMM -> top-3.

  1. HPSS keeps the harmonic part; beats come from librosa's tracker.
  2. CQT chroma (treble) and a low-octave chroma (bass) are aggregated per beat.
  3. Every label (12 roots x 9 qualities, plus N = no chord) is scored per beat:
     cosine against a root-weighted template, plus a small bonus when the root is
     in the bass, minus a complexity penalty for extended qualities, plus a key
     prior for chords diatonic to the detected key.
  4. Viterbi over beats with a sticky self-transition picks a first path. Where
     its chord changes fall (and where the onsets are strongest) gives the meter
     (3 or 4 beats per bar) and the downbeat; a second Viterbi then makes
     changes cheaper on downbeats and dearer mid-bar.
  5. Runs of equal labels become segments; each segment ranks all labels by mean
     score and keeps three candidates (the Viterbi pick is always first).
  6. If the bass sits on a chord tone other than the root, the label becomes a
     slash chord (C/E).
  7. Every chord change is then located between the beats: chroma windows before
     and after it are matched against the two chords' templates, and the offset
     with the best fit is the change time. A change in the last half beat before a
     beat is a "push" and counts for that beat when the downbeat is judged, so a
     song whose chords arrive early on the "and of 4" still gets its bar lines on
     the right beat.
  8. A bar-pointer HMM over (meter, position in bar) checks whether one meter
     explains the whole song. When a section with another meter (a 3/4 bridge in a
     4/4 song) or an odd bar fits clearly better, the meter gets `sections`.

Typical top-1 accuracy of the template recognizer on real pop recordings is ~65-75%
(lower for the extended qualities). The candidates exist to cover the misses.

A trained model can take part: pass `neural=` (per-frame chord probabilities from
`btc.neural_probs`) and its probabilities, mapped onto this module's labels, are
blended into the template scores before the smoothing, so beats, meter, keys,
pushes, bass and the top-3 candidates all work from the better scores.
"""

from __future__ import annotations

import numpy as np

from ..io_utils import to_sample_rate
from ..pitch import KK_MAJOR, KK_MINOR
from .theory import TIME_SIGNATURES, note_names_for_key
from .theory import key_label as theory_key_label

SR = 22_050
HOP = 512

# suffix -> intervals above the root, in template order (root first)
QUALITIES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
}

# Subtracted from the score so a plain triad wins unless the colour tone is
# clearly there. Harmonics of a triad already leak a little 7th and 9th energy.
COMPLEXITY_PENALTY = {
    "": 0.0, "m": 0.0, "7": 0.03, "maj7": 0.05, "m7": 0.04,
    "sus2": 0.045, "sus4": 0.04, "dim": 0.05, "aug": 0.06,
}

ROOT_WEIGHT = 1.15     # root counts a bit more than other chord tones
HARMONIC_DECAY = 0.6   # overtone amplitude ratio assumed by the templates
BASS_WEIGHT = 0.10     # bonus scale for root-in-bass
INVERSION_FACTOR = 0.6 # fraction of that bonus for another chord tone in the bass
KEY_BONUS = 0.025      # bonus for chords diatonic to the detected key
NO_CHORD_SCALE = 0.85  # N score = this * cosine(chroma, flat)
TAU = 0.04             # softmax temperature for emissions and confidences
P_STAY = 0.75          # Viterbi self-transition probability per beat
SILENCE_DB = -40.0     # beat energy below this (re 95th pct) counts as silent

# Meter-aware smoothing: self-transition probability by position in the bar.
P_STAY_DOWNBEAT = 0.55   # chords like to change on beat 1...
P_STAY_HALFBAR = 0.72    # ...or half way through a 4/4 bar
P_STAY_OFFBEAT = 0.88    # ...and rarely anywhere else
# Meter priors (added to the score): 4/4 wins near-ties, 2/4 needs clear evidence
METER_PRIOR = {2: -0.12, 3: 0.0, 4: 0.04}
TERNARY_RATIO = 1.25     # onsets at 1/3+2/3 of a beat vs 1/2 above this -> 6/8 feel
FAST_PULSE_BPM = 150.0   # a 3-grouped pulse this fast is probably 6/8 eighths

# Blending in a neural model's probabilities
NEURAL_WEIGHT = 0.5      # share of the score that comes from the model (rest: templates)

# Chord change timing between beats
REFINE_REACH = 0.8       # search +/- this many beats around the beat-aligned boundary
REFINE_WINDOW = 0.25     # beats of chroma averaged either side of a candidate change
REFINE_MIN_SCORE = 0.12  # template-fit gain needed to move a boundary off its beat
PUSH_THRESHOLD = 0.35    # a change this far past a beat (in beats) belongs to the next beat

# Tempo drift: how well the tracked beats keep up with the music
SYNC_TOLERANCE = 0.15    # a beat "has an onset" when one lies within this share of a period
SYNC_OK = 0.90           # at least this share of beats on an onset: the grid is trusted
SYNC_FREE = 0.55         # below this even the best tracker fails: treat as free time
SYNC_MARGIN = 0.05       # another tracker must beat the default by this much to replace it
CHUNK_S, CHUNK_HOP_S = 16.0, 8.0   # windows for the drift-tolerant tracker
MIN_PSEUDO_BEAT_S = 0.45 # free time: the closest two segment boundaries may be

# Meter sections (bar-pointer HMM)
SECTION_METERS = (2, 3, 4)
SECTION_SWITCH_P = 0.015  # probability that a new bar changes meter
SECTION_MIN_GAIN = 7.0    # log-likelihood a free path must gain over one fixed meter
P_CHANGE_DOWNBEAT = 0.55  # chance a chord changes on a downbeat...
P_CHANGE_HALFBAR = 0.18   # ...on the middle of a 4-beat bar...
P_CHANGE_OTHER = 0.06     # ...and anywhere else

# Key tracking (modulations): windowed key estimate, smoothed by a sticky HMM
KEY_WINDOW_BEATS = 32    # ~8 bars of chord tones per local key estimate
KEY_EMISSION_SCALE = 3.0 # per-beat weight of the key correlation
KEY_P_STAY = 0.995       # per-beat probability of staying in a key
KEY_MIN_BEATS = 16       # shorter key regions are folded into a neighbour
# "sensitive" notices shorter modulations (3-4 bars; "normal" needs about 8) from a
# narrower window of chords. A key read from three or four chords is less certain
# (D A Bm G fits D, A and G major alike), and a borrowed-chord passage can be
# taken for a change, so it is opt-in.
KEY_PROFILES = {
    "normal": {"window": KEY_WINDOW_BEATS, "min_beats": KEY_MIN_BEATS, "p_stay": KEY_P_STAY},
    "sensitive": {"window": 12, "min_beats": 8, "p_stay": 0.98},
}


def diatonic_labels(tonic: int, mode: str) -> set[tuple[int, str]]:
    """(root, quality) pairs that belong to the key, incl. common colourings."""
    if mode == "min":
        # natural minor = relative major's set, plus harmonic-minor V and V7
        rel = (tonic + 3) % 12
        out = diatonic_labels(rel, "maj")
        five = (tonic + 7) % 12
        return out | {(five, ""), (five, "7")}
    degree = lambda n: (tonic + n) % 12  # noqa: E731
    return {
        (degree(0), ""), (degree(0), "maj7"), (degree(0), "sus2"), (degree(0), "sus4"),
        (degree(2), "m"), (degree(2), "m7"),
        (degree(4), "m"), (degree(4), "m7"),
        (degree(5), ""), (degree(5), "maj7"), (degree(5), "sus2"),
        (degree(7), ""), (degree(7), "7"), (degree(7), "sus4"),
        (degree(9), "m"), (degree(9), "m7"),
        (degree(11), "dim"),
    }


def _labels() -> list[tuple[int, str]]:
    return [(root, q) for q in QUALITIES for root in range(12)]


# Pitch-class offsets of harmonics 1-6 (octave, fifth, octave, major third, fifth)
_HARMONIC_PCS = (0, 0, 7, 0, 4, 7)


def _templates(labels: list[tuple[int, str]]) -> np.ndarray:
    """Harmonic-aware templates: each chord tone also contributes the pitch
    classes of its overtones, decaying by HARMONIC_DECAY per harmonic. Without
    this, the 3rd harmonic of a triad's third (the major 7th above the root)
    makes every major chord look like a maj7.
    """
    t = np.zeros((len(labels), 12))
    for i, (root, q) in enumerate(labels):
        for k, iv in enumerate(QUALITIES[q]):
            weight = ROOT_WEIGHT if k == 0 else 1.0
            for h, offset in enumerate(_HARMONIC_PCS):
                t[i, (root + iv + offset) % 12] += weight * HARMONIC_DECAY ** h
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def chord_pcs(root: int, quality: str) -> list[int]:
    return [(root + iv) % 12 for iv in QUALITIES[quality]]


def _beat_aggregate(feat: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Median of each feature row over every [bounds[i], bounds[i+1]) span."""
    out = np.zeros((feat.shape[0], len(bounds) - 1))
    for i in range(len(bounds) - 1):
        a, b = bounds[i], max(bounds[i + 1], bounds[i] + 1)
        out[:, i] = np.median(feat[:, a:b], axis=1)
    return out


def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    z = (x - x.max(axis=axis, keepdims=True)) / TAU
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def recognize(
    harmony: np.ndarray,
    sr: int,
    bass: np.ndarray | None = None,
    key_sensitivity: str = "normal",
    neural: dict | None = None,
    neural_weight: float = NEURAL_WEIGHT,
) -> dict:
    """Detect chords in `harmony` (mono or stereo; drums and vocals ideally
    removed). `bass` is an optional isolated bass signal for root/slash detection.

    Returns {"key", "key_label", "tempo", "beat_period", "beat_times", "meter",
    "segments": [...]}, where
    each segment is {"id", "start", "end", "beats", "candidates", "selected",
    "custom"} and candidates are [{"label", "conf", "root", "quality", "bass",
    "pcs"}] best first.
    """
    import librosa

    y = harmony if harmony.ndim == 1 else harmony.mean(axis=1)
    y = to_sample_rate(y.astype(np.float32), sr, SR)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-6:
        return {"key": None, "key_label": None, "tempo": 0.0, "beat_period": 0.5,
                "key_regions": [], "beat_times": [],
                "meter": {"time_signature": "4/4", "beats_per_bar": 4, "downbeat": 0,
                          "subdivision": "binary", "confidence": 0.0},
                "segments": []}
    y = y / peak

    y_harm = librosa.effects.harmonic(y, margin=2.0)
    n_frames = 1 + len(y) // HOP

    # ---- beats ---------------------------------------------------------------
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beats = np.asarray(beats, dtype=int)
    free_time = False
    if len(beats) < 4:
        beats, tempo, free_time = _pseudo_beats(onset, n_frames), 0.0, True
    else:
        beats, tempo, free_time = choose_beats(y, onset, beats, tempo, n_frames)
    bounds = np.unique(np.concatenate([[0], beats, [n_frames]]))
    bounds = bounds[bounds <= n_frames]
    times = librosa.frames_to_time(bounds, sr=SR, hop_length=HOP)
    times[-1] = len(y) / SR
    beat_period = float(np.median(np.diff(times[1:-1]))) if len(times) > 3 else 0.5

    # ---- features ------------------------------------------------------------
    # one tuning estimate (in fractions of a bin) shared by treble and bass, so a
    # quiet or silent bass stem can't throw its own estimate off
    tuning = float(librosa.estimate_tuning(y=y_harm, sr=SR, bins_per_octave=36))
    chroma = librosa.feature.chroma_cqt(
        y=y_harm, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C2"),
        n_octaves=6, bins_per_octave=36, norm=None, tuning=tuning,
    )
    chroma = chroma[:, :n_frames]
    chroma = np.log1p(100.0 * chroma / (chroma.max() + 1e-12))
    treble = _beat_aggregate(chroma, bounds)
    chroma_cs = np.concatenate([np.zeros((12, 1)), np.cumsum(chroma, axis=1)], axis=1)

    if bass is not None:
        yb = bass if bass.ndim == 1 else bass.mean(axis=1)
        yb = to_sample_rate(yb.astype(np.float32), sr, SR)
        yb = np.pad(yb, (0, max(0, len(y) - len(yb))))[: len(y)] / peak
        bass_octaves = 3
    else:
        yb, bass_octaves = y_harm, 2
    bchroma = librosa.feature.chroma_cqt(
        y=yb, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("E1"),
        n_octaves=bass_octaves, bins_per_octave=36, norm=None, tuning=tuning,
    )[:, :n_frames]
    bass_beats = _beat_aggregate(bchroma, bounds)
    bass_energy = bass_beats.max(axis=0)
    bass_norm = bass_beats / np.maximum(bass_energy, 1e-12)
    bass_norm[:, bass_energy < 1e-3 * (bass_energy.max() + 1e-12)] = 0.0

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0][:n_frames]
    beat_rms = _beat_aggregate(rms[None, :], bounds)[0]
    ref = np.percentile(beat_rms, 95) + 1e-12
    silent = 20.0 * np.log10(beat_rms / ref + 1e-12) < SILENCE_DB

    # ---- scores --------------------------------------------------------------
    labels = _labels()
    templates = _templates(labels)
    unit = treble / np.maximum(np.linalg.norm(treble, axis=0, keepdims=True), 1e-12)
    scores = templates @ unit                                  # (n_labels, n_beats)
    # Bass bonus: full for the root, partial for any other chord tone, so first
    # inversions (C/E) aren't simply out-voted by the chord rooted on the bass.
    bass_bonus = np.zeros((len(labels), 12))
    for i, (root, q) in enumerate(labels):
        bass_bonus[i, chord_pcs(root, q)] = INVERSION_FACTOR
        bass_bonus[i, root] = 1.0
    scores += BASS_WEIGHT * (bass_bonus @ bass_norm)
    scores -= np.array([COMPLEXITY_PENALTY[q] for _, q in labels])[:, None]
    neural_n = None
    if neural is not None and neural_weight > 0 and len(neural.get("probs", [])):
        mixed = neural_scores(neural, labels, times)              # (labels + 1, intervals)
        scores = (1.0 - neural_weight) * scores + neural_weight * mixed[:-1]
        neural_n = mixed[-1]

    # ---- keys ----------------------------------------------------------------
    # Krumhansl on raw chroma is skewed by overtones, so correlate against
    # first-pass chord tones instead (root counted extra), in a sliding window,
    # and let a sticky HMM decide where (if anywhere) the song modulates.
    voiced = ~silent
    tones = np.zeros((12, scores.shape[1]))
    for beat in np.flatnonzero(voiced):
        root, q = labels[int(np.argmax(scores[:, beat]))]
        for k, pc in enumerate(chord_pcs(root, q)):
            tones[pc, beat] += 1.5 if k == 0 else 1.0
    regions = key_regions(tones, **KEY_PROFILES.get(key_sensitivity, KEY_PROFILES['normal']))
    beat_key = np.zeros(scores.shape[1], dtype=int)
    for key_idx, a, b in regions:
        beat_key[a:b] = key_idx
    diatonic_mask = {}
    for key_idx in {r[0] for r in regions}:
        dia = diatonic_labels(*_key_of(key_idx))
        diatonic_mask[key_idx] = np.array([lab in dia for lab in labels], dtype=float)
    for key_idx, a, b in regions:
        scores[:, a:b] += KEY_BONUS * diatonic_mask[key_idx][:, None]

    flat = np.full(12, 1.0 / np.sqrt(12.0))
    no_chord = NO_CHORD_SCALE * (flat @ unit)
    if neural_n is not None:
        no_chord = (1.0 - neural_weight) * no_chord + neural_weight * neural_n
    no_chord[silent] = 1.5
    scores[:, silent] = 0.0
    scores = np.vstack([scores, no_chord[None, :]])            # N is the last row
    n_idx = len(labels)

    # ---- smoothing + meter ---------------------------------------------------
    log_em = np.log(_softmax(scores, axis=0) + 1e-300)
    n_int = scores.shape[1]
    first_path = _viterbi_sticky(log_em, np.full(n_int, P_STAY))

    beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    # beat index of every interval (-1 for a pickup before the first beat)
    beat_of = np.searchsorted(beat_times, times[:-1] + 1e-3, side="right") - 1
    accent = np.array([onset[max(0, f - 2): f + 3].max() if f < len(onset) else 0.0
                       for f in beats])
    flat_t = np.full(12, 1.0 / np.sqrt(12.0))

    def template_of(state: int) -> np.ndarray:
        return templates[state] if state < n_idx else flat_t

    beat_frames = max(4.0, beat_period * SR / HOP)
    reach, win = int(round(REFINE_REACH * beat_frames)), max(2, int(round(REFINE_WINDOW * beat_frames)))

    def refined_times(path_runs: list[tuple[int, int, int]]) -> list[float]:
        """Start time of every run: the first at its beat, each later one where the
        chroma switches from the previous run's chord to its own."""
        out = [float(times[path_runs[0][1]])]
        for k in range(1, len(path_runs)):
            (s0, a0, _), (s1, a1, b1) = path_runs[k - 1], path_runs[k]
            nominal = int(bounds[a1])
            lo = int(bounds[a0])
            hi = int(bounds[min(b1, len(bounds) - 1)])
            f, score = _refine_boundary(chroma_cs, template_of(s0), template_of(s1),
                                        nominal, lo, hi, reach, win)
            out.append(float(times[a1]) if score < REFINE_MIN_SCORE
                       else float(librosa.frames_to_time(f, sr=SR, hop_length=HOP)))
        return out

    first_runs = _runs(first_path)
    first_times = refined_times(first_runs)
    # a change counts for the beat it lands on, or the next one if it arrives early (a push)
    changes = []
    for (_, a, _), t_change in zip(first_runs[1:], first_times[1:]):
        if beat_of[a] >= 0:
            b = metric_beat(_beat_position(t_change, beat_times, beat_period))
            if 0 <= b < len(beat_times):
                changes.append(b)
    sections = []
    if free_time:
        # no steady pulse to hang bars on: the boundaries are just where the sound changes
        meter = {"time_signature": "4/4", "beats_per_bar": 4, "downbeat": 0,
                 "subdivision": "binary", "confidence": 0.0, "free_time": True}
    else:
        subdivision, _ = beat_subdivision(onset, beats)
        meter = estimate_meter(changes, accent, subdivision=subdivision, tempo=tempo)
        if meter["time_signature"] != "6/8" and meter["beats_per_bar"] in SECTION_METERS:
            sections = meter_sections(changes, accent, len(beat_times), meter["beats_per_bar"])
    meter["sections"] = sections
    if sections:
        # the single-meter fields describe the longest section, for anything that ignores sections
        extents = [(sections[k + 1]["beat"] if k + 1 < len(sections) else len(beat_times))
                   - sections[k]["beat"] for k in range(len(sections))]
        main = sections[int(np.argmax(extents))]
        meter["beats_per_bar"] = main["beats_per_bar"]
        meter["downbeat"] = main["beat"] % main["beats_per_bar"]
        meter["time_signature"] = TIME_SIGNATURES[main["beats_per_bar"]]

    m, p = meter["beats_per_bar"], meter["downbeat"]
    section_starts = [sec["beat"] for sec in sections]

    def bar_position(beat: int) -> tuple[int, int]:
        """(position in the bar, beats in that bar) of a tracked beat."""
        if not sections:
            return (beat - p) % m, m
        k = max(0, int(np.searchsorted(section_starts, beat, side="right")) - 1)
        mk = sections[k]["beats_per_bar"]
        return (beat - sections[k]["beat"]) % mk, mk

    stay = np.full(n_int, P_STAY if free_time else P_STAY_OFFBEAT)
    for i in range(0 if not free_time else n_int, n_int):
        if beat_of[i] < 0:
            continue
        pos, bar_len = bar_position(int(beat_of[i]))
        if pos == 0:
            stay[i] = P_STAY_DOWNBEAT
        elif bar_len in (4, 6) and pos == bar_len // 2:
            stay[i] = P_STAY_HALFBAR
    path = _viterbi_sticky(log_em, stay)

    # absorb short no-chord blips (< 2 beats) into the preceding chord
    runs = _runs(path)
    for k, (state, a, b) in enumerate(runs):
        if state == n_idx and b - a < 2 and k > 0:
            path[a:b] = runs[k - 1][0]
    runs = _runs(path)
    run_times = refined_times(runs)          # where each chord really starts, between the beats

    # ---- segments + candidates -------------------------------------------------
    segments = []
    for k, (state, a, b) in enumerate(runs):
        key_idx = int(beat_key[a])
        names = note_names_for_key(*_key_of(key_idx))
        mean_scores = scores[:, a:b].mean(axis=1)
        picked = [int(state)]
        for cand in np.argsort(-mean_scores):
            cand = int(cand)
            if len(picked) == 3:
                break
            if cand in picked or cand == n_idx:
                continue
            root = labels[cand][0]
            same_root = sum(1 for p in picked if p != n_idx and labels[p][0] == root)
            if same_root >= 2:
                continue
            picked.append(cand)
        conf = _softmax(mean_scores[picked])
        seg_bass = bass_norm[:, a:b].mean(axis=1)
        candidates = [
            _candidate(p, labels, names, seg_bass, float(c), n_idx)
            for p, c in zip(picked, conf)
        ]
        t_end = run_times[k + 1] if k + 1 < len(runs) else float(times[b])
        segments.append({
            "start": round(float(run_times[k]), 3),
            "end": round(float(t_end), 3),
            "beats": int(b - a),
            "candidates": candidates,
            "selected": 0,
            "custom": None,
            "key": _key_string(key_idx),
        })

    segments = _merge_equal_neighbours(segments)
    for i, seg in enumerate(segments):
        seg["id"] = f"c{i}"

    # Key regions snapped to chord boundaries: a modulation starts on a chord.
    # The overview shows the key the song starts in; changes live on the chords.
    key_regs: list[dict] = []
    for seg in segments:
        if key_regs and key_regs[-1]["key"] == seg["key"]:
            key_regs[-1]["end"] = seg["end"]
        else:
            key_regs.append({"start": seg["start"], "end": seg["end"], "key": seg["key"],
                             "key_label": theory_key_label(seg["key"])})
    initial = key_regs[0]["key"] if key_regs else _key_string(regions[0][0])
    return {
        "key": initial,
        "key_label": theory_key_label(initial),
        "key_regions": key_regs,
        "tempo": round(tempo, 1),
        "beat_period": round(beat_period, 3),
        # in free time the boundaries aren't beats: no grid, so no bar lines or BPM
        "beat_times": [] if free_time else [round(float(t), 3) for t in beat_times],
        "meter": meter,
        "segments": segments,
    }


def neural_scores(neural: dict, labels: list[tuple[int, str]], times: np.ndarray) -> np.ndarray:
    """A neural model's per-frame probabilities as scores for our labels, one column per
    beat interval: shape (len(labels) + 1, n_intervals), the last row being "no chord".

    BTC's qualities are folded onto the nine this module has (a minor-sixth counts as
    minor, a half-diminished seventh as diminished...), probabilities of the
    classes that fold together are added, and each interval takes the time-weighted
    mean of the frames it overlaps."""
    from .btc import N_INDEX, QUALITIES, TO_OURS, X_INDEX

    probs, frame_s = np.asarray(neural["probs"], dtype=float), float(neural["frame_s"])
    n_labels = len(labels)
    fold = np.zeros((probs.shape[1], n_labels + 1))
    index = {lab: i for i, lab in enumerate(labels)}
    for cls in range(12 * len(QUALITIES)):
        fold[cls, index[(cls // len(QUALITIES), TO_OURS[QUALITIES[cls % len(QUALITIES)]])]] = 1.0
    fold[X_INDEX, n_labels] = fold[N_INDEX, n_labels] = 1.0       # unknown counts as no chord
    ours = probs @ fold                                           # (frames, labels + 1)
    cum = np.vstack([np.zeros((1, ours.shape[1])), np.cumsum(ours, axis=0)])

    def at(frac: np.ndarray) -> np.ndarray:                       # cumulative value at a fractional frame
        f = np.clip(frac, 0, len(ours))
        lo = np.minimum(f.astype(int), len(ours) - 1)
        return cum[lo] + (f - lo)[:, None] * ours[lo]

    a, b = times[:-1] / frame_s, np.maximum(times[1:] / frame_s, times[:-1] / frame_s + 1e-6)
    return ((at(b) - at(a)) / (b - a)[:, None]).T


def _onset_peaks(onset: np.ndarray) -> np.ndarray:
    if onset.size < 12 or onset.max() <= 0:
        return np.zeros(0, dtype=int)
    import librosa

    return np.asarray(librosa.util.peak_pick(onset, pre_max=3, post_max=3, pre_avg=6, post_avg=6,
                                             delta=0.1 * float(onset.max()), wait=4), dtype=int)


def beat_sync(onset: np.ndarray, beats: np.ndarray) -> float:
    """Share of the tracked beats that have an onset within SYNC_TOLERANCE of a beat
    period. Near 1.0 for a grid that keeps up with the music; lower under tempo drift
    or in free time, where a constant-tempo grid wanders off the pulse. (Counting
    beats rather than onsets means subdivisions, eighth-note strumming, don't count
    against the grid.)"""
    peaks = _onset_peaks(onset)
    if len(beats) < 4 or len(peaks) < 4:
        return 1.0                                   # nothing to judge by: trust the grid
    period = float(np.median(np.diff(beats)))
    nearest = np.abs(np.asarray(beats)[:, None] - peaks[None, :]).min(axis=1)
    return float(np.mean(nearest < SYNC_TOLERANCE * period))


def _chunked_beats(y: np.ndarray, tempo0: float) -> tuple[np.ndarray, float]:
    """Beats tracked in overlapping windows, each with its own tempo estimate (started
    from the previous window's), so a gradual acceleration or slowing is followed.
    A window that jumps to double or half the previous tempo is folded back."""
    import librosa

    win, hop_s = int(CHUNK_S * SR), int(CHUNK_HOP_S * SR)
    out: list[float] = []
    prev = tempo0 or 120.0
    starts = list(range(0, max(1, len(y) - win // 2), hop_s)) or [0]
    for k, a in enumerate(starts):
        chunk = y[a:a + win]
        if len(chunk) < SR * 4:
            break
        t, bt = librosa.beat.beat_track(y=chunk, sr=SR, hop_length=HOP, start_bpm=prev,
                                        tightness=60, units="time")
        t = float(np.atleast_1d(t)[0])
        bt = np.asarray(bt, dtype=float) + a / SR
        ratio = t / prev if prev else 1.0
        if 1.8 < ratio < 2.2:                      # locked on the double: keep every other beat
            bt = bt[1::2] if len(out) and abs(bt[0] - out[-1]) < abs(bt[1] - out[-1]) else bt[::2]
            t /= 2.0
        elif 0.45 < ratio < 0.55 and len(bt) > 1: # locked on the half: add the beats between
            bt = np.sort(np.concatenate([bt, (bt[:-1] + bt[1:]) / 2.0]))
            t *= 2.0
        prev = t if 40 <= t <= 240 else prev
        lo = a / SR + (0.0 if k == 0 else CHUNK_S / 4)
        hi = a / SR + (CHUNK_S if k == len(starts) - 1 else 3 * CHUNK_S / 4)
        out += [b for b in bt if lo <= b < hi]
    if len(out) < 4:
        return np.zeros(0, dtype=int), tempo0
    out = np.array(sorted(out))
    keep = [out[0]]
    for b in out[1:]:
        if b - keep[-1] > 0.4 * np.median(np.diff(out)):
            keep.append(b)
    frames = librosa.time_to_frames(np.array(keep), sr=SR, hop_length=HOP)
    return np.unique(frames), prev


def _pseudo_beats(onset: np.ndarray, n_frames: int) -> np.ndarray:
    """Boundaries for free time: the strongest onsets, at least MIN_PSEUDO_BEAT_S apart."""
    gap = int(round(MIN_PSEUDO_BEAT_S * SR / HOP))
    peaks = _onset_peaks(onset)
    picked: list[int] = []
    for pk in sorted(peaks, key=lambda i: -onset[i]):
        if all(abs(pk - q) >= gap for q in picked):
            picked.append(int(pk))
    return np.array(sorted(picked), dtype=int) if len(picked) >= 4 else \
        np.arange(gap, n_frames, gap, dtype=int)


def choose_beats(y: np.ndarray, onset: np.ndarray, beats: np.ndarray, tempo: float,
                 n_frames: int) -> tuple[np.ndarray, float, bool]:
    """(beats, tempo, free_time): keep the default tracker's beats when they keep up with
    the music, try a looser and a windowed tracker when they don't, and fall back to
    onset-based segment boundaries (free_time=True: no bar grid) when none fits."""
    sync = beat_sync(onset, beats)
    if sync >= SYNC_OK:
        return beats, tempo, False
    import librosa

    best, best_sync, best_tempo = beats, sync, tempo
    t2, b2 = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, tightness=20)
    b2 = np.asarray(b2, dtype=int)
    if len(b2) >= 4 and beat_sync(onset, b2) > best_sync + SYNC_MARGIN:
        best, best_sync, best_tempo = b2, beat_sync(onset, b2), float(np.atleast_1d(t2)[0])
    b3, t3 = _chunked_beats(y, tempo)
    if len(b3) >= 4 and beat_sync(onset, b3) > best_sync + SYNC_MARGIN:
        best, best_sync, best_tempo = b3, beat_sync(onset, b3), float(t3)
    if best_sync < SYNC_FREE:
        return _pseudo_beats(onset, n_frames), 0.0, True
    return best, best_tempo, False


def _refine_boundary(cs: np.ndarray, t_old: np.ndarray, t_new: np.ndarray, nominal: int,
                     lo: int, hi: int, reach: int, w: int) -> tuple[int, float]:
    """Frame (and fit score) where the chroma switches from chord `t_old` to
    `t_new`, searched around frame `nominal` but not past `lo`/`hi` (the
    neighbouring boundaries). `cs` is the cumulative sum of the chroma along time
    with a leading zero column. The score of a split at f is how much more the
    `w` frames after it look like the new chord than the old one, plus how much
    more the `w` frames before it look like the old chord than the new one."""
    n = cs.shape[1] - 1
    a, b = max(lo + w, nominal - reach), min(hi - w, nominal + reach)
    if b < a:
        return nominal, 0.0
    f = np.arange(a, b + 1)
    before = (cs[:, f] - cs[:, np.maximum(f - w, 0)]) / w
    after = (cs[:, np.minimum(f + w, n)] - cs[:, f]) / w

    def cos(x, t):
        return (t @ x) / np.maximum(np.linalg.norm(x, axis=0), 1e-12)

    score = cos(after, t_new) - cos(after, t_old) + cos(before, t_old) - cos(before, t_new)
    k = int(np.argmax(score))
    return int(f[k]), float(score[k])


def _beat_position(t: float, beat_times: np.ndarray, period: float) -> float:
    """Time -> fractional beat index (2.5 is halfway between beats 2 and 3)."""
    if len(beat_times) == 0:
        return t / period
    i = int(np.searchsorted(beat_times, t, side="right")) - 1
    if i < 0:
        return (t - beat_times[0]) / period
    if i >= len(beat_times) - 1:
        return i + (t - beat_times[-1]) / period
    return i + (t - beat_times[i]) / (beat_times[i + 1] - beat_times[i])


def metric_beat(position: float) -> int:
    """The beat a change at fractional beat `position` counts for: the nearest, except
    that anything more than PUSH_THRESHOLD past a beat is a push toward the next."""
    return int(np.floor(position + 1.0 - PUSH_THRESHOLD))


def meter_sections(change_beats: list[int], accent: np.ndarray, n_beats: int,
                   base_m: int) -> list[dict]:
    """Where the meter changes, as [{"beat", "beats_per_bar"}] (the first section
    starts at the beat of its first downbeat, which can be negative: a pickup), or []
    when one meter fits the song about as well.

    A bar-pointer HMM: the state is (meter m in 2/3/4, position in the bar); position
    advances one beat at a time and a new bar may switch meter at a small cost.
    A beat emits "a chord change here / not here" with a high chance on the
    downbeat, a smaller one half way through a 4-beat bar and a low one elsewhere,
    plus a mild reward for onset accents on downbeats. The best free path is
    compared with the best path that never leaves `base_m`: it must win by
    SECTION_MIN_GAIN so noise doesn't invent meter changes."""
    if n_beats < 12 or len(change_beats) < 6 or base_m not in SECTION_METERS:
        return []
    changes = set(c for c in change_beats if 0 <= c < n_beats)
    acc = np.asarray(accent, dtype=float)[:n_beats]
    acc = acc / acc.mean() if acc.size and acc.mean() > 0 else np.ones(n_beats)
    states = [(m, p) for m in SECTION_METERS for p in range(m)]
    index = {s: i for i, s in enumerate(states)}

    def emission(i: int, state) -> float:
        m, pos = state
        pc = P_CHANGE_DOWNBEAT if pos == 0 else P_CHANGE_HALFBAR if (m == 4 and pos == 2) \
            else P_CHANGE_OTHER
        lp = np.log(pc if i in changes else 1.0 - pc)
        a = acc[i] if i < acc.size else 1.0
        mu = 1.3 if pos == 0 else 0.9
        return lp - 0.5 * ((a - mu) / 0.6) ** 2 * 0.4
    log_em = np.array([[emission(i, st) for st in states] for i in range(n_beats)])

    log_stay, log_switch = np.log(1.0 - SECTION_SWITCH_P), np.log(SECTION_SWITCH_P / 2.0)
    # predecessors of each state: (previous state index, log transition)
    preds: list[list[tuple[int, float]]] = [[] for _ in states]
    for (m, pos), si in index.items():
        if pos > 0:
            preds[si].append((index[(m, pos - 1)], 0.0))
        else:                                    # a new bar starts: from the last beat of any bar
            for m2 in SECTION_METERS:
                preds[si].append((index[(m2, m2 - 1)], log_stay if m2 == m else log_switch))

    def viterbi(allowed: np.ndarray) -> tuple[float, list[int]]:
        neg = -1e18
        score = np.where(allowed, log_em[0], neg)
        back = np.zeros((n_beats, len(states)), dtype=int)
        for i in range(1, n_beats):
            new = np.full(len(states), neg)
            for si in range(len(states)):
                if not allowed[si]:
                    continue
                best, arg = neg, 0
                for pj, lt in preds[si]:
                    v = score[pj] + lt
                    if v > best:
                        best, arg = v, pj
                new[si], back[i, si] = best + log_em[i, si], arg
            score = new
        last = int(np.argmax(score))
        path = [last]
        for i in range(n_beats - 1, 0, -1):
            last = int(back[i, last])
            path.append(last)
        return float(np.max(score)), path[::-1]

    free_ll, free_path = viterbi(np.ones(len(states), dtype=bool))
    fixed_ll, _ = viterbi(np.array([m == base_m for m, _ in states]))
    if free_ll - fixed_ll < SECTION_MIN_GAIN:
        return []

    sections: list[dict] = []
    first_m, first_pos = states[free_path[0]]
    bar_start, cur_m = -first_pos, first_m
    sections.append({"beat": bar_start, "beats_per_bar": cur_m})
    for i in range(1, n_beats):
        m, pos = states[free_path[i]]
        if pos == 0:
            if m != cur_m:
                sections.append({"beat": i, "beats_per_bar": m})
                cur_m = m
    return sections if len(sections) > 1 or sections[0]["beats_per_bar"] != base_m else []


def _key_of(key_idx: int) -> tuple[int, str]:
    """Key index 0-23 -> (tonic pc, "maj" | "min")."""
    return key_idx % 12, ("maj" if key_idx < 12 else "min")


def _key_string(key_idx: int) -> str:
    tonic, mode = _key_of(key_idx)
    return f"{note_names_for_key(tonic, mode)[tonic]}:{mode}"


def _key_label(key_idx: int) -> str:
    tonic, mode = _key_of(key_idx)
    return f"{note_names_for_key(tonic, mode)[tonic]} {'major' if mode == 'maj' else 'minor'}"


def _signature(key_idx: int) -> int:
    """Relative keys (C major / A minor) share a signature: same spelling, no
    real modulation between them."""
    tonic, mode = _key_of(key_idx)
    return tonic if mode == "maj" else (tonic + 3) % 12


def key_regions(tones: np.ndarray, window: int = KEY_WINDOW_BEATS,
                min_beats: int = KEY_MIN_BEATS, p_stay: float = KEY_P_STAY
                ) -> list[tuple[int, int, int]]:
    """Split the song into key regions from per-beat chord-tone profiles.

    tones: (12, n_beats). Returns [(key index 0-23, start beat, end beat)], at
    least one region. A key change needs `min_beats` of sustained evidence (judged
    over a `window` of beats); moves between relative major/minor are not treated
    as modulations.
    """
    n = tones.shape[1]
    if n == 0 or not tones.any():
        return [(0, 0, max(n, 1))]
    profiles = np.array([np.roll(KK_MAJOR, t) for t in range(12)]
                        + [np.roll(KK_MINOR, t) for t in range(12)])
    profiles = (profiles - profiles.mean(axis=1, keepdims=True))
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True)

    cs = np.concatenate([np.zeros((12, 1)), np.cumsum(tones, axis=1)], axis=1)
    half = window // 2
    idx = np.arange(n)
    lo, hi = np.clip(idx - half, 0, n), np.clip(idx + half, 0, n)
    win = cs[:, hi] - cs[:, lo]                                   # (12, n)
    win = win - win.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(win, axis=0, keepdims=True)
    corr = profiles @ (win / np.maximum(norms, 1e-12))            # (24, n)
    corr[:, norms[0] < 1e-9] = 0.0

    path = _viterbi_sticky(KEY_EMISSION_SCALE * corr, np.full(n, p_stay))
    regions = [list(r) for r in _runs(path)]

    def merge(i: int, j: int) -> None:
        """Merge region j into region i (adjacent); keep the longer one's key."""
        a, b = regions[i], regions[j]
        key = a[0] if a[2] - a[1] >= b[2] - b[1] else b[0]
        regions[min(i, j)] = [key, min(a[1], b[1]), max(a[2], b[2])]
        del regions[max(i, j)]

    changed = True
    while changed and len(regions) > 1:
        changed = False
        for i in range(len(regions) - 1):          # same signature -> one region
            if _signature(regions[i][0]) == _signature(regions[i + 1][0]):
                merge(i, i + 1)
                changed = True
                break
        if changed:
            continue
        shortest = min(range(len(regions)), key=lambda k: regions[k][2] - regions[k][1])
        if regions[shortest][2] - regions[shortest][1] < min_beats:
            neighbour = shortest - 1 if shortest > 0 else shortest + 1
            # the short region is absorbed whole: its neighbour's key wins
            keep = regions[neighbour][0]
            merge(min(shortest, neighbour), max(shortest, neighbour))
            regions[min(shortest, neighbour)][0] = keep
            changed = True
    return [(int(k), int(a), int(b)) for k, a, b in regions]


def beat_subdivision(onset: np.ndarray, beats: np.ndarray) -> tuple[str, float]:
    """"ternary" when onsets inside each beat fall on thirds (6/8 feel), else
    "binary". Returns (kind, thirds-vs-half onset ratio)."""
    thirds, halves = [], []
    for a, b in zip(beats[:-1], beats[1:]):
        span = b - a
        if span < 6:
            continue

        def at(frac: float) -> float:
            i = int(round(a + frac * span))
            return float(onset[max(0, i - 1): i + 2].max()) if i < len(onset) else 0.0

        halves.append(at(0.5))
        thirds.append(0.5 * (at(1 / 3) + at(2 / 3)))
    if not halves:
        return "binary", 1.0
    h, t = float(np.mean(halves)), float(np.mean(thirds))
    if h <= 1e-6 and t <= 1e-6:
        return "binary", 1.0
    ratio = t / max(h, 1e-6)
    return ("ternary" if ratio > TERNARY_RATIO else "binary"), round(ratio, 3)


def estimate_meter(change_beats: list[int], accent: np.ndarray,
                   subdivision: str = "binary", tempo: float = 120.0) -> dict:
    """Time signature (2/4, 3/4, 4/4 or 6/8), beats per bar, and the downbeat.

    Chord changes cluster on downbeats, so for each grouping of tracked beats
    (2, 3, 4) and phase, score how far the share of changes landing on that
    phase exceeds chance, plus a smaller term for onset accents there. Then:
    a ternary beat subdivision means the tracked beat is a dotted quarter, so
    2 or 4 groups read as 6/8; a fast pulse grouped in 3 with changes every 6
    tracked beats reads as 6/8 counted in eighths.

    Returns {"time_signature", "beats_per_bar" (tracked beats per bar),
    "downbeat" (index into beat_times), "subdivision", "confidence"}.
    """
    accent = np.asarray(accent, dtype=float)
    mean_acc = accent.mean() if accent.size and accent.mean() > 0 else 1.0
    n = len(change_beats)

    def score(m: int, p: int) -> float:
        sc = METER_PRIOR.get(m, 0.0)
        if n >= 3:
            sc += sum(1 for b in change_beats if (b - p) % m == 0) / n - 1.0 / m
        if accent.size >= m:
            sc += 0.25 * (accent[p::m].mean() / mean_acc - 1.0)
        return sc

    scored = sorted(((score(m, p), m, p) for m in METER_PRIOR for p in range(m)),
                    reverse=True)
    best, m, p = scored[0]
    runner = next((sc for sc, mm, _ in scored if mm != m), best)

    ts = TIME_SIGNATURES[m]
    if subdivision == "ternary" and m in (2, 4):
        ts, m, p = "6/8", 2, p % 2
    elif m == 3 and tempo >= FAST_PULSE_BPM and n >= 3:
        gaps = np.diff(sorted(set(change_beats)))
        if gaps.size and np.mean(gaps % 6 == 0) >= 0.7:
            # pick the 6-phase (p or p+3) where more changes land
            p6 = max((p, p + 3), key=lambda q: sum(1 for b in change_beats if (b - q) % 6 == 0))
            ts, m, p = "6/8", 6, p6
    return {"time_signature": ts, "beats_per_bar": int(m), "downbeat": int(p),
            "subdivision": subdivision, "confidence": round(float(best - runner), 3)}


def _viterbi_sticky(log_em: np.ndarray, p_stay: np.ndarray) -> np.ndarray:
    """Viterbi for a "stay, or jump uniformly to any other state" chain whose
    stay probability can differ per step. O(T * N) instead of O(T * N^2).

    log_em: (n_states, n_steps) log emission. p_stay[t]: probability of keeping
    the state when entering step t.
    """
    n_states, n_steps = log_em.shape
    idx = np.arange(n_states)
    back = np.zeros((n_steps, n_states), dtype=np.int64)
    delta = log_em[:, 0] - np.log(n_states)
    for t in range(1, n_steps):
        log_stay = np.log(p_stay[t])
        log_move = np.log((1.0 - p_stay[t]) / (n_states - 1))
        order = np.argpartition(-delta, 1)[:2]
        best, second = (order if delta[order[0]] >= delta[order[1]] else order[::-1])
        from_other = np.where(idx == best, second, best)
        stay = delta + log_stay
        move = delta[from_other] + log_move
        take_stay = stay >= move
        back[t] = np.where(take_stay, idx, from_other)
        delta = np.where(take_stay, stay, move) + log_em[:, t]
    path = np.empty(n_steps, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _runs(path: np.ndarray) -> list[tuple[int, int, int]]:
    """[(state, start_idx, end_idx_exclusive)] for runs of equal states."""
    runs, start = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            runs.append((int(path[start]), start, i))
            start = i
    return runs


def _candidate(state, labels, names, seg_bass, conf, n_idx) -> dict:
    if state == n_idx:
        return {"label": "N.C.", "conf": round(conf, 3), "root": None,
                "quality": None, "bass": None, "pcs": []}
    root, quality = labels[state]
    pcs = chord_pcs(root, quality)
    label = names[root] + quality
    bass_pc = None
    if seg_bass.max() > 0:
        b = int(np.argmax(seg_bass))
        if b != root and b in pcs and seg_bass[b] >= 1.25 * seg_bass[root]:
            label += "/" + names[b]
            bass_pc = b
    return {"label": label, "conf": round(conf, 3), "root": root,
            "quality": quality, "bass": bass_pc, "pcs": pcs}


def _merge_equal_neighbours(segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for seg in segments:
        if out and out[-1]["candidates"][0]["label"] == seg["candidates"][0]["label"]:
            out[-1]["end"] = seg["end"]
            out[-1]["beats"] += seg["beats"]
            continue
        out.append(seg)
    return out
