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

Typical top-1 accuracy on real pop recordings is ~65-75% (lower for the extended
qualities). The candidates exist to cover the misses. Anything that returns the
same segment structure can replace this module.
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

# Key tracking (modulations): windowed key estimate, smoothed by a sticky HMM
KEY_WINDOW_BEATS = 32    # ~8 bars of chord tones per local key estimate
KEY_EMISSION_SCALE = 3.0 # per-beat weight of the key correlation
KEY_P_STAY = 0.995       # per-beat probability of staying in a key
KEY_MIN_BEATS = 16       # shorter key regions are folded into a neighbour


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
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beats = np.asarray(beats, dtype=int)
    if len(beats) < 4:
        step = max(1, int(round(0.5 * SR / HOP)))
        beats = np.arange(step, n_frames, step)
        tempo = 120.0
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
    regions = key_regions(tones)                         # [(key_idx, a, b)]
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
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    accent = np.array([onset[max(0, f - 2): f + 3].max() if f < len(onset) else 0.0
                       for f in beats])
    changes = [int(beat_of[i]) for i in range(1, n_int)
               if first_path[i] != first_path[i - 1] and beat_of[i] >= 0]
    subdivision, _ = beat_subdivision(onset, beats)
    meter = estimate_meter(changes, accent, subdivision=subdivision, tempo=tempo)

    m, p = meter["beats_per_bar"], meter["downbeat"]
    stay = np.full(n_int, P_STAY_OFFBEAT)
    for i in range(n_int):
        pos = (beat_of[i] - p) % m if beat_of[i] >= 0 else -1
        if pos == 0:
            stay[i] = P_STAY_DOWNBEAT
        elif m in (4, 6) and pos == m // 2:
            stay[i] = P_STAY_HALFBAR
    path = _viterbi_sticky(log_em, stay)

    # absorb short no-chord blips (< 2 beats) into the preceding chord
    runs = _runs(path)
    for k, (state, a, b) in enumerate(runs):
        if state == n_idx and b - a < 2 and k > 0:
            path[a:b] = runs[k - 1][0]
    runs = _runs(path)

    # ---- segments + candidates -------------------------------------------------
    segments = []
    for state, a, b in runs:
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
        segments.append({
            "start": round(float(times[a]), 3),
            "end": round(float(times[b]), 3),
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
        "beat_times": [round(float(t), 3) for t in beat_times],
        "meter": meter,
        "segments": segments,
    }


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


def key_regions(tones: np.ndarray) -> list[tuple[int, int, int]]:
    """Split the song into key regions from per-beat chord-tone profiles.

    tones: (12, n_beats). Returns [(key index 0-23, start beat, end beat)], at
    least one region. A key change needs ~KEY_MIN_BEATS of sustained evidence;
    moves between relative major/minor are not treated as modulations.
    """
    n = tones.shape[1]
    if n == 0 or not tones.any():
        return [(0, 0, max(n, 1))]
    profiles = np.array([np.roll(KK_MAJOR, t) for t in range(12)]
                        + [np.roll(KK_MINOR, t) for t in range(12)])
    profiles = (profiles - profiles.mean(axis=1, keepdims=True))
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True)

    cs = np.concatenate([np.zeros((12, 1)), np.cumsum(tones, axis=1)], axis=1)
    half = KEY_WINDOW_BEATS // 2
    idx = np.arange(n)
    lo, hi = np.clip(idx - half, 0, n), np.clip(idx + half, 0, n)
    win = cs[:, hi] - cs[:, lo]                                   # (12, n)
    win = win - win.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(win, axis=0, keepdims=True)
    corr = profiles @ (win / np.maximum(norms, 1e-12))            # (24, n)
    corr[:, norms[0] < 1e-9] = 0.0

    path = _viterbi_sticky(KEY_EMISSION_SCALE * corr, np.full(n, KEY_P_STAY))
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
        if regions[shortest][2] - regions[shortest][1] < KEY_MIN_BEATS:
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
