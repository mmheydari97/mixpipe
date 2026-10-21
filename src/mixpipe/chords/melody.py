"""Melody extraction: vocal stem -> pYIN pitch track -> notes -> top-3 pitches.

  1. pYIN (librosa) tracks f0 at 16 kHz with a 16 ms hop over C2-C6, kept only
     where the vocal stem is audibly active (transcribe.vocal_activity).
  2. Short unvoiced holes (consonants) are bridged, the semitone curve is
     median-filtered so vibrato and slides don't split notes, and a new note
     starts where the rounded pitch changes and holds, or at a vocal onset
     (so a repeated pitch on a new syllable is a new note).
  3. Each note ranks semitones by how many of its frames sit on them (weighted
     by pYIN's voicing probability), with a small bonus for notes in the key,
     and keeps three candidates; the gaps are padded with the neighbouring
     semitone and the octave pYIN most often confuses it with.
  4. A note an octave away from its neighbours is most likely a pitch-tracker
     octave error: the octave that fits the neighbours goes first, and the
     tracked pitch stays as the second candidate.
  5. Optionally, long stretches without vocals are tracked on the instrument
     stem as a rough lead-melody guess, marked "source": "instr". That stem is
     polyphonic, which pYIN can't follow (it locks onto the loudest voice), so
     it gets a salience tracker instead: CQT, harmonic summation, and a Viterbi
     path through the strongest peaks that prefers smooth, upper-register lines.

Octave stability: when the first pYIN pass wanders (a sizeable share of frames
more than ~11 semitones from the median), the track is repeated with the
search range clamped around that median, which removes most octave locks
before they become notes. `fast=True` keeps notes down to ~60 ms (grace notes
and quick runs) instead of 100 ms.

Notes have the chord segments' shape ({"id", "start", "end", "candidates",
"selected", "custom"}) so align.build_rows can place them on lyrics.
"""

from __future__ import annotations

import numpy as np

from ..io_utils import to_sample_rate
from .harmonica import note_label
from .theory import key_names, parse_key

SR = 16_000
HOP = 256                      # 16 ms
FRAME = 1024
FMIN, FMAX = 65.41, 1046.5     # C2 .. C6
MIN_NOTE_S = 0.10              # shorter notes are dropped
MIN_HOLD_S = 0.10              # a pitch change must hold this long to start a note
BRIDGE_S = 0.05                # unvoiced holes this short are filled
MEDIAN_S = 0.14                # median filter on the semitone curve
# fast=True: grace notes and quick runs survive
FAST = {"min_note": 0.06, "min_hold": 0.06, "median": 0.08, "onset_min": 0.07}
RANGE_SEMITONES = 11           # second pYIN pass clamps to the median +/- this
RANGE_TRIGGER = 0.05           # ... when this share of frames lies outside it
ONSET_MIN_S = 0.12             # onsets closer than this to a note edge don't split it
SILENCE_DB = 35.0              # frames this far below the loud level are unvoiced
KEY_BONUS = 0.12               # share bonus for in-key pitches (breaks flat/sharp ties)
OCTAVE_CONTEXT = 3             # notes either side that judge an octave jump
OCTAVE_JUMP = 10               # semitones from the neighbours' median that look like an octave error
CONF_TAU = 0.15                # softmax temperature for candidate confidences
INSTR_GAP_S = 2.0              # vocal-free stretches at least this long use the instruments
INSTR_CONF = 0.6               # confidence scale for instrument notes

_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10, 11)  # natural minor + the leading tone


def _scale(key: str | None) -> set[int]:
    parsed = parse_key(key)
    if parsed is None:
        return set()
    tonic, mode = parsed
    return {(tonic + s) % 12 for s in (_MINOR if mode == "min" else _MAJOR)}


def _mono16(audio: np.ndarray, sr: int) -> np.ndarray:
    y = audio if audio.ndim == 1 else audio.mean(axis=1)
    y = to_sample_rate(y.astype(np.float32), sr, SR)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return y / peak if peak > 1e-6 else y


def _pyin(y: np.ndarray, fmin: float, fmax: float):
    import librosa

    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=FRAME,
                                    hop_length=HOP)
    return f0, voiced, prob


def _track(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, midi float with NaN where unvoiced, voicing weight) per frame.

    One pYIN pass over C2-C6; if many frames sit far from the median pitch
    (octave locks, or the odd stray frame), a second pass searches only the
    median +/- RANGE_SEMITONES."""
    import librosa

    f0, voiced, prob = _pyin(y, FMIN, FMAX)
    ok0 = voiced & np.isfinite(f0)
    if ok0.sum() > 20:
        midi0 = librosa.hz_to_midi(f0[ok0])
        centre = float(np.median(midi0))
        if np.mean(np.abs(midi0 - centre) > RANGE_SEMITONES) > RANGE_TRIGGER:
            lo = float(librosa.midi_to_hz(max(centre - RANGE_SEMITONES, 24)))
            hi = float(librosa.midi_to_hz(centre + RANGE_SEMITONES))
            f0, voiced, prob = _pyin(y, max(lo, FMIN), min(hi, FMAX))
    times = librosa.times_like(f0, sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0][: len(f0)]
    db = 20 * np.log10(rms + 1e-9)
    loud = np.percentile(db, 95) if db.size else 0.0
    ok = voiced & np.isfinite(f0) & (db > loud - SILENCE_DB)
    midi = np.full(len(f0), np.nan)
    midi[ok] = librosa.hz_to_midi(f0[ok])
    return times, midi, np.where(ok, np.nan_to_num(prob), 0.0)


# ---- salience tracker for polyphonic material ------------------------------

SAL_BPO = 36                   # CQT bins per octave (3 per semitone)
SAL_FMIN_MIDI = 60             # C4: lead lines and anything a harmonica plays
SAL_OCTAVES = 3                # C4 .. C7
SAL_HARMONICS = 6
SAL_PEAKS = 6                  # candidate pitches kept per frame
SAL_UNVOICED = 0.22            # reward for staying silent, vs a peak's normalised salience
SAL_JUMP_COST = 0.07           # per semitone of jump between frames (capped at one octave)
SAL_SWITCH_COST = 0.45         # entering or leaving silence
SAL_TOP_VOICE = 0.15           # bonus per two octaves above middle C: melody rides on top


def salience_track(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lead-melody pitch of a polyphonic signal, same outputs as `_track`.

    CQT magnitudes are contrast-normalised (the local spectral envelope is
    subtracted, so a loud chord's broad bed doesn't outvote a single line), and
    each candidate fundamental sums its first six harmonics with decaying
    weights. The strongest few peaks per frame become states of a Viterbi
    search that rewards salience and the upper register, charges for jumps, and
    can stay silent.
    """
    import librosa
    from scipy.ndimage import median_filter

    n_bins = SAL_BPO * SAL_OCTAVES
    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(SAL_FMIN_MIDI),
                             n_bins=n_bins, bins_per_octave=SAL_BPO))
    times = librosa.times_like(cqt[0], sr=SR, hop_length=HOP)
    if not cqt.size or cqt.max() <= 0:
        return times, np.full(len(times), np.nan), np.zeros(len(times))
    mag = (cqt / cqt.max()) ** 0.6
    mag = np.maximum(mag - median_filter(mag, size=(SAL_BPO // 2 | 1, 1)), 0.0)
    harm = np.zeros_like(mag)
    for h in range(1, SAL_HARMONICS + 1):
        shift = int(round(SAL_BPO * np.log2(h)))
        if shift < n_bins:
            harm[: n_bins - shift] += (0.8 ** (h - 1)) * mag[shift:]
    # A plain harmonic sum rates a chord tone an octave below the melody above the
    # melody itself (the melody is its 2nd harmonic). Blending in the fundamental's
    # own magnitude (geometric mean) makes a candidate need energy where it says it is.
    sal = np.sqrt(harm * mag)
    sal /= max(np.percentile(sal.max(axis=0), 95), 1e-9)       # ~1 for a clear note

    n = sal.shape[1]
    top = np.minimum(np.argsort(-sal, axis=0)[: SAL_PEAKS * 3], n_bins - 1)   # strongest bins
    cand = np.full((n, SAL_PEAKS), -1, dtype=int)
    for t in range(n):
        picked: list[int] = []
        for b in top[:, t]:
            if sal[b, t] <= 0.02:
                break
            if all(abs(b - q) >= 4 for q in picked):          # peaks at least ~1 semitone apart
                picked.append(int(b))
                if len(picked) == SAL_PEAKS:
                    break
        cand[t, : len(picked)] = picked
    # emission reward per candidate (+ the top-voice nudge); -inf for empty slots
    reward = np.full((n, SAL_PEAKS), -np.inf)
    for k in range(SAL_PEAKS):
        valid = cand[:, k] >= 0
        b = cand[valid, k]
        midi = SAL_FMIN_MIDI + b / 3.0
        reward[valid, k] = sal[b, np.flatnonzero(valid)] * (
            1.0 + SAL_TOP_VOICE * np.clip((midi - 60.0) / 24.0, -1.0, 1.0))
    # Viterbi: states 0..K-1 candidates, K = silent
    K = SAL_PEAKS
    score = np.full(K + 1, -np.inf)
    score[:K] = reward[0]
    score[K] = SAL_UNVOICED
    back = np.zeros((n, K + 1), dtype=np.int16)
    for t in range(1, n):
        new = np.full(K + 1, -np.inf)
        prev_b = cand[t - 1]
        for k in range(K):
            if cand[t, k] < 0:
                continue
            jump = np.minimum(np.abs(cand[t, k] - prev_b) / 3.0, 12.0) * SAL_JUMP_COST
            from_voiced = score[:K] - np.where(prev_b >= 0, jump, np.inf)
            j = int(np.argmax(from_voiced))
            via_silence = score[K] - SAL_SWITCH_COST
            if from_voiced[j] >= via_silence:
                new[k], back[t, k] = from_voiced[j] + reward[t, k], j
            else:
                new[k], back[t, k] = via_silence + reward[t, k], K
        stay_silent = score[K] + SAL_UNVOICED
        from_voice = np.max(score[:K]) - SAL_SWITCH_COST + SAL_UNVOICED
        if stay_silent >= from_voice:
            new[K], back[t, K] = stay_silent, K
        else:
            new[K], back[t, K] = from_voice, int(np.argmax(score[:K]))
        score = new
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    midi = np.full(n, np.nan)
    weight = np.zeros(n)
    for t in range(n):
        if path[t] < K and cand[t, path[t]] >= 0:
            b = cand[t, path[t]]
            midi[t] = SAL_FMIN_MIDI + b / 3.0
            weight[t] = float(np.clip(sal[b, t], 0.0, 1.0))
    return times, midi, weight


def _onsets(y: np.ndarray, n: int) -> np.ndarray:
    import librosa

    frames = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP, units="frames")
    return frames[frames < n]


def _mask(times: np.ndarray, regions: list[tuple[float, float]]) -> np.ndarray:
    keep = np.zeros(len(times), dtype=bool)
    for a, b in regions:
        keep |= (times >= a) & (times < b)
    return keep


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] of True runs."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def _bridge(midi: np.ndarray, weight: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill unvoiced holes up to `frames` long between voiced frames."""
    midi, weight = midi.copy(), weight.copy()
    for a, b in _runs(np.isnan(midi)):
        if 0 < a and b < len(midi) and b - a <= frames:
            midi[a:b] = np.linspace(midi[a - 1], midi[b], b - a + 2)[1:-1]
            weight[a:b] = 0.0          # filled frames don't vote on the pitch
    return midi, weight


def segment(midi: np.ndarray, onsets: np.ndarray | None = None,
            fast: bool = False) -> list[tuple[int, int]]:
    """Split a per-frame semitone curve (NaN = unvoiced) into notes.
    Returns [(start_frame, end_frame_exclusive)]."""
    from scipy.ndimage import median_filter

    fps = SR / HOP
    hold = max(1, int(round((FAST["min_hold"] if fast else MIN_HOLD_S) * fps)))
    med = max(1, int(round((FAST["median"] if fast else MEDIAN_S) * fps)) | 1)
    edge = int(round((FAST["onset_min"] if fast else ONSET_MIN_S) * fps))
    onset_set = set(int(o) for o in (onsets if onsets is not None else []))
    notes: list[tuple[int, int]] = []
    for a, b in _runs(~np.isnan(midi)):
        q = np.round(median_filter(midi[a:b], size=min(med, b - a), mode="nearest")).astype(int)
        # runs of equal semitones; too-short runs join the neighbour nearest in pitch
        pieces = [[int(q[0]), 0, 1]]
        for i in range(1, len(q)):
            if q[i] == pieces[-1][0]:
                pieces[-1][2] = i + 1
            else:
                pieces.append([int(q[i]), i, i + 1])
        changed = True
        while changed and len(pieces) > 1:
            changed = False
            k = min(range(len(pieces)), key=lambda j: pieces[j][2] - pieces[j][1])
            if pieces[k][2] - pieces[k][1] >= hold:
                break
            if k == 0:
                j = 1
            elif k == len(pieces) - 1:
                j = k - 1
            else:
                j = min((k - 1, k + 1), key=lambda j: abs(pieces[j][0] - pieces[k][0]))
            lo, hi = min(j, k), max(j, k)
            pieces[lo] = [pieces[j][0], pieces[lo][1], pieces[hi][2]]
            del pieces[hi]
            # neighbours that now share a pitch become one
            merged = [pieces[0]]
            for p in pieces[1:]:
                if p[0] == merged[-1][0]:
                    merged[-1][2] = p[2]
                else:
                    merged.append(p)
            pieces = merged
            changed = True
        for _, s, e in pieces:
            cuts = [s]
            for o in sorted(onset_set):
                if cuts[-1] + edge <= o - a <= e - edge:
                    cuts.append(o - a)
            cuts.append(e)
            notes += [(a + c0, a + c1) for c0, c1 in zip(cuts[:-1], cuts[1:])]
    return notes


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp((x - x.max()) / CONF_TAU)
    return z / z.sum()


def candidates(midi: np.ndarray, weight: np.ndarray, scale: set[int],
               names: list[str]) -> list[dict]:
    """Three distinct pitches for one note, best first, with confidences."""
    voiced = np.isfinite(midi)
    m, w = midi[voiced], weight[voiced]
    if w.sum() <= 0:
        w = np.ones_like(m)
    q = np.round(m).astype(int)
    shares: dict[int, float] = {}
    for semi, wt in zip(q, w):
        shares[int(semi)] = shares.get(int(semi), 0.0) + float(wt)
    total = sum(shares.values())
    score = {s: v / total + (KEY_BONUS if s % 12 in scale else 0.0) for s, v in shares.items()}
    ranked = sorted(score, key=lambda s: -score[s])
    top = ranked[0]
    center = float(np.average(m, weights=w))
    # pad: neighbour semitone on the side the pitch leans to, the other side, the octave
    side = 1 if center >= top else -1
    fillers = [top + side, top - side, top - 12 if top >= 72 else top + 12]
    for k, f in enumerate(fillers):
        if len(ranked) >= 3:
            break
        if f not in score:  # no frames voted for it: it only ranks by key and order
            score[f] = (0.5 * KEY_BONUS if f % 12 in scale else 0.0) - 0.01 * k
            ranked.append(f)
    picked = ranked[:3]
    conf = _softmax(np.array([score[s] for s in picked]))
    return [{"label": note_label(s, names), "midi": int(s), "conf": round(float(c), 3)}
            for s, c in zip(picked, conf)]


def _notes(times, midi, weight, onsets, key: str | None, source: str,
           conf_scale: float = 1.0, fast: bool = False) -> list[dict]:
    scale, names = _scale(key), key_names(key)
    hop_s = HOP / SR
    midi, weight = _bridge(midi, weight, int(round(BRIDGE_S * SR / HOP)))
    out = []
    for a, b in segment(midi, onsets, fast):
        if (b - a) * hop_s < (FAST["min_note"] if fast else MIN_NOTE_S):
            continue
        cands = candidates(midi[a:b], weight[a:b], scale, names)
        if conf_scale != 1.0:
            for c in cands:
                c["conf"] = round(c["conf"] * conf_scale, 3)
        out.append({"start": round(float(times[a]), 3),
                    "end": round(float(times[b - 1] + hop_s), 3),
                    "candidates": cands, "selected": 0, "custom": None, "fingering": None,
                    "source": source})
    fix_octaves(out, names)
    return out


def fix_octaves(notes: list[dict], names: list[str]) -> int:
    """Move the octave that fits the neighbouring notes to the front where the
    tracked pitch jumps about an octave away from them. Returns how many moved."""
    tops = [n["candidates"][0]["midi"] for n in notes]
    moved = 0
    for i, n in enumerate(notes):
        around = tops[max(0, i - OCTAVE_CONTEXT): i] + tops[i + 1: i + 1 + OCTAVE_CONTEXT]
        if len(around) < 2:
            continue
        ref = float(np.median(around))
        off = tops[i] - ref
        if abs(off) < OCTAVE_JUMP:
            continue
        fixed = tops[i] - 12 * int(np.sign(off))
        if abs(fixed - ref) >= abs(off) - 6:
            continue
        cands = [c for c in n["candidates"] if c["midi"] != fixed]
        first = {"label": note_label(fixed, names), "midi": fixed, "conf": cands[0]["conf"]}
        cands[0] = {**cands[0], "conf": round(cands[0]["conf"] * 0.5, 3)}
        n["candidates"] = [first] + cands[:2]
        moved += 1
    return moved


def _gaps(regions: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """Stretches without vocal activity that are long enough for the instruments."""
    out, t = [], 0.0
    for a, b in sorted(regions):
        if a - t >= INSTR_GAP_S:
            out.append((t, a))
        t = max(t, b)
    if duration - t >= INSTR_GAP_S:
        out.append((t, duration))
    return out


def extract_melody(vocal: np.ndarray | None, sr: int, regions: list[tuple[float, float]],
                   fallback: np.ndarray | None = None, key: str | None = None,
                   progress=None, fast: bool = False) -> dict:
    """Notes of the sung melody (and, with `fallback`, a rough instrumental
    melody where nobody sings). `regions` is vocal activity in seconds; `fast`
    keeps grace notes and quick runs (down to ~60 ms).
    Returns {"notes": [...]} sorted by start, ids "n0", "n1", ..."""
    report = progress or (lambda frac, msg: None)
    notes: list[dict] = []
    duration = 0.0
    if vocal is not None and regions:
        y = _mono16(vocal, sr)
        duration = len(y) / SR
        report(0.1, "tracking the vocal pitch")
        times, midi, weight = _track(y)
        keep = _mask(times, [(a - 0.05, b + 0.05) for a, b in regions])
        midi[~keep] = np.nan
        weight[~keep] = 0.0
        notes += _notes(times, midi, weight, _onsets(y, len(times)), key, "vocal", fast=fast)
    if fallback is not None:
        y = _mono16(fallback, sr)
        duration = max(duration, len(y) / SR)
        gaps = _gaps(regions if vocal is not None else [], duration)
        if gaps:
            report(0.6, "guessing the melody where nobody sings")
            times, midi, weight = salience_track(y)
            keep = _mask(times, gaps)
            midi[~keep] = np.nan
            weight[~keep] = 0.0
            notes += _notes(times, midi, weight, _onsets(y, len(times)), key, "instr",
                            INSTR_CONF, fast=fast)
    notes.sort(key=lambda n: n["start"])
    for i, n in enumerate(notes):
        n["id"] = f"n{i}"
    report(1.0, f"{len(notes)} notes")
    return {"notes": notes}
