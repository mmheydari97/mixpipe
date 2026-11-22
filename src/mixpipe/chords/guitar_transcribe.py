"""Notes from the guitar stem, for the guitar tab (guitar.py lays them out).

    stereo guitar stem -> Basic Pitch (polyphonic notes + pitch bends)
                       -> overtone ghosts removed
                       -> per-note stereo position, bend and vibrato

Basic Pitch (Spotify, Apache-2.0) runs its ONNX model on onnxruntime, so no
TensorFlow is needed: `make install-guitar` installs it without its pinned
dependencies (they would downgrade numpy). Any polyphonic transcriber with the
same output can be passed in instead (tests use a fake).

Each note: {"start", "end", "midi", "amp" (0..1, the model's note strength),
"pan" (-1 left .. +1 right), "centre" (0..1: 1 = the same in both channels, a
centred part; ~0.5 = different in each, e.g. double-tracked guitars),
"bend" (largest bend up, semitones), "vibrato" (bool)}.

The stereo numbers are what tells a centred lead from guitars panned left and
right (guitar.parts); they are measured on the note's first three harmonics
in its first 0.3 s.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import numpy as np

# A transcriber: (mono float32, sr) -> [(start_s, end_s, midi, amp, bends)], bends in
# Basic Pitch's units (thirds of a semitone, one per model frame) or None.
Transcriber = Callable[[np.ndarray, int], list]

LOWEST_HZ = 70.0       # a little under E2 (82 Hz), for drop tunings
HIGHEST_HZ = 1400.0    # about F6, the 22nd fret of the top string with room for bends
MIN_NOTE_MS = 58       # Basic Pitch's minimum note length
HARMONICS = (12, 19, 24, 28, 31)   # overtones 2-6 of a string, in semitones
GHOST_WINDOW = 0.06    # an overtone ghost starts this close to its string's note
FRAME_S = 256 / 22050  # Basic Pitch's frame hop
BLIP_S, BLIP_AMP = 0.1, 0.4   # shorter and weaker than this: not a note


def basic_pitch(mono: np.ndarray, sr: int) -> list:
    """Basic Pitch on onnxruntime. Raises RuntimeError when it isn't installed."""
    try:
        import basic_pitch as bp
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise RuntimeError("the guitar tab needs Basic Pitch: make install-guitar") from exc
    import soundfile as sf

    model = os.path.join(os.path.dirname(bp.__file__), "saved_models", "icassp_2022", "nmp.onnx")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "guitar.wav")
        sf.write(path, mono, sr, subtype="PCM_16")
        _, _, events = predict(path, model, onset_threshold=0.5, frame_threshold=0.3,
                               minimum_note_length=MIN_NOTE_MS, minimum_frequency=LOWEST_HZ,
                               maximum_frequency=HIGHEST_HZ, multiple_pitch_bends=False,
                               melodia_trick=True)
    return events


def drop_ghosts(notes: list[dict]) -> list[dict]:
    """Remove what the model heard that isn't a played note:

    - overtones: a note an octave, an octave and a fifth, two octaves... above a
      clearly stronger note that starts with it (or was struck just before and
      still rings) is that string's overtone. Two octaves struck together at
      about the same strength are both kept: either may be real;
    - blips: very short and weak notes (a pick scrape, a fret buzz)."""
    keep: list[dict] = []
    for n in sorted(notes, key=lambda n: -n["amp"]):
        if n["end"] - n["start"] < BLIP_S and n["amp"] < BLIP_AMP:
            continue
        ghost = any(
            n["midi"] - k["midi"] in HARMONICS
            and ((abs(n["start"] - k["start"]) <= GHOST_WINDOW and n["amp"] < 0.95 * k["amp"])
                 or (k["start"] < n["start"] < k["end"] and n["start"] - k["start"] < 0.25
                     and n["amp"] < 0.8 * k["amp"]))
            for k in keep)
        if not ghost:
            keep.append(n)
    return sorted(keep, key=lambda n: (n["start"], n["midi"]))


def merge_wobbles(notes: list[dict]) -> list[dict]:
    """A bend or vibrato the model heard as quick notes a semitone or two apart
    (G G# G G#...): three or more short notes back to back on two neighbouring
    pitches become one note at the lower pitch, marked as bent ("bend") with
    vibrato, the upper pitch kept as "alt" (a trill looks the same; the note
    menu can switch it)."""
    out: list[dict] = []
    run: list[dict] = []

    def close():
        pitches = sorted({r["midi"] for r in run})
        if len(run) >= 3 and len(pitches) == 2 and pitches[1] - pitches[0] <= 2:
            low = min(run, key=lambda r: (r["midi"], r["start"]))
            out.append({**low, "start": run[0]["start"], "end": max(r["end"] for r in run),
                        "amp": max(r["amp"] for r in run), "bend": float(pitches[1] - pitches[0]),
                        "vibrato": True, "alt": pitches[1]})
        else:
            out.extend(run)

    for n in sorted(notes, key=lambda n: (n["start"], n["midi"])):
        short = n["end"] - n["start"] <= 0.18
        if run and short and n["start"] - run[-1]["end"] <= 0.04 and abs(n["midi"] - run[-1]["midi"]) <= 2 \
                and len({r["midi"] for r in run} | {n["midi"]}) <= 2:
            run.append(n)
            continue
        if run:
            close()
        run = [n] if short else []
        if not short:
            out.append(n)
    if run:
        close()
    return sorted(out, key=lambda n: (n["start"], n["midi"]))


def merge_bent_pairs(notes: list[dict]) -> list[dict]:
    """A bend heard as two notes at once: notes a semitone apart starting
    together (within GHOST_WINDOW) are one bent note at the lower pitch. A
    semitone cluster is almost never played on a guitar; a bend is."""
    notes = sorted(notes, key=lambda n: (n["start"], n["midi"]))
    gone: set[int] = set()
    out = []
    for i, a in enumerate(notes):
        if i in gone:
            continue
        for j in range(i + 1, len(notes)):
            b = notes[j]
            if b["start"] - a["start"] > GHOST_WINDOW:
                break
            if j not in gone and b["midi"] - a["midi"] == 1:
                gone.add(j)
                a = {**a, "end": max(a["end"], b["end"]), "amp": max(a["amp"], b["amp"]),
                     "bend": max(a.get("bend", 0.0), 1.0), "alt": b["midi"]}
                break
        out.append(a)
    return out


def _bend_shape(bends) -> tuple[float, bool]:
    """(largest bend up in semitones, vibrato?) from a note's pitch-bend curve."""
    if bends is None or len(bends) < 3:
        return 0.0, False
    semis = np.asarray(bends, dtype=float) / 3.0
    semis = semis - np.median(semis[: max(2, len(semis) // 5)])   # relative to how it started
    up = float(np.percentile(semis, 90))
    wobble = np.diff(np.sign(semis - np.median(semis)))
    crossings = int(np.count_nonzero(wobble))
    vibrato = len(semis) * FRAME_S >= 0.3 and crossings >= 4 and float(np.std(semis)) >= 0.12
    return round(max(0.0, up), 2), bool(vibrato)


def stereo_position(stereo: np.ndarray, sr: int, start: float, end: float, midi: int) -> tuple[float, float]:
    """(pan, centre) of one note: its first three harmonics in the left and
    right channels, and in their sum and difference."""
    a = int(start * sr)
    b = min(len(stereo), a + int(min(max(end - start, 0.05), 0.3) * sr))
    seg = stereo[a:b]
    if len(seg) < 256:
        return 0.0, 1.0
    n = 1 << int(np.ceil(np.log2(len(seg) * 2)))
    win = np.hanning(len(seg))[:, None]
    spec = np.fft.rfft(seg * win, n=n, axis=0)
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    bins = [int(round(f0 * k * n / sr)) for k in (1, 2, 3)]
    bins = [i for i in bins if i + 1 < len(spec)]
    if not bins:
        return 0.0, 1.0
    pick = np.concatenate([np.arange(i - 1, i + 2) for i in bins])
    left, right = spec[pick, 0], spec[pick, 1]
    el, er = float(np.abs(left).sum()), float(np.abs(right).sum())
    em, es = float(np.abs(left + right).sum()), float(np.abs(left - right).sum())
    pan = (er - el) / (er + el + 1e-12)
    centre = em / (em + es + 1e-12)
    return round(pan, 3), round(centre, 3)


def transcribe(stereo: np.ndarray, sr: int, transcriber: Transcriber | None = None,
               progress: Callable[[float, str], None] | None = None) -> list[dict]:
    """The guitar stem's notes (see the module docstring), sorted by start."""
    report = progress or (lambda f, msg: None)
    stereo = np.asarray(stereo, dtype=np.float32)
    if stereo.ndim == 1:
        stereo = np.stack([stereo, stereo], axis=1)
    report(0.05, "listening to the guitar (Basic Pitch)")
    events = (transcriber or basic_pitch)(stereo.mean(axis=1), sr)
    notes = []
    for ev in events:
        start, end, midi, amp = float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3])
        bends = ev[4] if len(ev) > 4 else None
        bend, vibrato = _bend_shape(bends)
        notes.append({"start": round(start, 3), "end": round(end, 3), "midi": midi,
                      "amp": round(amp, 3), "bend": bend, "vibrato": vibrato})
    report(0.7, f"{len(notes)} notes heard, removing overtones")
    notes = merge_bent_pairs(merge_wobbles(drop_ghosts(notes)))
    report(0.8, "measuring where each note sits in the stereo picture")
    for n in notes:
        n["pan"], n["centre"] = stereo_position(stereo, sr, n["start"], n["end"], n["midi"])
    report(1.0, f"{len(notes)} guitar notes")
    return notes
