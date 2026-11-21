"""Live guitar tuning: which string is ringing and how far off it is.

The browser streams microphone audio in short chunks; `StringTracker.feed`
keeps the last ~85 ms, finds the pitch with the shared YIN tracker
(`pitch.track_pitch`) and matches it to a string of the current tuning
(`chords.theory.guitar_tuning`).

One string at a time. The nearest string (in cents) wins, with three guards:
  - a different string must win a few chunks in a row before the display
    moves to it, so a decaying note or one odd frame doesn't flicker;
  - while a note keeps ringing (no new pluck), a reading an octave above the
    current string is taken as that string's octave error, not a jump to the
    string an octave up (Drop D's D2 vs D3) or the one near its octave;
  - a locked string (tapped in the page) is always the target, however far
    off it is, for a slack new string the nearest match would get wrong.
"""

from __future__ import annotations

import math

import numpy as np

from .chords.theory import SHARP, note_label
from .pitch import track_pitch

WINDOW = 4096            # samples kept for each estimate (~85 ms at 48 kHz)
HOP_MS = 10.0
FMIN, FMAX = 55.0, 450.0  # C2 a few semitones flat .. E4 a few sharp; keeps upper harmonics out
MIN_LEVEL_DB = -55.0
SWITCH_AFTER = 3         # chunks in a row a new string needs before the display moves to it
ONSET_RATIO = 1.4        # rms jump (vs the previous chunk) that counts as a new pluck
OCTAVE_SNAP = 1.0        # semitones: how close an octave-up reading must be to count as an error
TAIL_DB = 24.0           # a note this far below its peak is in the noise: hold the last reading
FORGET_AFTER = 4         # unclear chunks in a row before the ringing string is forgotten


class StringTracker:
    def __init__(self, sr: int, strings: list[int], a4: float = 440.0, lock: int | None = None):
        """`strings` are MIDI notes, string 6 first; `lock` is a string number (6..1)."""
        # YIN's 2048-sample frame must hold a low C's period: above 48 kHz, average
        # every `decim` samples down to a rate where it does
        self.decim = max(1, int(round(sr / 48000)))
        self.sr = int(sr) // self.decim
        self.buf = np.zeros(0, dtype=np.float64)
        self.current: int | None = None    # string number on display
        self.pending: int | None = None    # a different string, waiting to win SWITCH_AFTER chunks
        self.pending_n = 0
        self.prev_rms = 0.0
        self.peak_rms = 0.0                # loudest chunk of the note ringing now
        self.unclear = 0                   # chunks in a row without a clear pitch
        self.configure(strings=strings, a4=a4, lock=lock)

    def configure(self, strings: list[int] | None = None, a4: float | None = None,
                  lock: int | None | str = "keep") -> None:
        if strings is not None:
            self.targets = {6 - i: int(m) for i, m in enumerate(strings)}
            self.current = self.pending = None
        if a4 is not None:
            self.a4 = float(a4)
        if lock != "keep":
            self.lock = lock if lock in self.targets else None
            if self.lock is not None:
                self.current = self.lock

    def midi_of(self, hz: float) -> float:
        return 69.0 + 12.0 * math.log2(hz / self.a4)

    def estimate(self) -> tuple[float, float] | None:
        """(hz, confidence) of the buffered audio, or None when it isn't a clear note."""
        if self.buf.size < WINDOW:
            return None
        f0, conf = track_pitch(self.buf, self.sr, hop_ms=HOP_MS, fmin=FMIN, fmax=FMAX,
                               min_level_db=MIN_LEVEL_DB)
        hop = max(1, int(round(HOP_MS / 1000.0 * self.sr)))
        half = 1024                       # track_pitch's frame is 2048, centred on i * hop
        inside = [i for i in range(f0.size) if half <= i * hop <= self.buf.size - half]
        hz = f0[inside]
        voiced = ~np.isnan(hz)
        if voiced.sum() < max(2, len(inside) // 2):
            return None
        return float(np.median(hz[voiced])), float(np.median(conf[inside][voiced]))

    def feed(self, chunk: np.ndarray) -> dict:
        """Add a chunk of mono samples; returns the reading for the display:
        {"state": "silent"} | {"state": "pending"} | {"state": "note", ...}."""
        chunk = np.asarray(chunk, dtype=np.float64).ravel()
        if self.decim > 1:
            chunk = chunk[: chunk.size // self.decim * self.decim].reshape(-1, self.decim).mean(axis=1)
        self.buf = np.concatenate([self.buf, chunk])[-WINDOW:]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        onset = rms > ONSET_RATIO * self.prev_rms
        self.prev_rms = rms
        self.peak_rms = rms if onset else max(self.peak_rms, rms)

        est = self.estimate()
        if est is None:
            self.unclear += 1
            if self.unclear >= FORGET_AFTER and self.lock is None:
                self.current = self.pending = None
            return {"state": "silent"}
        self.unclear = 0
        if rms < self.peak_rms * 10 ** (-TAIL_DB / 20):
            return {"state": "pending"}
        hz, conf = est
        m = self.midi_of(hz)

        if self.lock is not None:
            string = self.lock
            m = self._fold(m, self.targets[string])
        else:
            nearest = min(self.targets, key=lambda s: abs(m - self.targets[s]))
            string = nearest
            if self.current is not None and nearest != self.current and not onset:
                cur = self.targets[self.current]
                if abs(m - 12 - cur) < OCTAVE_SNAP:
                    string, m = self.current, m - 12        # the current string's octave error
            if self.current is not None and string != self.current:
                if string == self.pending:
                    self.pending_n += 1
                else:
                    self.pending, self.pending_n = string, 1
                if self.pending_n < SWITCH_AFTER:
                    return {"state": "pending"}
            self.current, self.pending, self.pending_n = string, None, 0

        target = self.targets[string]
        offset = m - target
        return {
            "state": "note", "string": string, "hz": round(self.a4 * 2 ** ((m - 69) / 12), 2),
            "target_midi": target, "heard": note_label(int(round(m)), SHARP),
            "cents": round(100.0 * offset, 1),
            "semitones_off": int(round(offset)), "confidence": round(conf, 3),
            "locked": self.lock is not None,
        }

    @staticmethod
    def _fold(m: float, target: int) -> float:
        """An octave error of a locked string: a reading about an octave from the
        target (and far from the target itself) is moved back by that octave. A
        string that is just very slack, say 7 semitones flat, is left alone."""
        for k in (1, -1):
            if abs(m - 12 * k - target) < 2.0 and abs(m - target) > 9.0:
                return m - 12 * k
        return m
