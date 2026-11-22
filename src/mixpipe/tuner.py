"""Live guitar tuning: which string is ringing and how far off it is.

The browser streams microphone audio in short chunks; `StringTracker.feed`
keeps the last ~85 ms, finds the pitch with the shared YIN tracker
(`pitch.track_pitch`) and matches it to a string of the current tuning
(`chords.theory.guitar_tuning`).

Before YIN, every chunk goes through a low-pass (no string's fundamental is
near it; it takes off pick noise and an amp's fizz) and narrow notches on
mains hum. Through a distorting amp, hum mixes with the note, and as the note
dies the hum is what YIN would read. A linear filter keeps a periodic signal's
period, so the pitch itself doesn't move. Hum loud enough to get past them
(its distorted overtones still repeat at 50/60 Hz) is ignored unless a string
is locked.

One string at a time. The nearest string (in cents) wins, with these guards:
  - a different string must win a few chunks in a row before the display
    moves to it, so a decaying note or one odd frame doesn't flicker (not
    right after a pluck, when a new string is what's expected);
  - while a note keeps ringing (no new pluck), a reading an octave above the
    current string is taken as that string's octave error, not a jump to the
    string an octave up (Drop D's D2 vs D3) or the one near its octave;
  - a locked string (tapped in the page) is always the target, however far
    off it is, for a slack new string the nearest match would get wrong;
  - a ringing note keeps its pitch: until the next pluck, a reading that jumps
    away from the last few is dropped (sympathetic strings, hum, the noisy
    tail). A peg being turned moves the pitch gradually and is followed.
The cents shown are the median of the note's last few readings. After a
silence nothing is read until the next pluck: what rings on is the tail.

A pluck is a jump in level, or in high-band energy (a pick's attack still
shows through an amp's compression, where the level hardly moves).

Lock-in: a note that stays in tune (`TUNED_CENTS`) for `TUNED_HOLD_S`, with
little spread and no drift, is reported as "tuned" and then frozen until the
next pluck or silence, so the decaying tail can't undo it. A note with a
chorus or vibrato wobble gets the same verdict from its average over a longer
`WOBBLE_HOLD_S`.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from .chords.theory import SHARP, note_label
from .pitch import track_pitch

WINDOW = 4096            # samples kept for each estimate (~85 ms at 48 kHz)
HOP_MS = 10.0
FMIN, FMAX = 55.0, 450.0  # C2 a few semitones flat .. E4 a few sharp; keeps upper harmonics out
MIN_LEVEL_DB = -55.0
SWITCH_AFTER = 3         # chunks in a row a new string needs before the display moves to it
OCTAVE_SNAP = 1.0        # semitones: how close an octave-up reading must be to count as an error
TAIL_DB = 24.0           # a note this far below its peak is in the noise: hold the last reading
FORGET_AFTER = 4         # unclear chunks in a row before the ringing string is forgotten

# prefilter
LOWPASS_HZ = 1200.0      # keeps three harmonics of high E, drops pick noise and amp fizz
HUM_NOTCHES = (50.0, 60.0, 120.0, 180.0)   # mains and its harmonics; not 100/150 (next to G2, D3)
NOTCH_Q = 30.0
HUM_IGNORE_CENTS = 50.0  # an unlocked reading this close to 50 or 60 Hz is the hum itself
                         # (while the notches ring in, hum reads a little off); C2 is 150 away

# plucks
ONSET_JUMP = 2.8         # level jump (~9 dB) over the quietest of the last few chunks;
                         # a chorus's swells stay under it
HF_HZ = 2500.0           # the band a pick's attack shows up in
HF_JUMP = 3.0
REPLUCK_DB = 12.0        # a level jump counts while a note rings only this close to its peak
                         # (a chorus nulling the dying note and coming back is not a pluck)
HF_FLOOR_DB = -50.0
ONSET_GAP_S = 0.15       # one pluck, one onset
ONSET_LOOKBACK = 3       # chunks
ONSET_GRACE_S = 0.2      # after a pluck, no pitch yet isn't silence: YIN's window is still filling
FRESH_S = 0.4            # a note this young, with no readings yet, may be on any string

# a ringing note
MEDIAN_N = 5             # readings in the shown median (~210 ms)
CONTINUITY_ST = 0.6      # semitones a reading may jump from the last few
REESTABLISH = 8          # agreeing, loud readings that move the note anyway (a wrong start)
STRONG_DB = 12.0         # ... loud = within this of the note's peak

# lock-in
TUNED_CENTS = 5.0
TUNED_HOLD_S = 0.7
TUNED_SPREAD = 4.0       # cents (standard deviation)
TUNED_DRIFT = 4.0        # cents the readings' trend line may move across the hold: a turning peg moves more
WOBBLE_HOLD_S = 2.5      # chorus / vibrato: judge the average over this long instead
WOBBLE_SPREAD = 25.0
WOBBLE_DRIFT = 8.0
LATCH_GUARD_S = 0.6      # after a lock-in the page's chime (E7/B7) comes back through the mic.
                         # The low-pass takes it out of the level, so only the high-band pluck
                         # test is off this long; a pluck it hides is caught by `_missed_pluck`


def _db(db: float) -> float:
    return 10.0 ** (db / 20.0)


class StringTracker:
    def __init__(self, sr: int, strings: list[int], a4: float = 440.0, lock: int | None = None):
        """`strings` are MIDI notes, string 6 first; `lock` is a string number (6..1)."""
        raw_sr = int(sr)
        # YIN's 2048-sample frame must hold a low C's period: above 48 kHz, average
        # every `decim` samples down to a rate where it does (after the low-pass,
        # which keeps that from aliasing)
        self.decim = max(1, int(round(raw_sr / 48000)))
        self.raw_sr = raw_sr
        self.sr = raw_sr // self.decim
        nyq = raw_sr / 2.0
        stages = [butter(4, min(LOWPASS_HZ, 0.45 * raw_sr), "low", fs=raw_sr, output="sos")]
        stages += [tf2sos(*iirnotch(f, NOTCH_Q, fs=raw_sr)) for f in HUM_NOTCHES if f < 0.9 * nyq]
        self.sos = np.vstack(stages)
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.hf_sos = butter(2, min(HF_HZ, 0.4 * raw_sr), "high", fs=raw_sr, output="sos")
        self.hf_zi = sosfilt_zi(self.hf_sos) * 0.0

        self.buf = np.zeros(0, dtype=np.float64)
        self.t = 0.0                       # seconds of audio fed
        self.current: int | None = None    # string number on display
        self.pending: int | None = None    # a different string, waiting to win SWITCH_AFTER chunks
        self.pending_n = 0
        self.levels: deque[float] = deque(maxlen=ONSET_LOOKBACK)
        self.hf_levels: deque[float] = deque(maxlen=ONSET_LOOKBACK)
        self.last_onset = -math.inf
        self.guard_until = -math.inf
        self.peak_rms = 0.0                # loudest chunk of the note ringing now
        self.unclear = 0                   # chunks in a row without a clear pitch
        self.awaiting_onset = False        # after a silence (or `wait`): nothing counts until a pluck
        self.ignore_m: float | None = None  # pitch of the note that locked in, until the next pluck
        self.missed: list[float] = []      # readings that say a pluck was missed in the guard
        self._new_note()
        self.configure(strings=strings, a4=a4, lock=lock)

    def _new_note(self) -> None:
        self.recent: list[float] = []      # accepted midi readings of this note
        self.rejects: list[float] = []     # readings in a row that jumped away from them
        self.hist: list[tuple[float, float]] = []   # (time, cents) for the lock-in
        self.latched: dict | None = None

    def configure(self, strings: list[int] | None = None, a4: float | None = None,
                  lock: int | None | str = "keep", wait: bool = False) -> None:
        """`wait`: ignore what rings now and start on the next pluck (the page moved
        the lock on to the next string while the tuned one still rings)."""
        if strings is not None:
            self.targets = {6 - i: int(m) for i, m in enumerate(strings)}
            self.current = self.pending = None
            self._new_note()
        if a4 is not None:
            self.a4 = float(a4)
            self._new_note()
        if lock != "keep":
            self.lock = lock if lock in self.targets else None
            if self.lock is not None:
                self.current = self.lock
            self._new_note()
        if wait:
            self._new_note()
            self.awaiting_onset = True
            self.peak_rms = 0.0

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

    def _onset(self, rms: float, hf: float) -> bool:
        floor = _db(MIN_LEVEL_DB)
        quiet = max(min(self.levels), floor / 4) if self.levels else floor / 4
        hf_quiet = max(min(self.hf_levels), floor / 4) if self.hf_levels else floor / 4
        self.levels.append(rms)
        self.hf_levels.append(hf)
        if rms < floor or self.t - self.last_onset < ONSET_GAP_S:
            return False
        louder = rms > ONSET_JUMP * quiet and rms > self.peak_rms * _db(-REPLUCK_DB)
        attack = hf > _db(HF_FLOOR_DB) and hf > HF_JUMP * hf_quiet and self.t >= self.guard_until
        if louder or attack:
            self.last_onset = self.t
            return True
        return False

    def feed(self, chunk: np.ndarray) -> dict:
        """Add a chunk of mono samples; returns the reading for the display:
        {"state": "silent"} | {"state": "pending"} | {"state": "note", ...}
        | {"state": "tuned", ..., "new": first reading of this lock-in?}."""
        raw = np.asarray(chunk, dtype=np.float64).ravel()
        self.t += raw.size / self.raw_sr
        if raw.size == 0:
            return {"state": "pending"}
        hf_part, self.hf_zi = sosfilt(self.hf_sos, raw, zi=self.hf_zi)
        hf = float(np.sqrt(np.mean(hf_part ** 2)))
        x, self.zi = sosfilt(self.sos, raw, zi=self.zi)
        if self.decim > 1:
            x = x[: x.size // self.decim * self.decim].reshape(-1, self.decim).mean(axis=1)
        self.buf = np.concatenate([self.buf, x])[-WINDOW:]
        rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

        onset = self._onset(rms, hf)
        if onset:
            self.peak_rms = rms
            self.awaiting_onset = False
            self.ignore_m = None
            self._new_note()
        else:
            self.peak_rms = max(self.peak_rms, rms)

        est = self.estimate()
        if est is None and self.t - self.last_onset < ONSET_GRACE_S:
            # the pluck just started: not enough of it in the window for a pitch. Counting
            # this as silence would forget the pluck and then wait for another
            return self._frozen() or {"state": "pending"}
        if est is None:
            self.unclear += 1
            if self.unclear >= FORGET_AFTER:
                self._new_note()
                self.awaiting_onset = True       # what's heard next without a pluck is the tail
                self.peak_rms = 0.0
                self.ignore_m = None
                if self.lock is None:
                    self.current = self.pending = None
                return {"state": "silent"}
            return self._frozen() or {"state": "silent"}
        self.unclear = 0
        if self.latched is not None or self.awaiting_onset:
            if not self._missed_pluck(est[0], rms):
                return self._frozen() or {"state": "silent"}
            self._new_note()
            self.awaiting_onset, self.ignore_m = False, None
        if rms < self.peak_rms * _db(-TAIL_DB):
            return {"state": "pending"}
        hz, conf = est
        if self.lock is None and min(abs(1200 * math.log2(hz / f)) for f in HUM_NOTCHES[:2]) < HUM_IGNORE_CENTS:
            return {"state": "pending"}
        m = self.midi_of(hz)

        # which string, before anything is changed
        if self.lock is not None:
            string = self.lock
            m = self._fold(m, self.targets[string])
        else:
            string = min(self.targets, key=lambda s: abs(m - self.targets[s]))
            if self.current is not None and string != self.current and not onset:
                cur = self.targets[self.current]
                if abs(m - 12 - cur) < OCTAVE_SNAP:
                    string, m = self.current, m - 12        # the current string's octave error

        # a ringing note keeps its pitch (compared with the median of its last few
        # readings; with fewer than 3, the last one: the mean of two isn't either)
        ref = np.median(self.recent[-MEDIAN_N:]) if len(self.recent) >= 3 else (self.recent or [m])[-1]
        if len(self.recent) >= 2 and abs(m - ref) > CONTINUITY_ST:
            self.rejects.append(m)
            last = self.rejects[-REESTABLISH:]
            if (len(last) < REESTABLISH or np.ptp(last) > 0.3
                    or rms < self.peak_rms * _db(-STRONG_DB)):
                return {"state": "pending"}
            self._new_note()                               # it was the start that was wrong
        self.rejects = []

        if self.lock is None:
            # right after a pluck a new string is expected: no SWITCH_AFTER wait (the
            # two agreeing readings below still keep out a blip). With other strings
            # still ringing, the new one may give only a couple of clean readings
            fresh = not self.recent and self.t - self.last_onset < FRESH_S
            if self.current is not None and string != self.current and not fresh:
                if string == self.pending:
                    self.pending_n += 1
                else:
                    self.pending, self.pending_n = string, 1
                if self.pending_n < SWITCH_AFTER:
                    return {"state": "pending"}
                self._new_note()
            self.current, self.pending, self.pending_n = string, None, 0

        target = self.targets[string]
        self.recent.append(m)
        self.hist.append((self.t, 100.0 * (m - target)))
        if len(self.recent) == 2 and abs(self.recent[1] - self.recent[0]) > CONTINUITY_ST:
            del self.recent[0]                             # a first reading from the pick's attack
        if len(self.recent) < 2:
            return {"state": "pending"}                    # a note starts on two agreeing readings
        shown = float(np.median(self.recent[-MEDIAN_N:]))
        reading = self._reading(string, shown, conf)
        tuned = self._tuned_cents()
        if tuned is not None:
            self.latched = self._reading(string, target + tuned / 100.0, conf)
            self.latched["state"] = "tuned"
            self.guard_until = self.t + LATCH_GUARD_S
            self.ignore_m, self.missed = target + tuned / 100.0, []
            return {**self.latched, "new": True}
        return reading

    def _missed_pluck(self, hz: float, rms: float) -> bool:
        """A pluck the onset test didn't see (it came during the guard after a
        lock-in): a loud, steady pitch away from the note that locked in. The
        tuned string ringing on, its octave, or a quiet sympathetic string isn't."""
        if self.ignore_m is None or self.t < self.guard_until:
            self.missed = []
            return False
        m = self.midi_of(hz)
        d = m - self.ignore_m
        if (abs(d) <= CONTINUITY_ST or abs(d - 12) <= OCTAVE_SNAP
                or rms < self.peak_rms * _db(-STRONG_DB)):
            self.missed = []
            return False
        self.missed.append(m)
        return len(self.missed) >= SWITCH_AFTER and np.ptp(self.missed[-SWITCH_AFTER:]) < 0.3

    def _reading(self, string: int, m: float, conf: float) -> dict:
        target = self.targets[string]
        offset = m - target
        return {
            "state": "note", "string": string, "hz": round(self.a4 * 2 ** ((m - 69) / 12), 2),
            "target_midi": target, "heard": note_label(int(round(m)), SHARP),
            "cents": round(100.0 * offset, 1),
            "semitones_off": int(round(offset)), "confidence": round(conf, 3),
            "locked": self.lock is not None,
        }

    def _frozen(self) -> dict | None:
        return None if self.latched is None else {**self.latched, "new": False}

    def _tuned_cents(self) -> float | None:
        """The note's average offset once it has settled in tune, else None."""
        for hold, spread, drift in ((TUNED_HOLD_S, TUNED_SPREAD, TUNED_DRIFT),
                                    (WOBBLE_HOLD_S, WOBBLE_SPREAD, WOBBLE_DRIFT)):
            if not self.hist or self.t - self.hist[0][0] < hold:
                continue
            ts, cs = map(np.array, zip(*[(t, c) for t, c in self.hist if t > self.t - hold]))
            if cs.size < 4:
                continue
            mean = float(cs.mean())
            trend = float(np.polyfit(ts, cs, 1)[0]) * (ts[-1] - ts[0])
            if abs(mean) <= TUNED_CENTS and cs.std() <= spread and abs(trend) <= drift:
                return mean
        return None

    @staticmethod
    def _fold(m: float, target: int) -> float:
        """An octave error of a locked string: a reading about an octave from the
        target (and far from the target itself) is moved back by that octave. A
        string that is just very slack, say 7 semitones flat, is left alone."""
        for k in (1, -1):
            if abs(m - 12 * k - target) < 2.0 and abs(m - target) > 9.0:
                return m - 12 * k
        return m
