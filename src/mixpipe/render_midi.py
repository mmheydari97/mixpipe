"""MIDI rendering, for Seaboard captures saved as MIDI rather than audio.

`render` plays a MIDI file through FluidSynth and a SoundFont; `render_mpe` first
rewrites the file so FluidSynth plays the Seaboard's per-note expression.

MPE (MIDI Polyphonic Expression) gives every sounding note its own MIDI channel,
so each note can carry its own pitch bend (glide), channel pressure (press) and
CC74 (slide). FluidSynth has no MPE mode, but it does keep pitch bend, pressure
and CC74 per channel, which comes to the same thing once every member channel
has the instrument and a bend range matching the controller's (+/-48 semitones
by default for MPE). The Seaboard sends that setup itself, but a trimmed
capture often lost it, so `prepare_mpe` puts it back where it is missing. Slide
(CC74) drives FluidSynth's default brightness (filter cutoff) modulator and
pressure its default vibrato, so a SoundFont that responds to those sounds
expressive; one that doesn't will play the right notes with the right glides.

Needs the `fluidsynth` binary (brew install fluid-synth) and a .sf2 file. If you
would rather skip all of this, bounce audio out of Equator and drop the WAV in
stems/.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE


# --------------------------------------------------------------------------- #
# a small Standard MIDI File reader/writer
# --------------------------------------------------------------------------- #


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def _write_vlq(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def read_smf(path: str | os.PathLike) -> dict:
    """Parse a .mid file into {"format", "division", "tracks"}.

    Each track is a list of (absolute_tick, raw_event_bytes) in file order.
    Channel events are stored with their status byte (running status expanded),
    meta events as FF type len data, sysex as F0/F7 len data. The end-of-track
    marker is dropped; `write_smf` adds it back.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"{path}: not a MIDI file")
    fmt, n_tracks, division = struct.unpack(">HHH", data[8:14])
    pos = 8 + struct.unpack(">I", data[4:8])[0]
    tracks = []
    for _ in range(n_tracks):
        if data[pos:pos + 4] != b"MTrk":
            raise ValueError(f"{path}: damaged track header")
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        pos += 8
        end = pos + length
        events, tick, status = [], 0, 0
        while pos < end:
            delta, pos = _read_vlq(data, pos)
            tick += delta
            b = data[pos]
            if b == 0xFF:
                kind = data[pos + 1]
                size, p2 = _read_vlq(data, pos + 2)
                raw = data[pos:p2 + size]
                pos = p2 + size
                if kind != 0x2F:
                    events.append((tick, raw))
            elif b in (0xF0, 0xF7):
                size, p2 = _read_vlq(data, pos + 1)
                events.append((tick, data[pos:p2 + size]))
                pos = p2 + size
            else:
                if b & 0x80:
                    status = b
                    pos += 1
                n_data = 1 if (status & 0xF0) in (0xC0, 0xD0) else 2
                events.append((tick, bytes([status]) + data[pos:pos + n_data]))
                pos += n_data
        tracks.append(events)
        pos = end
    return {"format": fmt, "division": division, "tracks": tracks}


def write_smf(smf: dict, path: str | os.PathLike) -> str:
    out = [b"MThd", struct.pack(">IHHH", 6, smf["format"], len(smf["tracks"]), smf["division"])]
    for events in smf["tracks"]:
        body = bytearray()
        last = 0
        for tick, raw in sorted(events, key=lambda e: e[0]):   # stable: keeps file order
            body += _write_vlq(tick - last) + raw
            last = tick
        body += _write_vlq(0) + b"\xff\x2f\x00"
        out += [b"MTrk", struct.pack(">I", len(body)), bytes(body)]
    Path(path).write_bytes(b"".join(out))
    return str(path)


# --------------------------------------------------------------------------- #
# MPE setup
# --------------------------------------------------------------------------- #


def _channel_events(smf: dict):
    for events in smf["tracks"]:
        for tick, raw in events:
            if raw[0] < 0xF0:
                yield tick, raw[0] & 0x0F, raw[0] & 0xF0, raw[1:]


def prepare_mpe(smf: dict, bend_range: int = 48, program: int = 0) -> dict:
    """Make sure every channel that plays a note has an instrument and a bend range.

    For each such channel, at tick 0: a program change (copied from the first
    one found in the file, else `program`) unless it has its own, and RPN 0
    (pitch-bend sensitivity) = `bend_range` semitones unless the file already
    sets one. Returns a new dict; the input is not modified.
    """
    used, has_prog, has_rpn, first_prog = set(), set(), set(), None
    rpn_msb = {}
    for tick, ch, kind, data in _channel_events(smf):
        if kind == 0x90 and data[1] > 0:
            used.add(ch)
        elif kind == 0xC0:
            has_prog.add(ch)
            if first_prog is None:
                first_prog = data[0]
        elif kind == 0xB0:
            if data[0] == 101:
                rpn_msb[ch] = data[1]
            elif data[0] == 100 and rpn_msb.get(ch) == 0 and data[1] == 0:
                has_rpn.add(ch)
    prog = first_prog if first_prog is not None else program

    setup = []
    for ch in sorted(used):
        if ch not in has_prog:
            setup.append((0, bytes([0xC0 | ch, prog])))
        if ch not in has_rpn:
            for cc, val in ((101, 0), (100, 0), (6, int(bend_range)), (38, 0),
                            (101, 127), (100, 127)):
                setup.append((0, bytes([0xB0 | ch, cc, val])))
    tracks = [list(t) for t in smf["tracks"]]
    if not tracks:
        tracks = [[]]
    tracks[0] = setup + tracks[0]          # tick 0, ahead of everything else
    return {**smf, "tracks": tracks}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _fluidsynth() -> str:
    exe = shutil.which("fluidsynth")
    if not exe:
        raise RuntimeError(
            "rendering MIDI needs FluidSynth: brew install fluid-synth "
            "(or bounce audio from your synth and put the WAV in stems/)"
        )
    return exe


def render(
    midi_path: str, soundfont_path: str, sr: int = SAMPLE_RATE, gain: float = 0.8
) -> np.ndarray:
    """Render a MIDI file to stereo audio via FluidSynth and a SoundFont.

    Runs `fluidsynth -ni -g <gain> -r <sr> -F out.wav <sf2> <mid>` (the tail of
    reverb and release is included). Returns float32 (n, 2) at `sr`.
    """
    from .io_utils import normalize_shape, read_audio, to_sample_rate

    if not Path(midi_path).is_file():
        raise FileNotFoundError(f"no such MIDI file: {midi_path}")
    if not Path(soundfont_path).is_file():
        raise FileNotFoundError(f"no such SoundFont: {soundfont_path}")
    exe = _fluidsynth()
    with tempfile.TemporaryDirectory(prefix="mixpipe-midi-") as tmp:
        wav = Path(tmp) / "render.wav"
        proc = subprocess.run(
            [exe, "-ni", "-g", f"{gain}", "-r", str(sr), "-F", str(wav),
             str(soundfont_path), str(midi_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not wav.exists():
            raise RuntimeError(f"fluidsynth failed: {(proc.stderr or proc.stdout)[-300:]}")
        audio, file_sr = read_audio(wav)
    audio = to_sample_rate(normalize_shape(audio), file_sr, sr)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    return audio.astype(np.float32)


def render_mpe(
    midi_path: str, sr: int = SAMPLE_RATE, soundfont_path: str | None = None,
    bend_range: int = 48, program: int = 0,
) -> np.ndarray:
    """Render a Seaboard (MPE) capture with per-note glide, press and slide.

    See the module docstring: the file is rewritten with a program change and a
    +/-`bend_range` pitch-bend sensitivity on every channel that plays notes
    (where it doesn't already have them), then rendered with `render`. The
    SoundFont comes from `soundfont_path` or the MIXPIPE_SOUNDFONT environment
    variable.
    """
    soundfont_path = soundfont_path or os.environ.get("MIXPIPE_SOUNDFONT")
    if not soundfont_path:
        raise RuntimeError(
            "render_mpe needs a SoundFont: pass soundfont_path or set MIXPIPE_SOUNDFONT"
        )
    smf = prepare_mpe(read_smf(midi_path), bend_range=bend_range, program=program)
    with tempfile.TemporaryDirectory(prefix="mixpipe-mpe-") as tmp:
        prepared = write_smf(smf, Path(tmp) / "mpe.mid")
        return render(prepared, soundfont_path, sr)
