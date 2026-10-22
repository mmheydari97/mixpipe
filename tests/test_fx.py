"""fx.py: the effects the mixer and the splitter share (drive, chorus, tone,
the reverb/echo returns, tempo-synced echo times)."""

from __future__ import annotations

import numpy as np
import pytest

from mixpipe import fx
from mixpipe.eq import apply_eq_chain

SR = 48_000


def sine(f: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def magnitude(x: np.ndarray, f: float) -> float:
    x = x if x.ndim == 1 else x[:, 0]
    t = np.arange(len(x)) / SR
    return abs(np.dot(x, np.exp(-2j * np.pi * f * t))) / len(x)


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_drive_off_is_the_input():
    x = sine(440)
    assert fx.drive(x, SR, 0.0) is x


def test_drive_adds_odd_harmonics_at_the_same_level():
    x = sine(440)
    warm, hot = fx.drive(x, SR, 0.2, tone=1.0), fx.drive(x, SR, 0.9, tone=1.0)
    assert magnitude(warm, 1320) > 20 * magnitude(x, 1320)        # a third harmonic appears
    assert magnitude(hot, 1320) > 2 * magnitude(warm, 1320)       # and grows with the amount
    for y in (warm, hot):
        assert rms_db(y) == pytest.approx(rms_db(x), abs=0.3)     # grit, not volume
        assert y.shape == x.shape and y.dtype == np.float32


def test_drive_tone_darkens_the_grit():
    x = sine(440)
    dark, open_ = fx.drive(x, SR, 0.8, tone=0.0), fx.drive(x, SR, 0.8, tone=1.0)
    assert magnitude(dark, 440 * 9) < 0.5 * magnitude(open_, 440 * 9)


def test_chorus_off_is_the_input_and_on_it_moves():
    x = sine(440, 2.0)
    assert fx.chorus(x, SR, 0.0) is x
    y = fx.chorus(x, SR, 1.0, rate_hz=1.0)
    assert y.shape == x.shape
    # the delayed copies beat against the dry sound: the level swings over time
    blocks = np.sqrt(np.mean(y[: SR * 2 // 4800 * 4800].reshape(-1, 4800) ** 2, axis=1))
    assert blocks.max() - blocks.min() > 0.05


def test_chorus_widens_a_stereo_source():
    x = sine(440, 1.0)
    st = np.stack([x, x], axis=1)
    y = fx.chorus(st, SR, 0.8)
    assert y.shape == st.shape and not np.allclose(y[:, 0], y[:, 1], atol=1e-3)


def test_color_runs_drive_then_chorus_and_nothing_when_off():
    x = sine(220)
    assert fx.color(x, SR, {}) is x
    both = fx.color(x, SR, {"drive": {"amount": 0.5, "tone": 1.0}, "chorus": {"mix": 0.5}})
    by_hand = fx.chorus(fx.drive(x, SR, 0.5, 1.0), SR, 0.5)
    assert np.allclose(both, by_hand)


def test_tone_bands_are_shelves_the_eq_applies():
    assert fx.tone_bands(0, 0) == []
    bands = fx.tone_bands(-6.0, 3.0)
    assert [b["type"] for b in bands] == ["low_shelf", "high_shelf"]
    x = sine(30) + sine(16000)                     # well inside each shelf
    y = apply_eq_chain(x, hpf_hz=0, lpf_hz=0, bands=bands, sr=SR)
    assert 20 * np.log10(magnitude(y, 30) / magnitude(x, 30)) == pytest.approx(-6.0, abs=0.7)
    assert 20 * np.log10(magnitude(y, 16000) / magnitude(x, 16000)) == pytest.approx(3.0, abs=0.7)


def test_echo_times_follow_the_tempo():
    assert fx.delay_ms_for("1/8", 120) == pytest.approx(250.0)
    assert fx.delay_ms_for("1/8d", 120) == pytest.approx(375.0)
    assert fx.delay_ms_for("1/4t", 90) == pytest.approx(60000 / 90 * 2 / 3)
    assert fx.delay_ms_for("", 120) is None and fx.delay_ms_for("1/8", None) is None
    assert fx.delay_ms_for("1/5", 120) is None


BUSES = {"reverb": {"room_size": 0.5, "decay_s": 1.0, "damping": 0.4, "pre_delay_ms": 0.0,
                    "width": 1.0, "return_db": 0.0},
         "delay": {"delay_ms": 120.0, "feedback": 0.0, "damping": 0.0, "return_db": 0.0, "sync": ""}}


def test_send_returns_is_silent_without_sends():
    out = fx.send_returns([(sine(440), fx.OFF_DB, fx.OFF_DB)], BUSES, SR)
    assert out.shape == (SR, 2) and not out.any()


def test_a_synced_echo_lands_on_the_beat_and_tail_off_keeps_the_length():
    click = np.zeros(SR, dtype=np.float32)
    click[0] = 1.0
    buses = {**BUSES, "delay": {**BUSES["delay"], "sync": "1/8"}}
    out = fx.send_returns([(click, fx.OFF_DB, 0.0)], buses, SR, bpm=120, tail=False)
    assert out.shape == (SR, 2)                                   # tail off: as long as the input
    assert int(np.argmax(np.abs(out[:, 0]))) == pytest.approx(SR * 0.25, abs=2)   # 1/8 at 120 BPM
    free = fx.send_returns([(click, fx.OFF_DB, 0.0)], BUSES, SR, bpm=120, tail=False)
    assert int(np.argmax(np.abs(free[:, 0]))) == pytest.approx(SR * 0.12, abs=2)  # no sync: 120 ms
