"use strict";
// A small Web Audio synth shared by the pages: the chord finder's chord audition
// (synth.play, added by app.js), the harmonica's notes and the tuner's reference
// tones (synth.playNote) and its "in tune" chime (synth.chime). Load before the
// page script.

const synth = {
  ctx: null,
  nodes: [],
  ensure() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  },
  stop() {
    for (const n of this.nodes) {
      try { n.stop ? n.stop() : null; } catch { /* already stopped */ }
      try { n.disconnect(); } catch { /* already disconnected */ }
    }
    this.nodes = [];
  },
};

// A soft bell: sine notes one after another (`gap` seconds apart), each dying away
// over `ring` seconds. Doesn't stop a note already playing. The tuner's "in tune"
// confirmation; keep it high (above the tuner's low-pass) and short, since the
// microphone hears it too.
synth.chime = function (midis, { gap = 0.11, ring = 0.32, level = 0.07 } = {}) {
  const ctx = this.ensure();
  const out = ctx.createGain();
  out.gain.value = level;
  out.connect(ctx.destination);
  const t0 = ctx.currentTime + 0.02;
  const notes = [].concat(midis);
  notes.forEach((m, i) => {
    const start = t0 + i * gap;
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = 440 * 2 ** ((m - 69) / 12);
    const env = ctx.createGain();
    env.gain.setValueAtTime(0, start);
    env.gain.linearRampToValueAtTime(1, start + 0.006);
    env.gain.exponentialRampToValueAtTime(0.001, start + ring);
    osc.connect(env); env.connect(out);
    osc.start(start); osc.stop(start + ring + 0.02);
    osc.onended = () => {
      try { env.disconnect(); if (i === notes.length - 1) out.disconnect(); } catch { /* gone */ }
    };
  });
};

// A reed-ish tone: detuned saw + square through a lowpass, with a soft attack.
// `midi` is one note or a list (a chord: holes played together).
// Fractional MIDI is fine: the tuner passes its A4 offset that way.
synth.playNote = function (midi, duration) {
  this.stop();
  const midis = [].concat(midi);
  const ctx = this.ensure();
  const out = ctx.createGain();
  out.gain.value = 0.12 / Math.sqrt(midis.length);
  const lp = ctx.createBiquadFilter();
  lp.type = "lowpass"; lp.frequency.value = 2200; lp.Q.value = 0.7;
  lp.connect(out); out.connect(ctx.destination);
  this.nodes.push(out, lp);
  const t0 = ctx.currentTime + 0.02;
  const t1 = t0 + Math.max(0.3, duration);
  const env = ctx.createGain();
  env.gain.setValueAtTime(0, t0);
  env.gain.linearRampToValueAtTime(1, t0 + 0.04);
  env.gain.setValueAtTime(1, t1 - 0.08);
  env.gain.linearRampToValueAtTime(0, t1);
  env.connect(lp);
  this.nodes.push(env);
  midis.forEach((m) => {
    const hz = 440 * 2 ** ((m - 69) / 12);
    [["sawtooth", 1, 0.6], ["square", 1.003, 0.35]].forEach(([type, ratio, level]) => {
      const osc = ctx.createOscillator();
      osc.type = type;
      osc.frequency.value = hz * ratio;
      const g = ctx.createGain();
      g.gain.value = level;
      osc.connect(g); g.connect(env);
      osc.start(t0); osc.stop(t1 + 0.02);
      this.nodes.push(osc, g);
    });
  });
};

// A plucked string (Karplus-Strong): noise through a short delay line that loses a
// little each pass, so it rings and fades like a guitar. The guitar tab's "hear".
// A list of notes is strummed (a few ms apart).
synth.pluck = function (midi, duration = 1.2) {
  this.stop();
  const ctx = this.ensure();
  const midis = [].concat(midi);
  const out = ctx.createGain();
  out.gain.value = 0.35 / Math.sqrt(midis.length);
  out.connect(ctx.destination);
  this.nodes.push(out);
  const ring = Math.min(Math.max(duration, 0.4), 3);
  midis.forEach((m, k) => {
    const sr = ctx.sampleRate;
    const period = Math.max(2, Math.round(sr / (440 * 2 ** ((m - 69) / 12))));
    const len = Math.round(sr * (ring + 0.3));
    const buf = ctx.createBuffer(1, len, sr);
    const y = buf.getChannelData(0);
    for (let i = 0; i < period; i++) y[i] = Math.random() * 2 - 1;
    for (let i = period; i < len; i++) y[i] = 0.4985 * (y[i - period] + y[i - period - 1 >= 0 ? i - period - 1 : i - period]);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const env = ctx.createGain();
    const t0 = ctx.currentTime + 0.02 + k * 0.012;
    env.gain.setValueAtTime(1, t0);
    env.gain.setValueAtTime(1, t0 + ring);
    env.gain.linearRampToValueAtTime(0, t0 + ring + 0.25);
    src.connect(env); env.connect(out);
    src.start(t0); src.stop(t0 + ring + 0.3);
    this.nodes.push(src, env);
  });
};
