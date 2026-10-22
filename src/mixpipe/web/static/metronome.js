"use strict";
// The splitter's metronome: a click on the song's own beats and bars (the chord
// finder's beat_times and bar grid, not a fixed tempo, so it follows a song
// that drifts), scheduled a little ahead on the Web Audio clock so it lands on
// the beat at any speed, and a one-bar count-in.
//
//   const metro = createMetronome();
//   metro.attach(ctx, destination); metro.setGrid(beatTimes, bars);
//   metro.update(songNow, rate, playing)   every animation frame
//   await metro.countIn(songAt, rate)      before starting at songAt
//
// songNow / songAt are positions in the song as recorded (seconds), rate is
// how many song seconds play per second (the speed).

function createMetronome() {
  const LEAD = 0.12;     // seconds of clicks scheduled ahead of the playhead
  const m = { ctx: null, out: null, volume: 0.7, beats: [], bars: [], accents: new Set(),
              on: false, next: null, last: 0 };

  function attach(ctx, destination) {
    if (m.ctx === ctx) return;
    m.ctx = ctx;
    m.out = ctx.createGain();
    m.out.gain.value = m.volume;
    m.out.connect(destination);
  }

  function setGrid(beats, bars) {
    m.beats = beats || [];
    m.bars = bars || [];
    const starts = m.bars.map((b) => b.start);
    m.accents = new Set(m.beats.flatMap((t, i) => (starts.some((s) => Math.abs(s - t) < 0.002) ? [i] : [])));
    m.next = null;
  }

  function click(at, accent) {
    const osc = m.ctx.createOscillator();
    const g = m.ctx.createGain();
    osc.frequency.value = accent ? 1760 : 1175;
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(accent ? 1 : 0.55, at + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, at + 0.06);
    osc.connect(g).connect(m.out);
    osc.start(at);
    osc.stop(at + 0.07);
  }

  function update(songNow, rate, playing) {
    if (!m.on || !playing || !m.ctx || !m.beats.length) { m.next = null; return; }
    if (m.next === null || songNow < m.last - 0.05 || songNow - m.last > 0.5) {   // started, looped, seeked
      m.next = m.beats.findIndex((t) => t >= songNow - 0.01);
      if (m.next < 0) m.next = m.beats.length;
    }
    m.last = songNow;
    while (m.next < m.beats.length && (m.beats[m.next] - songNow) / rate < LEAD) {
      click(m.ctx.currentTime + Math.max(0, (m.beats[m.next] - songNow) / rate), m.accents.has(m.next));
      m.next++;
    }
  }

  // one bar of clicks, as long as the bar at songAt and at its beat spacing
  function countIn(songAt, rate) {
    if (!m.ctx || m.beats.length < 2) return Promise.resolve();
    const bar = [...m.bars].reverse().find((b) => b.start <= songAt + 0.01) || m.bars[0];
    const n = bar?.beats || 4;
    const i = Math.min(Math.max(1, m.beats.findIndex((t) => t >= songAt)), m.beats.length - 1);
    const period = (m.beats[i] - m.beats[i - 1]) / rate;
    const t0 = m.ctx.currentTime + 0.05;
    for (let k = 0; k < n; k++) click(t0 + k * period, k === 0);
    m.next = null;
    return new Promise((done) => setTimeout(done, (0.05 + n * period) * 1000));
  }

  return {
    attach, setGrid, update, countIn,
    get ready() { return m.beats.length > 1; },
    get on() { return m.on; },
    set on(v) { m.on = !!v; m.next = null; },
    setVolume(v) { m.volume = v; if (m.out) m.out.gain.value = v; },
  };
}
