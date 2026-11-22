"use strict";
// Stems played in sync in the browser: the splitter's player, the chord finder's
// Vocals/Guitar sliders and the mixer all use this one. Load after common.js.
//
// A "set" is one <audio> per stem, kept in sync and mixed through Web Audio
// gains. Each stem is fetched whole (compressed FLAC, ~12 MB for 4 minutes;
// decoded AudioBuffers would be ~0.5 GB for six) and played from a blob URL:
// six streaming <audio> elements would hold all six of the browser's
// connections to the server and stall every other request. The first stem is
// the clock, or an outside <audio> the set follows (follow()); the others are
// nudged back when they drift.
//
// opts:
//   rate(set)    playbackRate for a set's tracks (the splitter's Fast speed)
//   gain(name)   a track's level, read when a set is made and by applyGains()
//   onTick(t)    every animation frame while playing, after the resync (song time)
//   onEnded()    the clock stem reached the end (not called while following)
//   pan(name)    optional: a stereo position -1..1 per track (the mixer's pan)
//   meters       optional: an AnalyserNode per track (track.meter), for level meters

const LEADER_DRIFT = 0.12;  // s the silent outside player may wander from the set (follow())
const SEEK_SLACK = 0.3;     // s: a seek on the outside player closer than this to the set is ours

function createStemPlayer({ rate = () => 1, gain = () => 1, onTick = null, onEnded = null,
                            drift = 0.04, xfade = 0.08, pan = null, meters = false } = {}) {
  const S = { ctx: null, out: null, set: null, playing: false, raf: 0, token: 0,
              blobs: new Map(), leader: null };

  function context() {
    if (!S.ctx) {
      S.ctx = new AudioContext();
      S.out = S.ctx.createGain();
      S.out.connect(S.ctx.destination);
    }
    return S.ctx;
  }

  // where the set's tracks should be, in their own time (a set made at 70% runs longer)
  // Following an outside <audio> (follow()): it says where to start and stop, but
  // while the set plays, the set keeps its own clock (its first track, as without a
  // leader) and the outside player, silent then, is nudged to it. The two play
  // through different paths whose clocks drift; seeking the audible stems to the
  // leader made them stutter.
  const leading = () => S.leader && !S.playing;
  const trackTime = (set) => (leading() ? S.leader.currentTime / set.speed
    : set.tracks[0].audio.currentTime);
  // positions are in song time
  const now = () => (leading() ? S.leader.currentTime
    : (S.set?.tracks[0]?.audio.currentTime || 0) * (S.set?.speed || 1));

  function blob(url, label) {
    if (!S.blobs.has(url)) {
      S.blobs.set(url, fetch(apiUrl(url)).then((res) => {
        if (!res.ok) throw new Error(`couldn't load ${label} (${res.status})`);
        return res.blob();
      }));
      S.blobs.get(url).catch(() => S.blobs.delete(url));   // a failed fetch can be tried again
    }
    return S.blobs.get(url);
  }

  // keep only the stems these sets use
  function keep(...sets) {
    const used = new Set(sets.flatMap((s) => (s ? Object.values(s.urls) : [])));
    for (const url of S.blobs.keys()) if (!used.has(url)) S.blobs.delete(url);
  }

  // tracks: [{name, url, label}]; extra: fields kept on the set (speed, key, quality, ...)
  async function makeSet(tracks, extra = {}) {
    const blobs = await Promise.all(tracks.map((t) => blob(t.url, t.label || t.name)));
    const ctx = context();
    const set = { speed: 1, ...extra, bus: ctx.createGain(), tracks: [],
                  urls: Object.fromEntries(tracks.map((t) => [t.name, t.url])) };
    set.bus.connect(S.out);
    set.tracks = tracks.map((t, i) => {
      const audio = new Audio();
      audio.preload = "auto";
      audio.preservesPitch = true;
      audio.webkitPreservesPitch = true;
      audio.src = URL.createObjectURL(blobs[i]);
      audio.defaultPlaybackRate = audio.playbackRate = rate(set);
      const g = ctx.createGain();
      g.gain.value = gain(t.name);
      let last = ctx.createMediaElementSource(audio).connect(g);
      let panner = null, meter = null;
      if (pan) {
        panner = ctx.createStereoPanner();
        panner.pan.value = pan(t.name);
        last = last.connect(panner);
      }
      last.connect(set.bus);
      if (meters) {
        meter = ctx.createAnalyser();
        meter.fftSize = 1024;
        last.connect(meter);
      }
      return { name: t.name, audio, gain: g, panner, meter };
    });
    set.tracks[0].audio.addEventListener("ended", () => {
      if (set === S.set && !S.leader) onEnded?.();
    });
    return set;
  }

  function dropSet(set) {
    for (const t of set.tracks) {
      t.audio.pause();
      URL.revokeObjectURL(t.audio.src);
      t.audio.removeAttribute("src");
      t.audio.load();
      t.gain.disconnect();
      t.panner?.disconnect();
      t.meter?.disconnect();
    }
    set.bus.disconnect();
  }

  const ready = (a) => a.readyState >= 3 || new Promise((r) => {
    a.addEventListener("canplay", r, { once: true });
  });

  function teardown() {
    pause();
    S.blobs.clear();
    S.token++;
    if (S.set) dropSet(S.set);
    S.set = null;
  }

  // Make the first set (after a teardown). valid() can veto it, e.g. when the page
  // moved to another song meanwhile. False when superseded.
  async function load(build, valid = () => true) {
    teardown();
    const token = S.token;
    const set = await build();
    if (token !== S.token || !valid()) { dropSet(set); return false; }
    S.set = set;
    return true;
  }

  async function play() {
    const set = S.set;
    if (!set) return false;
    await context().resume();
    const t = trackTime(set);
    for (const tr of set.tracks) tr.audio.currentTime = t;
    await Promise.all(set.tracks.map((tr) => ready(tr.audio)));
    if (set !== S.set) return false;  // swapped meanwhile; that swap took over
    await Promise.all(set.tracks.map((tr) => tr.audio.play()));
    S.playing = true;
    cancelAnimationFrame(S.raf);
    S.raf = requestAnimationFrame(tick);
    return true;
  }

  function pause() {
    S.playing = false;
    cancelAnimationFrame(S.raf);
    if (S.set) {
      const t = trackTime(S.set);
      for (const tr of S.set.tracks) { tr.audio.pause(); tr.audio.currentTime = t; }
    }
  }

  // t in song time
  function seek(t) {
    if (S.set) for (const tr of S.set.tracks) tr.audio.currentTime = t / S.set.speed;
  }

  function resync() {
    const t = trackTime(S.set);
    for (const tr of S.set.tracks.slice(1)) {
      if (!tr.audio.seeking && Math.abs(tr.audio.currentTime - t) > drift) tr.audio.currentTime = t;
    }
    const m = S.leader;
    if (m && !m.paused && !m.seeking && Math.abs(m.currentTime - t * S.set.speed) > LEADER_DRIFT) {
      m.currentTime = t * S.set.speed;     // silent while the set plays: a jump isn't heard
    }
  }

  function tick() {
    if (!S.playing || !S.set) return;
    resync();
    onTick?.(now());
    if (S.playing) S.raf = requestAnimationFrame(tick);
  }

  // Swap to another set without a gap: the new set loads and lines up while the
  // old one keeps playing, then they crossfade. False when superseded or vetoed.
  async function swap(build, valid = () => true) {
    const token = ++S.token;
    const next = await build();
    const stale = () => token !== S.token || !valid();
    if (stale()) { dropSet(next); return false; }
    next.bus.gain.value = 0;
    for (const tr of next.tracks) tr.audio.currentTime = now() / next.speed;
    await Promise.all(next.tracks.map((tr) => ready(tr.audio)));
    if (stale()) { dropSet(next); return false; }
    for (const tr of next.tracks) {
      tr.audio.defaultPlaybackRate = tr.audio.playbackRate = rate(next);
      tr.gain.gain.value = gain(tr.name);
      if (tr.panner) tr.panner.pan.value = pan(tr.name);
    }
    const t = now() / next.speed;
    for (const tr of next.tracks) tr.audio.currentTime = t;
    const old = S.set;
    if (S.playing) {
      await Promise.all(next.tracks.map((tr) => tr.audio.play()));
      if (token !== S.token || !S.playing) {  // superseded or paused while starting
        if (token !== S.token) { dropSet(next); return false; }
        for (const tr of next.tracks) tr.audio.pause();
      }
    }
    S.set = next;
    const at = S.ctx.currentTime;
    if (S.playing && old) {
      old.bus.gain.setValueAtTime(1, at);
      old.bus.gain.linearRampToValueAtTime(0, at + xfade);
      next.bus.gain.setValueAtTime(0, at);
      next.bus.gain.linearRampToValueAtTime(1, at + xfade);
      setTimeout(() => dropSet(old), xfade * 1000 + 150);
    } else {
      next.bus.gain.value = 1;
      if (old) dropSet(old);
    }
    keep(next);
    return true;
  }

  function applyGains(instant = false) {
    for (const tr of S.set?.tracks || []) {
      const g = gain(tr.name);
      if (instant || !S.ctx) tr.gain.gain.value = g;
      else tr.gain.gain.setTargetAtTime(g, S.ctx.currentTime, 0.015);  // no clicks
    }
  }

  function applyPans() {
    if (!pan) return;
    for (const tr of S.set?.tracks || []) {
      if (tr.panner) tr.panner.pan.setTargetAtTime(pan(tr.name), S.ctx.currentTime, 0.015);
    }
  }

  function applyRate() {
    for (const tr of S.set?.tracks || []) tr.audio.defaultPlaybackRate = tr.audio.playbackRate = rate(S.set);
  }

  // Play in step with an outside <audio> (the chord finder's own player) instead of
  // a clock stem: its play / pause / seek drive the set, which is resynced to it.
  // active(): whether the set should sound at all right now.
  function follow(media, active = () => true) {
    S.leader = media;
    media.addEventListener("play", () => { if (S.set && active()) play(); });
    media.addEventListener("pause", () => pause());
    // a seek the listener made moves the set; one of ours (resync) lands where the set is
    media.addEventListener("seeked", () => {
      if (S.set && Math.abs(media.currentTime - now()) > SEEK_SLACK) seek(media.currentTime);
    });
  }

  return {
    context, makeSet, dropSet, load, swap, teardown, play, pause, seek, now,
    applyGains, applyPans, applyRate, follow, keep,
    get set() { return S.set; },
    get playing() { return S.playing; },
    get ctx() { return S.ctx; },
    get output() { return S.out; },
  };
}
