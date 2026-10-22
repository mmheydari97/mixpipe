"use strict";
// Stem splitter page. Helpers ($, el, api, store, createJobUI, ...) come from common.js.

const fmtClock = (t) => {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60);
  return `${m}:${String(Math.floor(t - m * 60)).padStart(2, "0")}`;
};
const typing = (e) => {
  const t = e.target;
  return t.isContentEditable || t.tagName === "TEXTAREA" || t.tagName === "SELECT"
    || (t.tagName === "INPUT" && t.type !== "range" && t.type !== "checkbox");
};

const { stageLabel, pollStatus } = createJobUI(["decode", "separate"],
  { decode: "Decode", separate: "Separate" });

// =========================================================================
// library view
// =========================================================================

let picker = null;

async function startSplit(rerun = false) {
  if (!picker.hasInput()) return;
  const link = picker.link;
  const btn = $("#split-btn");
  const box = $("#upload-progress");
  btn.disabled = true;
  box.replaceChildren(el("div", { class: "msg" }, link ? "Looking up the link…" : "Uploading…"));
  try {
    const { id } = await api(...picker.request({ rerun }));
    refreshSongs();
    await pollStatus(id, box, {
      onDone: () => { picker.clear(); box.replaceChildren(); location.hash = `#/song/${id}`; },
      onRetry: () => startSplit(true),
    });
  } catch (e) {
    box.replaceChildren(el("div", { class: "msg error" }, e.message));
  } finally {
    btn.disabled = !picker.hasInput();
    refreshSongs();
  }
}

function refreshSongs() {
  return refreshSongList($("#song-list"), { stageLabel, href: (s) => `#/song/${s.id}` });
}

// =========================================================================
// player
// =========================================================================
// A "set" is one <audio> per stem, kept in sync and mixed through Web Audio
// gains. Each stem is fetched whole (compressed FLAC, ~12 MB for 4 minutes;
// decoded AudioBuffers would be ~0.5 GB for six) and played from a blob URL:
// six streaming <audio> elements would hold all six of the browser's
// connections to the server and stall every other request. The first stem
// is the clock; the others are nudged back when they drift.
//
// Speed and key, two engines (P.engine):
// - "browser" (Fast): the browser stretches the audio itself (playbackRate +
//   preservesPitch). Instant, but it smears and warbles away from 100%.
// - "rubberband" (Accurate): the server makes the stems at that speed with
//   Rubber Band (a quick pass, then a cleaner one) and the page crossfades to
//   them. Until they arrive the browser stretches as a preview.
// The browser can't change the key, so a key change always comes from the
// server; with Fast it's made at 100% and the browser adds the speed on top.

const DRIFT = 0.04;   // seconds a stem may wander from the clock before it's resynced
const XFADE = 0.08;   // seconds of crossfade when swapping to another set
const SPEED_MIN = 0.5, SPEED_MAX = 3, SPEED_STEP = 0.1;
const KEY_MAX = 12;   // semitones (= frets) either way
const POLL_MS = 700;
const POLL_PENDING_MS = 400;  // faster while the change isn't audible yet, for the progress readout
const APPLIED_MS = 2000;      // how long "✓ applied" stays up
const ENGINE_KEY = "mixpipe.splitter.engine";
const P = {
  ctx: null, master: null, song: null, set: null, playing: false, dragging: false,
  speed: 1, semi: 0, engine: "rubberband", loop: { a: null, b: null, on: false }, raf: 0,
  versionReq: 0, swapToken: 0, requestTimer: 0, pollTimer: 0,
  hq: { state: "idle", msg: "", progress: null },  // state: idle | pending | fast | fine | error
  keyPending: false, keyFlash: 0,
  tempo: null,          // the song's BPM as recorded, once the chord finder has analysed it
  analysis: null, analysisTimer: 0, heardTimer: 0, counting: false,
  takes: [],            // recordings kept with the song (splitter/takes.py), played like stems
  fxTimer: 0, fxError: "",
  blobs: new Map(),     // url -> Promise<Blob>, so a new set reuses the stems it shares with the old
};
// mixer state, saved per song; fx = { stems: {name: {bass, treble, drive, chorus, reverb, echo}},
// room: {size, decay}, echo: {sync, ms, repeats} } (see effects.py on the server)
const M = { vol: {}, mute: {}, solo: {}, fx: null };
let FX_DEFAULTS = null;   // the room and echo a song starts with, from the server
// the chord finder's sheet for the song, as the chords & lyrics panel shows it (sheet.js)
const SH = { view: null, shift: 0, lyrics: false, timeline: [], chipNodes: new Map(), words: [],
             activeChord: null, activeWord: null };
const METRO = createMetronome();   // metronome.js

const clock = () => P.set?.tracks[0]?.audio;
// positions are in song time; a set made at 70% runs 1/0.7 times longer
const now = () => (clock()?.currentTime || 0) * (P.set?.speed || 1);
const rateFor = (set) => P.speed / set.speed;

function ensureContext() {
  if (!P.ctx) {
    P.ctx = new AudioContext();
    P.master = P.ctx.createGain();
    P.master.connect(P.ctx.destination);
    P.master.gain.value = Number($("#master").value);
    METRO.attach(P.ctx, P.master);
  }
  return P.ctx;
}

function blobFor(url, label) {
  if (!P.blobs.has(url)) {
    P.blobs.set(url, fetch(apiUrl(url)).then((res) => {
      if (!res.ok) throw new Error(`couldn't load ${label} (${res.status})`);
      return res.blob();
    }));
    P.blobs.get(url).catch(() => P.blobs.delete(url));   // a failed fetch can be tried again
  }
  return P.blobs.get(url);
}

// keep only the stems the playing set (and the one being made) use
function keepBlobs(...sets) {
  const used = new Set(sets.flatMap((s) => (s ? Object.values(s.urls) : [])));
  for (const url of P.blobs.keys()) if (!used.has(url)) P.blobs.delete(url);
}

// every channel the player plays: the six stems, then your takes ("take:<n>")
const channels = (song) => [
  ...song.stems.map((s) => ({ name: s.name, label: s.label, take: null })),
  ...P.takes.map((t) => ({ name: `take:${t.n}`, label: t.name, take: t })),
];

// the version the page plays now, to make a set again with the same one
const currentVersion = () => (P.set?.key
  ? { key: P.set.key, quality: P.set.quality, speed: P.set.speed, semi: P.set.semi } : null);

// v: { key, quality, speed, semi } of a server version, or null for the original stems.
// A stem with effects comes from the server's fx render of that version instead,
// and a take as the server stretches it for that version.
async function makeSet(song, v) {
  const base = v ? `/api/songs/${song.id}/versions/${v.key}/${v.quality}` : `/api/songs/${song.id}/stems`;
  const urls = Object.fromEntries(song.stems.map((s) => [s.name, `${base}/${s.name}`]));
  for (const t of P.takes) {
    // the take's placement in the URL, so a nudged take isn't taken from the blob cache
    const at = `at=${encodeURIComponent(t.latency_ms)}`;
    urls[`take:${t.n}`] = `/api/songs/${song.id}/takes/${t.n}?` +
      (v ? `version=${encodeURIComponent(v.key)}&quality=${v.quality}&${at}` : at);
  }
  const fxKey = fxActive() ? JSON.stringify(fxSettings()) : "";
  if (fxKey) {
    const res = await api(`/api/songs/${song.id}/fx`, jsonBody("POST", {
      version: v?.key ?? null, quality: v?.quality ?? null, settings: fxSettings(), tempo: P.tempo }));
    Object.assign(urls, res.stems);
  }
  const chans = channels(song);
  const blobs = await Promise.all(chans.map((c) => blobFor(urls[c.name], c.label)));
  const ctx = ensureContext();
  const set = { key: v?.key ?? null, quality: v?.quality ?? null, speed: v?.speed ?? 1,
                semi: v?.semi ?? 0, bus: ctx.createGain(), tracks: [], urls, fxKey };
  set.bus.connect(P.master);
  set.tracks = chans.map((s, i) => {
    const audio = new Audio();
    audio.preload = "auto";
    audio.preservesPitch = true;
    audio.webkitPreservesPitch = true;
    audio.src = URL.createObjectURL(blobs[i]);
    audio.defaultPlaybackRate = audio.playbackRate = rateFor(set);
    const gain = ctx.createGain();
    gain.gain.value = effectiveGain(s.name);
    ctx.createMediaElementSource(audio).connect(gain).connect(set.bus);
    return { name: s.name, audio, gain };
  });
  set.tracks[0].audio.addEventListener("ended", () => {
    if (set === P.set) { pause(); seek(P.loop.on && P.loop.a != null ? P.loop.a : 0); }
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
  }
  set.bus.disconnect();
}

function teardownPlayer() {
  pause();
  clearTimeout(P.fxTimer);
  P.blobs.clear();
  clearTimeout(P.requestTimer);
  clearTimeout(P.pollTimer);
  P.versionReq++;
  P.swapToken++;
  if (P.set) dropSet(P.set);
  P.set = null;
}

async function setupPlayer(song) {
  teardownPlayer();
  const token = P.swapToken;
  const set = await makeSet(song, null);
  if (token !== P.swapToken || song !== P.song) { dropSet(set); return; }  // left meanwhile
  P.set = set;
  applyGains(true);
  renderFx();
  if (P.semi !== 0 || (P.speed !== 1 && P.engine === "rubberband")) scheduleVersion(0);
}

function setDuration(d) {
  if (!Number.isFinite(d) || d <= 0) return;
  $("#seek").max = d;
  $("#time-dur").textContent = fmtClock(d);
  renderLoop();
}

const ready = (a) => a.readyState >= 3 || new Promise((r) => {
  a.addEventListener("canplay", r, { once: true });
});

async function play({ countIn = true } = {}) {
  if (!P.set || P.counting) return;
  await ensureContext().resume();
  const { a, b, on } = P.loop;
  if (on && a != null && b != null && (now() < a || now() >= b)) seek(a);
  if (countIn && $("#count-in").checked && METRO.ready) {
    P.counting = true;
    try { await METRO.countIn(now(), P.speed); } finally { P.counting = false; }
    if (!P.set) return;
  }
  const set = P.set;
  const t = clock().currentTime;
  for (const tr of set.tracks) tr.audio.currentTime = t;
  await Promise.all(set.tracks.map((tr) => ready(tr.audio)));
  if (set !== P.set) return;  // swapped meanwhile; that swap took over
  await Promise.all(set.tracks.map((tr) => tr.audio.play()));
  P.playing = true;
  $("#play-btn").textContent = "❚❚";
  $("#play-btn").setAttribute("aria-label", "Pause");
  cancelAnimationFrame(P.raf);
  P.raf = requestAnimationFrame(tick);
}

function pause() {
  P.playing = false;
  cancelAnimationFrame(P.raf);
  if (P.set) {
    const t = clock().currentTime;
    for (const tr of P.set.tracks) { tr.audio.pause(); tr.audio.currentTime = t; }
  }
  $("#play-btn").textContent = "▶";
  $("#play-btn").setAttribute("aria-label", "Play");
}

function seek(t) {
  const d = P.song?.duration;
  t = Math.max(0, Number.isFinite(d) ? Math.min(t, d - 0.05) : t);
  if (P.set) for (const tr of P.set.tracks) tr.audio.currentTime = t / P.set.speed;
  showTime(t);
}

function showTime(t) {
  $("#time-cur").textContent = fmtClock(t);
  if (!P.dragging) $("#seek").value = t;
}

function tick() {
  if (!P.playing || !P.set) return;
  const { a, b, on } = P.loop;
  if (on && a != null && b != null && now() >= b && !REC.rec) {
    if ($("#count-in").checked && METRO.ready) { loopCountIn(a); return; }
    seek(a);
  } else {
    const t = clock().currentTime;
    for (const tr of P.set.tracks.slice(1)) {
      if (!tr.audio.seeking && Math.abs(tr.audio.currentTime - t) > DRIFT) tr.audio.currentTime = t;
    }
    showTime(now());
  }
  METRO.update(now(), P.speed, P.playing);
  followSheet(now());
  P.raf = requestAnimationFrame(tick);
}

// a loop pass ended: back to A, a bar of clicks, then on
async function loopCountIn(a) {
  pause();
  seek(a);
  P.counting = true;
  try { await METRO.countIn(a, P.speed); } finally { P.counting = false; }
  if (P.set) play({ countIn: false });
}

// Swap to another set (a server version, or null = the original stems) without
// a gap: the new set loads and lines up while the old one keeps playing, then
// they crossfade.
async function swapTo(v) {
  const token = ++P.swapToken;
  const song = P.song;
  const next = await makeSet(song, v);
  if (token !== P.swapToken || song !== P.song) { dropSet(next); return false; }
  next.bus.gain.value = 0;
  for (const tr of next.tracks) tr.audio.currentTime = now() / next.speed;
  await Promise.all(next.tracks.map((tr) => ready(tr.audio)));
  if (token !== P.swapToken || song !== P.song) { dropSet(next); return false; }
  for (const tr of next.tracks) {
    tr.audio.defaultPlaybackRate = tr.audio.playbackRate = rateFor(next);
    tr.gain.gain.value = effectiveGain(tr.name);
  }
  const t = now() / next.speed;
  for (const tr of next.tracks) tr.audio.currentTime = t;
  const old = P.set;
  if (P.playing) {
    await Promise.all(next.tracks.map((tr) => tr.audio.play()));
    if (token !== P.swapToken || !P.playing) {  // superseded or paused while starting
      if (token !== P.swapToken) { dropSet(next); return false; }
      for (const tr of next.tracks) tr.audio.pause();
    }
  }
  P.set = next;
  const at = P.ctx.currentTime;
  if (P.playing) {
    old.bus.gain.setValueAtTime(1, at);
    old.bus.gain.linearRampToValueAtTime(0, at + XFADE);
    next.bus.gain.setValueAtTime(0, at);
    next.bus.gain.linearRampToValueAtTime(1, at + XFADE);
    setTimeout(() => dropSet(old), XFADE * 1000 + 150);
  } else {
    next.bus.gain.value = 1;
    dropSet(old);
  }
  keepBlobs(next);
  renderFx();
  return true;
}

// ---- speed, key, engine ----------------------------------------------------

function setSpeed(speed) {
  speed = Math.round(Math.min(SPEED_MAX, Math.max(SPEED_MIN, speed)) * 10) / 10;  // 10% steps
  if (speed === P.speed) return renderSpeed();
  P.speed = speed;
  if (P.set) for (const tr of P.set.tracks) tr.audio.defaultPlaybackRate = tr.audio.playbackRate = rateFor(P.set);
  renderSpeed();
  renderAnalysis();
  saveMix();
  if (P.engine === "rubberband") scheduleVersion();
}

function setKey(semi) {
  semi = Math.round(Math.min(KEY_MAX, Math.max(-KEY_MAX, semi)));
  if (semi === P.semi) return renderKey();
  P.semi = semi;
  renderKey();
  saveMix();
  scheduleVersion();
  clearTimeout(P.heardTimer);          // the key and the chords as you'll hear them
  P.heardTimer = setTimeout(() => { if (P.analysis?.state === "done") loadAnalysis(); }, 300);
}

function setEngine(engine) {
  P.engine = engine === "browser" ? "browser" : "rubberband";
  store.set(ENGINE_KEY, P.engine);
  renderEngine();
  if (P.song) scheduleVersion(0);
}

function renderSpeed() {
  const slider = $("#speed");
  if (Math.abs(Number(slider.value) - P.speed) > 1e-6) slider.value = P.speed;
  $("#speed-val").textContent = `${Math.round(P.speed * 100)}%`;
  $("#speed-reset").hidden = P.speed === 1;
  $("#speed-down").disabled = P.speed <= SPEED_MIN;
  $("#speed-up").disabled = P.speed >= SPEED_MAX;
  renderHQ();
}

// +1 = ♯, +2 = ♯♯, -1 = ♭ ...: one step is one semitone, one guitar fret
function keyLabel(n) {
  if (n === 0) return "0";
  const sign = n > 0 ? "+" : "−";
  const acc = Math.abs(n) <= 3 ? " " + (n > 0 ? "♯" : "♭").repeat(Math.abs(n)) : "";
  return `${sign}${Math.abs(n)}${acc}`;
}

function renderKey() {
  const slider = $("#key");
  if (Number(slider.value) !== P.semi) slider.value = P.semi;
  const out = $("#key-val");
  out.textContent = keyLabel(P.semi);
  out.title = P.semi === 0 ? "original key"
    : `${Math.abs(P.semi)} semitone${Math.abs(P.semi) > 1 ? "s" : ""} (fret${Math.abs(P.semi) > 1 ? "s" : ""}) ${P.semi > 0 ? "up" : "down"}`;
  $("#key-reset").hidden = P.semi === 0;
  $("#key-down").disabled = P.semi <= -KEY_MAX;
  $("#key-up").disabled = P.semi >= KEY_MAX;
  renderHQ();
}

function renderEngine() {
  for (const b of document.querySelectorAll("[data-engine]")) {
    b.setAttribute("aria-pressed", String(b.dataset.engine === P.engine));
  }
  renderHQ();
}

// Key changes only come from the server, so until the new version arrives the
// old key keeps playing: say so next to the key readout, with Rubber Band's progress.
function renderKeyStatus() {
  const chip = $("#key-status");
  const heard = P.set?.semi ?? 0;
  const pending = !!P.song && P.semi !== heard;
  const failed = pending && P.hq.state === "error";
  if (P.keyPending && !pending) {  // just arrived
    P.keyFlash = Date.now() + APPLIED_MS;
    setTimeout(renderKeyStatus, APPLIED_MS + 50);
  }
  P.keyPending = pending;
  $("#key-val").classList.toggle("pending", pending && !failed);
  chip.className = "key-status";
  if (failed) {
    chip.classList.add("error");
    chip.textContent = "✕ not applied";
  } else if (pending) {
    const pct = P.hq.state === "pending" && P.hq.progress != null ? ` ${Math.round(P.hq.progress * 100)}%` : "…";
    chip.classList.add("busy");
    chip.replaceChildren(el("span", { class: "spinner", "aria-hidden": "true" }),
      `applying${pct} · hearing ${heard === 0 ? "the original key" : keyLabel(heard)}`);
  } else if (Date.now() < P.keyFlash) {
    chip.classList.add("done");
    chip.textContent = "✓ applied";
  }
  chip.hidden = !chip.classList.contains("error") && !chip.classList.contains("busy")
    && !chip.classList.contains("done");
}

function renderHQ() {
  renderKeyStatus();
  const msg = $("#hq-msg");
  const serverSpeed = P.engine === "rubberband" && P.speed !== 1;
  if (!P.song || (!serverSpeed && P.semi === 0)) { msg.textContent = ""; return; }
  const what = serverSpeed && P.semi ? "speed and key" : serverSpeed ? "speed" : "key";
  msg.textContent = {
    idle: "",
    pending: `Rubber Band is changing the ${what}…`,
    fast: "quick version playing, cleaner one on the way…",
    fine: `✓ ${what} by Rubber Band`,
    error: `couldn't change the ${what}: ${P.hq.msg}`,
  }[P.hq.state];
  msg.classList.toggle("error", P.hq.state === "error");
}

// The version the page should be playing: speed only counts with Accurate.
function wantedVersion() {
  return { speed: P.engine === "rubberband" ? P.speed : 1, semi: P.semi };
}

function scheduleVersion(delay = 400) {
  clearTimeout(P.requestTimer);
  clearTimeout(P.pollTimer);
  P.versionReq++;  // whatever is in flight is outdated now
  P.hq = { state: "pending", msg: "", progress: null };
  renderHQ();
  P.requestTimer = setTimeout(requestVersion, delay);
}

async function requestVersion() {
  if (!P.song) return;
  const id = ++P.versionReq;
  const { speed, semi } = wantedVersion();
  try {
    const st = await api(`/api/songs/${P.song.id}/versions`, jsonBody("POST", { speed, semitones: semi }));
    await followVersion(st, id, speed, semi);
  } catch (e) {
    if (id === P.versionReq) { P.hq = { state: "error", msg: e.message, progress: null }; renderHQ(); }
  }
}

async function followVersion(st, id, speed, semi) {
  if (id !== P.versionReq || !P.song) return;
  const set = P.set;
  if (!st.key) {  // the original stems
    if (set.key !== null) await swapTo(null);
    if (id === P.versionReq) { P.hq = { state: "idle", msg: "", progress: null }; renderHQ(); }
    return;
  }
  const better = st.ready && (set.key !== st.key || (set.quality === "fast" && st.ready === "fine"));
  if (better) await swapTo({ key: st.key, quality: st.ready, speed, semi });
  if (id !== P.versionReq) return;
  const loaded = P.set.key === st.key ? P.set.quality : null;
  P.hq = st.error ? { state: "error", msg: st.error, progress: null }
    : { state: loaded || "pending", msg: "", progress: st.progress };
  renderHQ();
  if (st.working || (st.ready && !loaded)) {
    P.pollTimer = setTimeout(async () => {
      try {
        await followVersion(await api(`/api/songs/${P.song.id}/versions/${st.key}`), id, speed, semi);
      } catch (e) {
        if (id === P.versionReq) { P.hq = { state: "error", msg: e.message, progress: null }; renderHQ(); }
      }
    }, loaded ? POLL_MS : POLL_PENDING_MS);
  }
}

// ---- mixer -------------------------------------------------------------

function effectiveGain(name) {
  const anySolo = Object.values(M.solo).some(Boolean);
  if (anySolo ? !M.solo[name] : M.mute[name]) return 0;
  return M.vol[name] ?? 1;
}

function applyGains(instant = false) {
  for (const tr of P.set?.tracks || []) {
    const g = effectiveGain(tr.name);
    if (instant || !P.ctx) tr.gain.gain.value = g;
    else tr.gain.gain.setTargetAtTime(g, P.ctx.currentTime, 0.015);  // no clicks
  }
  for (const strip of document.querySelectorAll(".strip")) {
    const name = strip.dataset.name;
    strip.classList.toggle("silent", effectiveGain(name) === 0);
    strip.querySelector(".mute").setAttribute("aria-pressed", String(!!M.mute[name]));
    strip.querySelector(".solo").setAttribute("aria-pressed", String(!!M.solo[name]));
    strip.querySelector("input").value = M.vol[name] ?? 1;
    strip.querySelector("output").textContent = `${Math.round((M.vol[name] ?? 1) * 100)}%`;
  }
}

function renderStrips(song) {
  $("#strips").replaceChildren(...channels(song).flatMap((s) => {
    const vol = el("input", { type: "range", min: "0", max: "1", step: "0.01", "aria-label": `${s.label} volume` });
    vol.addEventListener("input", () => { M.vol[s.name] = Number(vol.value); M.mute[s.name] = false; applyGains(); saveMix(); });
    const toggle = (kind) => () => { M[kind][s.name] = !M[kind][s.name]; applyGains(); saveMix(); };
    const get = el("button", { class: "get", title: `Download ${s.label} on its own (full level, current speed, key and effects)`,
      "aria-label": `Download ${s.label}`, onclick: (e) => download(s.name, e.currentTarget) }, "⤓");
    const common = [
      el("button", { class: "mute", title: `Mute ${s.label}`, "aria-pressed": "false", onclick: toggle("mute") }, "M"),
      el("button", { class: "solo", title: `Solo ${s.label}`, "aria-pressed": "false", onclick: toggle("solo") }, "S"),
      vol, el("output", {}, "100%")];
    if (s.take) {
      const n = s.take.n;
      return [el("div", { class: "strip take", dataset: { name: s.name } },
        el("span", { class: "name" }, s.label,
          el("button", { class: "link del", title: `Delete ${s.label}`, onclick: (e) => deleteTake(n, e.currentTarget) }, "✕")),
        ...common,
        el("span", { class: "nudge-take", title: "Move the take if it's early or late against the backing" },
          el("button", { onclick: () => nudgeTake(n, -10), "aria-label": `${s.label} 10 ms earlier` }, "−"),
          el("button", { onclick: () => nudgeTake(n, 10), "aria-label": `${s.label} 10 ms later` }, "+")),
        get)];
    }
    return [el("div", { class: "strip", dataset: { name: s.name } },
      el("span", { class: "name" }, s.label), ...common,
      el("button", { class: "fx-btn", title: `Effects on ${s.label}`, "aria-expanded": "false",
        onclick: (e) => toggleFxRow(s.name, e.currentTarget) }, "FX"),
      get), fxRow(s.name)];
  }));
  renderFxButtons();
}

// ---- takes: you, recorded over the backing (splitter/takes.py) -----------------

const REC = { stream: null, rec: null, chunks: [], start: 0, speed: 1, semi: 0, latency: 0 };

async function toggleRecord() {
  if (REC.rec) {
    if (REC.rec.state !== "inactive") REC.rec.stop();   // saveTake runs when it has stopped
    else { REC.rec = null; renderRecord(); }             // still counting in: just cancel
    return;
  }
  const msg = $("#rec-msg");
  try {
    REC.stream ||= await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) { msg.textContent = `no microphone: ${e.message}`; return; }
  const ctx = ensureContext();
  // how late the mic hears you against what you heard: the output's delay, the input's
  const input = REC.stream.getAudioTracks()[0]?.getSettings?.().latency || 0;
  REC.latency = Math.round(((ctx.outputLatency || 0) + (ctx.baseLatency || 0) + input) * 1000);
  REC.chunks = [];
  REC.rec = new MediaRecorder(REC.stream);
  REC.rec.ondataavailable = (e) => { if (e.data.size) REC.chunks.push(e.data); };
  REC.rec.onstop = saveTake;
  renderRecord();
  const rec = REC.rec;
  if (!P.playing) await play();               // with its count-in
  if (REC.rec !== rec) return;                // cancelled meanwhile
  if (!P.playing) { REC.rec = null; renderRecord(); return; }
  REC.start = now(); REC.speed = P.speed; REC.semi = P.semi;
  REC.rec.start();
  msg.textContent = "recording… ■ or R stops";
}

async function saveTake() {
  const rec = REC.rec;
  REC.rec = null;
  pause();
  renderRecord();
  const msg = $("#rec-msg");
  const blob = new Blob(REC.chunks, { type: rec.mimeType || "audio/webm" });
  if (!blob.size) { msg.textContent = "nothing was recorded"; return; }
  const ext = blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm";
  const fd = new FormData();
  fd.append("file", blob, `take.${ext}`);
  fd.append("start_s", REC.start); fd.append("speed", REC.speed);
  fd.append("semitones", REC.semi); fd.append("latency_ms", REC.latency);
  msg.textContent = "saving the take…";
  try {
    const take = await api(`/api/songs/${P.song.id}/takes`, { method: "POST", body: fd });
    P.takes.push(take);
    await takesChanged();
    msg.textContent = `saved ${take.name} (${REC.latency} ms of audio delay taken off; use − / + on it if it's early or late)`;
  } catch (e) { msg.textContent = `couldn't save the take: ${e.message}`; }
}

async function nudgeTake(n, ms) {
  try {
    const take = await api(`/api/songs/${P.song.id}/takes/${n}`, jsonBody("PATCH", { ms }));
    P.takes = P.takes.map((t) => (t.n === n ? take : t));
    await takesChanged();
  } catch (e) { $("#rec-msg").textContent = e.message; }
}

async function deleteTake(n, btn) {
  if (!btn.classList.contains("sure")) {            // two clicks, no dialog
    btn.classList.add("sure"); btn.textContent = "delete?";
    setTimeout(() => { btn.classList.remove("sure"); btn.textContent = "✕"; }, 3000);
    return;
  }
  try {
    await api(`/api/songs/${P.song.id}/takes/${n}`, { method: "DELETE" });
    P.takes = P.takes.filter((t) => t.n !== n);
    await takesChanged();
  } catch (e) { $("#rec-msg").textContent = e.message; }
}

async function takesChanged() {
  renderStrips(P.song);
  applyGains(true);
  await swapTo(currentVersion());
}

function renderRecord() {
  const btn = $("#rec-btn");
  btn.textContent = REC.rec ? "■ Stop" : "● Record";
  btn.classList.toggle("recording", !!REC.rec);
  btn.setAttribute("aria-pressed", String(!!REC.rec));
}

// ---- key, tempo, beats, chords: the chord finder's analysis (splitter/analysis.py) ----

const busy = (st) => ["queued", "running"].includes(st?.state);

async function loadAnalysis() {
  if (!P.song) return;
  const id = P.song.id;
  clearTimeout(P.analysisTimer);
  let st;
  try { st = await api(`/api/songs/${id}/analysis?shift=${P.semi}`); } catch (e) { st = { state: "error", job: { error: e.message } }; }
  if (P.song?.id !== id) return;
  P.analysis = st;
  if (st.state === "done") {
    P.tempo = st.tempo || null;
    METRO.setGrid(st.beat_times, st.bars);
    if (!SH.view || SH.shift !== P.semi || SH.lyrics !== st.lyrics) loadSheet();
  }
  if (busy(st) || (st.state === "done" && busy(st.job))) P.analysisTimer = setTimeout(loadAnalysis, 1500);
  renderAnalysis(); renderFxShared(); renderClick();
}

async function analyse(lyrics = false) {
  try {
    P.analysis = await api(`/api/songs/${P.song.id}/analyse`, jsonBody("POST", { lyrics }));
  } catch (e) { P.analysis = { state: "error", job: { error: e.message } }; }
  renderAnalysis();
  loadAnalysis();
}

function renderAnalysis() {
  const box = $("#song-facts");
  const st = P.analysis;
  if (!box) return;
  if (!st || st.state === "none") {
    box.replaceChildren(el("button", { onclick: () => analyse(false) }, "Find key, tempo & chords"),
      el("small", { class: "muted" }, " the chord finder works it out from these stems (about a minute)"));
  } else if (st.state === "done") {
    const bpm = st.tempo ? Math.round(st.tempo * P.speed) : null;
    const items = [el("span", { class: "badge", title: P.semi ? `Recorded in ${st.original_key_label}` : null },
      `Key ${st.key_label || "?"}`)];
    if (bpm) {
      items.push(el("span", { class: "badge", title: P.speed !== 1 ? `${Math.round(st.tempo)} BPM as recorded` : null }, `${bpm} BPM`));
      if (st.meter?.time_signature) items.push(el("span", { class: "badge" }, st.meter.time_signature));
    } else {
      items.push(el("span", { class: "badge", title: "The chord finder found no steady beat, so there's no tempo, bar grid or metronome" },
        "no steady beat"));
    }
    if (busy(st.job)) items.push(el("small", { class: "muted" }, `adding lyrics… ${st.job.message || ""}`));
    if (BASE && st.chords_id) items.push(el("a", { href: `/chords/#/song/${encodeURIComponent(st.chords_id)}` }, "open in the chord finder →"));
    box.replaceChildren(...items);
  } else if (st.state === "error" || st.state === "cancelled") {
    box.replaceChildren(el("span", { class: "msg error" }, `Couldn't find the key and chords: ${st.job?.error || st.job?.message || st.state}`),
      " ", el("button", { class: "link", onclick: () => analyse(false) }, "try again"));
  } else {
    const pct = st.job?.progress != null ? ` ${Math.round(st.job.progress * 100)}%` : "";
    box.replaceChildren(el("span", { class: "muted small" }, `Finding key, tempo and chords… ${st.job?.message || st.state}${pct}`));
  }
  $("#chords-card").hidden = st?.state !== "done";
  $("#lyrics-hint").hidden = !(st?.state === "done" && !st.lyrics && !busy(st.job));
}

async function loadSheet() {
  const id = P.song.id, shift = P.semi;
  let view;
  try { view = await api(`/api/songs/${id}/sheet?shift=${shift}`); } catch { return; }
  if (P.song?.id !== id || P.semi !== shift) return;
  const byId = new Map(view.chords.map((c) => [c.id, c]));
  Object.assign(SH, { view, shift, lyrics: !!P.analysis?.lyrics, timeline: chordTimeline(view),
                      chipNodes: new Map(), activeChord: null, activeWord: null });
  const chip = (cid, { carry = false } = {}) => {
    const c = byId.get(cid);
    const node = el("button", { class: "chip" + (carry ? " carry" : ""), onclick: () => seek(c.start),
      title: carry ? "Still sounding from before this line" : `${fmtTime(c.start)}: play from here` }, displayOf(c));
    if (!SH.chipNodes.has(cid)) SH.chipNodes.set(cid, []);
    SH.chipNodes.get(cid).push(node);
    return node;
  };
  const { rows, words } = sheetRows(view, chip, (t) => seek(t));
  SH.words = words;
  const sheet = $("#split-sheet");
  sheet.dir = sheetDir(view);
  sheet.replaceChildren(...(rows.length ? rows : [el("p", { class: "muted" }, "No chords found.")]));
  followSheet(now());
}

function followSheet(t) {
  if (!SH.view) return;
  const { cur, next } = followChord(SH.timeline, SH.chipNodes, t, SH);
  $("#now-chord").textContent = cur ? displayOf(cur) : "–";
  $("#next-chord").textContent = next
    ? `${displayOf(next)} in ${Math.max(0, (next.start - t) / P.speed).toFixed(1)}s` : "–";
  followWord(SH.words, t, SH, $("#follow").checked && P.playing);
}

function renderClick() {
  const ready = METRO.ready;
  for (const id of ["#click-on", "#count-in"]) {
    $(id).disabled = !ready;
    $(id).closest("label").title = ready ? "" : P.analysis?.state === "done"
      ? "The chord finder found no steady beat in this song" : "Needs the song's beats: Find key, tempo & chords";
  }
  if (!ready) { $("#click-on").checked = false; METRO.on = false; }
}

// ---- effects (the mixer's fx.py, rendered per stem on the server) -------------

// [key, label, min, max, step, how it reads]
const FX_KNOBS = [
  ["bass", "Bass", -12, 12, 1, (v) => (v ? `${v > 0 ? "+" : ""}${v} dB` : "0")],
  ["treble", "Treble", -12, 12, 1, (v) => (v ? `${v > 0 ? "+" : ""}${v} dB` : "0")],
  ["drive", "Drive", 0, 1, 0.05, (v) => (v ? `${Math.round(v * 100)}%` : "off")],
  ["chorus", "Chorus", 0, 1, 0.05, (v) => (v ? `${Math.round(v * 100)}%` : "off")],
  ["reverb", "Reverb", 0, 1, 0.05, (v) => (v ? `${Math.round(v * 100)}%` : "off")],
  ["echo", "Echo", 0, 1, 0.05, (v) => (v ? `${Math.round(v * 100)}%` : "off")],
];
const stemFx = (name) => (M.fx.stems[name] ||= {});
const hasFx = (e) => FX_KNOBS.some(([k]) => e?.[k]);
const fxActive = () => !!P.song && P.song.stems.some((s) => hasFx(M.fx.stems[s.name]));
// a send knob's 0..1 as the send level in dB (off below 0.5%), like a mixer's send
const sendDb = (a) => (a > 0.005 ? Math.round((20 * Math.log10(a) - 6) * 10) / 10 : -60);

function fxRow(name) {
  const e = stemFx(name);
  return el("div", { class: "fx-row", dataset: { fxFor: name }, hidden: true },
    FX_KNOBS.map(([key, label, min, max, step, fmt]) => {
      const out = el("output", {}, fmt(e[key] || 0));
      const input = el("input", { type: "range", min, max, step, value: e[key] || 0, "aria-label": `${label}, ${name}` });
      input.addEventListener("input", () => {
        stemFx(name)[key] = Number(input.value);
        out.textContent = fmt(Number(input.value));
        renderFxButtons(); saveMix(); scheduleFx();
      });
      return el("label", { class: "knob" }, el("span", {}, label), input, out);
    }));
}

function toggleFxRow(name, btn) {
  const row = document.querySelector(`.fx-row[data-fx-for="${name}"]`);
  row.hidden = !row.hidden;
  btn.setAttribute("aria-expanded", String(!row.hidden));
}

function renderFxButtons() {
  for (const strip of document.querySelectorAll(".strip")) {
    strip.querySelector(".fx-btn")?.classList.toggle("on", hasFx(M.fx.stems[strip.dataset.name]));
  }
  $("#fx-reset").hidden = !fxActive();
}

function fxSettings() {
  const stems = {};
  for (const s of P.song.stems) {
    const e = M.fx.stems[s.name] || {};
    stems[s.name] = { bass_db: e.bass || 0, treble_db: e.treble || 0, drive: e.drive || 0,
                      chorus: e.chorus || 0, reverb_db: sendDb(e.reverb || 0), delay_db: sendDb(e.echo || 0) };
  }
  const { room, echo } = M.fx;
  return { stems, send_buses: {
    reverb: { room_size: room.size, decay_s: room.decay },
    delay: { delay_ms: echo.ms, feedback: echo.repeats, sync: P.tempo ? echo.sync : "" } } };
}

// effects changed: make a set with them (same version), after a short pause
function scheduleFx() {
  clearTimeout(P.fxTimer);
  P.fxError = "";
  renderFx();
  if (!P.set) return;
  P.fxTimer = setTimeout(async () => {
    try { await swapTo(currentVersion()); } catch (e) { P.fxError = e.message; renderFx(); }
  }, 400);
}

function renderFx() {
  const msg = $("#fx-msg");
  const want = fxActive() ? JSON.stringify(fxSettings()) : "";
  msg.classList.toggle("error", !!P.fxError);
  msg.textContent = P.fxError ? `effects failed: ${P.fxError}`
    : P.set && want !== P.set.fxKey ? "applying effects…" : want ? "✓ effects on" : "";
}

function renderFxShared() {
  const { room, echo } = M.fx;
  $("#room-size").value = room.size;
  $("#room-decay").value = room.decay;
  $("#room-decay-val").textContent = `${Number(room.decay).toFixed(1)} s`;
  $("#echo-sync").value = P.tempo ? echo.sync : "";
  for (const o of $("#echo-sync").options) {
    o.disabled = !!o.value && !P.tempo;
    o.title = o.disabled ? "Needs the song's tempo: analyse it first" : "";
  }
  $("#echo-ms").value = echo.ms;
  $("#echo-ms-val").textContent = `${echo.ms} ms`;
  $("#echo-ms-wrap").hidden = !!(P.tempo && echo.sync);
  $("#echo-repeats").value = echo.repeats;
}

function bindFxShared() {
  const on = (sel, apply) => $(sel).addEventListener("input", () => {
    apply($(sel).value); renderFxShared(); saveMix(); scheduleFx();
  });
  on("#room-size", (v) => { M.fx.room.size = Number(v); });
  on("#room-decay", (v) => { M.fx.room.decay = Number(v); });
  on("#echo-sync", (v) => { M.fx.echo.sync = v; });
  on("#echo-ms", (v) => { M.fx.echo.ms = Number(v); });
  on("#echo-repeats", (v) => { M.fx.echo.repeats = Number(v); });
  $("#fx-reset").addEventListener("click", () => {
    M.fx = freshFx();
    renderStrips(P.song); applyGains(true); renderFxShared(); saveMix(); scheduleFx();
  });
}

function freshFx(saved = {}) {
  const d = FX_DEFAULTS || { reverb: { room_size: 0.5, decay_s: 1.4 }, delay: { delay_ms: 250, feedback: 0.25 } };
  return {
    stems: saved.stems || {},
    room: { size: d.reverb.room_size, decay: d.reverb.decay_s, ...(saved.room || {}) },
    echo: { sync: "", ms: d.delay.delay_ms, repeats: d.delay.feedback, ...(saved.echo || {}) },
  };
}

const PRESETS = {
  "all": () => ({}),
  "no-vocals": () => ({ mute: { vocals: true } }),
  "no-guitar": () => ({ mute: { guitar: true } }),
  "guitar-only": () => ({ solo: { guitar: true } }),
  "vocals-only": () => ({ solo: { vocals: true } }),
};

function applyPreset(name) {
  const p = PRESETS[name]();
  M.vol = {}; M.mute = p.mute || {}; M.solo = p.solo || {};
  applyGains(); saveMix();
}

// ---- A-B loop -----------------------------------------------------------

function setLoopPoint(which) {
  const t = now();
  P.loop[which] = t;
  const { a, b } = P.loop;
  if (a != null && b != null && b <= a) P.loop[which === "a" ? "b" : "a"] = null;
  if (P.loop.a != null && P.loop.b != null) P.loop.on = true;
  renderLoop(); saveMix();
}

function renderLoop() {
  const { a, b, on } = P.loop;
  $("#a-val").textContent = a == null ? "–" : fmtClock(a);
  $("#b-val").textContent = b == null ? "–" : fmtClock(b);
  $("#loop-on").checked = on;
  $("#loop-on").disabled = a == null || b == null;
  const d = Number($("#seek").max);
  const band = $("#loop-band");
  band.hidden = a == null || b == null || !(d > 1);
  if (!band.hidden) {
    band.style.left = `${(a / d) * 100}%`;
    band.style.width = `${((b - a) / d) * 100}%`;
    band.style.opacity = on ? "1" : ".4";
  }
}

// ---- persistence (per viewer, per song) ----------------------------------

function saveMix() {
  if (!P.song) return;
  store.set(`mixpipe.splitter.${P.song.id}`, JSON.stringify({ ...M, speed: P.speed, semi: P.semi, loop: P.loop }));
}

function loadMix(id) {
  let saved = {};
  try { saved = JSON.parse(store.get(`mixpipe.splitter.${id}`) || "{}") || {}; } catch { /* corrupt */ }
  M.vol = saved.vol || {}; M.mute = saved.mute || {}; M.solo = saved.solo || {};
  M.fx = freshFx(saved.fx || {});
  P.fxError = "";
  P.speed = Math.round(Math.min(SPEED_MAX, Math.max(SPEED_MIN, Number(saved.speed) || 1)) * 10) / 10;
  P.semi = Math.round(Math.min(KEY_MAX, Math.max(-KEY_MAX, Number(saved.semi) || 0)));
  P.loop = { a: null, b: null, on: false, ...(saved.loop || {}) };
  P.hq = { state: "idle", msg: "", progress: null };
  P.keyPending = false; P.keyFlash = 0;  // nothing carried over from the previous song
  renderSpeed();
  renderKey();
  renderFxShared();
}

// ---- download -------------------------------------------------------------

// what = "mix" (stems x faders), a stem name (that stem at full level) or
// "stems" (every stem at full level, zipped); always at the page's speed and key
async function download(what, btn) {
  const msg = $("#download-msg");
  const fmt = $("#fmt").value;
  const id = P.song.id;
  const body = { speed: P.speed, semitones: P.semi, fmt, fx: fxActive() ? fxSettings() : null, tempo: P.tempo };
  let path = `/api/songs/${id}/stems/${what}/render`;
  if (what === "mix") {
    path = `/api/songs/${id}/render`;
    body.gains = Object.fromEntries(channels(P.song).map((s) => [s.name, effectiveGain(s.name)]));
  } else if (what === "stems") path = `/api/songs/${id}/stems.zip`;
  if (btn) btn.disabled = true;
  // always Rubber Band: a file can't use the browser's stretch
  const cached = P.set?.quality === "fine" && P.set.speed === P.speed && P.set.semi === P.semi;
  msg.textContent = (P.speed !== 1 || P.semi !== 0) && !cached
    ? "rendering with Rubber Band (about 5 s per minute of song)…" : "rendering…";
  try {
    msg.textContent = `saved ${await downloadFile(path, jsonBody("POST", body), `${id}.${fmt}`)}`;
  } catch (e) {
    msg.textContent = `failed: ${e.message}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

// =========================================================================
// mixer view
// =========================================================================

async function openSong(id) {
  teardownPlayer();
  P.song = null;
  $("#mixer").hidden = true;
  $("#transport").hidden = true;
  $("#mixer-progress").replaceChildren();
  $("#song-title").textContent = "";
  $("#song-source").hidden = true;
  let song;
  try {
    song = await api(`/api/songs/${id}`);
  } catch (e) {
    if (e.status === 409) {
      await pollStatus(id, $("#mixer-progress"), { onDone: () => openSong(id) });
      return;
    }
    $("#mixer-progress").replaceChildren(el("p", { class: "msg error" }, e.message));
    return;
  }
  P.song = song;
  P.takes = song.takes || [];
  document.title = `${song.title} · MixPipe Splitter`;
  $("#song-title").textContent = song.title;
  const src = song.source;
  if (src) {
    $("#song-source").hidden = false;
    $("#song-source").replaceChildren(
      "From ", el("a", { href: src.page_url, target: "_blank", rel: "noopener noreferrer" }, src.site),
      src.note ? ` · ${src.note}` : "");
  }
  // under the hub the mixer shares this library and this port
  $("#song-links").hidden = !BASE;
  $("#mix-link").href = `/mixer/#/new?splitter_song=${encodeURIComponent(id)}`;
  FX_DEFAULTS ||= await api("/api/fx/defaults").catch(() => null);
  clearTimeout(P.analysisTimer);
  Object.assign(SH, { view: null, timeline: [], chipNodes: new Map(), words: [], activeChord: null, activeWord: null });
  P.analysis = null; P.tempo = null;
  METRO.setGrid([], []);
  $("#split-sheet").replaceChildren();
  loadMix(id);
  renderStrips(song);
  renderAnalysis(); renderClick();
  loadAnalysis();
  $("#mixer-progress").replaceChildren(el("div", { class: "msg" }, "Loading the stems…"));
  try {
    await setupPlayer(song);
  } catch (e) {
    $("#mixer-progress").replaceChildren(el("p", { class: "msg error" }, e.message));
    return;
  }
  if (P.song !== song) return;
  $("#mixer-progress").replaceChildren();
  setDuration(song.duration);
  showTime(0);
  renderLoop();
  $("#mixer").hidden = false;
  $("#transport").hidden = false;
}

function initMixer() {
  const speed = $("#speed");
  speed.addEventListener("input", () => setSpeed(Number(speed.value)));
  $("#speed-down").addEventListener("click", () => setSpeed(P.speed - SPEED_STEP));
  $("#speed-up").addEventListener("click", () => setSpeed(P.speed + SPEED_STEP));
  $("#speed-reset").addEventListener("click", () => setSpeed(1));
  const key = $("#key");
  key.addEventListener("input", () => setKey(Number(key.value)));
  $("#key-down").addEventListener("click", () => setKey(P.semi - 1));
  $("#key-up").addEventListener("click", () => setKey(P.semi + 1));
  $("#key-reset").addEventListener("click", () => setKey(0));
  document.querySelectorAll("[data-engine]").forEach((b) =>
    b.addEventListener("click", () => setEngine(b.dataset.engine)));
  P.engine = store.get(ENGINE_KEY) === "browser" ? "browser" : "rubberband";
  renderEngine();

  document.querySelectorAll("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => applyPreset(b.dataset.preset)));

  $("#play-btn").addEventListener("click", () => (P.playing ? pause() : play()));
  const seekBar = $("#seek");
  seekBar.addEventListener("input", () => { P.dragging = true; $("#time-cur").textContent = fmtClock(Number(seekBar.value)); });
  seekBar.addEventListener("change", () => { P.dragging = false; seek(Number(seekBar.value)); });
  $("#master").addEventListener("input", (e) => { if (P.master) P.master.gain.value = Number(e.target.value); });

  $("#set-a").addEventListener("click", () => setLoopPoint("a"));
  $("#set-b").addEventListener("click", () => setLoopPoint("b"));
  $("#loop-on").addEventListener("change", (e) => { P.loop.on = e.target.checked; renderLoop(); saveMix(); });
  $("#loop-clear").addEventListener("click", () => { P.loop = { a: null, b: null, on: false }; renderLoop(); saveMix(); });
  $("#download-btn").addEventListener("click", (e) => download("mix", e.currentTarget));
  $("#stems-btn").addEventListener("click", (e) => download("stems", e.currentTarget));
  bindFxShared();
  $("#click-on").addEventListener("change", (e) => { METRO.on = e.target.checked; });
  $("#click-vol").addEventListener("input", (e) => METRO.setVolume(Number(e.target.value)));
  $("#add-lyrics").addEventListener("click", () => analyse(true));
  $("#rec-btn").addEventListener("click", toggleRecord);

  document.addEventListener("keydown", (e) => {
    if ($("#mixer").hidden || typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); P.playing ? pause() : play(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); seek(now() - 5); }
    else if (e.key === "ArrowRight") { e.preventDefault(); seek(now() + 5); }
    else if (e.key === "-" || e.key === "_") setSpeed(P.speed - SPEED_STEP);
    else if (e.key === "+" || e.key === "=") setSpeed(P.speed + SPEED_STEP);
    else if (e.key === "r" || e.key === "R") toggleRecord();
    else if (e.key === "m" || e.key === "M") {
      if (METRO.ready) { $("#click-on").checked = !$("#click-on").checked; METRO.on = $("#click-on").checked; }
    } else if (e.key === ",") setKey(P.semi - 1);
    else if (e.key === ".") setKey(P.semi + 1);
    else if (e.key === "[") setLoopPoint("a");
    else if (e.key === "]") setLoopPoint("b");
    else if (e.key === "l" || e.key === "L") {
      if (P.loop.a != null && P.loop.b != null) { P.loop.on = !P.loop.on; renderLoop(); saveMix(); }
    }
  });
}

// =========================================================================
// routing
// =========================================================================

function route() {
  const m = /^#\/song\/([\w-]+)/.exec(location.hash);
  if (m) {
    $("#library-view").hidden = true;
    $("#mixer-view").hidden = false;
    openSong(m[1]);
  } else {
    teardownPlayer();
    P.song = null;
    document.title = "MixPipe Splitter";
    $("#mixer-view").hidden = true;
    $("#transport").hidden = true;
    $("#library-view").hidden = false;
    refreshSongs();
  }
}

picker = createSourcePicker($("#source-picker"), {
  multiple: false,
  hint: "One audio file: the full song. It's split into vocals, guitar, drums, bass, keys and other.",
  onChange: (p) => { $("#split-btn").disabled = !p.hasInput(); },
  onSubmit: () => startSplit(false),
});
$("#split-btn").addEventListener("click", () => startSplit(false));
initMixer();
window.addEventListener("hashchange", route);
route();
