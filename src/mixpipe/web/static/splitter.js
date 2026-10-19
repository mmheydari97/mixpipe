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
};
const M = { vol: {}, mute: {}, solo: {} };  // mixer state, saved per song

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
  }
  return P.ctx;
}

// v: { key, quality, speed, semi } of a server version, or null for the original stems
async function makeSet(song, v) {
  const base = v ? `/api/songs/${song.id}/versions/${v.key}/${v.quality}` : `/api/songs/${song.id}/stems`;
  const blobs = await Promise.all(song.stems.map(async (s) => {
    const res = await fetch(`${base}/${s.name}`);
    if (!res.ok) throw new Error(`couldn't load ${s.label} (${res.status})`);
    return res.blob();
  }));
  const ctx = ensureContext();
  const set = { key: v?.key ?? null, quality: v?.quality ?? null, speed: v?.speed ?? 1,
                semi: v?.semi ?? 0, bus: ctx.createGain(), tracks: [] };
  set.bus.connect(P.master);
  set.tracks = song.stems.map((s, i) => {
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

async function play() {
  if (!P.set) return;
  await ensureContext().resume();
  const { a, b, on } = P.loop;
  if (on && a != null && b != null && (now() < a || now() >= b)) seek(a);
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
  if (on && a != null && b != null && now() >= b) {
    seek(a);
  } else {
    const t = clock().currentTime;
    for (const tr of P.set.tracks.slice(1)) {
      if (!tr.audio.seeking && Math.abs(tr.audio.currentTime - t) > DRIFT) tr.audio.currentTime = t;
    }
    showTime(now());
  }
  P.raf = requestAnimationFrame(tick);
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
  return true;
}

// ---- speed, key, engine ----------------------------------------------------

function setSpeed(speed) {
  speed = Math.round(Math.min(SPEED_MAX, Math.max(SPEED_MIN, speed)) * 10) / 10;  // 10% steps
  if (speed === P.speed) return renderSpeed();
  P.speed = speed;
  if (P.set) for (const tr of P.set.tracks) tr.audio.defaultPlaybackRate = tr.audio.playbackRate = rateFor(P.set);
  renderSpeed();
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
  $("#strips").replaceChildren(...song.stems.map((s) => {
    const vol = el("input", { type: "range", min: "0", max: "1", step: "0.01", "aria-label": `${s.label} volume` });
    vol.addEventListener("input", () => { M.vol[s.name] = Number(vol.value); M.mute[s.name] = false; applyGains(); saveMix(); });
    const toggle = (kind) => () => { M[kind][s.name] = !M[kind][s.name]; applyGains(); saveMix(); };
    return el("div", { class: "strip", dataset: { name: s.name } },
      el("span", { class: "name" }, s.label),
      el("button", { class: "mute", title: `Mute ${s.label}`, "aria-pressed": "false", onclick: toggle("mute") }, "M"),
      el("button", { class: "solo", title: `Solo ${s.label}`, "aria-pressed": "false", onclick: toggle("solo") }, "S"),
      vol,
      el("output", {}, "100%"));
  }));
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
  P.speed = Math.round(Math.min(SPEED_MAX, Math.max(SPEED_MIN, Number(saved.speed) || 1)) * 10) / 10;
  P.semi = Math.round(Math.min(KEY_MAX, Math.max(-KEY_MAX, Number(saved.semi) || 0)));
  P.loop = { a: null, b: null, on: false, ...(saved.loop || {}) };
  P.hq = { state: "idle", msg: "", progress: null };
  P.keyPending = false; P.keyFlash = 0;  // nothing carried over from the previous song
  renderSpeed();
  renderKey();
}

// ---- download -------------------------------------------------------------

async function downloadMix() {
  const btn = $("#download-btn");
  const msg = $("#download-msg");
  const fmt = $("#fmt").value;
  const gains = Object.fromEntries(P.song.stems.map((s) => [s.name, effectiveGain(s.name)]));
  btn.disabled = true;
  // always Rubber Band: a file can't use the browser's stretch
  const cached = P.set?.quality === "fine" && P.set.speed === P.speed && P.set.semi === P.semi;
  msg.textContent = (P.speed !== 1 || P.semi !== 0) && !cached
    ? "rendering with Rubber Band (about 5 s per minute of song)…" : "rendering…";
  try {
    const res = await fetch(`/api/songs/${P.song.id}/render`,
      jsonBody("POST", { gains, speed: P.speed, semitones: P.semi, fmt }));
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
      throw new Error(detail);
    }
    const name = /filename="([^"]+)"/.exec(res.headers.get("Content-Disposition") || "")?.[1]
      || `${P.song.id}.${fmt}`;
    const url = URL.createObjectURL(await res.blob());
    const a = el("a", { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    msg.textContent = `saved ${name}`;
  } catch (e) {
    msg.textContent = `failed: ${e.message}`;
  } finally {
    btn.disabled = false;
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
  document.title = `${song.title} · MixPipe Splitter`;
  $("#song-title").textContent = song.title;
  const src = song.source;
  if (src) {
    $("#song-source").hidden = false;
    $("#song-source").replaceChildren(
      "From ", el("a", { href: src.page_url, target: "_blank", rel: "noopener noreferrer" }, src.site),
      src.note ? ` · ${src.note}` : "");
  }
  loadMix(id);
  renderStrips(song);
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
  $("#download-btn").addEventListener("click", downloadMix);

  document.addEventListener("keydown", (e) => {
    if ($("#mixer").hidden || typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); P.playing ? pause() : play(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); seek(now() - 5); }
    else if (e.key === "ArrowRight") { e.preventDefault(); seek(now() + 5); }
    else if (e.key === "-" || e.key === "_") setSpeed(P.speed - SPEED_STEP);
    else if (e.key === "+" || e.key === "=") setSpeed(P.speed + SPEED_STEP);
    else if (e.key === ",") setKey(P.semi - 1);
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
