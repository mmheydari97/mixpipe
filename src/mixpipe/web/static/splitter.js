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
// player: one <audio> per stem, kept in sync, mixed through Web Audio gains
// =========================================================================
// <audio> streams from disk (6 decoded stems would be ~0.5 GB for a 4-minute
// song) and slows down without changing pitch (playbackRate + preservesPitch).
// The first stem is the clock; the others are nudged back when they drift.

const DRIFT = 0.04;   // seconds a stem may wander from the clock before it's resynced
const P = {
  ctx: null, master: null, song: null, tracks: [], playing: false, dragging: false,
  speed: 1, loop: { a: null, b: null, on: false }, raf: 0,
};
const M = { vol: {}, mute: {}, solo: {} };  // mixer state, saved per song

const clock = () => P.tracks[0]?.audio;
const now = () => clock()?.currentTime || 0;

function ensureContext() {
  if (!P.ctx) {
    P.ctx = new AudioContext();
    P.master = P.ctx.createGain();
    P.master.connect(P.ctx.destination);
    P.master.gain.value = Number($("#master").value);
  }
  return P.ctx;
}

function teardownPlayer() {
  pause();
  for (const t of P.tracks) {
    t.audio.removeAttribute("src");
    t.audio.load();
    t.gain.disconnect();
  }
  P.tracks = [];
}

function setupPlayer(song) {
  teardownPlayer();
  const ctx = ensureContext();
  P.tracks = song.stems.map((s) => {
    const audio = new Audio();
    audio.preload = "auto";
    audio.preservesPitch = true;
    audio.webkitPreservesPitch = true;
    audio.src = `/api/songs/${song.id}/stems/${s.name}`;
    audio.playbackRate = P.speed;
    const gain = ctx.createGain();
    ctx.createMediaElementSource(audio).connect(gain).connect(P.master);
    return { ...s, audio, gain };
  });
  clock().addEventListener("ended", () => { pause(); seek(P.loop.on && P.loop.a != null ? P.loop.a : 0); });
  clock().addEventListener("durationchange", () => setDuration(clock().duration));
  applyGains(true);
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
  if (!P.tracks.length) return;
  await ensureContext().resume();
  const { a, b, on } = P.loop;
  if (on && a != null && b != null && (now() < a || now() >= b)) seek(a);
  const t = now();
  for (const tr of P.tracks) tr.audio.currentTime = t;
  await Promise.all(P.tracks.map((tr) => ready(tr.audio)));
  await Promise.all(P.tracks.map((tr) => tr.audio.play()));
  P.playing = true;
  $("#play-btn").textContent = "❚❚";
  $("#play-btn").setAttribute("aria-label", "Pause");
  cancelAnimationFrame(P.raf);
  P.raf = requestAnimationFrame(tick);
}

function pause() {
  P.playing = false;
  cancelAnimationFrame(P.raf);
  const t = now();
  for (const tr of P.tracks) { tr.audio.pause(); tr.audio.currentTime = t; }
  $("#play-btn").textContent = "▶";
  $("#play-btn").setAttribute("aria-label", "Play");
}

function seek(t) {
  const d = clock()?.duration;
  t = Math.max(0, Number.isFinite(d) ? Math.min(t, d - 0.05) : t);
  for (const tr of P.tracks) tr.audio.currentTime = t;
  showTime(t);
}

function showTime(t) {
  $("#time-cur").textContent = fmtClock(t);
  if (!P.dragging) $("#seek").value = t;
}

function tick() {
  if (!P.playing) return;
  const t = now();
  const { a, b, on } = P.loop;
  if (on && a != null && b != null && t >= b) {
    seek(a);
  } else {
    for (const tr of P.tracks.slice(1)) {
      if (!tr.audio.seeking && Math.abs(tr.audio.currentTime - t) > DRIFT) tr.audio.currentTime = t;
    }
    showTime(t);
  }
  P.raf = requestAnimationFrame(tick);
}

function setSpeed(speed) {
  P.speed = speed;
  for (const tr of P.tracks) tr.audio.playbackRate = speed;
  saveMix();
}

// ---- mixer -------------------------------------------------------------

function effectiveGain(name) {
  const anySolo = P.tracks.some((t) => M.solo[t.name]);
  if (anySolo ? !M.solo[name] : M.mute[name]) return 0;
  return M.vol[name] ?? 1;
}

function applyGains(instant = false) {
  for (const tr of P.tracks) {
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
  store.set(`mixpipe.splitter.${P.song.id}`, JSON.stringify({ ...M, speed: P.speed, loop: P.loop }));
}

function loadMix(id) {
  let saved = {};
  try { saved = JSON.parse(store.get(`mixpipe.splitter.${id}`) || "{}") || {}; } catch { /* corrupt */ }
  M.vol = saved.vol || {}; M.mute = saved.mute || {}; M.solo = saved.solo || {};
  P.speed = saved.speed || 1;
  P.loop = { a: null, b: null, on: false, ...(saved.loop || {}) };
  $("#speed").value = String(P.speed);
}

// ---- download -------------------------------------------------------------

async function downloadMix() {
  const btn = $("#download-btn");
  const msg = $("#download-msg");
  const fmt = $("#fmt").value;
  const gains = Object.fromEntries(P.tracks.map((t) => [t.name, effectiveGain(t.name)]));
  btn.disabled = true;
  msg.textContent = P.speed < 1 ? "rendering (slowing down takes a few seconds)…" : "rendering…";
  try {
    const res = await fetch(`/api/songs/${P.song.id}/render`,
      jsonBody("POST", { gains, speed: P.speed, fmt }));
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
  setupPlayer(song);
  setDuration(song.duration);
  showTime(0);
  renderLoop();
  $("#mixer").hidden = false;
  $("#transport").hidden = false;
}

function initMixer() {
  const speed = $("#speed");
  for (let p = 100; p >= 50; p -= 5) speed.append(el("option", { value: String(p / 100) }, `${p}%`));
  speed.addEventListener("change", () => setSpeed(Number(speed.value)));

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
