"use strict";
// Guitar tuner. Needs common.js ($, el, api, store, micStream, BASE) and synth.js.
// The microphone goes to the server in chunks (tuner-capture.js -> WebSocket /ws);
// the server (tuner.py) answers each with which string rings and how far off it
// is. Strings are numbered 6 (lowest) to 1, as the server numbers them.
// When a string has stayed in tune a moment the server says "tuned" and freezes
// it: the chip is marked done, a chime plays and the page asks for the next one.

const IN_TUNE = 5;       // cents: green
const CLOSE = 15;        // cents: amber; beyond is red
const QUIET_MS = 600;    // no note this long: the needle greys out
const SCALE = 3.6;       // gauge units per cent (±50 cents = ±180)
// E7, B7 (2.6, 4 kHz): the tuner hears them through the mic, and they're far enough
// above its 1.2 kHz low-pass not to count as a pluck or a string
const CHIME = [100, 107];
const CHIME_ALL = [100, 104, 107];

const T = {
  tunings: [],
  tuning: null,                   // {name, strings: [{string, midi, label}]}
  wholeDown: store.get("mixpipe.tunerWholeDown") === "1",
  drop6: store.get("mixpipe.tunerDrop6") === "1",
  a4: Number(store.get("mixpipe.tunerA4")) || 440,
  lock: null,                     // string number, or null for automatic
  readings: new Map(),            // string number -> last cents heard
  done: new Set(),                // strings that locked in as tuned
  active: null,                   // string on display
  shown: null,                    // smoothed cents on the needle
  lastNote: 0,
  // live audio
  ctx: null, stream: null, node: null, ws: null, quietTimer: null,
};

const targetOf = (n) => T.tuning?.strings.find((s) => s.string === n);
const bare = (label) => label.replace(/-?\d+$/, "");

// ---------------------------------------------------------------- tuning --

function pickTuning() {
  T.tuning = T.tunings.find((t) => t.whole_down === T.wholeDown && t.drop6 === T.drop6) || T.tunings[0];
  $("#tuning-name").textContent = T.tuning.name;
  $("#tuning-notes").textContent = T.tuning.strings.map((s) => bare(s.label)).join(" ");
  T.readings.clear();
  T.done.clear();
  T.active = null;
  T.shown = null;
  renderStrings();
  showIdle();
}

function setSwitch(name, value) {
  $(`input[name=${name}][value="${value ? 1 : 0}"]`).checked = true;
}

function onSwitch() {
  T.wholeDown = $("input[name=whole]:checked").value === "1";
  T.drop6 = $("input[name=drop6]:checked").value === "1";
  store.set("mixpipe.tunerWholeDown", T.wholeDown ? "1" : "0");
  store.set("mixpipe.tunerDrop6", T.drop6 ? "1" : "0");
  pickTuning();
  send({ whole_down: T.wholeDown, drop6: T.drop6 });
}

function onA4() {
  const v = Math.round(Number($("#a4").value));
  if (!(v >= 420 && v <= 460)) { $("#a4").value = T.a4; return; }
  T.a4 = v;
  store.set("mixpipe.tunerA4", String(v));
  T.done.clear();
  updateStrings();
  send({ a4: v });
}

// --------------------------------------------------------------- strings --

const LOCK_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="7" width="10" height="7" rx="1.5"/><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" fill="none" stroke-width="1.6"/></svg>';

function chipState(cents) {
  if (cents == null) return "none";
  const a = Math.abs(cents);
  return a <= IN_TUNE ? "ok" : a <= CLOSE ? "near" : "off";
}

function readingText(cents) {
  if (cents == null) return "·";
  if (Math.abs(cents) <= IN_TUNE) return "✓";
  const a = Math.abs(cents);
  const amount = a >= 100 ? `${(a / 100).toFixed(1)} st` : `${Math.round(a)}¢`;
  return `${cents < 0 ? "↑" : "↓"} ${amount}`;
}

// built once per tuning; updateStrings() changes them in place, so a tap isn't
// lost to a chip rebuilt between press and release
function renderStrings() {
  $("#strings").replaceChildren(...T.tuning.strings.map((s) => el("div", {
    class: "string", dataset: { string: s.string },
  },
  el("button", { class: "pick", type: "button", onclick: () => toggleLock(s.string) },
    el("small", { class: "num" }, String(s.string)),
    el("b", { class: "label" }, s.label),
    el("span", { class: "reading" }),
    el("span", { class: "lock-icon", "aria-hidden": "true" })),
  el("button", { class: "ref", type: "button", title: `Play ${s.label}`, "aria-label": `Play ${s.label}`,
    onclick: () => synth.playNote(s.midi + 12 * Math.log2(T.a4 / 440), 2.5) }, "▶"))));
  document.querySelectorAll("#strings .lock-icon").forEach((i) => { i.innerHTML = LOCK_ICON; });
  updateStrings();
}

function updateStrings() {
  document.querySelectorAll("#strings .string").forEach((chip) => {
    const n = Number(chip.dataset.string);
    const cents = T.readings.get(n);
    const locked = T.lock === n;
    const done = T.done.has(n);
    chip.className = `string ${done ? "ok done" : chipState(cents)}${T.active === n ? " active" : ""}${locked ? " locked" : ""}`;
    chip.querySelector(".reading").textContent = done ? "✓ tuned" : readingText(cents);
    const pick = chip.querySelector(".pick");
    pick.setAttribute("aria-pressed", locked ? "true" : "false");
    pick.title = locked ? "Locked: tap for automatic" : `Lock string ${n}`;
  });
}

function toggleLock(n) {
  T.lock = T.lock === n ? null : n;
  if (T.lock != null) { T.active = T.lock; T.shown = null; }
  send({ lock: T.lock });
  updateStrings();
  if (!T.ws) showIdle();                  // shows the locked string's note, or clears it
}

// ---------------------------------------------------------------- readout --

function buildGauge() {
  const NS = "http://www.w3.org/2000/svg";
  const g = $("#ticks");
  const add = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text) n.textContent = text;
    g.append(n);
  };
  for (let c = -50; c <= 50; c += 5) {
    const x = 200 + c * SCALE;
    const major = c % 25 === 0;
    add("line", { x1: x, x2: x, y1: major ? 44 : 50, y2: 60, class: major ? "tick major" : "tick" });
  }
  add("text", { x: 20, y: 78, "text-anchor": "start" }, "−50¢ flat");
  add("text", { x: 200, y: 78, "text-anchor": "middle" }, "0");
  add("text", { x: 380, y: 78, "text-anchor": "end" }, "sharp +50¢");
}

function setNeedle(cents) {
  const c = Math.max(-50, Math.min(50, cents));
  $("#needle").style.transform = `translateX(${(c * SCALE).toFixed(1)}px)`;
}

function showIdle() {
  const r = $("#readout");
  r.dataset.state = T.ws ? "quiet" : "idle";
  setNeedle(0);
  if (T.lock != null) { showTarget(T.lock); return; }
  $("#note").textContent = "—";
  $("#string-no").textContent = "";
  $("#hz").textContent = "";
  $("#advice").textContent = T.ws ? "Play one string." : "Start the microphone, then play one string.";
}

function showTarget(n) {
  const s = targetOf(n);
  $("#note").textContent = s.label;
  $("#string-no").textContent = `string ${n}${T.lock === n ? " · locked" : ""}`;
  $("#hz").textContent = "";
  $("#advice").textContent = T.ws ? `Play string ${n}.` : "Start the microphone, then play this string.";
}

function advice(cents, semis, heard) {
  const a = Math.abs(cents);
  if (a <= IN_TUNE) return { text: "✓ In tune", tone: "ok" };
  const dir = cents < 0 ? "up" : "down";
  const arrow = cents < 0 ? "▲" : "▼";
  const tone = a <= CLOSE ? "near" : "off";
  if (Math.abs(semis) >= 1) {
    const n = Math.abs(semis);
    return { text: `${arrow} ${n} semitone${n === 1 ? "" : "s"} ${cents < 0 ? "flat" : "sharp"} (hearing ${heard}): tune ${dir}`, tone };
  }
  return { text: `${arrow} Tune ${dir} ${Math.round(a)}¢`, tone };
}

function onReading(r) {
  if (r.state === "error") { $("#advice").textContent = r.detail; return; }
  if (r.state !== "note") {
    if (r.state === "silent" && !T.quietTimer && T.lastNote) {
      T.quietTimer = setTimeout(() => {
        T.quietTimer = null;
        const box = $("#readout");
        // a tuned string's verdict and the next-string prompt stay up
        if (performance.now() - T.lastNote >= QUIET_MS && box.dataset.state !== "tuned") box.dataset.state = "quiet";
      }, QUIET_MS);
    }
    return;
  }
  if (r.state === "tuned") { onTuned(r); return; }
  T.lastNote = performance.now();
  const s = targetOf(r.string);
  if (!s) return;
  if (r.string !== T.active) { T.active = r.string; T.shown = null; }
  // ease the needle; the server already sends a median of the last few readings
  T.shown = T.shown == null ? r.cents : T.shown * 0.5 + r.cents * 0.5;
  T.readings.set(r.string, T.shown);
  if (T.done.has(r.string) && Math.abs(r.cents) > IN_TUNE) T.done.delete(r.string);   // gone out again

  const a = advice(T.shown, r.semitones_off, r.heard);
  const box = $("#readout");
  box.dataset.state = "note";
  box.dataset.tone = a.tone;
  box.dataset.all = "";
  $("#note").textContent = s.label;
  $("#string-no").textContent = `string ${r.string}${r.locked ? " · locked" : ""}`;
  $("#hz").textContent = `${r.hz.toFixed(1)} Hz`;
  $("#advice").textContent = a.text;
  setNeedle(T.shown);
  updateStrings();
}

// the first string, 6 to 1, not tuned yet; null when all are
const nextString = () => T.tuning.strings.map((s) => s.string).find((n) => !T.done.has(n)) ?? null;

function onTuned(r) {
  T.lastNote = performance.now();
  const s = targetOf(r.string);
  if (!s) return;
  const first = !T.done.has(r.string);
  T.done.add(r.string);
  T.active = r.string;
  T.shown = r.cents;
  T.readings.set(r.string, r.cents);
  const next = nextString();
  if (r.new) {
    if (first && next == null) synth.chime(CHIME_ALL, { gap: 0.13, ring: 0.5 });
    else synth.chime(CHIME);
    // locked: the lock moves on, and the server waits for that string's pluck
    if (first && T.lock != null && next != null) { T.lock = next; send({ lock: next, wait: true }); }
  }
  const box = $("#readout");
  box.dataset.state = "tuned";
  box.dataset.tone = "ok";
  box.dataset.all = next == null ? "1" : "";
  $("#note").textContent = s.label;
  $("#string-no").textContent = `string ${r.string} · tuned`;
  $("#hz").textContent = `${r.hz.toFixed(1)} Hz`;
  $("#advice").textContent = next == null
    ? "✓ All six strings are tuned. Ready to play!"
    : `✓ String ${r.string} is tuned. Play string ${next} (${bare(targetOf(next).label)}) next.`;
  setNeedle(r.cents);
  updateStrings();
}

// ------------------------------------------------------------ microphone --

function send(msg) {
  if (T.ws?.readyState === WebSocket.OPEN) T.ws.send(JSON.stringify(msg));
}

async function audioContext() {
  // the server's pitch tracker likes 44.1-48 kHz; ask for 48 kHz, take what the device gives
  try { return new AudioContext({ sampleRate: 48000 }); } catch { return new AudioContext(); }
}

async function start() {
  const btn = $("#mic");
  btn.disabled = true;
  try {
    T.stream = await micStream();
    T.ctx = await audioContext();
    await T.ctx.audioWorklet.addModule("/static/tuner-capture.js");
    let src;
    try { src = T.ctx.createMediaStreamSource(T.stream); } catch {
      // a browser that can't resample the mic to 48 kHz: use the device's rate
      await T.ctx.close();
      T.ctx = new AudioContext();
      await T.ctx.audioWorklet.addModule("/static/tuner-capture.js");
      src = T.ctx.createMediaStreamSource(T.stream);
    }
    T.node = new AudioWorkletNode(T.ctx, "tuner-capture");
    const mute = T.ctx.createGain();
    mute.gain.value = 0;                     // keep the graph pulled, play nothing
    src.connect(T.node); T.node.connect(mute); mute.connect(T.ctx.destination);

    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${BASE}/ws`);
    ws.binaryType = "arraybuffer";
    T.ws = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({ sr: T.ctx.sampleRate, whole_down: T.wholeDown, drop6: T.drop6,
                               a4: T.a4, lock: T.lock }));
    };
    ws.onmessage = (e) => { try { onReading(JSON.parse(e.data)); } catch { /* not a reading */ } };
    ws.onclose = () => {
      if (T.ws === ws) { stop(); $("#advice").textContent = "Lost the tuner server. Is it still running?"; }
    };
    T.node.port.onmessage = (e) => {
      if (ws.readyState === WebSocket.OPEN && ws.bufferedAmount < 65536) ws.send(e.data);
    };
    btn.textContent = "Stop";
    btn.classList.remove("primary");
    showIdle();
  } catch (e) {
    stop();
    $("#advice").textContent = e.name === "NotAllowedError"
      ? "The microphone is blocked. Allow it for this page and try again."
      : `No microphone: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

function stop() {
  const ws = T.ws;
  T.ws = null;
  try { ws?.close(); } catch { /* already closed */ }
  T.stream?.getTracks().forEach((t) => t.stop());
  T.ctx?.close().catch(() => {});
  T.stream = T.ctx = T.node = null;
  clearTimeout(T.quietTimer);
  T.quietTimer = null;
  const btn = $("#mic");
  btn.textContent = "Start tuning";
  btn.classList.add("primary");
  showIdle();
}

// ------------------------------------------------------------------ start --

async function init() {
  buildGauge();
  setSwitch("whole", T.wholeDown);
  setSwitch("drop6", T.drop6);
  $("#a4").value = T.a4;
  document.querySelectorAll("input[name=whole], input[name=drop6]")
    .forEach((i) => i.addEventListener("change", onSwitch));
  $("#a4").addEventListener("change", onA4);
  $("#mic").addEventListener("click", () => (T.ws ? stop() : start()));
  try {
    T.tunings = await api("/api/tunings");
  } catch (e) {
    $("#advice").textContent = `Could not load the tunings: ${e.message}`;
    return;
  }
  pickTuning();
}

init();
