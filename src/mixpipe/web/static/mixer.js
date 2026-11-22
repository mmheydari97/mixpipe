"use strict";
// Mix by chat. Needs common.js ($, el, api, jsonBody, store, createSourcePicker).

const views = { start: $("#start-view"), session: $("#session-view") };
const MODEL_KEY = "mixpipe.mix.model";      // the OpenRouter model last picked here
let picker = null;
let startModelPicker = null, sessionModelPicker = null;
let appConfig = null;
let chosenSplitterSong = null;
let current = null;          // last session state from the server
let pollTimer = null;

// =========================================================================
// routing
// =========================================================================

function route() {
  const m = location.hash.match(/^#\/s\/([\w-]+)/);
  clearTimeout(pollTimer);
  stopAudio();
  // #/new?splitter_song=<id>: the splitter's "Mix these stems" link
  const pick = location.hash.match(/^#\/new\?splitter_song=([^&]+)/);
  if (pick) { chosenSplitterSong = decodeURIComponent(pick[1]); tab = "splitter"; }
  if (m) { show("session"); openSession(m[1]); } else { show("start"); loadStart(); }
}
function show(name) { for (const [k, v] of Object.entries(views)) v.hidden = k !== name; }
window.addEventListener("hashchange", route);

// =========================================================================
// start
// =========================================================================

const tabs = { splitter: [$("#tab-splitter"), $("#pane-splitter")], upload: [$("#tab-upload"), $("#pane-upload")] };
let tab = store.get("mixer.tab") || "splitter";
function setTab(name) {
  tab = name; store.set("mixer.tab", name);
  for (const [k, [btn, pane]] of Object.entries(tabs)) {
    btn.setAttribute("aria-selected", String(k === name)); pane.hidden = k !== name;
  }
  updateMixButton();
}
tabs.splitter[0].addEventListener("click", () => setTab("splitter"));
tabs.upload[0].addEventListener("click", () => setTab("upload"));

function updateMixButton() {
  $("#mix-btn").disabled = tab === "splitter" ? !chosenSplitterSong : !(picker && picker.files.length);
}

async function getConfig() {
  if (!appConfig) appConfig = await api("/api/config");
  return appConfig;
}
const shortModel = (id) => (id || "").split("/").pop();
const MODEL_PLACEHOLDER = "Search chat models (try: claude, gemini, free)…";

function updateModelField() {
  $("#model-field").hidden = !(startModelPicker && $("#use-llm").checked);
}
$("#use-llm").addEventListener("change", updateModelField);

async function loadStart() {
  if (!picker) {
    picker = createSourcePicker($("#source-picker"), {
      multiple: true, hint: "WAV, FLAC, AIFF, MP3 … one per instrument", onChange: updateMixButton,
    });
  }
  setTab(tab);
  try {
    const cfg = await getConfig();
    const or = cfg.openrouter;
    const canUse = !!cfg.llm || !!or?.has_key;
    $("#use-llm").disabled = !canUse;
    if (!canUse) $("#use-llm").checked = false;
    if (or?.has_key) {
      startModelPicker ||= createModelPicker($("#model-picker"), "chat", {
        placeholder: MODEL_PLACEHOLDER, storeKey: MODEL_KEY, defaultValue: or.default_model,
      });
      $("#llm-hint").textContent = "";
    } else {
      $("#llm-hint").textContent = cfg.llm ? `(${cfg.llm.model})` : "(no API key set: keyword rules only)";
    }
    updateModelField();
  } catch { /* the model choice is optional */ }

  const ul = $("#splitter-list");
  try {
    const songs = await api("/api/splitter-songs");
    if (!songs.length) {
      ul.replaceChildren(el("li", { class: "muted" }, "No separated songs yet. Split one in the splitter (make split-ui), or upload stems."));
    } else {
      ul.replaceChildren(...songs.map((s) => el("li", {},
        el("label", {},
          el("input", { type: "radio", name: "splitter-song", value: s.id, checked: s.id === chosenSplitterSong,
            onchange: () => { chosenSplitterSong = s.id; updateMixButton(); } }),
          el("span", { class: "title" }, s.title),
          el("small", { class: "muted" }, s.stems.join(" · "))))));
    }
  } catch (e) { ul.replaceChildren(el("li", { class: "msg error" }, e.message)); }

  const sl = $("#session-list");
  try {
    const sessions = await api("/api/sessions");
    sl.replaceChildren(...(sessions.length ? sessions.map((s) => el("li", {},
      el("a", { href: `#/s/${s.id}` }, s.title),
      el("span", { class: "badge" + (s.state === "ready" ? " ok" : s.state === "error" ? " err" : "") }, s.state),
      el("span", { class: "badge" }, new Date(s.created * 1000).toLocaleTimeString())))
      : [el("li", { class: "muted" }, "None yet.")]));
  } catch { /* list is optional */ }
}

$("#mix-btn").addEventListener("click", async () => {
  const btn = $("#mix-btn");
  const err = $("#start-error");
  err.hidden = true; btn.disabled = true;
  const notes = $("#first-notes").value.trim();
  const llm = $("#use-llm").checked;
  const model = llm && startModelPicker ? startModelPicker.value : null;
  try {
    let res;
    if (tab === "splitter") {
      res = await api("/api/sessions", jsonBody("POST", { splitter_song: chosenSplitterSong, notes, llm, model }));
    } else {
      const fd = new FormData();
      picker.files.forEach((f) => fd.append("files", f, f.name));
      fd.append("notes", notes); fd.append("llm", String(llm));
      if (model) fd.append("model", model);
      res = await api("/api/sessions", { method: "POST", body: fd });
    }
    location.hash = `#/s/${res.id}`;
  } catch (e) {
    err.textContent = e.message; err.hidden = false; btn.disabled = false;
  }
});

// =========================================================================
// session
// =========================================================================
// Two tabs. Mix: one lane per channel, played live in the page from the
// server's per-channel renders (session.console: each channel through its EQ,
// dynamics and effects, before fader, regions and pan), so faders, pans, mutes
// and solos are heard at once; a change is saved as a version on release.
// Master: the whole song mixed and mastered by the server, measured and
// compared. The chat and the version history sit in a drawer.

const fmtDb = (v, digits = 1) => (v == null ? "–" : `${v > 0 ? "+" : ""}${Number(v).toFixed(digits)}`);
const fmtTime = (s) => (isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "0:00");
const fmtClock = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const dbToGain = (d) => Math.pow(10, (d || 0) / 20);
let sessionId = null;
const cur = () => current?.versions.find((v) => v.n === current.current);

async function openSession(id) {
  if (sessionId !== id) resetConsole();
  sessionId = id;
  $("#session").hidden = true;
  let st;
  try { st = await api(`/api/sessions/${id}`); } catch (e) {
    $("#session-progress").replaceChildren(el("div", { class: "msg error" }, e.message)); return;
  }
  $("#session-title").textContent = st.title;
  if (st.state === "ready") { renderSession(st, { fresh: true }); return; }
  $("#session-progress").replaceChildren(el("div", { class: "progress" },
    st.state === "error"
      ? el("div", { class: "msg error" }, `The first mix failed: ${st.error}`)
      : el("div", { class: "msg" }, st.state === "queued" && st.ahead
        ? `Waiting · ${st.ahead} ahead…` : "Cleaning the stems and making the first mix…"),
    st.state === "error" ? null : el("div", { class: "bar indeterminate" }, el("div"))));
  if (st.state !== "error") pollTimer = setTimeout(() => sessionId === id && openSession(id), 1500);
}

function renderSession(st, { fresh = false } = {}) {
  const before = current?.current;
  current = st;
  $("#session-progress").replaceChildren();
  $("#session").hidden = false;
  const badges = [];
  if (st.tempo_bpm) badges.push(el("span", { class: "badge" }, `${Math.round(st.tempo_bpm)} BPM`));
  if (st.preview) {
    badges.push(el("span", { class: "badge", title: "The chat measures this stretch; the console, the master and Export use the whole song" },
      `chat listens to ${fmtTime(st.preview.start_s)}–${fmtTime(st.preview.start_s + st.preview.duration_s)}`));
  }
  $("#session-badges").replaceChildren(...badges);
  if (fresh) {
    $("#chat-log").replaceChildren(...(st.turns || st.versions).flatMap(chatEntry).filter(Boolean));
    scrollChat();
    checkAiStatus();
  }
  renderModelControl(st);
  renderVersions(st);
  renderSuggestions(st);
  if (fresh || before !== st.current) {
    loadConsole(st.current);
    if (stage === "master") loadMaster();
  }
}

// ------------------------------------------------------------------ tabs, drawer

let stage = "mix";
function setStage(name) {
  stage = name;
  $("#tab-mix").setAttribute("aria-selected", String(name === "mix"));
  $("#tab-master").setAttribute("aria-selected", String(name === "master"));
  $("#pane-mix").hidden = name !== "mix";
  $("#pane-master").hidden = name !== "master";
  if (name === "master") { pauseConsole(); loadMaster(); } else { stopMaster(); requestAnimationFrame(drawLanes); }
}
$("#tab-mix").addEventListener("click", () => setStage("mix"));
$("#tab-master").addEventListener("click", () => setStage("master"));

function toggleDrawer(open) {
  $("#drawer").hidden = !open;
  $("#chat-btn").setAttribute("aria-expanded", String(open));
  if (open) scrollChat();
}
$("#chat-btn").addEventListener("click", () => toggleDrawer($("#drawer").hidden));
$("#drawer-close").addEventListener("click", () => toggleDrawer(false));

// ------------------------------------------------------------------ the console (Mix tab)

const C = {
  con: null,            // session.console(): channels, fx return, sr, speed, duration
  peaks: null,          // {name: [[min, max], ...]}
  live: {},             // name -> channel as the page hears it now (fader_db, pan, mute, points...)
  solo: new Set(),      // soloed channels (monitoring only, not saved)
  sel: null,            // the channel in the inspector
  req: 0,
  lanes: new Map(),     // name -> {canvas, wave (offscreen), meter canvas}
  drag: null,           // {name, a, b} while selecting a stretch
};
const meterBuf = new Float32Array(1024);
const PL = createStemPlayer({
  gain: (name) => trackGain(name),
  pan: (name) => C.live[name]?.pan ?? 0,
  meters: true,
  onTick: consoleTick,
  onEnded: () => { pauseConsole(); PL.seek(0); },
});

function resetConsole() {
  PL.teardown();
  Object.assign(C, { con: null, peaks: null, live: {}, solo: new Set(), sel: null, drag: null });
  C.lanes.clear();
  $("#lanes").replaceChildren();
  $("#inspector").hidden = true;
}

// the level along a channel's regions at song time t (the server's breakpoints)
function envAt(points, t) {
  if (!points?.length || t <= points[0][0] || t >= points[points.length - 1][0]) return 1;
  for (let i = 1; i < points.length; i++) {
    const [t1, g1] = points[i];
    if (t <= t1) {
      const [t0, g0] = points[i - 1];
      return t1 > t0 ? g0 + (g1 - g0) * (t - t0) / (t1 - t0) : g1;
    }
  }
  return 1;
}

function audible(name) {
  const ch = C.live[name];
  if (!ch) return false;
  return C.solo.size ? C.solo.has(name) : !ch.mute;
}

function trackGain(name) {
  if (name === "fx") return C.con?.fx ? dbToGain(C.con.fx.scale_db) : 0;
  const ch = C.live[name];
  if (!ch || !audible(name)) return 0;
  return dbToGain(ch.scale_db + ch.fader_db) * envAt(ch.points, PL.now());
}

const fileUrl = (file) => `/api/sessions/${sessionId}/console/files/${file}`;

async function loadConsole(n) {
  const sid = sessionId;
  const req = ++C.req;
  const msg = $("#console-msg");
  msg.textContent = C.con ? "Updating the channels…" : "Preparing the channels (the first time takes a little while)…";
  let con;
  try { con = await api(`/api/sessions/${sid}/console?n=${n}`); } catch (e) {
    if (req === C.req) msg.textContent = e.message;
    return;
  }
  if (req !== C.req || sid !== sessionId) return;
  C.con = con;
  for (const ch of con.channels) C.live[ch.name] = { ...ch };
  if (!C.peaks) C.peaks = await api(`/api/sessions/${sid}/peaks`).catch(() => ({}));
  if (req !== C.req) return;
  renderLanes();
  renderInspector();
  renderMasterControls();
  $("#t-total").textContent = fmtTime(con.duration_s);
  const tracks = con.channels.map((ch) => ({ name: ch.name, label: ch.name, url: fileUrl(ch.file) }));
  if (con.fx) tracks.push({ name: "fx", label: "reverb & echo", url: fileUrl(con.fx.file) });
  const same = PL.set && PL.set.speed === con.speed && tracks.length === PL.set.tracks.length
    && tracks.every((t) => PL.set.urls[t.name] === t.url);
  if (!same) {
    msg.textContent = "Loading the channels…";
    PL.context();
    const build = () => PL.makeSet(tracks, { speed: con.speed });
    try {
      const ok = PL.set ? await PL.swap(build, () => sid === sessionId && req === C.req)
        : await PL.load(build, () => sid === sessionId && req === C.req);
      if (!ok) return;
    } catch (e) { msg.textContent = e.message; return; }
  }
  applyMonitor();
  PL.applyGains();
  PL.applyPans();
  msg.textContent = "";
}

function applyMonitor() {
  if (PL.output) PL.output.gain.value = dbToGain(C.con?.bus_trim_db || 0) * Number($("#monitor").value);
}
$("#monitor").addEventListener("input", applyMonitor);

async function playConsole() {
  if (!PL.set) return;
  if (!await PL.play()) return;
  $("#play-btn").textContent = "❚❚"; $("#play-btn").setAttribute("aria-label", "Pause");
}
function pauseConsole() {
  PL.pause();
  $("#play-btn").textContent = "▶"; $("#play-btn").setAttribute("aria-label", "Play");
  requestAnimationFrame(drawLanes);
}
$("#play-btn").addEventListener("click", () => (PL.playing ? pauseConsole() : playConsole()));
function seekConsole(t) {
  PL.seek(Math.max(0, Math.min(t, (C.con?.duration_s || 0) - 0.05)));
  $("#t-now").textContent = fmtTime(PL.now());
  requestAnimationFrame(drawLanes);
}

function consoleTick(t) {
  $("#t-now").textContent = fmtTime(t);
  if (Object.values(C.live).some((ch) => ch.points?.length)) PL.applyGains();
  drawLanes();
  drawMeters();
}

// ---- lanes

function renderLanes() {
  const box = $("#lanes");
  const order = C.con.channels.map((c) => c.name);
  const existing = new Set(C.lanes.keys());
  if (order.length !== existing.size || order.some((n) => !existing.has(n))) {
    box.replaceChildren();
    C.lanes.clear();
    for (const name of order) box.append(makeLane(name));
  }
  for (const name of order) updateLaneHead(name);
  sizeLanes();
}

function makeLane(name) {
  const ch = C.live[name];
  const fader = el("input", { type: "range", min: "-40", max: "12", step: "0.5", class: "fader", "aria-label": `${name} level` });
  const faderOut = el("output", {});
  fader.addEventListener("input", () => { C.live[name].fader_db = Number(fader.value); faderOut.textContent = fmtDb(Number(fader.value)); PL.applyGains(); });
  fader.addEventListener("change", () => commit([{ target: name, param: "gain_db", value: Number(fader.value) }], `${name} level`));
  fader.addEventListener("dblclick", () => { fader.value = 0; fader.dispatchEvent(new Event("input")); fader.dispatchEvent(new Event("change")); });
  const pan = el("input", { type: "range", min: "-1", max: "1", step: "0.05", class: "pan", "aria-label": `${name} pan` });
  const panOut = el("output", {});
  pan.addEventListener("input", () => { C.live[name].pan = Number(pan.value); panOut.textContent = panLabel(Number(pan.value)); PL.applyPans(); });
  pan.addEventListener("change", () => commit([{ target: name, param: "pan", value: Number(pan.value) }], `${name} pan`));
  pan.addEventListener("dblclick", () => { pan.value = 0; pan.dispatchEvent(new Event("input")); pan.dispatchEvent(new Event("change")); });
  const mute = el("button", { class: "m", title: `Mute ${name} (saved in the mix)`, onclick: () => {
    C.live[name].mute = !C.live[name].mute; updateLaneHead(name); PL.applyGains();
    commit([{ target: name, param: "mute", value: C.live[name].mute }], `${name} mute`);
  } }, "M");
  const solo = el("button", { class: "s", title: `Hear ${name} alone (only while you listen; not saved)`, onclick: () => {
    if (C.solo.has(name)) C.solo.delete(name); else C.solo.add(name);
    for (const n of C.lanes.keys()) updateLaneHead(n);
    PL.applyGains();
  } }, "S");
  const meter = el("canvas", { class: "meter", width: 8, height: 44, "aria-hidden": "true" });
  const nameBtn = el("button", { class: "lane-name", title: "EQ, compressor and effects", onclick: () => selectChannel(name) },
    name.replace(/_/g, " "), el("small", {}, ch.role || ""));
  const head = el("div", { class: "lane-head" }, nameBtn,
    el("div", { class: "lane-ctl" }, mute, solo,
      el("label", { class: "knob", title: "Level (dB). Double-click: 0" }, fader, faderOut),
      el("label", { class: "knob", title: "Pan. Double-click: centre" }, pan, panOut)),
    meter);
  const canvas = el("canvas", { class: "lane-wave", height: 56 });
  bindLaneMouse(canvas, name);
  const row = el("div", { class: "lane", dataset: { name } }, head, canvas);
  C.lanes.set(name, { row, canvas, meter, fader, faderOut, pan, panOut, mute, solo, wave: null });
  return row;
}

const panLabel = (p) => (Math.abs(p) < 0.01 ? "C" : `${Math.round(Math.abs(p) * 100)}${p < 0 ? "L" : "R"}`);

function updateLaneHead(name) {
  const L = C.lanes.get(name), ch = C.live[name];
  if (!L || !ch) return;
  L.fader.value = ch.fader_db; L.faderOut.textContent = fmtDb(ch.fader_db);
  L.pan.value = ch.pan; L.panOut.textContent = panLabel(ch.pan);
  L.mute.setAttribute("aria-pressed", String(!!ch.mute));
  L.solo.setAttribute("aria-pressed", String(C.solo.has(name)));
  L.row.classList.toggle("silent", !audible(name));
  L.row.classList.toggle("selected", C.sel === name);
  L.wave = null;                                  // regions may have changed: redraw the wave
}

function sizeLanes() {
  for (const L of C.lanes.values()) {
    const w = Math.max(200, L.canvas.clientWidth);
    if (L.canvas.width !== w * devicePixelRatio) { L.canvas.width = w * devicePixelRatio; L.wave = null; }
    L.canvas.height = 56 * devicePixelRatio;
  }
  const ruler = $("#ruler");
  ruler.width = Math.max(200, ruler.clientWidth) * devicePixelRatio;
  ruler.height = 22 * devicePixelRatio;
  drawRuler();
  drawLanes();
}
addEventListener("resize", () => { if (C.con) sizeLanes(); });

const css = (v) => getComputedStyle(document.body).getPropertyValue(v).trim() || "#888";

function drawRuler() {
  const c = $("#ruler"), g = c.getContext("2d");
  const d = C.con?.duration_s || 1;
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = css("--muted");
  g.font = `${11 * devicePixelRatio}px sans-serif`;
  const step = [5, 10, 15, 30, 60].find((s) => (c.width / devicePixelRatio) / (d / s) > 60) || 60;
  for (let t = 0; t < d; t += step) {
    const x = (t / d) * c.width;
    g.fillRect(x, c.height - 6 * devicePixelRatio, 1, 6 * devicePixelRatio);
    g.fillText(fmtTime(t), x + 3, 12 * devicePixelRatio);
  }
}
$("#ruler").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  seekConsole(((e.clientX - r.left) / r.width) * (C.con?.duration_s || 0));
});

function laneWave(name, L) {
  if (L.wave) return L.wave;
  const w = L.canvas.width, h = L.canvas.height;
  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const g = off.getContext("2d");
  const d = C.con.duration_s;
  // regions under the wave
  for (const r of C.live[name].regions || []) {
    const x0 = (r.start_s / d) * w, x1 = (r.end_s / d) * w;
    g.fillStyle = r.gain_db <= -60 ? "rgba(200,60,60,.28)" : "rgba(220,150,40,.25)";
    g.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    g.fillStyle = css("--text");
    g.font = `${10 * devicePixelRatio}px sans-serif`;
    if (x1 - x0 > 30 * devicePixelRatio) g.fillText(r.gain_db <= -60 ? "muted" : `${r.gain_db} dB`, x0 + 4, 12 * devicePixelRatio);
  }
  const peaks = C.peaks?.[name] || [];
  g.fillStyle = css("--accent");
  const mid = h / 2;
  for (let x = 0; x < w; x++) {
    const i = Math.floor((x / w) * peaks.length);
    const p = peaks[i];
    if (!p) continue;
    const ch = C.live[name];
    const env = envAt(ch.points, (x / w) * d);
    const a = Math.min(1, Math.abs(p[0]) * 1.8) * env, b = Math.min(1, Math.abs(p[1]) * 1.8) * env;
    g.fillRect(x, mid - b * mid, 1, Math.max(1, (a + b) * mid));
  }
  L.wave = off;
  return off;
}

function drawLanes() {
  if (!C.con) return;
  const d = C.con.duration_s, t = PL.now();
  for (const [name, L] of C.lanes) {
    const g = L.canvas.getContext("2d");
    const w = L.canvas.width, h = L.canvas.height;
    g.clearRect(0, 0, w, h);
    g.globalAlpha = audible(name) ? 1 : 0.35;
    g.drawImage(laneWave(name, L), 0, 0);
    g.globalAlpha = 1;
    if (C.drag && C.drag.name === name) {
      const [a, b] = [Math.min(C.drag.a, C.drag.b), Math.max(C.drag.a, C.drag.b)];
      g.fillStyle = "rgba(80,140,255,.3)";
      g.fillRect((a / d) * w, 0, ((b - a) / d) * w, h);
    }
    g.fillStyle = css("--text");
    g.fillRect((t / d) * w, 0, Math.max(1, devicePixelRatio), h);
  }
}

function drawMeters() {
  for (const tr of PL.set?.tracks || []) {
    const L = C.lanes.get(tr.name);
    if (!L || !tr.meter) continue;
    tr.meter.getFloatTimeDomainData(meterBuf);
    let peak = 0;
    for (let i = 0; i < meterBuf.length; i++) peak = Math.max(peak, Math.abs(meterBuf[i]));
    const db = 20 * Math.log10(peak || 1e-6);
    const frac = Math.max(0, Math.min(1, (db + 48) / 48));
    const g = L.meter.getContext("2d");
    g.clearRect(0, 0, 8, 44);
    g.fillStyle = db > -1 ? "#d33" : db > -9 ? "#da3" : "#3a6";
    g.fillRect(0, 44 * (1 - frac), 8, 44 * frac);
  }
}

// drag across a lane: a stretch to mute or turn down; click: move the playhead, or edit a region
function bindLaneMouse(canvas, name) {
  const timeAt = (e) => {
    const r = canvas.getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (C.con?.duration_s || 0);
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (!C.con) return;
    canvas.setPointerCapture(e.pointerId);
    const t = timeAt(e);
    C.drag = { name, a: t, b: t, x: e.clientX };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!C.drag || C.drag.name !== name) return;
    C.drag.b = timeAt(e);
    drawLanes();
  });
  canvas.addEventListener("pointerup", (e) => {
    if (!C.drag || C.drag.name !== name) return;
    const { a, b } = C.drag;
    const moved = Math.abs(e.clientX - C.drag.x) > 4;
    C.drag = null;
    if (!moved) {
      const r = (C.live[name].regions || []).find((x) => a >= x.start_s && a <= x.end_s);
      if (r) openRegionPopup(name, r.start_s, r.end_s, r, e);
      else seekConsole(a);
      return;
    }
    openRegionPopup(name, Math.min(a, b), Math.max(a, b), null, e);
  });
}

function openRegionPopup(name, a, b, existing, e) {
  closePopup();
  const start = el("input", { value: fmtClock(a), "aria-label": "From", size: 6 });
  const end = el("input", { value: fmtClock(b), "aria-label": "To", size: 6 });
  const fade = el("input", { type: "number", min: "5", max: "2000", step: "5", value: existing?.fade_ms ?? 30, "aria-label": "Fade (ms)" });
  const custom = el("input", { type: "number", min: "-59", max: "6", step: "0.5", value: existing && existing.gain_db > -60 ? existing.gain_db : -6, "aria-label": "Level (dB)" });
  const apply = (gain_db) => {
    let s, t;
    try { s = parseClock(start.value); t = parseClock(end.value); } catch (err) { alertLine(err.message); return; }
    closePopup();
    const adjs = [];
    if (existing && (s !== existing.start_s || t !== existing.end_s)) {
      adjs.push({ target: name, param: "region_remove", value: { start_s: existing.start_s, end_s: existing.end_s } });
    }
    adjs.push({ target: name, param: "region_set", value: { start_s: s, end_s: t, gain_db, fade_ms: Number(fade.value) } });
    commit(adjs, `${name} ${gain_db <= -60 ? "muted" : gain_db + " dB"} ${fmtClock(s)}–${fmtClock(t)}`);
  };
  const pop = el("div", { class: "popup", role: "dialog", "aria-label": "Change this stretch" },
    el("div", { class: "head" }, el("b", {}, name.replace(/_/g, " ")), " ", start, " – ", end),
    el("div", { class: "row" },
      el("button", { class: "primary", onclick: () => apply(-60) }, "Mute"),
      el("button", { onclick: () => apply(-12) }, "−12 dB"),
      el("button", { onclick: () => apply(-6) }, "−6 dB"),
      el("button", { onclick: () => apply(-3) }, "−3 dB")),
    el("div", { class: "row" }, el("label", {}, "Level ", custom, " dB"),
      el("button", { onclick: () => apply(Number(custom.value)) }, "Set")),
    el("div", { class: "row" }, el("label", {}, "Fade ", fade, " ms"),
      el("button", { class: "link", onclick: () => { seekConsole(Math.max(0, a - 2)); playConsole(); } }, "▶ listen from just before")),
    el("div", { class: "row end" },
      existing ? el("button", { class: "danger", onclick: () => {
        closePopup();
        commit([{ target: name, param: "region_remove", value: { start_s: existing.start_s, end_s: existing.end_s } }],
          `${name} back to normal ${fmtClock(existing.start_s)}–${fmtClock(existing.end_s)}`);
      } }, "Back to normal") : null,
      el("button", { onclick: closePopup }, "Cancel")));
  document.body.append(pop);
  const x = Math.min(e.clientX, innerWidth - pop.offsetWidth - 12);
  const y = Math.min(e.clientY + 12, innerHeight - pop.offsetHeight - 12);
  pop.style.left = `${Math.max(8, x)}px`; pop.style.top = `${Math.max(8, y)}px`;
  requestAnimationFrame(drawLanes);
}
function closePopup() { document.querySelectorAll(".popup").forEach((p) => p.remove()); }
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePopup(); });
document.addEventListener("pointerdown", (e) => {
  if (!e.target.closest(".popup") && !e.target.closest(".lane-wave")) closePopup();
});

// ---- saving a change (the same versioned path as the chat)

async function commit(adjustments, label) {
  try {
    const res = await api(`/api/sessions/${sessionId}/set`, jsonBody("POST", { adjustments, label }));
    if (res.turn) { $("#chat-log").append(...chatEntry(res.turn).filter(Boolean)); scrollChat(); }
    renderSession(res);
  } catch (e) {
    alertLine(e.message);
    if (C.con) loadConsole(current.current);      // back to what is saved
  }
}

// ---- inspector: one channel's EQ, dynamics and effects

function selectChannel(name) {
  C.sel = C.sel === name ? null : name;
  for (const n of C.lanes.keys()) C.lanes.get(n).row.classList.toggle("selected", C.sel === n);
  renderInspector();
  if (C.sel) $("#inspector").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function slider({ label, min, max, step, value, unit = "", fmt, title, onCommit, off }) {
  const input = el("input", { type: "range", min, max, step, value });
  const show = (v) => (off != null && v <= off ? "off" : fmt ? fmt(v) : `${Number(v).toFixed(step < 1 ? 1 : 0)}${unit}`);
  const out = el("output", {}, show(value));
  input.addEventListener("input", () => { out.textContent = show(Number(input.value)); });
  input.addEventListener("change", () => onCommit(Number(input.value)));
  return el("label", { class: "ctl", title: title || "" }, el("span", {}, label), input, out);
}

const LOGF = (u) => Math.round(20 * Math.pow(1000, u / 1000));        // 0..1000 -> 20 Hz..20 kHz
const ULOG = (f) => Math.round((Math.log(f / 20) / Math.log(1000)) * 1000);
const fmtHz = (f) => (f >= 1000 ? `${(f / 1000).toFixed(f >= 10000 ? 0 : 1)} kHz` : `${Math.round(f)} Hz`);

// an EQ: the curve the audio gets (drawn from the server's filters) and its bands
function eqEditor(target, bands, extra = []) {
  const canvas = el("canvas", { class: "eq-curve", height: 140 });
  const list = el("div", { class: "bands-edit" });
  bands.forEach((b, i) => {
    const kind = el("select", { "aria-label": "Band type" },
      el("option", { value: "peaking" }, "bell"), el("option", { value: "low_shelf" }, "low shelf"),
      el("option", { value: "high_shelf" }, "high shelf"));
    kind.value = b.type;
    const set = (patch) => commit([{ target, param: "eq_set", value: { index: i, ...b, ...patch } }], `${target} EQ`);
    kind.addEventListener("change", () => set({ type: kind.value }));
    list.append(el("div", { class: "band-row" },
      el("div", { class: "band-top" }, el("b", {}, `Band ${i + 1}`), kind,
        el("button", { class: "link", title: "Remove this band", onclick: () => commit([{ target, param: "eq_remove", value: i }], `${target} EQ`) }, "✕ remove")),
      slider({ label: "Freq", min: 0, max: 1000, step: 1, value: ULOG(b.freq), fmt: (u) => fmtHz(LOGF(u)), onCommit: (u) => set({ freq: LOGF(u) }) }),
      slider({ label: "Gain", min: -12, max: 9, step: 0.5, value: b.gain_db, unit: " dB", onCommit: (v) => set({ gain_db: v }) }),
      slider({ label: "Q", min: 0.2, max: 10, step: 0.1, value: b.q, title: "Width: low = wide, high = narrow", onCommit: (v) => set({ q: v }) })));
  });
  const add = el("button", { disabled: bands.length >= 6, title: "A new bell at 1 kHz; move it where it's needed",
    onclick: () => commit([{ target, param: "eq_set", value: { index: bands.length, freq: 1000, gain_db: -3, q: 1, type: "peaking" } }], `${target} EQ`) }, "+ Band");
  const box = el("div", { class: "eq" }, canvas, ...extra, list, add);
  requestAnimationFrame(() => drawCurve(canvas, target, bands));
  return box;
}

// curves are redrawn whenever their canvas changes size (a tab shown, the window resized)
const curveSizes = new ResizeObserver((entries) => {
  for (const e of entries) if (e.target._curve && e.target.clientWidth) paintCurve(e.target);
});

async function drawCurve(canvas, target, bands) {
  let curve;
  try { curve = await api(`/api/sessions/${sessionId}/eq-curve?target=${encodeURIComponent(target)}&n=${current.current}`); } catch { return; }
  canvas._curve = { curve, bands };
  curveSizes.observe(canvas);
  if (canvas.clientWidth) paintCurve(canvas);
}

function paintCurve(canvas) {
  const { curve, bands } = canvas._curve;
  const w = canvas.width = Math.max(200, canvas.clientWidth) * devicePixelRatio;
  const h = canvas.height = 140 * devicePixelRatio;
  const g = canvas.getContext("2d");
  const fx = (f) => (Math.log(f / 20) / Math.log(1000)) * w;
  const fy = (db) => h / 2 - (db / 18) * (h / 2);
  g.clearRect(0, 0, w, h);
  g.strokeStyle = css("--border"); g.lineWidth = 1;
  g.fillStyle = css("--muted"); g.font = `${10 * devicePixelRatio}px sans-serif`;
  for (const f of [50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
    g.beginPath(); g.moveTo(fx(f), 0); g.lineTo(fx(f), h); g.stroke();
    g.fillText(fmtHz(f).replace(" ", ""), fx(f) + 2, h - 3);
  }
  for (const db of [-12, -6, 0, 6, 12]) {
    g.beginPath(); g.moveTo(0, fy(db)); g.lineTo(w, fy(db)); g.stroke();
    g.fillText(`${db > 0 ? "+" : ""}${db}`, 2, fy(db) - 2);
  }
  g.strokeStyle = css("--accent"); g.lineWidth = 2 * devicePixelRatio;
  g.beginPath();
  curve.freq.forEach((f, i) => { const y = fy(Math.max(-18, Math.min(18, curve.db[i]))); i ? g.lineTo(fx(f), y) : g.moveTo(fx(f), y); });
  g.stroke();
  g.fillStyle = css("--accent");
  bands.forEach((b, i) => {
    g.beginPath(); g.arc(fx(b.freq), fy(b.gain_db), 5 * devicePixelRatio, 0, 7); g.fill();
    g.fillStyle = css("--bg"); g.fillText(`${i + 1}`, fx(b.freq) - 3 * devicePixelRatio, fy(b.gain_db) + 3 * devicePixelRatio);
    g.fillStyle = css("--accent");
  });
}

function renderInspector() {
  const box = $("#inspector");
  const ch = C.sel && C.live[C.sel];
  box.hidden = !ch;
  if (!ch) return;
  const name = C.sel;
  const set = (param) => (value) => commit([{ target: name, param, value }], `${name} ${param}`);
  const comp = ch.comp || {}, sends = ch.sends || {};
  box.replaceChildren(
    el("div", { class: "card-head" },
      el("h2", {}, name.replace(/_/g, " "), " ", el("small", { class: "muted" }, ch.role || "")),
      el("span", { class: "muted small" }, "A change here is processed on the server, then heard (a second or two)."),
      el("button", { onclick: () => selectChannel(name), "aria-label": "Close" }, "✕")),
    el("div", { class: "insp-grid" },
      el("section", {}, el("h3", { class: "small-h" }, "EQ"),
        eqEditor(name, ch.eq || [], [el("div", { class: "row" },
          slider({ label: "High-pass", min: 0, max: 1000, step: 5, value: ch.hpf_hz, unit: " Hz", off: 0, title: "Cuts rumble below this", onCommit: set("hpf_hz") }),
          slider({ label: "Low-pass", min: 0, max: 22000, step: 100, value: ch.lpf_hz, off: 0, fmt: fmtHz, title: "Cuts hiss above this", onCommit: set("lpf_hz") }))])),
      el("section", {}, el("h3", { class: "small-h" }, "Dynamics"),
        slider({ label: "Input trim", min: -24, max: 12, step: 0.5, value: ch.gain_db, unit: " dB", title: "Level into the chain (the compressor hears this)", onCommit: set("trim_db") }),
        slider({ label: "Threshold", min: -60, max: 0, step: 1, value: comp.threshold_db ?? 0, unit: " dB", title: "Compression starts above this", onCommit: set("comp_threshold_db") }),
        slider({ label: "Ratio", min: 1, max: 20, step: 0.5, value: comp.ratio ?? 1, fmt: (v) => `${v}:1`, title: "How hard it compresses", onCommit: set("comp_ratio") }),
        slider({ label: "De-ess", min: 0, max: 18, step: 0.5, value: ch.deess?.max_reduction_db ?? 0, unit: " dB", off: 0, title: "Tames S sounds", onCommit: set("deess_db") }),
        el("h3", { class: "small-h" }, "Effects"),
        slider({ label: "Reverb", min: -60, max: 0, step: 1, value: sends.reverb_db ?? -60, unit: " dB", off: -60, title: "Send to the shared reverb", onCommit: set("reverb_db") }),
        slider({ label: "Echo", min: -60, max: 0, step: 1, value: sends.delay_db ?? -60, unit: " dB", off: -60, title: "Send to the shared echo", onCommit: set("delay_db") }),
        slider({ label: "Drive", min: 0, max: 1, step: 0.05, value: ch.drive?.amount ?? 0, off: 0, fmt: (v) => v.toFixed(2), onCommit: set("drive") }),
        slider({ label: "Chorus", min: 0, max: 1, step: 0.05, value: ch.chorus?.mix ?? 0, off: 0, fmt: (v) => v.toFixed(2), onCommit: set("chorus") }),
        (ch.regions || []).length ? el("div", { class: "row" },
          el("span", { class: "muted small" }, `${ch.regions.length} stretch${ch.regions.length > 1 ? "es" : ""} turned down`),
          el("button", { class: "link", onclick: () => commit([{ target: name, param: "regions_clear" }], `${name} regions`) }, "all back to normal")) : null)));
}

// ------------------------------------------------------------------ the master tab

const M = { stats: {}, ctx: null, analyser: null, raf: 0, active: "a", b: null };

function renderMasterControls() {
  const box = $("#master-controls");
  if (!C.con) return;
  const m = C.con.master || {}, buses = C.con.send_buses || {}, rev = buses.reverb || {}, echo = buses.delay || {};
  const set = (param) => (value) => commit([{ target: "master", param, value }], `master ${param}`);
  box.replaceChildren(
    el("h3", { class: "small-h" }, "Loudness"),
    slider({ label: "Target", min: -24, max: -8, step: 0.5, value: m.target_lufs ?? -14, unit: " LUFS", title: "Streaming services play at about -14 LUFS", onCommit: set("target_lufs") }),
    slider({ label: "Ceiling", min: -3, max: -0.1, step: 0.1, value: m.ceiling_db ?? -1, unit: " dBTP", title: "The limiter keeps true peaks under this", onCommit: set("ceiling_db") }),
    el("h3", { class: "small-h" }, "Glue compressor"),
    slider({ label: "Threshold", min: -40, max: 0, step: 0.5, value: m.glue_comp?.threshold_db ?? -18, unit: " dB", onCommit: set("glue_threshold_db") }),
    slider({ label: "Ratio", min: 1, max: 10, step: 0.1, value: m.glue_comp?.ratio ?? 1.5, fmt: (v) => `${v}:1`, onCommit: set("glue_ratio") }),
    el("h3", { class: "small-h" }, "Master EQ"),
    eqEditor("master", m.master_eq || [], [el("div", { class: "row" },
      slider({ label: "High-pass", min: 0, max: 120, step: 1, value: m.master_hpf_hz ?? 0, unit: " Hz", off: 0, onCommit: set("master_hpf_hz") }))]),
    el("h3", { class: "small-h" }, "Reverb & echo"),
    slider({ label: "Room size", min: 0, max: 1, step: 0.05, value: rev.room_size ?? 0.5, fmt: (v) => v.toFixed(2), onCommit: set("reverb_size") }),
    slider({ label: "Decay", min: 0.1, max: 8, step: 0.1, value: rev.decay_s ?? 1.5, unit: " s", onCommit: set("reverb_decay_s") }),
    slider({ label: "Reverb level", min: -40, max: 6, step: 0.5, value: rev.return_db ?? 0, unit: " dB", onCommit: set("reverb_return_db") }),
    slider({ label: "Echo time", min: 10, max: 1000, step: 5, value: echo.delay_ms ?? 300, unit: " ms", onCommit: set("delay_ms") }),
    slider({ label: "Repeats", min: 0, max: 0.9, step: 0.05, value: echo.feedback ?? 0.3, fmt: (v) => v.toFixed(2), onCommit: set("delay_feedback") }),
    slider({ label: "Echo level", min: -40, max: 6, step: 0.5, value: echo.return_db ?? 0, unit: " dB", onCommit: set("delay_return_db") }));
}

const MA = $("#audio-a"), MB = $("#audio-b");

async function renderMasterOf(n) {
  if (M.stats[n]) return M.stats[n];
  M.stats[n] = await api(`/api/sessions/${sessionId}/master?n=${n}`, { method: "POST" });
  return M.stats[n];
}

async function loadMaster() {
  const n = current.current, sid = sessionId;
  const msg = $("#master-msg");
  const sel = $("#m-b-select");
  sel.replaceChildren(...current.versions.filter((v) => v.n !== n).map((v) => el("option", { value: v.n }, `v${v.n}`)));
  const parent = cur()?.parent;
  if (parent != null) sel.value = String(parent);
  $("#m-b").disabled = !sel.options.length;
  if (M.stats[n]) { showMaster(M.stats[n]); return; }
  msg.textContent = "Mixing and mastering the whole song…";
  $("#m-play").disabled = true;
  try {
    const st = await renderMasterOf(n);
    if (sid !== sessionId || current.current !== n) return;
    showMaster(st);
  } catch (e) { msg.textContent = e.message; }
}
$("#master-render").addEventListener("click", () => { delete M.stats[current.current]; loadMaster(); });

function showMaster(st) {
  $("#master-msg").textContent = st.warning ? `⚠ ${st.warning}` : `v${st.n}, the whole song (${fmtTime(st.duration_s)}).`;
  const src = apiUrl(fileUrl(st.file));
  if (MA.dataset.src !== src) { MA.src = src; MA.dataset.src = src; }
  $("#m-play").disabled = false;
  const o = st.output || {};
  const item = (label, value, title) => el("div", { class: "stat", title }, el("span", {}, label), el("strong", {}, value));
  $("#m-stats").replaceChildren(
    item("Loudness", o.lufs != null ? `${o.lufs.toFixed(1)} LUFS` : "–", "Integrated loudness"),
    item("True peak", o.true_peak_db != null ? `${o.true_peak_db.toFixed(1)} dBTP` : "–", "Keep it under -1 for streaming"),
    item("Crest", o.crest_db != null ? `${o.crest_db.toFixed(1)} dB` : "–", "Peak to RMS: lower is more squashed"),
    item("Range", o.loudness_range_db != null ? `${o.loudness_range_db.toFixed(1)} dB` : "–", "How much the loudness moves through the song"),
    item("Stereo", o.lr_correlation != null ? o.lr_correlation.toFixed(2) : "–", "1 is mono, below 0 has phase problems"),
    item("Limiter", st.crest_lost_db != null ? `${st.crest_lost_db.toFixed(1)} dB` : "–", "Dynamic range the limiter took away"));
  drawLevel(st);
  const bands = Object.entries(o.band_energy_db || {});
  $("#m-bands").replaceChildren(...bands.map(([name, db]) => el("div", { class: "band", title: `${name}: ${db.toFixed(1)} dB` },
    el("div", { class: "bar" }, el("div", { style: `height:${Math.max(2, 100 + db * 2.5)}%` })), el("small", {}, name.replace(/_/g, " ")))));
  applyMasterVolumes();
}

function drawLevel(st) {
  const c = $("#m-level");
  const w = c.width = Math.max(200, c.clientWidth) * devicePixelRatio, h = c.height = 90 * devicePixelRatio;
  const g = c.getContext("2d");
  g.clearRect(0, 0, w, h);
  const lv = st.level_db || [];
  const lo = -40, hi = 0, y = (v) => h - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * h;
  g.fillStyle = css("--accent");
  lv.forEach((v, i) => { const x = (i / lv.length) * w; g.fillRect(x, y(v), Math.max(1, w / lv.length), h - y(v)); });
  if (MA.duration) {
    g.fillStyle = css("--text");
    g.fillRect((MA.currentTime / MA.duration) * w, 0, devicePixelRatio, h);
  }
}

function masterContext() {
  if (!M.ctx) {
    M.ctx = new AudioContext();
    M.analyser = M.ctx.createAnalyser();
    M.analyser.fftSize = 4096;
    M.analyser.smoothingTimeConstant = 0.8;
    M.analyser.connect(M.ctx.destination);
    for (const a of [MA, MB]) M.ctx.createMediaElementSource(a).connect(M.analyser);
  }
  return M.ctx;
}

function drawSpectrum() {
  const c = $("#m-spectrum");
  const w = c.width = Math.max(200, c.clientWidth) * devicePixelRatio, h = c.height = 120 * devicePixelRatio;
  const g = c.getContext("2d");
  g.clearRect(0, 0, w, h);
  if (!M.analyser) return;
  const data = new Float32Array(M.analyser.frequencyBinCount);
  M.analyser.getFloatFrequencyData(data);
  const sr = M.ctx.sampleRate;
  g.strokeStyle = css("--accent"); g.lineWidth = 1.5 * devicePixelRatio;
  g.beginPath();
  for (let x = 0; x < w; x++) {
    const f = 20 * Math.pow(1000, x / w);
    const v = data[Math.min(data.length - 1, Math.round((f / (sr / 2)) * data.length))];
    const yv = h - ((Math.max(-100, Math.min(-10, v)) + 100) / 90) * h;
    x ? g.lineTo(x, yv) : g.moveTo(x, yv);
  }
  g.stroke();
  g.fillStyle = css("--muted"); g.font = `${10 * devicePixelRatio}px sans-serif`;
  for (const f of [100, 1000, 10000]) g.fillText(fmtHz(f), (Math.log(f / 20) / Math.log(1000)) * w + 2, h - 3);
}

function masterLoop() {
  drawSpectrum();
  if (M.stats[current?.current]) drawLevel(M.stats[current.current]);
  $("#m-seek").value = MA.duration ? String(Math.round((MA.currentTime / MA.duration) * 1000)) : "0";
  if (!MA.paused) {
    if (M.active === "b" && Math.abs(MB.currentTime - MA.currentTime) > 0.08) MB.currentTime = MA.currentTime;
    M.raf = requestAnimationFrame(masterLoop);
  }
}

function applyMasterVolumes() {
  const la = M.stats[current?.current]?.output?.lufs, lb = M.b != null ? M.stats[M.b]?.output?.lufs : null;
  let ga = 1, gb = 1;
  if ($("#match-loudness").checked && la != null && lb != null) {
    const ref = Math.min(la, lb);
    ga = dbToGain(ref - la); gb = dbToGain(ref - lb);
  }
  MA.volume = M.active === "a" ? ga : 0;
  MB.volume = M.active === "b" ? gb : 0;
}
$("#match-loudness").addEventListener("change", applyMasterVolumes);

async function setMasterActive(which) {
  if (which === "b") {
    const n = Number($("#m-b-select").value);
    if (!n) return;
    if (M.b !== n || !M.stats[n]) {
      $("#master-msg").textContent = `Mastering v${n} to compare…`;
      try { await renderMasterOf(n); } catch (e) { $("#master-msg").textContent = e.message; return; }
      M.b = n;
      MB.src = apiUrl(fileUrl(M.stats[n].file));
      MB.currentTime = MA.currentTime;
      if (!MA.paused) MB.play().catch(() => {});
      showMaster(M.stats[current.current]);
    }
  }
  M.active = which;
  $("#m-a").classList.toggle("on", which === "a"); $("#m-a").setAttribute("aria-pressed", String(which === "a"));
  $("#m-b").classList.toggle("on", which === "b"); $("#m-b").setAttribute("aria-pressed", String(which === "b"));
  applyMasterVolumes();
}
$("#m-a").addEventListener("click", () => setMasterActive("a"));
$("#m-b").addEventListener("click", () => setMasterActive("b"));
$("#m-b-select").addEventListener("change", () => { if (M.active === "b") setMasterActive("b"); });

$("#m-play").addEventListener("click", async () => {
  if (MA.paused) {
    await masterContext().resume();
    MB.currentTime = MA.currentTime;
    await Promise.all([MA.play(), M.b != null ? MB.play() : null]).catch(() => {});
    $("#m-play").textContent = "❚❚";
    cancelAnimationFrame(M.raf); M.raf = requestAnimationFrame(masterLoop);
  } else stopMaster();
});
function stopMaster() {
  MA.pause(); MB.pause();
  $("#m-play").textContent = "▶";
}
MA.addEventListener("ended", stopMaster);
$("#m-seek").addEventListener("input", () => {
  if (!MA.duration) return;
  MA.currentTime = (Number($("#m-seek").value) / 1000) * MA.duration;
  MB.currentTime = MA.currentTime;
  if (M.stats[current.current]) drawLevel(M.stats[current.current]);
});
$("#m-level").addEventListener("click", (e) => {
  if (!MA.duration) return;
  const r = e.currentTarget.getBoundingClientRect();
  MA.currentTime = ((e.clientX - r.left) / r.width) * MA.duration;
  MB.currentTime = MA.currentTime;
  drawLevel(M.stats[current.current]);
});

function stopAudio() { pauseConsole(); stopMaster(); }

// ------------------------------------------------------------------ versions

function renderVersions(st) {
  const ol = $("#version-list");
  ol.replaceChildren(...[...st.versions].reverse().map((v) => el("li", { class: v.n === st.current ? "on" : "" },
    el("button", { class: "link", onclick: () => gotoVersion(v.n) }, `v${v.n}`),
    el("span", { class: "vmsg" },
      v.kind === "ai" ? el("span", { class: "badge ai", title: `AI edit of v${v.parent}; its plan is v${v.base}'s` }, "AI") : "",
      v.source === "manual" ? (v.changes || []).map((c) => c.text).join("; ") : v.message))));
  const sel = $("#version-select");
  sel.replaceChildren(...[...st.versions].reverse().map((v) => el("option", { value: v.n },
    `v${v.n} · ${(v.source === "manual" ? (v.changes || []).map((c) => c.text).join("; ") : v.message || "").slice(0, 40)}`)));
  sel.value = String(st.current);
}
$("#version-select").addEventListener("change", (e) => gotoVersion(Number(e.target.value)));

async function gotoVersion(n) {
  setBusy(true);
  try { renderSession(await api(`/api/sessions/${sessionId}/goto/${n}`, { method: "POST" })); }
  catch (e) { alertLine(e.message); } finally { setBusy(false); }
}
$("#undo-btn").addEventListener("click", async () => {
  setBusy(true);
  try { renderSession(await api(`/api/sessions/${sessionId}/undo`, { method: "POST" })); }
  finally { setBusy(false); }
});
$("#redo-btn").addEventListener("click", async () => {
  setBusy(true);
  try { renderSession(await api(`/api/sessions/${sessionId}/redo`, { method: "POST" })); }
  finally { setBusy(false); }
});
function alertLine(text, kind = "error") {
  $("#chat-log").append(el("li", { class: `msg-bot ${kind}` }, text)); scrollChat();
  if (kind === "error") { const m = $("#console-msg"); m.textContent = text; setTimeout(() => { if (m.textContent === text) m.textContent = ""; }, 6000); }
}

function setBusy(b) {
  for (const id of ["#chat-send", "#undo-btn", "#redo-btn", "#export-btn", "#model-rules"]) $(id).disabled = b;
  $("#chat-input").disabled = b;
}

$("#export-btn").addEventListener("click", async () => {
  const depth = Number($("#export-depth").value);
  const msg = $("#export-msg");
  const ai = cur()?.kind === "ai";
  msg.textContent = ai ? "Writing the AI edit…" : "Rendering the whole song…";
  setBusy(true);
  try {
    const name = await downloadFile(`/api/sessions/${sessionId}/export`,
      jsonBody("POST", { bit_depth: depth, full: true }), "mix.wav");
    msg.textContent = ai && current.preview
      ? `Saved ${name}. An AI edit is the length of the stretch it was made from.` : `Saved ${name}.`;
  } catch (e) { msg.textContent = e.message; } finally { setBusy(false); }
});

document.addEventListener("keydown", (e) => {
  if (views.session.hidden || e.target.closest("input, textarea, select")) return;
  if (e.key === " ") { e.preventDefault(); stage === "mix" ? $("#play-btn").click() : $("#m-play").click(); }
  if (stage === "master" && e.key.toLowerCase() === "a") setMasterActive("a");
  if (stage === "master" && e.key.toLowerCase() === "b") setMasterActive("b");
});

// ------------------------------------------------------------------ chat

function chatEntry(v) {
  const user = v.message && v.message !== "(first mix)" && v.source !== "manual"
    ? el("li", { class: "msg-user" }, v.kind === "ai" ? `AI edit: ${v.message}` : v.message) : null;
  const chips = (v.changes || []).map((c) => el("span", { class: "chip" }, c.text));
  const src = v.source || "";
  const byModel = src.startsWith("chat:llm") || src.startsWith("llm");
  const via = src === "manual" ? "console" : src.startsWith("ai:") ? `ACE-Step${v.model ? ` (${v.model})` : ""}`
    : byModel ? (v.model ? shortModel(v.model) : "model")
    : src.startsWith("chat:rules") ? "keyword rules" : src.startsWith("rules") ? "rule-based advisor" : src;
  const notes = [];
  if (v.llm_error) notes.push(el("small", { class: "warn" }, `Model unavailable (${v.llm_error}); used the keyword rules.`));
  if (v.rejected?.length) notes.push(el("small", { class: "muted" }, `Ignored: ${v.rejected.join(", ")}`));
  return [user, el("li", { class: "msg-bot" },
    el("div", { class: "reply" }, v.reply || ""),
    chips.length ? el("div", { class: "chips" }, chips) : null,
    notes,
    el("div", { class: "meta" },
      v.n ? el("button", { class: "link", onclick: () => gotoVersion(v.n) }, `v${v.n}`) : el("span", {}, "no change"),
      via ? el("span", { title: byModel && v.model ? v.model : null }, ` · ${via}`) : "",
      v.render_s ? ` · ${v.kind === "ai" ? "made" : "rendered"} in ${v.render_s.toFixed(1)} s` : ""))];
}
function scrollChat() { const log = $("#chat-log"); log.scrollTop = log.scrollHeight; }

function renderSuggestions(st) {
  const stems = Object.keys(cur()?.stems || {});
  const pick = (re) => stems.find((s) => re.test(s));
  const ideas = [];
  const ac = pick(/acoustic|guitar|gtr/i), vox = pick(/vocal|vox|voice/i), bass = pick(/bass/i);
  if (ac) ideas.push(`the ${ac.replace(/_/g, " ")} is too loud`);
  if (vox) ideas.push("more reverb on the vocals", "vocals up a little");
  ideas.push("a little slower");
  if (bass) ideas.push("the bass is boomy");
  ideas.push("it sounds muddy", "make it louder overall");
  $("#suggestions").replaceChildren(...ideas.slice(0, 5).map((t) =>
    el("button", { class: "chip suggestion", onclick: () => { $("#chat-input").value = t; $("#chat-input").focus(); } }, t)));
}

$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const message = $("#chat-input").value.trim();
  if (!message || !sessionId) return;
  if (engine === "ai") aiEdit(message); else mixTurn(message);
});

async function mixTurn(message) {
  $("#chat-input").value = "";
  const log = $("#chat-log");
  log.append(el("li", { class: "msg-user" }, message));
  const thinking = el("li", { class: "msg-bot thinking" },
    current?.llm ? `Asking ${shortModel(current.llm.model)} and re-rendering…` : "Re-rendering…");
  log.append(thinking); scrollChat();
  setBusy(true);
  try {
    const res = await api(`/api/sessions/${sessionId}/chat`, jsonBody("POST", { message }));
    const entry = chatEntry(res.turn);
    thinking.replaceWith(entry[1]);           // the user line is already there
    renderSession(res);
  } catch (err) {
    thinking.replaceWith(el("li", { class: "msg-bot error" }, err.message));
  } finally { setBusy(false); scrollChat(); }
}

// ------------------------------------------------------------------ who answers (the gear)

// "mix": the message changes the plan (OpenRouter model or keyword rules);
// "ai": the message is a prompt for ACE-Step, which rewrites the audio
const ENGINE_KEY = "mixer.engine";
let engine = store.get(ENGINE_KEY) === "ai" ? "ai" : "mix";
let aiStatus = null;

function setEngine(name) {
  engine = name;
  store.set(ENGINE_KEY, name);
  renderEngine();
}

function renderEngine() {
  const ai = engine === "ai";
  const btn = $("#engine-btn");
  $("#engine-label").textContent = ai ? "AI edit · ACE-Step (local)"
    : current?.llm ? `Mix · ${shortModel(current.llm.model)}` : "Mix · keyword rules";
  btn.classList.toggle("ai", ai);
  btn.title = ai ? "Your messages are prompts for ACE-Step on this machine"
    : current?.llm ? `Your messages change the mix, read by ${current.llm.provider}: ${current.llm.model}`
    : "Your messages change the mix, read by the keyword rules";
  document.querySelectorAll('input[name="engine"]').forEach((r) => { r.checked = r.value === engine; });
  $("#engine-mix-settings").hidden = ai;
  $("#engine-ai-settings").hidden = !ai;
  $("#ai-options").hidden = !ai;
  $("#suggestions").hidden = ai;
  $("#chat-input").placeholder = ai ? "Describe the music you want, e.g. warm cello and strings swell"
    : "e.g. the acoustic guitar is too loud";
  $("#chat-send").textContent = ai ? "Edit with AI" : "Send";
  renderAiStatus();
}

function renderAiStatus() {
  const badge = $("#ai-status");
  const note = $("#engine-ai-settings");
  if (!aiStatus) { badge.textContent = ""; note.textContent = "Checking for ACE-Step…"; return; }
  badge.textContent = aiStatus.ok ? "running" : "not running";
  badge.className = `badge ${aiStatus.ok ? "ok" : "err"}`;
  note.textContent = aiStatus.ok
    ? `ACE-Step ${aiStatus.model || ""} is running at ${aiStatus.url}. Pick a range or "Re-style all of it" above the message box.`
    : `ACE-Step isn't running at ${aiStatus.url}. Start it with make studio (or make acestep), then reopen this panel.`;
}

async function checkAiStatus() {
  try { aiStatus = await api("/api/ai-edit/status"); } catch { aiStatus = null; }
  renderAiStatus();
}

async function renderModelControl(st) {
  let hasKey = false;
  try { hasKey = !!(await getConfig()).openrouter?.has_key; } catch { /* stays off */ }
  $("#session-model-picker").hidden = !hasKey;
  $("#model-rules").hidden = !hasKey || !st.llm;
  $("#no-key-note").hidden = hasKey;
  if (sessionModelPicker) sessionModelPicker.set(st.llm?.provider === "openrouter" ? st.llm.model : null);
  renderEngine();
}

function toggleEnginePanel(open) {
  $("#engine-panel").hidden = !open;
  $("#engine-btn").setAttribute("aria-expanded", String(open));
  if (!open) return;
  checkAiStatus();
  if (!sessionModelPicker && appConfig?.openrouter?.has_key) {
    sessionModelPicker = createModelPicker($("#session-model-picker"), "chat", {
      placeholder: MODEL_PLACEHOLDER, storeKey: MODEL_KEY,
      value: current?.llm?.provider === "openrouter" ? current.llm.model : null,
      defaultValue: appConfig?.openrouter?.default_model,
      onChange: (id) => switchModel(id),
    });
  }
}
$("#engine-btn").addEventListener("click", () => toggleEnginePanel($("#engine-panel").hidden));
$("#engine-close").addEventListener("click", () => toggleEnginePanel(false));
document.querySelectorAll('input[name="engine"]').forEach((r) =>
  r.addEventListener("change", () => { if (r.checked) setEngine(r.value); }));
$("#model-rules").addEventListener("click", () => switchModel(null));

async function switchModel(id) {
  const before = current?.llm;
  setBusy(true);
  try {
    renderSession(await api(`/api/sessions/${sessionId}/model`, jsonBody("POST", { model: id })));
    alertLine(id ? `Switched to ${id}. It answers from your next message on.` : "Switched to the keyword rules.", "note");
  } catch (e) {
    alertLine(e.message);
    sessionModelPicker?.set(before?.provider === "openrouter" ? before.model : null);
  } finally { setBusy(false); }
}

function setBusy(b) {
  for (const id of ["#chat-send", "#undo-btn", "#redo-btn", "#export-btn", "#model-rules"]) $(id).disabled = b;
  $("#chat-input").disabled = b;
  document.querySelectorAll(".channels input").forEach((i) => { i.disabled = b; });
}

// ------------------------------------------------------------------ AI edit

const aiMode = () => document.querySelector('input[name="ai-mode"]:checked').value;
document.querySelectorAll('input[name="ai-mode"]').forEach((r) => r.addEventListener("change", () => {
  $("#ai-range").hidden = aiMode() !== "repaint";
  $("#ai-strength-row").hidden = aiMode() !== "cover";
}));
$("#ai-strength").addEventListener("input", () => { $("#ai-strength-val").textContent = $("#ai-strength").value; });
// the playhead, in the clip an AI edit works on (a long song's chat stretch starts later)
const clipTime = () => Math.max(0, PL.now() - (current?.preview?.start_s || 0));
$("#ai-start-here").addEventListener("click", () => { $("#ai-start").value = fmtClock(clipTime()); });
$("#ai-end-here").addEventListener("click", () => { $("#ai-end").value = fmtClock(clipTime()); });

function parseClock(text) {
  // "1:05.5", "65.5" -> seconds; "" -> null
  const t = text.trim();
  if (!t) return null;
  const m = t.match(/^(?:(\d+):)?(\d+(?:\.\d+)?)$/);
  if (!m) throw new Error(`not a time: ${t} (use 0:20 or 20)`);
  return (m[1] ? Number(m[1]) * 60 : 0) + Number(m[2]);
}

async function aiEdit(prompt) {
  const mode = aiMode();
  const body = { prompt, mode };
  try {
    if (mode === "repaint") { body.start = parseClock($("#ai-start").value) ?? 0; body.end = parseClock($("#ai-end").value); }
    else body.strength = Number($("#ai-strength").value);
  } catch (err) { alertLine(err.message); return; }
  $("#chat-input").value = "";
  const from = current.current;
  if (from !== current.current) {
    // the edit is made from the version being listened to
    try { renderSession(await api(`/api/sessions/${sessionId}/goto/${from}`, { method: "POST" })); }
    catch (err) { alertLine(err.message); return; }
  }
  const what = mode === "repaint"
    ? `AI edit · repaint ${fmtClock(body.start)}–${body.end != null ? fmtClock(body.end) : "end"} of v${from}: ${prompt}`
    : `AI edit · re-style v${from} (strength ${body.strength}): ${prompt}`;
  const log = $("#chat-log");
  log.append(el("li", { class: "msg-user" }, what));
  const thinking = el("li", { class: "msg-bot thinking" }, "ACE-Step is making it (about half a minute)…");
  log.append(thinking); scrollChat();
  setBusy(true);
  try {
    const res = await api(`/api/sessions/${sessionId}/ai-edit`, jsonBody("POST", body));
    thinking.replaceWith(chatEntry(res.turn)[1]);
    renderSession(res);
  } catch (err) {
    thinking.replaceWith(el("li", { class: "msg-bot error" }, err.message));
    checkAiStatus();
  } finally { setBusy(false); scrollChat(); }
}


route();
