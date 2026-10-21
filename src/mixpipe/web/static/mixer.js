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

const fmtDb = (v, digits = 1) => (v == null ? "–" : `${v > 0 ? "+" : ""}${Number(v).toFixed(digits)}`);
const fmtTime = (s) => (isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "0:00");
let sessionId = null;

async function openSession(id) {
  sessionId = id;
  $("#chat-log").replaceChildren();
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
  current = st;
  $("#session-progress").replaceChildren();
  $("#session").hidden = false;
  const badges = [];
  if (st.preview) {
    badges.push(el("span", { class: "badge", title: "Chat turns render this slice; Export renders the whole song" },
      `preview ${fmtTime(st.preview.start_s)}–${fmtTime(st.preview.start_s + st.preview.duration_s)} of ${fmtTime(st.preview.full_duration_s)}`));
  }
  $("#session-badges").replaceChildren(...badges);
  if (fresh) {
    $("#chat-log").replaceChildren(...(st.turns || st.versions).flatMap(chatEntry).filter(Boolean));
    scrollChat();
  }
  renderModelControl(st);
  renderVersions(st);
  renderChannels(st);
  renderStats(st);
  setupPlayer(st);
  renderSuggestions(st);
}

const cur = () => current.versions.find((v) => v.n === current.current);

// ------------------------------------------------------------------ chat

function chatEntry(v) {
  const user = v.message && v.message !== "(first mix)" && v.source !== "manual"
    ? el("li", { class: "msg-user" }, v.message) : null;
  const chips = (v.changes || []).map((c) => el("span", { class: "chip" }, c.text));
  const src = v.source || "";
  const byModel = src.startsWith("chat:llm") || src.startsWith("llm");
  const via = src === "manual" ? "fader" : byModel ? (v.model ? shortModel(v.model) : "model")
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
      v.render_s ? ` · rendered in ${v.render_s.toFixed(1)} s` : ""))];
}
function scrollChat() { const log = $("#chat-log"); log.scrollTop = log.scrollHeight; }

function renderSuggestions(st) {
  const stems = Object.keys(cur()?.stems || {});
  const pick = (re) => stems.find((s) => re.test(s));
  const ideas = [];
  const ac = pick(/acoustic|guitar|gtr/i), vox = pick(/vocal|vox|voice/i), bass = pick(/bass/i);
  if (ac) ideas.push(`the ${ac.replace(/_/g, " ")} is too loud`);
  if (vox) ideas.push("more reverb on the vocals", "vocals up a little");
  if (bass) ideas.push("the bass is boomy");
  ideas.push("it sounds muddy", "make it louder overall");
  $("#suggestions").replaceChildren(...ideas.slice(0, 5).map((t) =>
    el("button", { class: "chip suggestion", onclick: () => { $("#chat-input").value = t; $("#chat-input").focus(); } }, t)));
}

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message || !sessionId) return;
  input.value = "";
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
    if (res.turn.n) selectVersionA(res.turn.n);
  } catch (err) {
    thinking.replaceWith(el("li", { class: "msg-bot error" }, err.message));
  } finally { setBusy(false); scrollChat(); }
});

// ------------------------------------------------------------------ model

async function renderModelControl(st) {
  const btn = $("#model-btn");
  btn.textContent = st.llm ? `Model: ${shortModel(st.llm.model)} ▾` : "Keyword rules ▾";
  btn.title = st.llm ? `${st.llm.provider}: ${st.llm.model}` : "Your messages are read by the keyword rules";
  let canSwitch = false;
  try { canSwitch = !!(await getConfig()).openrouter?.has_key; } catch { /* stays off */ }
  btn.disabled = !canSwitch;
  if (!canSwitch) btn.textContent = btn.textContent.replace(" ▾", "");
  if (sessionModelPicker) sessionModelPicker.set(st.llm?.provider === "openrouter" ? st.llm.model : null);
}

function toggleModelPanel(open) {
  const panel = $("#model-panel");
  panel.hidden = !open;
  $("#model-btn").setAttribute("aria-expanded", String(open));
  if (open && !sessionModelPicker) {
    sessionModelPicker = createModelPicker($("#session-model-picker"), "chat", {
      placeholder: MODEL_PLACEHOLDER, storeKey: MODEL_KEY,
      value: current?.llm?.provider === "openrouter" ? current.llm.model : null,
      defaultValue: appConfig?.openrouter?.default_model,
      onChange: (id) => switchModel(id),
    });
  }
  if (open) $("#session-model-picker input")?.focus();
}
$("#model-btn").addEventListener("click", () => toggleModelPanel($("#model-panel").hidden));
$("#model-close").addEventListener("click", () => toggleModelPanel(false));
$("#model-rules").addEventListener("click", () => switchModel(null));

async function switchModel(id) {
  const before = current?.llm;
  setBusy(true);
  try {
    renderSession(await api(`/api/sessions/${sessionId}/model`, jsonBody("POST", { model: id })));
    alertLine(id ? `Switched to ${id}. It answers from your next message on.` : "Switched to the keyword rules.", "note");
    toggleModelPanel(false);
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

// ------------------------------------------------------------------ versions

function renderVersions(st) {
  const ol = $("#version-list");
  ol.replaceChildren(...[...st.versions].reverse().map((v) => el("li", { class: v.n === st.current ? "on" : "" },
    el("button", { class: "link", onclick: () => gotoVersion(v.n) }, `v${v.n}`),
    el("span", { class: "vmsg" }, v.source === "manual" ? (v.changes || []).map((c) => c.text).join("; ") : v.message),
    el("span", { class: "muted small" }, v.output?.lufs != null ? `${v.output.lufs.toFixed(1)} LUFS` : ""))));
  const sel = $("#ab-b-select");
  const prevB = sel.value;
  sel.replaceChildren(...st.versions.map((v) => el("option", { value: v.n }, `v${v.n}`)));
  const parent = cur()?.parent;
  sel.value = prevB && st.versions.some((v) => String(v.n) === prevB) && Number(prevB) !== st.current
    ? prevB : String(parent ?? st.versions[0].n);
}

async function gotoVersion(n) {
  setBusy(true);
  try { renderSession(await api(`/api/sessions/${sessionId}/goto/${n}`, { method: "POST" })); selectVersionA(n); }
  catch (e) { alertLine(e.message); } finally { setBusy(false); }
}
$("#undo-btn").addEventListener("click", async () => {
  setBusy(true);
  try { const st = await api(`/api/sessions/${sessionId}/undo`, { method: "POST" }); renderSession(st); selectVersionA(st.current); }
  finally { setBusy(false); }
});
$("#redo-btn").addEventListener("click", async () => {
  setBusy(true);
  try { const st = await api(`/api/sessions/${sessionId}/redo`, { method: "POST" }); renderSession(st); selectVersionA(st.current); }
  finally { setBusy(false); }
});
function alertLine(text, kind = "error") {
  $("#chat-log").append(el("li", { class: `msg-bot ${kind}` }, text)); scrollChat();
}

$("#export-btn").addEventListener("click", async () => {
  const depth = Number($("#export-depth").value);
  const msg = $("#export-msg");
  msg.textContent = current.preview ? "Rendering the whole song…" : "Rendering…";
  setBusy(true);
  try {
    const res = await fetch(`/api/sessions/${sessionId}/export`, jsonBody("POST", { bit_depth: depth, full: true }));
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const blob = await res.blob();
    const name = (res.headers.get("content-disposition") || "").match(/filename="?([^"]+)"?/)?.[1] || "mix.wav";
    const a = el("a", { href: URL.createObjectURL(blob), download: name });
    document.body.append(a); a.click(); a.remove();
    msg.textContent = `Saved ${name}.`;
  } catch (e) { msg.textContent = e.message; } finally { setBusy(false); }
});

// ------------------------------------------------------------------ channels

function renderChannels(st) {
  const v = cur();
  if (!v) return;
  $("#channel-rows").replaceChildren(...Object.entries(v.stems).map(([name, s]) => {
    const level = v.levels?.[name];
    const fader = el("input", { type: "range", min: "-24", max: "12", step: "0.5", value: s.fader_db ?? 0,
      "aria-label": `${name} fader` });
    const val = el("output", {}, fmtDb(s.fader_db ?? 0));
    fader.addEventListener("input", () => { val.textContent = fmtDb(Number(fader.value)); });
    fader.addEventListener("change", () => setParam(name, "gain_db", Number(fader.value)));
    const mute = el("input", { type: "checkbox", checked: !!s.mute, "aria-label": `Mute ${name}` });
    mute.addEventListener("change", () => setParam(name, "mute", mute.checked));
    return el("tr", { class: s.mute ? "muted-row" : "" },
      el("td", {}, el("span", { class: "stem-name" }, name), el("small", { class: "muted" }, s.role)),
      el("td", { class: "num", title: level == null && !s.mute ? "No audible signal in this stretch" : null },
        s.mute ? "–" : level != null ? `${fmtDb(level)} dB` : el("span", { class: "muted" }, "silent")),
      el("td", { class: "fader" }, fader, val),
      el("td", {}, mute));
  }));
}

async function setParam(target, param, value) {
  setBusy(true);
  try {
    const res = await api(`/api/sessions/${sessionId}/set`,
      jsonBody("POST", { adjustments: [{ target, param, value }], label: `${target} ${param}` }));
    if (res.turn) {
      $("#chat-log").append(...chatEntry(res.turn).filter(Boolean)); scrollChat();
      renderSession(res); selectVersionA(res.turn.n);
    } else renderSession(res);
  } catch (e) { alertLine(e.message); } finally { setBusy(false); }
}

function renderStats() {
  const o = cur()?.output || {};
  const item = (label, value, title) => el("div", { class: "stat", title }, el("span", {}, label), el("strong", {}, value));
  $("#output-stats").replaceChildren(
    item("Loudness", o.lufs != null ? `${o.lufs.toFixed(1)} LUFS` : "–", "Integrated loudness of this version"),
    item("True peak", o.true_peak_db != null ? `${o.true_peak_db.toFixed(1)} dBTP` : "–"),
    item("Crest", o.crest_db != null ? `${o.crest_db.toFixed(1)} dB` : "–", "Peak to RMS: lower is more squashed"),
    item("Stereo", o.lr_correlation != null ? o.lr_correlation.toFixed(2) : "–", "L/R correlation: 1 is mono, below 0 has phase problems"));
}

// ------------------------------------------------------------------ A/B player

const A = $("#audio-a"), B = $("#audio-b");
let active = "a", versionA = null, versionB = null;

function setupPlayer(st) {
  if (versionA == null || !st.versions.some((v) => v.n === versionA)) versionA = st.current;
  versionB = Number($("#ab-b-select").value);
  loadInto(A, versionA); loadInto(B, versionB);
  $("#ab-a-label").textContent = `v${versionA}`;
  $("#ab-b-label").textContent = `v${versionB}`;
  applyVolumes();
}
function loadInto(audio, n) {
  const src = `/api/sessions/${sessionId}/audio/${n}.wav`;
  if (audio.dataset.src === src) return;
  const t = audio.currentTime || 0, playing = !A.paused;
  audio.dataset.src = src; audio.src = src;
  audio.addEventListener("loadedmetadata", () => { audio.currentTime = Math.min(t, audio.duration || t); if (playing) audio.play().catch(() => {}); }, { once: true });
}
function selectVersionA(n) {
  versionA = n; loadInto(A, n); $("#ab-a-label").textContent = `v${n}`;
  setActive("a"); applyVolumes();
}
$("#ab-b-select").addEventListener("change", () => {
  versionB = Number($("#ab-b-select").value); loadInto(B, versionB);
  $("#ab-b-label").textContent = `v${versionB}`; applyVolumes();
});
function lufsOf(n) { return current?.versions.find((v) => v.n === n)?.output?.lufs; }
function applyVolumes() {
  // loudness-matched: turn the louder one down to the quieter one, so "louder" can't win the comparison
  let ga = 1, gb = 1;
  const la = lufsOf(versionA), lb = lufsOf(versionB);
  if ($("#match-loudness").checked && la != null && lb != null) {
    const ref = Math.min(la, lb);
    ga = Math.pow(10, (ref - la) / 20); gb = Math.pow(10, (ref - lb) / 20);
  }
  A.volume = active === "a" ? ga : 0;
  B.volume = active === "b" ? gb : 0;
}
$("#match-loudness").addEventListener("change", applyVolumes);
function setActive(which) {
  active = which;
  $("#ab-a").classList.toggle("on", which === "a"); $("#ab-a").setAttribute("aria-pressed", String(which === "a"));
  $("#ab-b").classList.toggle("on", which === "b"); $("#ab-b").setAttribute("aria-pressed", String(which === "b"));
  applyVolumes();
}
$("#ab-a").addEventListener("click", () => setActive("a"));
$("#ab-b").addEventListener("click", () => setActive("b"));

$("#play-btn").addEventListener("click", () => {
  if (A.paused) {
    B.currentTime = A.currentTime;
    Promise.all([A.play(), B.play()]).catch(() => {});
    $("#play-btn").textContent = "❚❚"; $("#play-btn").setAttribute("aria-label", "Pause");
  } else stopAudio(false);
});
function stopAudio(rewind = false) {
  A.pause(); B.pause();
  if (rewind) { A.currentTime = 0; B.currentTime = 0; }
  $("#play-btn").textContent = "▶"; $("#play-btn").setAttribute("aria-label", "Play");
}
A.addEventListener("timeupdate", () => {
  if (Math.abs(B.currentTime - A.currentTime) > 0.08 && !A.paused) B.currentTime = A.currentTime;
  $("#seek").value = A.duration ? String(Math.round((A.currentTime / A.duration) * 1000)) : "0";
  $("#t-now").textContent = fmtTime(A.currentTime);
  $("#t-total").textContent = fmtTime(A.duration);
});
A.addEventListener("ended", () => stopAudio(true));
$("#seek").addEventListener("input", () => {
  if (!A.duration) return;
  const t = (Number($("#seek").value) / 1000) * A.duration;
  A.currentTime = t; B.currentTime = Math.min(t, B.duration || t);
});
document.addEventListener("keydown", (e) => {
  if (views.session.hidden || e.target.closest("input, textarea, select")) return;
  if (e.key === " ") { e.preventDefault(); $("#play-btn").click(); }
  if (e.key.toLowerCase() === "a") setActive("a");
  if (e.key.toLowerCase() === "b") setActive("b");
});

route();
