"use strict";

// =========================================================================
// helpers
// =========================================================================

// $, el, api, jsonBody, store: common.js; fmtTime, labelOf, displayOf, lyricCell,
// sheetRows, findAt, followWord, followChord: sheet.js (shared with the splitter)


// =========================================================================
// chord parsing + audition synth
// =========================================================================

const NOTE_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const QUALITY_IVS = {
  "": [0, 4, 7], maj: [0, 4, 7], M: [0, 4, 7], m: [0, 3, 7], min: [0, 3, 7], "-": [0, 3, 7],
  "7": [0, 4, 7, 10], maj7: [0, 4, 7, 11], M7: [0, 4, 7, 11], m7: [0, 3, 7, 10], min7: [0, 3, 7, 10],
  sus2: [0, 2, 7], sus4: [0, 5, 7], sus: [0, 5, 7], dim: [0, 3, 6], dim7: [0, 3, 6, 9],
  m7b5: [0, 3, 6, 10], aug: [0, 4, 8], "+": [0, 4, 8], "6": [0, 4, 7, 9], m6: [0, 3, 7, 9],
  "9": [0, 4, 7, 10, 2], add9: [0, 4, 7, 2], madd9: [0, 3, 7, 2], "5": [0, 7],
  "7sus4": [0, 5, 7, 10], maj9: [0, 4, 7, 11, 2], m9: [0, 3, 7, 10, 2],
};

function parseChord(label) {
  const m = /^([A-G])([#b]?)([^/]*)(?:\/([A-G])([#b]?))?$/.exec((label || "").trim());
  if (!m) return null;
  const acc = (a) => (a === "#" ? 1 : a === "b" ? -1 : 0);
  const root = (NOTE_PC[m[1]] + acc(m[2]) + 12) % 12;
  const ivs = QUALITY_IVS[m[3]];
  if (!ivs) return null;
  const bass = m[4] ? (NOTE_PC[m[4]] + acc(m[5]) + 12) % 12 : null;
  return { root, pcs: ivs.map((i) => (root + i) % 12), bass };
}

const synth = {
  ctx: null,
  nodes: [],
  timer: null,
  ensure() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  },
  // Re-strike the chord on every beat, piano-ish decay, quietly under the track.
  play(chordInfo, duration, beat) {
    this.stop();
    const ctx = this.ensure();
    const out = ctx.createGain();
    out.gain.value = 0.16;
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass"; lp.frequency.value = 2400;
    lp.connect(out); out.connect(ctx.destination);
    this.nodes.push(out, lp);

    const bassPc = chordInfo.bass ?? chordInfo.root;
    const midis = [36 + bassPc, ...chordInfo.pcs.map((pc) => 60 + ((pc - 0 + 12) % 12))];
    const hz = (m) => 440 * 2 ** ((m - 69) / 12);
    const t0 = ctx.currentTime + 0.02;
    beat = Math.max(0.25, beat || 0.5);
    const hits = Math.max(1, Math.floor(duration / beat));
    for (let h = 0; h < hits; h++) {
      const start = t0 + h * beat;
      midis.forEach((m, i) => {
        const osc = ctx.createOscillator();
        osc.type = i === 0 ? "sine" : "triangle";
        osc.frequency.value = hz(m);
        const g = ctx.createGain();
        const peak = i === 0 ? 0.9 : 0.5;
        const strum = start + i * 0.012;
        g.gain.setValueAtTime(0, strum);
        g.gain.linearRampToValueAtTime(peak, strum + 0.01);
        g.gain.exponentialRampToValueAtTime(0.001, strum + Math.min(beat * 1.6, 1.8));
        osc.connect(g); g.connect(lp);
        osc.start(strum); osc.stop(strum + Math.min(beat * 1.6, 1.8) + 0.05);
        this.nodes.push(osc, g);
      });
    }
  },
  stop() {
    for (const n of this.nodes) {
      try { n.stop ? n.stop() : null; } catch { /* already stopped */ }
      try { n.disconnect(); } catch { /* already disconnected */ }
    }
    this.nodes = [];
  },
};

// =========================================================================
// progress panel
// =========================================================================

const { stageLabel, pollStatus } = createJobUI(
  ["decode", "separate", "transcribe", "chords", "melody", "align"],
  { decode: "Decode", separate: "Separate", transcribe: "Lyrics", chords: "Chords", melody: "Melody",
    align: "Align" });

// =========================================================================
// library view
// =========================================================================

const lib = { picker: null, sttPicker: null, revisePicker: null, hasKey: null };

function onSourceChange(picker) {
  $("#analyze-btn").disabled = !picker.hasInput();
  const multi = picker.files.length > 1;
  $("#separate").disabled = multi;
  $("#separate-field").title = multi ? "You uploaded stems, so separation isn't needed." : "";
}

async function orStatus() {
  if (lib.hasKey === null) {
    try { lib.hasKey = (await api("/api/openrouter/status")).has_key; } catch { lib.hasKey = false; }
  }
  return lib.hasKey;
}

function sttChoice() { return document.querySelector("input[name=stt]:checked").value; }

async function onSttChange() {
  const or = sttChoice() === "openrouter";
  $("#stt-model-field").hidden = !or;
  if (or) {
    if (!lib.sttPicker) {
      lib.sttPicker = createModelPicker($("#stt-picker"), "transcription", {
        placeholder: "Search transcription models (try: whisper, free)…",
      });
    }
    $("#stt-key-warning").hidden = await orStatus();
  }
}

// the whole-lyric LLM revision after transcription (optional)
async function onReviseChange() {
  const on = $("#revise").checked;
  store.set("mixpipe.revise", on ? "1" : "0");
  $("#revise-model-field").hidden = !on;
  if (on) {
    if (!lib.revisePicker) {
      lib.revisePicker = createModelPicker($("#revise-picker"), "chat", {
        placeholder: "Search chat models (try: free, gemini, claude)…", initialQuery: "free",
      });
    }
    $("#revise-key-warning").hidden = await orStatus();
  }
}

async function analyze(rerun = false) {
  const picker = lib.picker;
  if (!picker.hasInput()) return;
  const stt = sttChoice();
  if (stt === "openrouter" && !lib.sttPicker?.value) {
    $("#upload-progress").replaceChildren(el("div", { class: "msg error" }, "Pick an OpenRouter model first."));
    return;
  }
  const revise = $("#revise").checked;
  if (revise && !lib.revisePicker?.value) {
    $("#upload-progress").replaceChildren(el("div", { class: "msg error" }, "Pick a model for the lyric revision first."));
    return;
  }
  const link = picker.link;
  const request = picker.request({
    language: $("#language").value,
    stt,
    stt_model: stt === "openrouter" ? lib.sttPicker.value : null,
    separate: $("#separate").checked,
    key_changes: $("#key-changes").value,
    recognizer: $("#recognizer").value,
    revise_model: revise ? lib.revisePicker.value : null,
    rerun,
  });

  const btn = $("#analyze-btn");
  btn.disabled = true;
  const box = $("#upload-progress");
  box.replaceChildren(el("div", { class: "msg" }, link ? "Looking up the link…" : "Uploading…"));
  try {
    const { id } = await api(...request);
    refreshSongs();
    await pollStatus(id, box, {
      onDone: () => { picker.clear(); box.replaceChildren(); location.hash = `#/song/${id}`; },
      onRetry: () => analyze(true),
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

function initLibrary() {
  lib.picker = createSourcePicker($("#source-picker"), {
    multiple: true,
    hint: "One file = a full song (vocals are separated automatically). Several files = stems: "
      + "name them like vox.wav, bass.wav, guitar.wav so vocals feed the lyrics and the rest feeds the chords.",
    onChange: onSourceChange,
    onSubmit: () => analyze(false),
  });
  document.querySelectorAll("input[name=stt]").forEach((r) => r.addEventListener("change", onSttChange));
  $("#revise").checked = store.get("mixpipe.revise") === "1";
  $("#revise").addEventListener("change", onReviseChange);
  onReviseChange();
  const savedLang = store.get("mixpipe.language");
  if (savedLang) $("#language").value = savedLang;
  $("#language").addEventListener("change", (e) => store.set("mixpipe.language", e.target.value));
  $("#analyze-btn").addEventListener("click", () => analyze(false));
}

// =========================================================================
// song view
// =========================================================================

const S = {
  song: null,
  tab: "guitar",       // "guitar" | "harmonica" (harmonica.js)
  chords: new Map(),   // id -> chord
  timeline: [],        // visible chords sorted by start
  words: [],           // [{start, end, node}]
  chipNodes: new Map(), // chord id -> [nodes]
  activeChord: null,
  activeWord: null,
  menuFor: null,
  auditionStop: null,
};

const audio = $("#audio");

// concert-pitch label (what the recording plays)
function chordInfoOf(c) {
  if (c.custom) return parseChord(c.custom);
  const cand = c.candidates[c.selected || 0];
  return cand && cand.root != null ? { root: cand.root, pcs: cand.pcs, bass: cand.bass } : null;
}

function makeChip(id, { carry = false } = {}) {
  const c = S.chords.get(id);
  const chip = el("button", {
    class: "chip" + (carry ? " carry" : "") + (!carry && c.custom ? " custom" : "") +
      (!carry && !c.custom && c.selected ? " alt" : ""),
    dataset: { cid: id },
    title: carry ? "Chord still sounding from before this line" : `${fmtTime(c.start)} – click for alternatives`,
    "aria-haspopup": "menu",
  }, displayOf(c));
  chip.addEventListener("click", (e) => { e.stopPropagation(); openChordMenu(id, chip); });
  if (!S.chipNodes.has(id)) S.chipNodes.set(id, []);
  S.chipNodes.get(id).push(chip);
  if (!carry && c.key_change) {
    // the song modulates here: mark it on the chord, not in the overview
    return el("span", { class: "keychg-wrap" },
      el("span", { class: "keychg", title: "The song changes key here" }, `→ ${c.key_change}`), chip);
  }
  return chip;
}

// has the user changed the meter away from what was detected?
function meterIsManual(song) {
  const det = song.meter?.detected;
  if (!det) return false;
  return det.time_signature !== song.meter.time_signature ||
    det.beats_per_bar !== song.meter.beats_per_bar || det.downbeat !== song.meter.downbeat ||
    JSON.stringify(det.sections || []) !== JSON.stringify(song.meter.sections || []);
}

function renderSong() {
  const song = S.song;
  S.chords = new Map(song.chords.map((c) => [c.id, c]));
  S.timeline = chordTimeline(song);
  S.chipNodes = new Map();
  S.words = [];
  S.activeChord = S.activeWord = null;

  $("#song-title").textContent = song.title;
  const t = song.transcript || {};
  const meta = [];
  const disp = song.display || {};
  const keyLabel = disp.key_label || song.key_label;
  if (keyLabel) {
    meta.push(el("span", { class: "badge", title: song.transpose ? `Recording is in ${song.key_label}` : null },
      `Key ${keyLabel}${song.transpose ? ` (transposed ${song.transpose > 0 ? "+" : ""}${song.transpose})` : ""}`));
  }
  if (song.capo) meta.push(el("span", { class: "badge" }, `Capo ${song.capo} · ${disp.shape_key_label} shapes`));
  if (disp.time_signature && song.beat_times?.length) {
    const manual = meterIsManual(song);
    meta.push(el("span", { class: "badge" }, `${disp.time_signature}${manual ? " · manual" : " · detected"}`));
    for (const mc of disp.meter_changes || []) {
      meta.push(el("span", { class: "badge", title: `The meter changes to ${mc.time_signature} at bar ${mc.bar}. Setting a time signature by hand replaces the detected changes.` },
        `${mc.time_signature} from bar ${mc.bar}`));
    }
  }
  if (song.tempo) meta.push(el("span", { class: "badge" }, `${Math.round(song.tempo)} BPM`));
  meta.push(el("span", { class: "badge" }, `${S.timeline.length} chord changes`));
  if (song.recognizer === "btc") {
    meta.push(el("span", { class: "badge", title: "Chords found by the BTC neural model combined with the standard chroma templates." }, "neural chords"));
  }
  if (t.backend && t.backend !== "none") {
    meta.push(el("span", { class: "badge" }, `lyrics: ${t.backend === "mlx" ? "local Whisper" : t.model}${t.language ? " · " + t.language : ""}`));
  } else {
    meta.push(el("span", { class: "badge" }, "instrumental"));
  }
  const rev = t.revised;
  if (rev?.applied) {
    meta.push(el("span", { class: "badge", title: `${rev.model} read the whole lyric and fixed misheard lines. Edit lyrics to change them back.` },
      `lyrics revised · ${rev.changed} word${rev.changed === 1 ? "" : "s"}`));
  } else if (rev) {
    meta.push(el("span", { class: "badge", title: `Revision by ${rev.model} not applied: ${rev.reason}` }, "lyrics as heard"));
  }
  if (t.timing === "approx") meta.push(el("span", { class: "badge err", title: "This model didn't return word timestamps; positions are estimated." }, "word timing approximate"));
  if (t.timing === "aligned") meta.push(el("span", { class: "badge", title: "The model returned no word timestamps; word times come from a local Whisper pass matched to its text." }, "word timing aligned locally"));
  if (t.timing === "snapped") meta.push(el("span", { class: "badge", title: "The model returned no word timestamps; word starts were moved onto the nearest sung onsets in the vocal." }, "word timing snapped to the vocal"));
  $("#song-meta").replaceChildren(...meta);
  const src = song.source;
  $("#song-source").hidden = !src;
  if (src) {
    $("#song-source").replaceChildren(
      "From ", el("a", { href: src.page_url, target: "_blank", rel: "noopener noreferrer" }, src.site),
      src.note ? ` · ${src.note}` : "");
  }
  $("#export-cho").href = apiUrl(`/api/songs/${song.id}/export?fmt=cho`);
  $("#export-txt").href = apiUrl(`/api/songs/${song.id}/export?fmt=txt`);
  $("#export-harp").href = apiUrl(`/api/songs/${song.id}/export?fmt=harp`);
  $("#tab-guitar").href = `#/song/${song.id}`;
  $("#tab-harmonica").href = `#/song/${song.id}/harmonica`;

  renderArrangement();
  const sheet = $("#sheet");
  sheet.classList.toggle("show-bars", $("#show-bars").checked);
  sheet.dir = sheetDir(song);
  const { rows, words } = sheetRows(song, (id, opts) => makeChip(id, opts), seekPlay);
  S.words = words;
  if (!rows.length) rows.push(el("p", { class: "muted" }, "No chords detected."));
  sheet.replaceChildren(...rows);
  renderHarmonica();
}

function setTab(tab) {
  S.tab = tab;
  const guitar = tab === "guitar";
  $("#tab-guitar").setAttribute("aria-selected", String(guitar));
  $("#tab-harmonica").setAttribute("aria-selected", String(!guitar));
  $("#sheet").hidden = !guitar;
  $("#harp-panel").hidden = guitar;
  $("#capo-group").hidden = !guitar;
  $("#instrument-group").hidden = !guitar;
  $("#octave-group").hidden = guitar || !S.song?.harmonica;
  $("#harp-group").hidden = guitar || !S.song?.harmonica;
  $("#export-cho").hidden = $("#export-txt").hidden = !guitar;
  $("#export-harp").hidden = guitar;
  closeChordMenu();
  S.activeChord = S.activeWord = null;
  document.querySelectorAll(".chip.active, .word.active").forEach((n) => n.classList.remove("active"));
  resetHarpFollow();
  tick();
}

async function openSong(id, tab = "guitar") {
  if (S.song?.id === id && $("#song-progress").childElementCount === 0) {
    setTab(tab);  // same song, other tab: nothing to fetch
    return;
  }
  closeChordMenu();
  $("#editor").hidden = true;
  $("#sheet").replaceChildren();
  $("#harp-sheet").replaceChildren();
  $("#melody-progress").replaceChildren();
  $("#song-progress").replaceChildren();
  $("#song-title").textContent = "";
  $("#song-meta").replaceChildren();
  $("#song-source").hidden = true;
  try {
    S.song = await api(`/api/songs/${id}`);
  } catch (e) {
    if (e.status === 409) {
      await pollStatus(id, $("#song-progress"), { onDone: () => openSong(id) });
      return;
    }
    $("#sheet").replaceChildren(el("p", { class: "msg error" }, e.message));
    return;
  }
  renderSong();
  setTab(tab);
  const src = apiUrl(`/api/songs/${id}/audio`);
  if (S.song.audio && audio.dataset.src !== src) {
    audio.src = src;
    audio.dataset.src = src;
  }
  $("#player").hidden = !S.song.audio;
}

function seekPlay(t) {
  stopAudition();
  audio.currentTime = Math.max(0, t - 0.15);
  audio.play().catch(() => {});
}

// ---- playback follow -----------------------------------------------------

function tick() {
  if (!S.song) return;
  const t = audio.currentTime;
  if (S.tab === "harmonica") { harpTick(t); return; }
  const { cur, next } = followChord(S.timeline, S.chipNodes, t, S);
  $("#now-chord").textContent = cur ? displayOf(cur) : "–";
  $("#next-chord").textContent = next ? `${displayOf(next)} in ${Math.max(0, next.start - t).toFixed(1)}s` : "–";
  followWord(S.words, t, S, $("#follow").checked && !audio.paused);
}

function loop() {
  tick();
  if (!audio.paused) requestAnimationFrame(loop);
}
audio.addEventListener("play", () => requestAnimationFrame(loop));
audio.addEventListener("seeked", tick);

// ---- chord menu ------------------------------------------------------------

function closeChordMenu() {
  const menu = $("#chord-menu");
  menu.hidden = true;
  document.querySelectorAll(".chip[aria-expanded=true]").forEach((c) => c.setAttribute("aria-expanded", "false"));
  S.menuFor = null;
}

async function saveChord(id, body) {
  const updated = await api(`/api/songs/${S.song.id}/chords/${id}`, jsonBody("PATCH", body));
  const idx = S.song.chords.findIndex((c) => c.id === id);
  S.song.chords[idx] = updated;
  // labels can turn N.C. <-> chord, which changes rows: refetch the view
  S.song = await api(`/api/songs/${S.song.id}`);
  renderSong();
  tick();
}

function stopAudition() {
  synth.stop();
  if (S.auditionStop) { clearTimeout(S.auditionStop.timer); S.auditionStop = null; audio.pause(); }
}

// play [start, start + dur) of the song, with `overlay()` starting the synth on top
function playSection(start, dur, overlay) {
  stopAudition();
  if (S.song.audio) {
    audio.currentTime = start;
    audio.play().catch(() => {});
  }
  overlay?.();
  S.auditionStop = { timer: setTimeout(stopAudition, dur * 1000 + 150) };
}

function audition(chord, info) {
  const dur = Math.min(Math.max(chord.end - chord.start, 1), 10);
  playSection(chord.start, dur, info && (() => synth.play(info, dur, S.song.beat_period)));
}

// show the shared popup menu under `anchor`; `key` identifies what it's for
function showMenu(rows, anchor, key) {
  const menu = $("#chord-menu");
  menu.replaceChildren(...rows);
  menu.hidden = false;
  const r = anchor.getBoundingClientRect();
  const left = Math.min(window.scrollX + r.left, window.scrollX + document.documentElement.clientWidth - menu.offsetWidth - 8);
  menu.style.left = `${Math.max(8, left)}px`;
  menu.style.top = `${window.scrollY + r.bottom + 6}px`;
  anchor.setAttribute("aria-expanded", "true");
  S.menuFor = key;
  menu.querySelector(".cand.sel .pick, .pick")?.focus();
}

function openChordMenu(id, anchor) {
  if (S.menuFor === id && !$("#chord-menu").hidden) { closeChordMenu(); return; }
  closeChordMenu();
  const c = S.chords.get(id);
  const rows = [el("div", { class: "head" },
    el("span", {}, `${fmtTime(c.start)} – ${fmtTime(c.end)}`),
    el("span", {}, `${c.beats} beat${c.beats === 1 ? "" : "s"}`))];

  c.candidates.forEach((cand, i) => {
    const isSel = !c.custom && (c.selected || 0) === i;
    const info = cand.root != null ? { root: cand.root, pcs: cand.pcs, bass: cand.bass } : null;
    rows.push(el("div", { class: "cand" + (isSel ? " sel" : ""), role: "menuitemradio", "aria-checked": String(isSel) },
      el("button", { class: "pick", onclick: () => { closeChordMenu(); saveChord(id, { selected: i }); } },
        `${isSel ? "✓ " : ""}${cand.display ?? cand.label}`),
      el("div", { class: "conf", title: `${Math.round(cand.conf * 100)}% relative confidence` },
        el("div", { style: `width:${Math.round(cand.conf * 100)}%` })),
      el("button", { class: "play", title: "Play this section with this chord on top",
        onclick: () => audition(c, info) }, "▶ hear")));
  });

  const customInput = el("input", { type: "text", placeholder: "Other chord, e.g. Cadd9 or D/F#", value: c.custom ? displayOf(c) : "" });
  const applyCustom = () => {
    closeChordMenu();
    saveChord(id, { custom: customInput.value.trim() });
  };
  customInput.addEventListener("keydown", (e) => { if (e.key === "Enter") applyCustom(); if (e.key === "Escape") closeChordMenu(); });
  rows.push(el("div", { class: "custom-row" }, customInput, el("button", { onclick: applyCustom }, "Set")));
  rows.push(el("div", { class: "hint" }, "Empty + Set reverts to the detected candidates."));
  rows.push(el("div", { class: "foot" },
    el("button", { onclick: () => audition(c, null) }, "▶ Section only"),
    el("button", { onclick: () => audition(c, chordInfoOf(c)) }, "▶ With current"),
    el("button", { onclick: stopAudition }, "■ Stop")));

  showMenu(rows, anchor, id);
}

document.addEventListener("click", (e) => {
  if (!$("#chord-menu").hidden && !e.target.closest("#chord-menu")) closeChordMenu();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeChordMenu(); });

// ---- lyric editor ------------------------------------------------------------

function linesToText(lines) {
  const out = [];
  for (const ln of lines) {
    if (ln.gap_before && out.length) out.push("");
    out.push(ln.words.map((w) => w.text).join(" "));
  }
  return out.join("\n");
}

async function saveLyrics(text) {
  S.song = await api(`/api/songs/${S.song.id}/lyrics`, jsonBody("PUT", { text }));
  renderSong();
  tick();
}

// ---- arrangement: transpose, capo, meter -------------------------------------

function renderArrangement() {
  const song = S.song;
  const disp = song.display || {};
  $("#tr-val").textContent = (song.transpose > 0 ? "+" : "") + song.transpose;
  const instSel = $("#instrument");
  if (!instSel.options.length) {
    (disp.instruments || []).forEach((i) => instSel.append(el("option", { value: i.id }, i.label)));
  }
  instSel.value = disp.instrument || "guitar";
  const short = (disp.instrument_label || "Guitar").replace(/ \(.*$/, "");
  $("#chord-instrument").textContent = short;
  const capo = $("#capo");
  if (!capo.options.length) {
    for (let i = 0; i <= 9; i++) capo.append(el("option", { value: i }, i ? `${i}` : "none"));
  }
  capo.value = String(song.capo || 0);
  const sug = disp.capo_suggestion;
  const btn = $("#capo-suggest");
  if (sug) {
    btn.hidden = false;
    btn.textContent = `Try capo ${sug.capo || "none"} (${Math.round(sug.easy * 100)}% open shapes)`;
    btn.dataset.capo = sug.capo;
  } else {
    btn.hidden = true;
  }
  const hasBars = !!song.beat_times?.length;
  $("#meter").value = song.meter?.time_signature || "4/4";
  ["#meter", "#db-left", "#db-right", "#show-bars"].forEach((sel) => { $(sel).disabled = !hasBars; });
  $("#meter-reset").hidden = !song.meter?.detected || !meterIsManual(song);
}

async function saveSettings(body) {
  try {
    S.song = await api(`/api/songs/${S.song.id}/settings`, jsonBody("PATCH", body));
    closeChordMenu();
    renderSong();
    tick();
  } catch (e) {
    alertInline($("#arrange"), e.message);
  }
}

function initArrangement() {
  $("#tr-down").addEventListener("click", () => saveSettings({ transpose: Math.max(-11, (S.song.transpose || 0) - 1) }));
  $("#tr-up").addEventListener("click", () => saveSettings({ transpose: Math.min(11, (S.song.transpose || 0) + 1) }));
  $("#instrument").addEventListener("change", (e) => saveSettings({ instrument: e.target.value }));
  $("#capo").addEventListener("change", (e) => saveSettings({ capo: Number(e.target.value) }));
  $("#capo-suggest").addEventListener("click", (e) => saveSettings({ capo: Number(e.target.dataset.capo) }));
  $("#meter").addEventListener("change", (e) => saveSettings({ time_signature: e.target.value }));
  $("#db-left").addEventListener("click", () => saveSettings({ shift_downbeat: -1 }));
  $("#db-right").addEventListener("click", () => saveSettings({ shift_downbeat: 1 }));
  $("#meter-reset").addEventListener("click", () => saveSettings({ reset_meter: true }));
  const bars = store.get("mixpipe.showBars");
  if (bars != null) $("#show-bars").checked = bars === "1";
  $("#show-bars").addEventListener("change", (e) => {
    store.set("mixpipe.showBars", e.target.checked ? "1" : "0");
    document.querySelectorAll(".sheet").forEach((n) => n.classList.toggle("show-bars", e.target.checked));
  });
}

function initSongView() {
  initArrangement();
  $("#edit-btn").addEventListener("click", () => {
    const ed = $("#editor");
    ed.hidden = !ed.hidden;
    if (!ed.hidden) {
      $("#editor-text").value = linesToText(S.song.lines);
      $("#editor-text").dir = $("#sheet").dir;
      $("#editor-text").focus();
    }
  });
  $("#editor-cancel").addEventListener("click", () => { $("#editor").hidden = true; });
  $("#editor-save").addEventListener("click", async () => {
    const btn = $("#editor-save");
    btn.disabled = true;
    try {
      await saveLyrics($("#editor-text").value);
      $("#editor").hidden = true;
    } catch (e) {
      alertInline($("#editor"), e.message);
    } finally { btn.disabled = false; }
  });

  // revise dialog: an LLM suggests fixes to the editor's text
  let chatPicker = null;
  let proposal = null;
  $("#revise-btn").addEventListener("click", async () => {
    const dlg = $("#revise-dialog");
    if (!chatPicker) {
      chatPicker = createModelPicker($("#chat-picker"), "chat", {
        placeholder: "Search chat models (try: free, gemini, llama)…", initialQuery: "free",
      });
    }
    proposal = null;
    $("#revise-result").replaceChildren();
    $("#revise-accept").hidden = true;
    $("#chat-key-warning").hidden = await orStatus();
    dlg.showModal();
  });
  $("#revise-run").addEventListener("click", async () => {
    const box = $("#revise-result");
    if (!chatPicker?.value) { box.replaceChildren(el("div", { class: "msg error" }, "Pick a model first.")); return; }
    if (!$("#editor-text").value.trim()) { box.replaceChildren(el("div", { class: "msg error" }, "The editor has no lyrics to revise.")); return; }
    const btn = $("#revise-run");
    btn.disabled = true;
    box.replaceChildren(el("div", { class: "msg" }, `Asking ${chatPicker.value}…`));
    try {
      const res = await api(`/api/songs/${S.song.id}/revise`,
        jsonBody("POST", { model: chatPicker.value, text: $("#editor-text").value }));
      proposal = res.text;
      const changed = res.diff.some(([op]) => op !== "equal");
      const parts = res.diff.map(([op, text]) => {
        const node = op === "insert" ? el("ins", {}) : op === "delete" ? el("del", {}) : el("span", {});
        text.split("\n").forEach((piece, i) => { if (i) node.append(el("br")); node.append(piece.trim() + " "); });
        return node;
      });
      box.replaceChildren(changed ? el("div", { class: "diff", dir: $("#sheet").dir }, parts)
        : el("div", { class: "msg" }, "No changes suggested."));
      $("#revise-accept").hidden = !changed;
    } catch (e) {
      box.replaceChildren(el("div", { class: "msg error" }, e.message));
    } finally { btn.disabled = false; }
  });
  $("#revise-accept").addEventListener("click", () => {
    if (proposal == null) return;
    $("#editor-text").value = proposal;
    $("#revise-dialog").close();
    $("#editor-text").focus();
  });
}

function alertInline(container, msg) {
  container.querySelector(".msg.error")?.remove();
  container.append(el("div", { class: "msg error" }, msg));
}

// =========================================================================
// routing
// =========================================================================

function route() {
  const m = /^#\/song\/([^/?#]+)(\/harmonica)?/.exec(location.hash);
  stopAudition();
  if (m) {
    $("#library-view").hidden = true;
    $("#song-view").hidden = false;
    openSong(decodeURIComponent(m[1]), m[2] ? "harmonica" : "guitar");
  } else {
    $("#song-view").hidden = true;
    $("#library-view").hidden = false;
    $("#player").hidden = true;
    audio.pause();
    S.song = null;
    refreshSongs();
  }
}

initLibrary();
initSongView();
window.addEventListener("hashchange", route);
route();
