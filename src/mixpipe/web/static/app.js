"use strict";

// =========================================================================
// helpers
// =========================================================================

// $, el, api, jsonBody, store: common.js

const fmtTime = (t) => {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
};
const RTL = /[֐-ࣿיִ-﷿ﹰ-﻿]/;

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
// OpenRouter model picker (search bar; "free" puts free models first)
// =========================================================================

const modelCache = new Map();
async function loadModels(kind) {
  if (!modelCache.has(kind)) {
    modelCache.set(kind, api(`/api/openrouter/models?kind=${kind}`).catch((e) => {
      modelCache.delete(kind);
      throw e;
    }));
  }
  return modelCache.get(kind);
}

function rankModels(models, query) {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const wantFree = tokens.includes("free");
  const terms = tokens.filter((t) => t !== "free");
  const scored = [];
  for (const m of models) {
    const id = m.id.toLowerCase();
    const name = (m.name || "").toLowerCase();
    const desc = (m.description || "").toLowerCase();
    const idTail = id.split("/").pop();
    let score = 0;
    let ok = true;
    for (const t of terms) {
      let s = 0;
      if (idTail.startsWith(t) || id.startsWith(t)) s = 30;
      else if (name.startsWith(t)) s = 25;
      else if (id.includes(t)) s = 20;
      else if (name.includes(t)) s = 15;
      else if (desc.includes(t)) s = 4;
      if (!s) { ok = false; break; }
      score += s;
    }
    if (ok) scored.push({ m, score });
  }
  scored.sort((a, b) =>
    (wantFree ? Number(b.m.is_free) - Number(a.m.is_free) : 0) ||
    b.score - a.score ||
    a.m.sort_price - b.m.sort_price ||
    a.m.name.localeCompare(b.m.name));
  return { list: scored.map((s) => s.m), wantFree };
}

function createModelPicker(container, kind, { placeholder, initialQuery = "", onChange } = {}) {
  const storeKey = `mixpipe.model.${kind}`;
  let selected = store.get(storeKey);
  let results = [];
  let hl = -1;

  const input = el("input", { type: "search", placeholder, autocomplete: "off", spellcheck: "false",
    "aria-autocomplete": "list", role: "combobox", "aria-expanded": "false" });
  input.value = initialQuery;
  const list = el("ul", { class: "picker-list", role: "listbox", hidden: true });
  const chosen = el("div", { class: "picker-selected" });
  container.replaceChildren(el("div", { class: "picker" }, input, list), chosen);

  const showChosen = () => {
    chosen.replaceChildren(selected ? el("span", {}, "Selected: ", el("b", {}, selected)) : "No model selected");
  };
  showChosen();

  const badgeRow = (m) => {
    const badges = [];
    badges.push(m.is_free ? el("span", { class: "badge free" }, "FREE") : el("span", { class: "badge" }, m.price_label));
    if (kind === "transcription") {
      if (m.word_timing === "word") badges.push(el("span", { class: "badge ok" }, "word timing ✓"));
      else if (m.word_timing === "approx") badges.push(el("span", { class: "badge" }, "word timing ~"));
    } else {
      if ((m.input_modalities || []).includes("audio")) badges.push(el("span", { class: "badge" }, "audio in"));
      if (m.context_length) badges.push(el("span", { class: "badge" }, `${Math.round(m.context_length / 1000)}k ctx`));
    }
    return badges;
  };

  async function render() {
    let models;
    try {
      models = await loadModels(kind);
    } catch (e) {
      list.replaceChildren(el("li", { class: "hint" }, `Couldn't load models: ${e.message}`));
      list.hidden = false;
      return;
    }
    const { list: ranked, wantFree } = rankModels(models, input.value);
    results = ranked.slice(0, 60);
    hl = results.length ? 0 : -1;
    const items = [];
    if (wantFree && kind === "transcription" && !models.some((m) => m.is_free)) {
      items.push(el("li", { class: "hint" },
        "No free transcription models on OpenRouter. Local Whisper is free; paid models are listed below."));
    }
    results.forEach((m, i) => {
      items.push(el("li", {
        role: "option", class: i === hl ? "hl" : "", dataset: { i },
        onmousedown: (ev) => { ev.preventDefault(); choose(m); },
      },
      el("div", { class: "row1" }, el("b", {}, m.name), ...badgeRow(m)),
      el("span", { class: "id" }, m.id)));
    });
    if (!results.length) items.push(el("li", { class: "hint" }, "No matching models."));
    list.replaceChildren(...items);
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }

  function choose(m) {
    selected = m.id;
    store.set(storeKey, selected);
    input.value = m.name;
    close();
    showChosen();
    onChange?.(selected);
  }
  function close() { list.hidden = true; input.setAttribute("aria-expanded", "false"); }
  function moveHl(d) {
    if (!results.length) return;
    hl = (hl + d + results.length) % results.length;
    list.querySelectorAll("li[role=option]").forEach((li) => li.classList.toggle("hl", Number(li.dataset.i) === hl));
    list.querySelector("li.hl")?.scrollIntoView({ block: "nearest" });
  }

  input.addEventListener("focus", render);
  input.addEventListener("input", render);
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); moveHl(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveHl(-1); }
    else if (e.key === "Enter") { e.preventDefault(); if (results[hl]) choose(results[hl]); }
    else if (e.key === "Escape") close();
  });

  return { get value() { return selected; } };
}

// =========================================================================
// progress panel
// =========================================================================

const { stageLabel, pollStatus } = createJobUI(
  ["decode", "separate", "transcribe", "chords", "align"],
  { decode: "Decode", separate: "Separate", transcribe: "Lyrics", chords: "Chords", align: "Align" });

// =========================================================================
// library view
// =========================================================================

const lib = { picker: null, sttPicker: null, hasKey: null };

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

async function analyze(rerun = false) {
  const picker = lib.picker;
  if (!picker.hasInput()) return;
  const stt = sttChoice();
  if (stt === "openrouter" && !lib.sttPicker?.value) {
    $("#upload-progress").replaceChildren(el("div", { class: "msg error" }, "Pick an OpenRouter model first."));
    return;
  }
  const link = picker.link;
  const request = picker.request({
    language: $("#language").value,
    stt,
    stt_model: stt === "openrouter" ? lib.sttPicker.value : null,
    separate: $("#separate").checked,
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
function labelOf(c) {
  if (c.custom) return c.custom;
  const cands = c.candidates || [];
  return cands.length ? cands[Math.min(c.selected || 0, cands.length - 1)].label : "N.C.";
}

// label as shown: after transpose and capo
function displayOf(c) {
  return c.display ?? labelOf(c);
}

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

function renderSong() {
  const song = S.song;
  S.chords = new Map(song.chords.map((c) => [c.id, c]));
  S.timeline = song.chords.filter((c) => labelOf(c) !== "N.C.").sort((a, b) => a.start - b.start);
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
    const det = song.meter?.detected;
    const manual = det && (det.time_signature !== song.meter.time_signature ||
      det.beats_per_bar !== song.meter.beats_per_bar || det.downbeat !== song.meter.downbeat);
    meta.push(el("span", { class: "badge" }, `${disp.time_signature}${manual ? " · manual" : " · detected"}`));
  }
  if (song.tempo) meta.push(el("span", { class: "badge" }, `${Math.round(song.tempo)} BPM`));
  meta.push(el("span", { class: "badge" }, `${S.timeline.length} chord changes`));
  if (t.backend && t.backend !== "none") {
    meta.push(el("span", { class: "badge" }, `lyrics: ${t.backend === "mlx" ? "local Whisper" : t.model}${t.language ? " · " + t.language : ""}`));
  } else {
    meta.push(el("span", { class: "badge" }, "instrumental"));
  }
  if (t.timing === "approx") meta.push(el("span", { class: "badge err", title: "This model didn't return word timestamps; positions are estimated." }, "word timing approximate"));
  $("#song-meta").replaceChildren(...meta);
  const src = song.source;
  $("#song-source").hidden = !src;
  if (src) {
    $("#song-source").replaceChildren(
      "From ", el("a", { href: src.page_url, target: "_blank", rel: "noopener noreferrer" }, src.site),
      src.note ? ` · ${src.note}` : "");
  }
  $("#export-cho").href = `/api/songs/${song.id}/export?fmt=cho`;
  $("#export-txt").href = `/api/songs/${song.id}/export?fmt=txt`;
  $("#polish-btn").disabled = !song.lines.length;

  renderArrangement();
  const sheet = $("#sheet");
  sheet.classList.toggle("show-bars", $("#show-bars").checked);
  const allText = song.lines.map((l) => l.words.map((w) => w.text).join(" ")).join(" ");
  sheet.dir = RTL.test(allText) ? "rtl" : "ltr";

  const rows = song.rows.map((row) => {
    if (row.type === "instrumental") {
      const items = [el("span", { class: "tag" }, "instr.")];
      if (row.bars) {
        items.push(el("span", { class: "bars" }, row.bars.map((bar) => {
          const parts = [];
          if (bar.chords.length) {
            if (bar.hold) parts.push(makeChip(bar.hold, { carry: true }));
            bar.chords.forEach((id) => parts.push(makeChip(id)));
          } else {
            parts.push(el("span", { class: "repeat", title: bar.hold ? `${displayOf(S.chords.get(bar.hold))} continues` : "no chord" },
              bar.hold ? "%" : "–"));
          }
          return el("span", { class: "barcell", title: `Bar ${bar.bar + 1}` }, parts);
        })));
      } else {
        row.chords.forEach((id, i) => {
          if (i) items.push(el("span", { class: "sep" }, "|"));
          items.push(makeChip(id));
        });
      }
      return el("div", { class: "line instrumental" }, items);
    }
    const cells = row.words.map((w, i) => {
      const chips = w.chords.map((id) => makeChip(id));
      if (i === 0 && row.carry) chips.unshift(makeChip(row.carry, { carry: true }));
      const src = song.lines[row.line].words[w.i];
      const word = el("span", { class: "word" + (src.approx ? " approx" : ""), title: fmtTime(w.start) }, w.text);
      word.addEventListener("click", () => seekPlay(w.start));
      S.words.push({ start: w.start, end: w.end, node: word });
      return el("span", { class: "cell" + (w.bar != null ? " barline" : ""), title: w.bar != null ? `Bar ${w.bar + 1}` : null },
        el("span", { class: "chords" }, chips), word);
    });
    return el("div", { class: "line" + (row.gap_before ? " stanza" : "") }, cells);
  });
  if (!rows.length) rows.push(el("p", { class: "muted" }, "No chords detected."));
  sheet.replaceChildren(...rows);
  S.words.sort((a, b) => a.start - b.start);
}

async function openSong(id) {
  closeChordMenu();
  $("#editor").hidden = true;
  $("#sheet").replaceChildren();
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
  const src = `/api/songs/${id}/audio`;
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

function findAt(list, t) {
  let lo = 0, hi = list.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].start <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

function tick() {
  if (!S.song) return;
  const t = audio.currentTime;
  const ci = findAt(S.timeline, t);
  const cur = ci >= 0 && t < S.timeline[ci].end ? S.timeline[ci] : null;
  if (cur?.id !== S.activeChord) {
    (S.chipNodes.get(S.activeChord) || []).forEach((n) => n.classList.remove("active"));
    (S.chipNodes.get(cur?.id) || []).forEach((n) => n.classList.add("active"));
    S.activeChord = cur?.id ?? null;
    $("#now-chord").textContent = cur ? displayOf(cur) : "–";
  }
  const next = S.timeline[ci + 1];
  $("#next-chord").textContent = next ? `${displayOf(next)} in ${Math.max(0, next.start - t).toFixed(1)}s` : "–";

  const wi = findAt(S.words, t);
  const w = wi >= 0 && t <= S.words[wi].end + 0.3 ? S.words[wi] : null;
  if (w !== S.activeWord) {
    S.activeWord?.node.classList.remove("active");
    w?.node.classList.add("active");
    S.activeWord = w;
    if (w && $("#follow").checked && !audio.paused) {
      const r = w.node.getBoundingClientRect();
      if (r.top < 80 || r.bottom > window.innerHeight - 140) {
        w.node.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    }
  }
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

function audition(chord, info) {
  stopAudition();
  const dur = Math.min(Math.max(chord.end - chord.start, 1), 10);
  if (S.song.audio) {
    audio.currentTime = chord.start;
    audio.play().catch(() => {});
  }
  if (info) synth.play(info, dur, S.song.beat_period);
  S.auditionStop = { timer: setTimeout(stopAudition, dur * 1000 + 150) };
}

function openChordMenu(id, anchor) {
  if (S.menuFor === id && !$("#chord-menu").hidden) { closeChordMenu(); return; }
  closeChordMenu();
  const c = S.chords.get(id);
  const menu = $("#chord-menu");
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

  menu.replaceChildren(...rows);
  menu.hidden = false;
  const r = anchor.getBoundingClientRect();
  const left = Math.min(window.scrollX + r.left, window.scrollX + document.documentElement.clientWidth - menu.offsetWidth - 8);
  menu.style.left = `${Math.max(8, left)}px`;
  menu.style.top = `${window.scrollY + r.bottom + 6}px`;
  anchor.setAttribute("aria-expanded", "true");
  S.menuFor = id;
  menu.querySelector(".cand.sel .pick, .pick")?.focus();
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
  const det = song.meter?.detected;
  $("#meter-reset").hidden = !det || (det.time_signature === song.meter.time_signature &&
    det.beats_per_bar === song.meter.beats_per_bar && det.downbeat === song.meter.downbeat);
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
    $("#sheet").classList.toggle("show-bars", e.target.checked);
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

  // polish dialog
  let chatPicker = null;
  let proposal = null;
  $("#polish-btn").addEventListener("click", async () => {
    const dlg = $("#polish-dialog");
    if (!chatPicker) {
      chatPicker = createModelPicker($("#chat-picker"), "chat", {
        placeholder: "Search chat models (try: free, gemini, llama)…", initialQuery: "free",
      });
    }
    proposal = null;
    $("#polish-result").replaceChildren();
    $("#polish-accept").hidden = true;
    $("#chat-key-warning").hidden = await orStatus();
    dlg.showModal();
  });
  $("#polish-run").addEventListener("click", async () => {
    const box = $("#polish-result");
    if (!chatPicker?.value) { box.replaceChildren(el("div", { class: "msg error" }, "Pick a model first.")); return; }
    const btn = $("#polish-run");
    btn.disabled = true;
    box.replaceChildren(el("div", { class: "msg" }, `Asking ${chatPicker.value}…`));
    try {
      const res = await api(`/api/songs/${S.song.id}/polish`, jsonBody("POST", { model: chatPicker.value }));
      proposal = res.text;
      const changed = res.diff.some(([op]) => op !== "equal");
      const parts = res.diff.map(([op, text]) => {
        const node = op === "insert" ? el("ins", {}) : op === "delete" ? el("del", {}) : el("span", {});
        text.split("\n").forEach((piece, i) => { if (i) node.append(el("br")); node.append(piece.trim() + " "); });
        return node;
      });
      box.replaceChildren(changed ? el("div", { class: "diff", dir: $("#sheet").dir }, parts)
        : el("div", { class: "msg" }, "No changes suggested."));
      $("#polish-accept").hidden = !changed;
    } catch (e) {
      box.replaceChildren(el("div", { class: "msg error" }, e.message));
    } finally { btn.disabled = false; }
  });
  $("#polish-accept").addEventListener("click", async () => {
    if (proposal == null) return;
    await saveLyrics(proposal);
    $("#polish-dialog").close();
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
  const m = /^#\/song\/([^/?#]+)/.exec(location.hash);
  stopAudition();
  if (m) {
    $("#library-view").hidden = true;
    $("#song-view").hidden = false;
    openSong(decodeURIComponent(m[1]));
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
