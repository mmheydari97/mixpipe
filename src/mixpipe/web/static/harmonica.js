"use strict";
// Harmonica tab: the melody as tab for a 12-hole chromatic or a 10-hole diatonic
// harmonica in any key.
// Load after app.js; uses its S, audio, synth, fmtTime, findAt, followWord,
// showMenu, closeChordMenu, playSection, stopAudition, saveSettings, alertInline.

const H = {
  notes: new Map(),     // id -> note (from the server's harmonica view)
  timeline: [],         // notes that sound, sorted by start
  words: [],            // [{start, end, node}] in the harmonica sheet
  chipNodes: new Map(), // note id -> [nodes]
  activeNote: null,
  activeWord: null,
};

const harpJob = createJobUI(["decode", "melody"], { decode: "Decode", melody: "Melody" });

// A reed-ish tone: detuned saw + square through a lowpass, with a soft attack.
synth.playNote = function (midi, duration) {
  this.stop();
  const ctx = this.ensure();
  const out = ctx.createGain();
  out.gain.value = 0.12;
  const lp = ctx.createBiquadFilter();
  lp.type = "lowpass"; lp.frequency.value = 2200; lp.Q.value = 0.7;
  lp.connect(out); out.connect(ctx.destination);
  this.nodes.push(out, lp);
  const hz = 440 * 2 ** ((midi - 69) / 12);
  const t0 = ctx.currentTime + 0.02;
  const t1 = t0 + Math.max(0.3, duration);
  const env = ctx.createGain();
  env.gain.setValueAtTime(0, t0);
  env.gain.linearRampToValueAtTime(1, t0 + 0.04);
  env.gain.setValueAtTime(1, t1 - 0.08);
  env.gain.linearRampToValueAtTime(0, t1);
  env.connect(lp);
  this.nodes.push(env);
  [["sawtooth", 1, 0.6], ["square", 1.003, 0.35]].forEach(([type, ratio, level]) => {
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.value = hz * ratio;
    const g = ctx.createGain();
    g.gain.value = level;
    osc.connect(g); g.connect(env);
    osc.start(t0); osc.stop(t1 + 0.02);
    this.nodes.push(osc, g);
  });
};

const noteLabel = (n) => (n.rest ? "rest" : `${n.display} ${n.tab || "—"}`);

// ---- instrument ------------------------------------------------------------------

const HARP_KEYS = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const prettyKey = (k) => k.replace("#", "♯").replace("b", "♭");

function renderInstrument(hv) {
  const inst = hv.instrument;
  const keySel = $("#harp-key");
  if (!keySel.options.length) {
    HARP_KEYS.forEach((k) => keySel.append(el("option", { value: k }, prettyKey(k))));
  }
  $("#harp-type").value = inst.type;
  keySel.value = inst.key;
  $("#harp-name").textContent = inst.short_name;
  $("#harp-legend").replaceChildren(...(inst.diatonic
    ? [el("b", {}, "3↑"), " blow · ", el("b", {}, "3↓"), " draw · ", el("b", {}, "′ ″ ‴"), " bend ½ / 1 / 1½ steps"]
    : [el("b", {}, "3↑"), " blow · ", el("b", {}, "3↓"), " draw · ", el("b", {}, "●"), " button in"]));
  $("#fast-notes").checked = !!hv.fast;
  const sug = (hv.suggestions || []).slice(0, 4);
  const box = $("#harp-suggest");
  box.hidden = !sug.length;
  if (sug.length) {
    box.replaceChildren(el("span", { class: "lbl" }, "Diatonic harps that suit this melody:"),
      ...sug.map((r) => el("button", {
        class: r.current && inst.diatonic ? "on" : "",
        title: `${r.position ? r.position + ". " : ""}${Math.round(r.playable * 100)}% of the melody is playable, `
          + `${Math.round(r.unbent * 100)}% without bending.`,
        onclick: () => saveSettings({ harmonica_type: "diatonic10", harmonica_key: r.key }),
      }, el("b", {}, prettyKey(r.key)), el("small", {}, `${r.position ? r.position.split(" ")[0] + " pos · " : ""}${Math.round(r.unbent * 100)}% no bends`))));
  }
}

// ---- rendering ---------------------------------------------------------------

function makeNoteChip(id) {
  const n = H.notes.get(id);
  const cls = ["chip", "note"];
  if (n.rest) cls.push("rest");
  else {
    if (!n.in_range) cls.push("oor");
    if (n.custom) cls.push("custom");
    else if (n.selected) cls.push("alt");
  }
  if (n.source === "instr") cls.push("instr");
  const title = n.rest ? `${fmtTime(n.start)} – removed; click to restore`
    : !n.in_range ? `${fmtTime(n.start)} – ${n.name} is out of the harmonica's range`
    : `${fmtTime(n.start)} – ${n.name}${n.source === "instr" ? " (from the instruments)" : ""} – click to edit`;
  const chip = el("button", { class: cls.join(" "), dataset: { nid: id }, title, "aria-haspopup": "menu" },
    el("span", { class: "nm" }, n.rest ? "×" : n.display),
    el("span", { class: "tb" }, n.rest ? "" : n.tab || "—"));
  chip.addEventListener("click", (e) => { e.stopPropagation(); openNoteMenu(id, chip); });
  if (!H.chipNodes.has(id)) H.chipNodes.set(id, []);
  H.chipNodes.get(id).push(chip);
  return chip;
}

function renderHarmonica() {
  const song = S.song;
  const hv = song.harmonica;
  H.notes = new Map();
  H.timeline = [];
  H.words = [];
  H.chipNodes = new Map();
  H.activeNote = H.activeWord = null;
  const sheet = $("#harp-sheet");
  $("#melody-btn").textContent = hv ? "Re-detect melody" : "Find melody";
  $("#melody-btn").title = hv ? "Find the melody again (your note edits on this tab are replaced)" : "";
  $("#octave-group").hidden = S.tab !== "harmonica" || !hv;
  $("#harp-group").hidden = S.tab !== "harmonica" || !hv;
  $("#add-note-btn").disabled = !hv;
  $("#harp-suggest").hidden = true;
  if (!hv) {
    $("#harp-meta").replaceChildren();
    sheet.replaceChildren(el("p", { class: "muted" },
      "This song was analysed before the harmonica tab existed. Press ",
      el("b", {}, "Find melody"), " to find the notes (a few seconds per minute of audio)."));
    return;
  }

  renderInstrument(hv);
  H.notes = new Map(hv.notes.map((n) => [n.id, n]));
  H.timeline = hv.notes.filter((n) => !n.rest).sort((a, b) => a.start - b.start);
  const instr = H.timeline.filter((n) => n.source === "instr").length;
  const meta = [el("span", { class: "badge" }, `${H.timeline.length} notes`)];
  meta.push(el("span", { class: "badge" + (hv.in_range < 1 ? " err" : ""),
    title: `The harmonica plays ${hv.range[0]} to ${hv.range[1]}` },
  hv.in_range < 1 ? `${Math.round(hv.in_range * 100)}% playable` : "all playable"));
  if (instr) {
    meta.push(el("span", { class: "badge", title: "Where nobody sings, the notes are a rough guess from the instruments (shown faded)." },
      `${instr} from instruments`));
  }
  $("#harp-meta").replaceChildren(...meta);

  const sign = (k) => (k > 0 ? `+${k}` : `${k}`);
  $("#oct-val").textContent = sign(hv.octave);
  $("#oct-val").title = hv.user_octave ? `automatic would be ${sign(hv.auto_octave)}` : "automatic";
  $("#oct-auto").hidden = !hv.user_octave;

  sheet.dir = $("#sheet").dir;
  const rows = hv.rows.map((row) => {
    if (row.type === "instrumental") {
      const items = [el("span", { class: "tag" }, "instr.")];
      if (row.bars) {
        items.push(el("span", { class: "bars" }, row.bars.map((bar) => el("span", { class: "barcell", title: `Bar ${bar.bar + 1}` },
          bar.chords.length ? bar.chords.map(makeNoteChip) : el("span", { class: "repeat" }, "–")))));
      } else {
        items.push(...row.chords.map(makeNoteChip));
      }
      return el("div", { class: "line instrumental" }, items);
    }
    const cells = row.words.map((w) => {
      const src = song.lines[row.line].words[w.i];
      const { node, timing } = lyricCell(w, src, makeNoteChip, null);
      H.words.push({ start: w.start, end: w.end, node: timing });
      return node;
    });
    return el("div", { class: "line" + (row.gap_before ? " stanza" : "") }, cells);
  });
  if (!rows.length) rows.push(el("p", { class: "muted" }, "No melody found."));
  sheet.replaceChildren(...rows);
  sheet.classList.toggle("show-bars", $("#show-bars").checked);
  H.words.sort((a, b) => a.start - b.start);
}

// ---- playback follow -----------------------------------------------------------

function resetHarpFollow() {
  H.activeNote = H.activeWord = null;
}

function harpTick(t) {
  const i = findAt(H.timeline, t);
  const cur = i >= 0 && t < H.timeline[i].end + 0.05 ? H.timeline[i] : null;
  if (cur?.id !== H.activeNote) {
    (H.chipNodes.get(H.activeNote) || []).forEach((n) => n.classList.remove("active"));
    (H.chipNodes.get(cur?.id) || []).forEach((n) => n.classList.add("active"));
    H.activeNote = cur?.id ?? null;
    $("#now-chord").textContent = cur ? noteLabel(cur) : "–";
  }
  const next = H.timeline[i + 1];
  $("#next-chord").textContent = next ? `${noteLabel(next)} in ${Math.max(0, next.start - t).toFixed(1)}s` : "–";
  followWord(H.words, t, H);
}

// ---- note menu -----------------------------------------------------------------

async function saveNote(id, body) {
  try {
    S.song = await api(`/api/songs/${S.song.id}/notes/${id}`, jsonBody("PATCH", body));
    renderSong();
    tick();
  } catch (e) {
    alertInline($("#harp-panel"), e.message);
  }
}

async function editNotes(body) {
  try {
    S.song = await api(`/api/songs/${S.song.id}/notes`, jsonBody("POST", body));
    renderSong();
    tick();
  } catch (e) {
    alertInline($("#harp-panel"), e.message);
  }
}

// add a note the tracker missed: starts at the playhead, lasts until the next note (max 1 s)
function openAddNote(anchor) {
  const key = "add-note";
  if (S.menuFor === key && !$("#chord-menu").hidden) { closeChordMenu(); return; }
  closeChordMenu();
  const start = Math.round(audio.currentTime * 100) / 100;
  const later = [...H.notes.values()].filter((n) => n.start > start + 0.05).sort((a, b) => a.start - b.start)[0];
  const end = Math.round(Math.min(start + 1, later ? later.start - 0.02 : start + 1) * 100) / 100;
  const input = el("input", { type: "text", placeholder: "Note, e.g. C5, F# or Bb" });
  const apply = () => {
    const note = input.value.trim();
    if (!note) return;
    closeChordMenu();
    editNotes({ action: "add", start, end, note });
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") apply(); if (e.key === "Escape") closeChordMenu(); });
  showMenu([
    el("div", { class: "head" }, el("span", {}, `New note at ${fmtTime(start)}`), el("span", {}, `${(end - start).toFixed(2)} s`)),
    el("div", { class: "custom-row" }, input, el("button", { onclick: apply }, "Add")),
    el("div", { class: "hint" }, "Move the playhead to where the note starts first. No octave = the nearest one."),
  ], anchor, key);
  input.focus();
}

function auditionNote(n, sungMidi) {
  const dur = Math.min(Math.max(n.end - n.start, 0.6), 10);
  playSection(n.start, dur, sungMidi != null && (() => synth.playNote(sungMidi, n.end - n.start)));
}

function openNoteMenu(id, anchor) {
  const key = `note:${id}`;
  if (S.menuFor === key && !$("#chord-menu").hidden) { closeChordMenu(); return; }
  closeChordMenu();
  const n = H.notes.get(id);
  const rows = [el("div", { class: "head" },
    el("span", {}, `${fmtTime(n.start)} – ${fmtTime(n.end)}`),
    el("span", {}, n.source === "instr" ? "from the instruments (rough)" : "sung"))];

  n.candidates.forEach((c, i) => {
    const isSel = !n.custom && (n.selected || 0) === i;
    rows.push(el("div", { class: "cand" + (isSel ? " sel" : ""), role: "menuitemradio", "aria-checked": String(isSel) },
      el("button", { class: "pick", onclick: () => { closeChordMenu(); saveNote(id, { selected: i }); } },
        `${isSel ? "✓ " : ""}${c.name}`, el("span", { class: "tab" + (c.in_range ? "" : " oor") }, c.tab || "out of range")),
      el("div", { class: "conf", title: `${Math.round(c.conf * 100)}% relative confidence` },
        el("div", { style: `width:${Math.round(c.conf * 100)}%` })),
      el("button", { class: "play", title: "Play this part of the song with this note on top",
        onclick: () => auditionNote(n, c.midi) }, "▶ hear")));
  });

  if (!n.rest && n.fingerings.length > 1) {
    rows.push(el("div", { class: "fingering-row" },
      el("span", { class: "lbl" }, "Play as"),
      n.fingerings.map((f, i) => el("button", {
        class: i === n.fingering_index ? "on" : "",
        title: `Hole ${f.hole}, ${f.breath}${f.bend ? `, bend ${f.bend} semitone${f.bend > 1 ? "s" : ""}` : ""}${f.button ? ", button in" : ""}`,
        "aria-pressed": String(i === n.fingering_index),
        onclick: () => { closeChordMenu(); saveNote(id, { fingering: i }); },
      }, f.tab)),
      n.fingering != null
        ? el("button", { class: "link", onclick: () => { closeChordMenu(); saveNote(id, { auto_fingering: true }); } }, "auto")
        : null));
  }
  if (!n.rest && !n.in_range) {
    rows.push(el("div", { class: "hint warn" },
      `${n.name} is outside the harmonica (${S.song.harmonica.range.join("–")}). Try Octave − / + or type another note.`));
  }

  const customInput = el("input", { type: "text", placeholder: "Other note, e.g. F#5 or Bb",
    value: n.custom && !n.rest ? n.name.replace("♯", "#").replace("♭", "b") : "" });
  const applyCustom = () => { closeChordMenu(); saveNote(id, { custom: customInput.value.trim() }); };
  customInput.addEventListener("keydown", (e) => { if (e.key === "Enter") applyCustom(); if (e.key === "Escape") closeChordMenu(); });
  rows.push(el("div", { class: "custom-row" }, customInput, el("button", { onclick: applyCustom }, "Set")));
  rows.push(el("div", { class: "hint" }, "No octave = the nearest one. Empty + Set reverts to the detected notes."));
  rows.push(el("div", { class: "foot" },
    el("button", { onclick: () => auditionNote(n, null) }, "▶ Section only"),
    el("button", { onclick: () => auditionNote(n, n.sung) }, "▶ With current"),
    el("button", { onclick: stopAudition }, "■ Stop")));
  const sorted = [...H.notes.values()].sort((a, b) => a.start - b.start);
  const next = sorted[sorted.findIndex((x) => x.id === id) + 1];
  const t = audio.currentTime;
  const splitAt = t > n.start + 0.05 && t < n.end - 0.05 ? t : null;
  rows.push(el("div", { class: "foot" },
    el("button", {
      title: splitAt ? "Cut this note in two at the playhead" : "Cut this note in two in the middle (or move the playhead inside it first)",
      disabled: n.end - n.start < 0.12,
      onclick: () => { closeChordMenu(); editNotes({ action: "split", id, at: splitAt }); },
    }, splitAt ? "✂ Split at playhead" : "✂ Split in half"),
    el("button", {
      title: "Join this note and the next into one",
      disabled: !next,
      onclick: () => { closeChordMenu(); editNotes({ action: "merge", ids: [id, next.id] }); },
    }, "⇥ Merge with next")));
  rows.push(el("div", { class: "foot" },
    n.rest
      ? el("button", { onclick: () => { closeChordMenu(); saveNote(id, { custom: "" }); } }, "↺ Restore note")
      : el("button", { onclick: () => { closeChordMenu(); saveNote(id, { custom: "rest" }); } }, "✕ Not a note")));

  showMenu(rows, anchor, key);
}

// ---- melody job + octave -----------------------------------------------------------

async function findMelody() {
  const btn = $("#melody-btn");
  const box = $("#melody-progress");
  const id = S.song.id;
  btn.disabled = true;
  try {
    await api(`/api/songs/${id}/melody`, jsonBody("POST", { fast: $("#fast-notes").checked }));
    await harpJob.pollStatus(id, box, {
      onDone: async () => {
        box.replaceChildren();
        if (S.song?.id !== id) return;
        S.song = await api(`/api/songs/${id}`);
        renderSong();
        tick();
      },
      onRetry: findMelody,
    });
  } catch (e) {
    box.replaceChildren(el("div", { class: "msg error" }, e.message));
  } finally {
    btn.disabled = false;
  }
}

function initHarmonica() {
  $("#melody-btn").addEventListener("click", () => {
    const btn = $("#melody-btn");
    if (S.song.harmonica && btn.dataset.armed !== "1") {
      // re-detecting replaces note edits: ask for a second click
      btn.dataset.armed = "1";
      btn.textContent = "Replace my note edits?";
      setTimeout(() => { btn.dataset.armed = ""; if (S.song?.harmonica) btn.textContent = "Re-detect melody"; }, 3000);
      return;
    }
    btn.dataset.armed = "";
    findMelody();
  });
  const user = () => S.song.harmonica?.user_octave || 0;
  $("#oct-down").addEventListener("click", () => saveSettings({ harmonica_octave: Math.max(-3, user() - 1) }));
  $("#oct-up").addEventListener("click", () => saveSettings({ harmonica_octave: Math.min(3, user() + 1) }));
  $("#oct-auto").addEventListener("click", () => saveSettings({ harmonica_octave: 0 }));
  $("#harp-type").addEventListener("change", (e) => saveSettings({ harmonica_type: e.target.value }));
  $("#harp-key").addEventListener("change", (e) => saveSettings({ harmonica_key: e.target.value }));
  $("#add-note-btn").addEventListener("click", (e) => openAddNote(e.currentTarget));
}

initHarmonica();
