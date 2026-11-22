"use strict";
// Harmonica tab: the melody as tab for a 12-hole chromatic or a 10-hole diatonic
// harmonica in any key.
// Load after app.js; uses its S, audio, synth (synth.js), and sheet.js's fmtTime, findAt, followWord,
// showMenu, closeChordMenu, playSection, stopAudition, saveSettings, alertInline.

const H = {
  notes: new Map(),     // id -> note (from the server's harmonica view)
  chords: new Map(),    // id -> harmonica chord ("h<chord id>")
  timeline: [],         // notes that sound (main notes only, with mainOnly), sorted by start
  mode: store.get("mixpipe.harpMain") === "strong" ? "strong" : "beat",  // main note per beat / strong beat
  mainOnly: store.get("mixpipe.harpMainOnly") === "1",
  showChords: store.get("mixpipe.harpChords") !== "0",
  words: [],            // [{start, end, node}] in the harmonica sheet
  chipNodes: new Map(), // note id -> [nodes]
  activeNote: null,
  activeWord: null,
};

const harpJob = createJobUI(["decode", "melody"], { decode: "Decode", melody: "Melody" });

const noteLabel = (n) => (n.rest ? "rest" : `${n.display} ${n.tab || "—"}`);
// a main note at the chosen density (per beat, or per strong beat)
const isMain = (n) => !n.rest && n.level >= (H.mode === "strong" ? 2 : 1);
const holdEnd = (n) => n.hold?.[H.mode] ?? n.end;

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
    : [el("b", {}, "3↑"), " blow · ", el("b", {}, "3↓"), " draw · ", el("b", {}, "●"), " button in"]),
  " · ", el("b", {}, "1-3↑"), " holes 1–3 together");
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
  if (!n.rest) cls.push(isMain(n) ? "main" : "passing");
  const role = n.rest ? "" : isMain(n)
    ? ` – main note, hold until ${fmtTime(holdEnd(n))} to skip the passing notes`
    : " – passing note";
  const title = n.rest ? `${fmtTime(n.start)} – removed; click to restore`
    : !n.in_range ? `${fmtTime(n.start)} – ${n.name} is out of the harmonica's range`
    : `${fmtTime(n.start)} – ${n.name}${n.source === "instr" ? " (from the instruments)" : ""}${role} – click to edit`;
  const chip = el("button", { class: cls.join(" "), dataset: { nid: id }, title, "aria-haspopup": "menu" },
    el("span", { class: "nm" }, n.rest ? "×" : n.display),
    el("span", { class: "tb" }, n.rest ? "" : n.tab || "—"));
  chip.addEventListener("click", (e) => { e.stopPropagation(); openNoteMenu(id, chip); });
  if (!H.chipNodes.has(id)) H.chipNodes.set(id, []);
  H.chipNodes.get(id).push(chip);
  return chip;
}

// a harmonica chord: neighbouring holes played together
function makeHarpChordChip(id) {
  const c = H.chords.get(id);
  const cls = ["chip", "hchord"];
  if (!c.tab) cls.push("none");
  else if (!c.full) cls.push("partial");
  const where = c.tab && `${c.breath} holes ${c.holes[0]}–${c.holes[1]} together${c.button ? ", button in" : ""}`;
  const title = !c.tab ? `${fmtTime(c.start)} – ${c.label}: no chord on this harp fits, play the melody`
    : `${fmtTime(c.start)} – ${c.label}: ${where} (${c.notes.join(" ")})`
      + `${c.full ? "" : " – a partial chord, only some of its notes"}. Click to hear it.`;
  const chip = el("button", { class: cls.join(" "), dataset: { hid: id }, title },
    el("span", { class: "lb" }, c.label),
    el("span", { class: "tb" }, c.tab ? `${c.tab}${c.full ? "" : "~"}` : "—"));
  if (c.tab) {
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      const dur = Math.min(Math.max(c.end - c.start, 0.8), 6);
      playSection(c.start, dur, () => synth.playNote(c.midis, dur));
    });
  }
  return chip;
}

const makeHarpChip = (id) => (H.chords.has(id) ? makeHarpChordChip(id) : makeNoteChip(id));

// put a lyric cell's chord chips on their own line above its notes
function liftChords(node) {
  node.querySelectorAll(".chords").forEach((box) => {
    const chips = [...box.querySelectorAll(".hchord")];
    if (chips.length) box.before(el("span", { class: "hchords" }, chips));
  });
}

function renderHarmonica() {
  const song = S.song;
  const hv = song.harmonica;
  H.notes = new Map();
  H.chords = new Map();
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
  H.chords = new Map((hv.chords || []).map((c) => [c.id, c]));
  const sounding = hv.notes.filter((n) => !n.rest).sort((a, b) => a.start - b.start);
  H.timeline = H.mainOnly ? sounding.filter(isMain) : sounding;
  const instr = sounding.filter((n) => n.source === "instr").length;
  const mains = sounding.filter(isMain).length;
  const meta = [el("span", { class: "badge" }, `${sounding.length} notes`),
    el("span", { class: "badge", title: "Hold each main note until the next one for a simpler melody, or play the passing notes in between." },
      `${mains} main`)];
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
          bar.chords.length ? bar.chords.map(makeHarpChip) : el("span", { class: "repeat" }, "–")))));
      } else {
        items.push(...row.chords.map(makeHarpChip));
      }
      return el("div", { class: "line instrumental" }, items);
    }
    const cells = row.words.map((w) => {
      const src = song.lines[row.line].words[w.i];
      const { node, timing } = lyricCell(w, src, makeHarpChip, null, seekPlay);
      liftChords(node);
      H.words.push({ start: w.start, end: w.end, node: timing });
      return node;
    });
    return el("div", { class: "line" + (row.gap_before ? " stanza" : "") }, cells);
  });
  if (!rows.length) rows.push(el("p", { class: "muted" }, "No melody found."));
  sheet.replaceChildren(...rows);
  sheet.classList.toggle("show-bars", $("#show-bars").checked);
  sheet.classList.toggle("main-only", H.mainOnly);
  sheet.classList.toggle("no-hchords", !H.showChords);
  H.words.sort((a, b) => a.start - b.start);
}

// ---- playback follow -----------------------------------------------------------

function resetHarpFollow() {
  H.activeNote = H.activeWord = null;
}

function harpTick(t) {
  const i = findAt(H.timeline, t);
  // with main notes only, a main note stays lit while you hold it
  const end = (n) => (H.mainOnly ? holdEnd(n) : n.end);
  const cur = i >= 0 && t < end(H.timeline[i]) + 0.05 ? H.timeline[i] : null;
  if (cur?.id !== H.activeNote) {
    (H.chipNodes.get(H.activeNote) || []).forEach((n) => n.classList.remove("active"));
    (H.chipNodes.get(cur?.id) || []).forEach((n) => n.classList.add("active"));
    H.activeNote = cur?.id ?? null;
    $("#now-chord").textContent = cur ? `${H.mainOnly ? "hold " : ""}${noteLabel(cur)}` : "–";
  }
  const next = H.timeline[i + 1];
  $("#next-chord").textContent = next ? `${noteLabel(next)} in ${Math.max(0, next.start - t).toFixed(1)}s` : "–";
  followWord(H.words, t, H, $("#follow").checked && !audio.paused);
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

function openAddNote(anchor) {
  openAddNoteMenu(anchor, [...H.notes.values()], "Note, e.g. C5, F# or Bb", (body) => editNotes({ action: "add", ...body }));
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
  const rows = [];
  if (!n.rest && n.fingerings.length > 1) {
    rows.push({ label: "Play as", auto: n.fingering != null ? () => saveNote(id, { auto_fingering: true }) : null,
      buttons: n.fingerings.map((f, i) => ({
        text: f.tab, on: i === n.fingering_index,
        title: `Hole ${f.hole}, ${f.breath}${f.bend ? `, bend ${f.bend} semitone${f.bend > 1 ? "s" : ""}` : ""}${f.button ? ", button in" : ""}`,
        onclick: () => saveNote(id, { fingering: i }) })) });
  }
  if (!n.rest) {
    const main = isMain(n);
    rows.push({ label: "Role", title: "Hold a main note until the next one, or play the passing notes in between",
      auto: n.main_pinned ? () => saveNote(id, { auto_main: true }) : null,
      buttons: [
        { text: "★ Main note", on: main, title: "A main note on every beat and strong beat: hold it until the next main note",
          onclick: () => saveNote(id, { main: true }) },
        { text: "Passing note", on: !main, title: "A note between the main ones: add it for nuance, or skip it",
          onclick: () => saveNote(id, { main: false }) }] });
  }
  const sorted = [...H.notes.values()].sort((a, b) => a.start - b.start);
  const next = sorted[sorted.findIndex((x) => x.id === id) + 1];
  const splitAt = splitPoint(n);
  showMenu(noteMenuRows({
    head: [`${fmtTime(n.start)} – ${fmtTime(n.end)}`, n.source === "instr" ? "from the instruments (rough)" : "sung"],
    cands: n.candidates.map((c, i) => ({ name: c.name, extra: c.tab || "out of range", extraClass: c.in_range ? "" : "oor",
      conf: c.conf, selected: !n.custom && (n.selected || 0) === i })),
    onPick: (i) => saveNote(id, { selected: i }),
    onHear: (i) => auditionNote(n, n.candidates[i].midi),
    rows,
    hints: !n.rest && !n.in_range ? [{ warn: true,
      text: `${n.name} is outside the harmonica (${S.song.harmonica.range.join("–")}). Try Octave − / + or type another note.` }] : [],
    custom: { placeholder: "Other note, e.g. F#5 or Bb", value: n.custom && !n.rest ? n.name.replace("♯", "#").replace("♭", "b") : "",
      onSet: (text) => saveNote(id, { custom: text }) },
    listen: [
      { text: "▶ Section only", onclick: () => auditionNote(n, null) },
      { text: "▶ With current", onclick: () => auditionNote(n, n.sung) },
      { text: "■ Stop", onclick: stopAudition }],
    split: { at: splitAt, disabled: n.end - n.start < 0.12, onclick: () => editNotes({ action: "split", id, at: splitAt }) },
    merge: { disabled: !next, onclick: () => editNotes({ action: "merge", ids: [id, next.id] }) },
    rest: n.rest,
    onRemove: () => saveNote(id, { custom: "rest" }),
    onRestore: () => saveNote(id, { custom: "" }),
  }), anchor, key);
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
  // main notes, main-only and the chord line only change the view: no round trip
  $("#main-mode").value = H.mode;
  $("#main-only").checked = H.mainOnly;
  $("#harp-chords").checked = H.showChords;
  const rerender = () => { if (S.song?.harmonica) { renderHarmonica(); tick(); } };
  $("#main-mode").addEventListener("change", (e) => {
    H.mode = e.target.value;
    store.set("mixpipe.harpMain", H.mode);
    rerender();
  });
  $("#main-only").addEventListener("change", (e) => {
    H.mainOnly = e.target.checked;
    store.set("mixpipe.harpMainOnly", H.mainOnly ? "1" : "0");
    rerender();
  });
  $("#harp-chords").addEventListener("change", (e) => {
    H.showChords = e.target.checked;
    store.set("mixpipe.harpChords", H.showChords ? "1" : "0");
    $("#harp-sheet").classList.toggle("no-hchords", !H.showChords);
  });
}

initHarmonica();
