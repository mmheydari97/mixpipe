"use strict";
// Guitar tab: the recorded guitar note by note, drawn as staff notation (treble
// clef, an octave above the sound as guitar music is written) and/or tablature
// with VexFlow (static/vendor/vexflow.js). The server does the music
// (chords/guitar.py): which notes are the lead, strings and frets, the rhythm per
// bar; this page draws it, follows playback and edits notes.
// Load after app.js, note-menu.js and playalong.js; uses S, audio, synth, showMenu,
// closeChordMenu, playSection, stopAudition, fmtTime, alertInline, PA functions.

const G = {
  view: null,
  part: store.get("mixpipe.guitarPart") || "lead",       // lead | rhythm | both
  mode: store.get("mixpipe.guitarMode") || "both",       // staff | tab | both
  all: store.get("mixpipe.guitarAll") === "1",           // whole song, not just the solos
  open: new Set(),        // indices of sung stretches opened by hand
  notes: new Map(),       // id -> note (from the view)
  timeline: [],           // [{t, els}] sorted, for following playback
  active: null,
  loading: false,
  songId: null,
};
const soloJob = createJobUI(["separate", "transcribe"], { separate: "Split stems", transcribe: "Listen to the guitar" });
const VF = Vex.Flow;
const STRING_NAMES = ["", "1st", "2nd", "3rd", "4th", "5th", "6th"];

// ---- loading ---------------------------------------------------------------------

async function openSolo() {
  const id = S.song?.id;
  if (!id) return;
  $("#export-tab").href = apiUrl(`/api/songs/${id}/export?fmt=tab&part=${G.part}`);
  if (!S.song.has_guitar) {
    G.view = null;
    renderSolo();
    const st = await api(`/api/songs/${id}/guitar/status`).catch(() => null);
    if (st && (st.state === "queued" || st.state === "running")) followSoloJob(id);
    return;
  }
  if (G.songId === id && G.view && G.view.part === G.part) { renderSolo(); return; }
  await loadSolo();
}

async function loadSolo() {
  const id = S.song.id;
  G.loading = true;
  $("#solo-progress").replaceChildren(el("div", { class: "msg" }, "Writing out the guitar…"));
  try {
    const v = await api(`/api/songs/${id}/guitar?part=${G.part}`);
    if (S.song?.id !== id) return;
    G.view = v;
    G.songId = id;
  } catch (e) {
    $("#solo-progress").replaceChildren(el("div", { class: "msg error" }, e.message));
    return;
  } finally {
    G.loading = false;
  }
  $("#solo-progress").replaceChildren();
  renderSolo();
}

async function findGuitar(redo = false) {
  const id = S.song.id;
  $("#solo-btn").disabled = true;
  try {
    await api(`/api/songs/${id}/guitar`, jsonBody("POST", { redo }));
  } catch (e) {
    $("#solo-progress").replaceChildren(el("div", { class: "msg error" }, e.message));
    $("#solo-btn").disabled = false;
    return;
  }
  followSoloJob(id);
}

async function followSoloJob(id) {
  const box = $("#solo-progress");
  await soloJob.pollStatus(id, box, {
    status: `/api/songs/${id}/guitar/status`, cancel: false,
    onDone: async () => {
      box.replaceChildren();
      if (S.song?.id !== id) return;
      S.song.has_guitar = true;
      G.songId = null;
      await loadSolo();
    },
    onRetry: () => findGuitar(false),
  });
  $("#solo-btn").disabled = false;
}

// every edit returns the whole view for the part shown
async function soloEdit(path, opts) {
  const id = S.song.id;
  try {
    const v = await api(`${path}${path.includes("?") ? "&" : "?"}part=${G.part}`, opts);
    if (S.song?.id !== id) return;
    G.view = v;
    renderSolo();
  } catch (e) {
    alertInline($("#solo-panel"), e.message);
  }
}
const saveGuitarNote = (id, body) => soloEdit(`/api/songs/${S.song.id}/guitar/notes/${id}`, jsonBody("PATCH", body));
const editGuitar = (body) => soloEdit(`/api/songs/${S.song.id}/guitar/notes`, jsonBody("POST", body));

// ---- the head of the panel ----------------------------------------------------------

function renderSoloHead() {
  const v = G.view;
  const has = !!v;
  $("#solo-btn").textContent = has ? "Listen again" : "Write out the guitar";
  $("#solo-btn").title = has
    ? "Transcribe the guitar stem again (replaces your note edits)"
    : "Split the song into stems if that isn't done yet (about half a minute), then write out what the guitar plays";
  for (const b of document.querySelectorAll("[data-gmode]")) b.setAttribute("aria-pressed", String(b.dataset.gmode === G.mode));
  $("#solo-part").value = G.part;
  $("#solo-all").checked = G.all;
  for (const id of ["solo-add", "solo-part", "solo-all"]) $(`#${id}`).disabled = !has;
  const meta = [];
  if (has) {
    meta.push(`${v.time_signature} · ${v.key || ""} · tuning ${v.tuning.join(" ")}${v.capo ? ` · capo ${v.capo}` : ""}`);
    meta.push(`${v.notes.length} notes${v.hidden ? ` (${v.hidden} in the other part)` : ""}`);
  }
  $("#solo-meta").textContent = meta.join(" · ");
  // where the solos are
  const nav = $("#solo-nav");
  nav.hidden = !has || !v.solos.length;
  if (has && v.solos.length) {
    nav.replaceChildren(el("span", { class: "lbl" }, "Without singing:"),
      ...v.solos.map((s, i) => el("button", {
        title: "Jump there", onclick: () => jumpTo(s.start),
      }, el("b", {}, i === 0 && s.start < 1 ? "Intro" : s.end >= (S.song.duration || 0) - 1 ? "Outro" : `Solo ${i}`),
      el("small", {}, `${fmtTime(s.start)}–${fmtTime(s.end)}`))));
  }
  // two guitars: which one is followed
  const lock = $("#solo-lock");
  lock.hidden = !has || (!v.two_guitars && !v.lock);
  if (has && v.lock) {
    lock.replaceChildren(`Following the guitar you marked as ${v.lock.as === "rhythm" ? "rhythm" : "the solo"}`
      + ` (${v.lock.centre > 0.75 ? "centred" : "to the sides"}, around ${noteName(Math.round(v.lock.pitch * 24))}). `,
    el("button", { class: "link", onclick: () => editGuitar({ action: "unlock" }) }, "forget it"));
  } else if (has && v.two_guitars) {
    lock.replaceChildren("Two guitars play at once in places; the lead is guessed from where it sits in the stereo "
      + "picture and how high it plays. If it picks the wrong one, click a note of the solo and choose ",
    el("b", {}, "This is the solo"), ".");
  }
}

const NAMES = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
const noteName = (m) => `${NAMES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`;

function jumpTo(t) {
  const seg = [...document.querySelectorAll("#solo-sheet .seg")].find((s) => Number(s.dataset.start) <= t + 0.5
    && Number(s.dataset.end) > t);
  seg?.scrollIntoView({ behavior: "smooth", block: "start" });
  seekPlay(t);
}

// ---- drawing ---------------------------------------------------------------------------

function renderSolo() {
  renderSoloHead();
  const sheet = $("#solo-sheet");
  const v = G.view;
  G.notes = new Map((v?.notes || []).map((n) => [n.id, n]));
  G.timeline = [];
  G.active = null;
  if (!v) {
    sheet.replaceChildren(el("p", { class: "muted" },
      "Write out what the guitar plays, note by note: as notes on a staff, as tab, or both. "
      + "The song is split into stems first (once; the Vocals and Guitar sliders use the same split). "
      + "Strummed chords show as chords; picked notes, double stops and arpeggios are written out. "
      + "Where two guitars play, the lead is shown and the rhythm part is a click away."));
    return;
  }
  if (!v.measures.length) {
    sheet.replaceChildren(el("p", { class: "muted" }, v.notes.length
      ? "This song has no beat grid (free time), so the guitar can't be written in bars. The text tab still works."
      : "No guitar notes found here."));
    return;
  }
  const solos = v.solos;
  const inSolo = (t) => solos.some((s) => t >= s.start - 0.3 && t < s.end - 0.05);
  const segs = [];
  for (const bar of v.measures) {
    const kind = inSolo(bar.start) ? "solo" : "sung";
    if (!segs.length || segs[segs.length - 1].kind !== kind) segs.push({ kind, bars: [] });
    segs[segs.length - 1].bars.push(bar);
  }
  const width = Math.max(320, sheet.clientWidth || 900);
  const nodes = [];
  segs.forEach((seg, i) => {
    const start = seg.bars[0].start;
    const last = seg.bars[seg.bars.length - 1];
    const end = last.start + (seg.bars.length > 1 ? last.start - seg.bars[seg.bars.length - 2].start : 2);
    const box = el("div", { class: `seg ${seg.kind}`, dataset: { start, end } });
    const notesIn = seg.bars.reduce((k, b) => k + b.items.filter((it) => !it.rest).length, 0);
    const open = G.all || seg.kind === "solo" || G.open.has(i);
    box.append(el("div", { class: "seg-head" },
      el("button", { class: "link", onclick: () => seekPlay(start) }, `${fmtTime(start)}–${fmtTime(end)}`),
      el("span", {}, seg.kind === "solo" ? " no singing" : " sung", ` · ${seg.bars.length} bars · ${notesIn} notes`),
      seg.kind === "sung" && !G.all ? el("button", { class: "link", onclick: () => {
        if (G.open.has(i)) G.open.delete(i); else G.open.add(i);
        renderSolo();
      } }, open ? "hide" : "show") : null));
    if (open) drawBars(box, seg.bars, width);
    nodes.push(box);
  });
  sheet.replaceChildren(...nodes);
  // the drawing is in the page now: link notes to their SVG elements
  for (const fn of G.pendingLinks || []) fn();
  G.pendingLinks = [];
  G.timeline.sort((a, b) => a.t - b.t);
  tick();
}

function barWidth(bar, first) {
  return Math.max(120, 40 + 30 * bar.items.length) + (first ? 70 : 0);
}

function drawBars(box, bars, width) {
  // rows of bars that fit the width; one SVG per row
  const rows = [];
  let row = [], used = 0;
  for (const bar of bars) {
    const w = barWidth(bar, !row.length);
    if (row.length && used + w > width - 10) { rows.push(row); row = []; used = 0; }
    row.push(bar);
    used += barWidth(bar, row.length === 1);
  }
  if (row.length) rows.push(row);
  const staff = G.mode !== "tab", tab = G.mode !== "staff";
  for (const r of rows) {
    const div = el("div", { class: "solo-row" });
    box.append(div);
    const natural = r.reduce((k, b, i) => k + barWidth(b, i === 0), 0);
    const stretch = Math.min(1.6, (width - 10) / natural);
    const height = (staff ? 120 : 0) + (tab ? 110 : 0) + 20;
    const renderer = new VF.Renderer(div, VF.Renderer.Backends.SVG);
    renderer.resize(width, height);
    const ctx = renderer.getContext();
    // the page's own colours (light or dark theme); the white boxes VexFlow puts
    // behind fret numbers take the page background
    const cs = getComputedStyle(document.body);
    ctx.setFillStyle(cs.color);
    ctx.setStrokeStyle(cs.color);
    ctx.setBackgroundFillStyle?.(cs.backgroundColor);
    let x = 5;
    r.forEach((bar, i) => {
      const w = barWidth(bar, i === 0) * stretch;
      drawBar(ctx, bar, x, w, i === 0, staff, tab);
      x += w;
    });
  }
}

function vfDuration(it) {
  return it.dur + (it.dots ? "d" : "") + (it.rest ? "r" : "");
}

function drawBar(ctx, bar, x, w, first, staff, tab) {
  const v = G.view;
  const [beats, value] = v.time_signature.split("/").map(Number);
  const staves = [];
  let sStave = null, tStave = null;
  if (staff) {
    sStave = new VF.Stave(x, 10, w);
    if (first) {
      sStave.addClef("treble", "default", "8vb");
      if (v.key) { try { sStave.addKeySignature(v.key); } catch { /* unknown key name */ } }
    }
    if (first && bar === v.measures[0]) sStave.addTimeSignature(v.time_signature);
    sStave.setContext(ctx).draw();
    staves.push(sStave);
  }
  if (tab) {
    tStave = new VF.TabStave(x, staff ? 130 : 10, w);
    if (first) tStave.addTabGlyph();
    tStave.setContext(ctx).draw();
    staves.push(tStave);
  }
  const sNotes = [], tNotes = [];
  const chordAt = new Map();
  for (const c of bar.chords) {
    let best = null;
    for (const it of bar.items) if (!best || Math.abs(it.t - c.t) < Math.abs(best.t - c.t)) best = it;
    if (best) chordAt.set(best, c.label);
  }
  for (const it of bar.items) {
    const dur = vfDuration(it);
    if (staff) {
      const sn = new VF.StaveNote({ clef: "treble", keys: it.rest ? ["b/4"] : it.keys, duration: dur, auto_stem: true });
      if (it.dots) VF.Dot.buildAndAttach([sn], { all: true });
      const label = it.strum ? `${it.strum} ↓` : chordAt.get(it);
      if (label) sn.addModifier(new VF.Annotation(label).setFont("sans-serif", 11, it.strum ? "bold" : "normal")
        .setVerticalJustification(VF.Annotation.VerticalJustify.TOP), 0);
      sNotes.push(sn);
    }
    if (tab) {
      let tn;
      const pos = it.rest ? [] : it.notes.filter((n) => n.string).map((n) => ({ str: n.string, fret: n.fret }));
      if (!pos.length) {
        tn = new VF.GhostNote(dur.replace("r", ""));
      } else {
        tn = new VF.TabNote({ positions: pos, duration: dur.replace("r", "") });
        it.notes.forEach((n, k) => {
          if (n.tech.includes("full") || n.tech.includes("½")) {
            try { tn.addModifier(new VF.Bend(n.tech.includes("full") ? "Full" : "½"), k); } catch { /* old api */ }
          }
          if (n.tech.includes("~")) { try { tn.addModifier(new VF.Vibrato(), k); } catch { /* old api */ } }
        });
        if (!staff && (it.strum || chordAt.get(it))) {
          tn.addModifier(new VF.Annotation(it.strum ? `${it.strum} ↓` : chordAt.get(it)).setFont("sans-serif", 11)
            .setVerticalJustification(VF.Annotation.VerticalJustify.TOP), 0);
        }
      }
      if (it.dots) { try { VF.Dot.buildAndAttach([tn], { all: true }); } catch { /* ghost notes have no dots */ } }
      tNotes.push(tn);
    }
  }
  // triplets: the pieces of one beat that are in triplet time
  const tuplets = [];
  let group = [];
  bar.items.forEach((it, i) => {
    if (it.triplet && group.length && bar.items[group[0]].beat === it.beat) group.push(i);
    else {
      if (group.length) tuplets.push(group);
      group = it.triplet ? [i] : [];
    }
  });
  if (group.length) tuplets.push(group);
  const sTuplets = [];
  for (const g of tuplets) {
    if (staff) sTuplets.push(new VF.Tuplet(g.map((i) => sNotes[i]), { num_notes: 3, notes_occupied: 2, ratioed: false }));
    if (tab) new VF.Tuplet(g.map((i) => tNotes[i]), { num_notes: 3, notes_occupied: 2 });  // timing only, not drawn
  }
  const voices = [];
  const voice = (notes) => new VF.Voice({ num_beats: beats, beat_value: value }).setMode(VF.Voice.Mode.SOFT).addTickables(notes);
  let sv = null, tv = null;
  if (staff) { sv = voice(sNotes); voices.push(sv); }
  if (tab) { tv = voice(tNotes); voices.push(tv); }
  if (staff && v.key) { try { VF.Accidental.applyAccidentals([sv], v.key); } catch { /* unknown key */ } }
  const fmt = new VF.Formatter();
  voices.forEach((vo) => fmt.joinVoices([vo]));
  const ref = sStave || tStave;
  fmt.format(voices, Math.max(40, ref.getNoteEndX() - ref.getNoteStartX() - 12));
  if (sv) sv.draw(ctx, sStave);
  if (tv) tv.draw(ctx, tStave);
  sTuplets.forEach((t) => t.setContext(ctx).draw());
  // ties: within the bar here, across bar lines from the last item of the previous bar
  bar.items.forEach((it, i) => {
    if (!it.tie || it.rest || !staff) return;
    const next = sNotes[i + 1];
    const idx = it.keys.map((_, k) => k);
    new VF.StaveTie({ first_note: sNotes[i], last_note: next || null, first_indices: idx, last_indices: idx })
      .setContext(ctx).draw();
  });
  // link notes and items to the drawing once it's in the page
  G.pendingLinks = G.pendingLinks || [];
  G.pendingLinks.push(() => {
    bar.items.forEach((it, i) => {
      if (it.rest) return;
      const els = [sNotes[i]?.getSVGElement?.(), tNotes[i]?.getSVGElement?.()].filter(Boolean);
      for (const e of els) {
        e.classList.add("gnote");
        e.addEventListener("click", (ev) => { ev.stopPropagation(); openItemMenu(it, e); });
      }
      G.timeline.push({ t: it.t, els });
    });
  });
}

// ---- following playback -------------------------------------------------------------

function soloTick(t) {
  if (!G.timeline.length) return;
  let lo = 0, hi = G.timeline.length - 1, k = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (G.timeline[mid].t <= t + 0.03) { k = mid; lo = mid + 1; } else hi = mid - 1;
  }
  const cur = k >= 0 ? G.timeline[k] : null;
  if (cur === G.active) return;
  G.active?.els.forEach((e) => e.classList.remove("playing"));
  G.active = cur;
  if (!cur) return;
  cur.els.forEach((e) => e.classList.add("playing"));
  if ($("#follow").checked && !audio.paused && cur.els[0]) {
    const r = cur.els[0].getBoundingClientRect();
    if (r.top < 80 || r.bottom > innerHeight - 140) cur.els[0].scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

// ---- note menu -----------------------------------------------------------------------

function openItemMenu(item, anchor) {
  const ids = item.notes.map((n) => n.id).filter((id) => G.notes.has(id));
  if (ids.length === 1) { openGuitarNoteMenu(ids[0], anchor); return; }
  const key = `gitem:${ids.join(",")}`;
  if (S.menuFor === key && !$("#chord-menu").hidden) { closeChordMenu(); return; }
  closeChordMenu();
  showMenu([
    el("div", { class: "head" }, el("span", {}, `${ids.length} notes together`), el("span", {}, fmtTime(item.t))),
    ...ids.map((id) => {
      const n = G.notes.get(id);
      return el("div", { class: "cand" }, el("button", { class: "pick", onclick: () => openGuitarNoteMenu(id, anchor) },
        n.label, el("span", { class: "tab" }, n.string ? `${STRING_NAMES[n.string]} string, fret ${n.fret}` : "out of reach")));
    }),
    el("div", { class: "foot" }, el("button", { onclick: () => auditionGuitar(item.t, 1.5, item.notes.map((n) => n.midi)) }, "▶ hear")),
  ], anchor, key);
}

function auditionGuitar(start, dur, midis) {
  playSection(start, Math.min(Math.max(dur, 0.8), 8), midis && (() => synth.pluck(midis, Math.min(dur, 2))));
}

function openGuitarNoteMenu(id, anchor) {
  const key = `gnote:${id}`;
  if (S.menuFor === key && !$("#chord-menu").hidden) { closeChordMenu(); return; }
  closeChordMenu();
  const n = G.notes.get(id);
  const rest = n.custom === "rest";
  const rows = [];
  if (n.positions.length > 1) {
    rows.push({ label: "Play on", auto: n.fingering != null ? () => saveGuitarNote(id, { auto_fingering: true }) : null,
      buttons: n.positions.map((p, i) => ({
        text: `${p.string}·${p.fret}`, on: p.string === n.string && p.fret === n.fret,
        title: `${STRING_NAMES[p.string]} string, fret ${p.fret}`,
        onclick: () => saveGuitarNote(id, { fingering: i }) })) });
  }
  rows.push({ label: "Part", title: "Which guitar plays this note",
    auto: n.pinned_part ? () => saveGuitarNote(id, { auto_part: true }) : null,
    buttons: [
      { text: "Lead", on: n.part === "lead", title: "Show this note with the lead guitar", onclick: () => saveGuitarNote(id, { part: "lead" }) },
      { text: "Rhythm", on: n.part === "rhythm", title: "Show this note with the rhythm guitar", onclick: () => saveGuitarNote(id, { part: "rhythm" }) }] });
  rows.push({ label: "Guitar", title: "Where two guitars play at once, follow the one this note belongs to",
    buttons: [
      { text: "★ This is the solo", title: "Follow this guitar (its place in the stereo picture and its range) as the lead everywhere",
        onclick: () => editGuitar({ action: "lock", id, part: "lead" }) },
      { text: "This is the rhythm", title: "This guitar plays the rhythm part; follow the other one",
        onclick: () => editGuitar({ action: "lock", id, part: "rhythm" }) }] });
  const shown = [...G.notes.values()].sort((a, b) => a.start - b.start || a.midi - b.midi);
  const next = shown[shown.findIndex((x) => x.id === id) + 1];
  const splitAt = splitPoint(n);
  const hints = [];
  if (n.tech.length) hints.push({ text: `Heard as ${n.tech.map((t) => (t === "~" ? "vibrato" : `a ${t === "full" ? "whole-step" : "half-step"} bend`)).join(" with ")}.` });
  if (!n.string) hints.push({ warn: true, text: `${n.label} is out of reach on this guitar${S.song.capo ? " with the capo" : ""}.` });
  showMenu(noteMenuRows({
    head: [`${fmtTime(n.start)} – ${fmtTime(n.end)}`, n.string ? `${STRING_NAMES[n.string]} string, fret ${n.fret}` : ""],
    cands: n.candidates.map((c, i) => ({ name: c.label, conf: c.conf, selected: !n.custom && (n.selected || 0) === i })),
    onPick: (i) => saveGuitarNote(id, { selected: i }),
    onHear: (i) => auditionGuitar(n.start, n.end - n.start, [n.candidates[i].midi]),
    rows, hints,
    custom: { placeholder: "Other note, e.g. F# or A4", value: n.custom && !rest ? n.label.replace("♯", "#").replace("♭", "b") : "",
      onSet: (text) => saveGuitarNote(id, { custom: text }) },
    listen: [
      { text: "▶ Here", onclick: () => auditionGuitar(n.start, Math.max(1.5, n.end - n.start), null) },
      { text: "▶ With this note", onclick: () => auditionGuitar(n.start, n.end - n.start, [n.midi]) },
      { text: "■ Stop", onclick: stopAudition }],
    split: { at: splitAt, disabled: n.end - n.start < 0.12, onclick: () => editGuitar({ action: "split", id, at: splitAt }) },
    merge: { disabled: !next, onclick: () => editGuitar({ action: "merge", ids: [id, next.id] }) },
    rest,
    onRemove: () => editGuitar({ action: "delete", id }),
    onRestore: () => saveGuitarNote(id, { custom: "" }),
  }), anchor, key);
}

// ---- wiring ------------------------------------------------------------------------------

function initGuitar() {
  $("#solo-btn").addEventListener("click", () => {
    const btn = $("#solo-btn");
    if (G.view && btn.dataset.armed !== "1") {
      btn.dataset.armed = "1";
      btn.textContent = "Replace my note edits?";
      setTimeout(() => { btn.dataset.armed = ""; renderSoloHead(); }, 3000);
      return;
    }
    btn.dataset.armed = "";
    findGuitar(!!G.view);
  });
  for (const b of document.querySelectorAll("[data-gmode]")) {
    b.addEventListener("click", () => {
      G.mode = b.dataset.gmode;
      store.set("mixpipe.guitarMode", G.mode);
      renderSolo();
    });
  }
  $("#solo-part").addEventListener("change", (e) => {
    G.part = e.target.value;
    store.set("mixpipe.guitarPart", G.part);
    $("#export-tab").href = apiUrl(`/api/songs/${S.song.id}/export?fmt=tab&part=${G.part}`);
    loadSolo();
  });
  $("#solo-all").addEventListener("change", (e) => {
    G.all = e.target.checked;
    store.set("mixpipe.guitarAll", G.all ? "1" : "0");
    renderSolo();
  });
  $("#solo-add").addEventListener("click", (e) => openAddNoteMenu(e.currentTarget, [...G.notes.values()],
    "Note, e.g. A4 or F#", (body) => editGuitar({ action: "add", ...body, part: G.part === "both" ? null : G.part })));
  $("#solo-hear").addEventListener("click", (e) => {
    const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
    e.currentTarget.setAttribute("aria-pressed", String(on));
    paGuitarOnly(on);
  });
  let resize = 0;
  addEventListener("resize", () => {
    clearTimeout(resize);
    resize = setTimeout(() => { if (S.tab === "solo" && G.view) renderSolo(); }, 250);
  });
}

initGuitar();
