"use strict";
// The note menu shared by the harmonica tab and the guitar tab: the detected
// pitches with their confidence and a "hear" button, rows of choices (how to
// play it, main/passing, lead/rhythm), a box to type another note, listening,
// split / merge, and remove / restore. Each tab says what its rows hold and
// what the buttons do; showMenu / closeChordMenu come from app.js.
// Load after app.js.

// spec: {
//   head: [left, right],
//   cands: [{name, extra, extraClass, conf, selected}], onPick(i), onHear(i),
//   rows: [{label, title, buttons: [{text, title, on, onclick}], auto: fn | null}],
//   hints: [{text, warn}],
//   custom: {value, placeholder, onSet(text)}, customHint,
//   listen: [{text, onclick}],
//   split: {at, disabled, onclick}, merge: {disabled, onclick},
//   rest: bool, onRemove(), onRestore(),
//   extra: [nodes] (more foot buttons)
// }
function noteMenuRows(spec) {
  const close = (fn) => () => { closeChordMenu(); fn(); };
  const rows = [el("div", { class: "head" }, el("span", {}, spec.head[0]), el("span", {}, spec.head[1] || ""))];
  (spec.cands || []).forEach((c, i) => {
    rows.push(el("div", { class: "cand" + (c.selected ? " sel" : ""), role: "menuitemradio", "aria-checked": String(!!c.selected) },
      el("button", { class: "pick", onclick: close(() => spec.onPick(i)) },
        `${c.selected ? "✓ " : ""}${c.name}`,
        c.extra != null ? el("span", { class: "tab" + (c.extraClass ? ` ${c.extraClass}` : "") }, c.extra) : null),
      el("div", { class: "conf", title: `${Math.round(c.conf * 100)}% relative confidence` },
        el("div", { style: `width:${Math.round(c.conf * 100)}%` })),
      spec.onHear ? el("button", { class: "play", title: "Play this part of the song with this note on top",
        onclick: () => spec.onHear(i) }, "▶ hear") : null));
  });
  for (const r of spec.rows || []) {
    rows.push(el("div", { class: "fingering-row" },
      el("span", { class: "lbl", title: r.title || "" }, r.label),
      r.buttons.map((b) => el("button", {
        class: b.on ? "on" : "", title: b.title || "", "aria-pressed": String(!!b.on), onclick: close(b.onclick),
      }, b.text)),
      r.auto ? el("button", { class: "link", onclick: close(r.auto) }, "auto") : null));
  }
  for (const h of spec.hints || []) rows.push(el("div", { class: "hint" + (h.warn ? " warn" : "") }, h.text));
  if (spec.custom) {
    const input = el("input", { type: "text", placeholder: spec.custom.placeholder, value: spec.custom.value || "" });
    const apply = close(() => spec.custom.onSet(input.value.trim()));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") apply(); if (e.key === "Escape") closeChordMenu(); });
    rows.push(el("div", { class: "custom-row" }, input, el("button", { onclick: apply }, "Set")));
    rows.push(el("div", { class: "hint" }, spec.customHint || "No octave = the nearest one. Empty + Set reverts to the detected notes."));
  }
  if (spec.listen?.length) {
    rows.push(el("div", { class: "foot" }, spec.listen.map((b) => el("button", { onclick: b.onclick }, b.text))));
  }
  const foot = [];
  if (spec.split) {
    foot.push(el("button", {
      title: spec.split.at != null ? "Cut this note in two at the playhead" : "Cut this note in two in the middle (or move the playhead inside it first)",
      disabled: spec.split.disabled, onclick: close(spec.split.onclick),
    }, spec.split.at != null ? "✂ Split at playhead" : "✂ Split in half"));
  }
  if (spec.merge) {
    foot.push(el("button", { title: "Join this note and the next into one", disabled: spec.merge.disabled,
      onclick: close(spec.merge.onclick) }, "⇥ Merge with next"));
  }
  if (foot.length) rows.push(el("div", { class: "foot" }, foot));
  rows.push(el("div", { class: "foot" },
    spec.rest
      ? el("button", { onclick: close(spec.onRestore) }, "↺ Restore note")
      : el("button", { onclick: close(spec.onRemove) }, "✕ Not a note"),
    spec.extra || []));
  return rows;
}

// where the playhead could split a note, or null (then the middle)
function splitPoint(n) {
  const t = audio.currentTime;
  return t > n.start + 0.05 && t < n.end - 0.05 ? t : null;
}

// a note the tracker missed: starts at the playhead, lasts until the next note (max 1 s)
function openAddNoteMenu(anchor, notes, placeholder, onAdd) {
  const key = "add-note";
  if (S.menuFor === key && !$("#chord-menu").hidden) { closeChordMenu(); return; }
  closeChordMenu();
  const start = Math.round(audio.currentTime * 100) / 100;
  const later = notes.filter((n) => n.start > start + 0.05).sort((a, b) => a.start - b.start)[0];
  const end = Math.round(Math.min(start + 1, later ? later.start - 0.02 : start + 1) * 100) / 100;
  const input = el("input", { type: "text", placeholder });
  const apply = () => {
    const note = input.value.trim();
    if (!note) return;
    closeChordMenu();
    onAdd({ start, end, note });
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") apply(); if (e.key === "Escape") closeChordMenu(); });
  showMenu([
    el("div", { class: "head" }, el("span", {}, `New note at ${fmtTime(start)}`), el("span", {}, `${(end - start).toFixed(2)} s`)),
    el("div", { class: "custom-row" }, input, el("button", { onclick: apply }, "Add")),
    el("div", { class: "hint" }, "Move the playhead to where the note starts first. No octave = the nearest one."),
  ], anchor, key);
  input.focus();
}
