"use strict";
// Chord sheets as the chord finder serves them (chords/sheet.py view()): chord
// labels, lyric cells, the rows of a sheet, and following playback. The chord
// finder's page (app.js, harmonica.js) and the splitter's chords & lyrics panel
// both draw with these. Needs common.js (el).

const fmtTime = (t) => {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
};
const RTL = /[֐-ࣿיִ-﷿ﹰ-﻿]/;

function labelOf(c) {
  if (c.custom) return c.custom;
  const cands = c.candidates || [];
  return cands.length ? cands[Math.min(c.selected || 0, cands.length - 1)].label : "N.C.";
}

// label as shown: after transpose and capo
function displayOf(c) {
  return c.display ?? labelOf(c);
}

// One lyric word with its chips. A word the server split into syllables (w.syl)
// puts each chip over its own syllable; `mk(id)` makes a chip, `carry` is a
// "still sounding" chip for the line's first word, `onSeek(t)` jumps there.
// Returns {node, timing}: the cell to insert and the node that is highlighted
// while the word is sung.
function lyricCell(w, src, mk, carry, onSeek) {
  const approx = src.approx ? " approx" : "";
  const barCls = w.bar != null ? " barline" : "";
  const barTitle = w.bar != null ? `Bar ${w.bar + 1}` : null;
  if (!w.syl) {
    const chips = w.chords.map(mk);
    if (carry) chips.unshift(carry);
    const word = el("span", { class: "word" + approx, title: fmtTime(w.start) }, w.text);
    word.addEventListener("click", () => onSeek(w.start));
    return { node: el("span", { class: "cell" + barCls, title: barTitle }, el("span", { class: "chords" }, chips), word), timing: word };
  }
  const groups = w.syl.map(() => []);
  w.chords.forEach((id) => groups[w.at[id] ?? 0].push(mk(id)));
  if (carry) groups[0].unshift(carry);
  const parts = w.syl.map((sp, k) => {
    const end = k + 1 < w.syl.length ? w.syl[k + 1].c : undefined;
    const part = el("span", { class: "word part" + approx, title: `${fmtTime(sp.start)} · syllable ${k + 1}` }, w.text.slice(sp.c, end));
    part.addEventListener("click", () => onSeek(sp.start));
    return el("span", { class: "cell syl" + (k === 0 ? barCls : ""), title: k === 0 ? barTitle : null },
      el("span", { class: "chords" }, groups[k]), part);
  });
  const group = el("span", { class: "wordgroup", title: fmtTime(w.start) }, parts);
  return { node: group, timing: group };
}

// The rows of a sheet view: lyric lines with chord chips over their words, and
// instrumental stretches as bars. chip(id, {carry}) makes a chord chip and
// onSeek(t) jumps playback there. Returns {rows, words}: the nodes, and the
// words [{start, end, node}] in time order for followWord.
function sheetRows(song, chip, onSeek) {
  const chords = new Map(song.chords.map((c) => [c.id, c]));
  const words = [];
  const rows = song.rows.map((row) => {
    if (row.type === "instrumental") {
      const items = [el("span", { class: "tag" }, "instr.")];
      if (row.bars) {
        items.push(el("span", { class: "bars" }, row.bars.map((bar) => {
          const parts = [];
          if (bar.chords.length) {
            if (bar.hold) parts.push(chip(bar.hold, { carry: true }));
            bar.chords.forEach((id) => parts.push(chip(id)));
          } else {
            parts.push(el("span", { class: "repeat", title: bar.hold ? `${displayOf(chords.get(bar.hold))} continues` : "no chord" },
              bar.hold ? "%" : "–"));
          }
          return el("span", { class: "barcell", title: `Bar ${bar.bar + 1}` }, parts);
        })));
      } else {
        row.chords.forEach((id, i) => {
          if (i) items.push(el("span", { class: "sep" }, "|"));
          items.push(chip(id));
        });
      }
      return el("div", { class: "line instrumental" }, items);
    }
    const cells = row.words.map((w, i) => {
      const src = song.lines[row.line].words[w.i];
      const carry = i === 0 && row.carry ? chip(row.carry, { carry: true }) : null;
      const { node, timing } = lyricCell(w, src, (id) => chip(id), carry, onSeek);
      words.push({ start: w.start, end: w.end, node: timing });
      return node;
    });
    return el("div", { class: "line" + (row.gap_before ? " stanza" : "") }, cells);
  });
  words.sort((a, b) => a.start - b.start);
  return { rows, words };
}

// the text direction a sheet's lyrics need
function sheetDir(song) {
  return RTL.test(song.lines.map((l) => l.words.map((w) => w.text).join(" ")).join(" ")) ? "rtl" : "ltr";
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

// highlight the word being sung (and scroll to it when `scroll`, i.e. following playback)
function followWord(words, t, holder, scroll) {
  const wi = findAt(words, t);
  const w = wi >= 0 && t <= words[wi].end + 0.3 ? words[wi] : null;
  if (w !== holder.activeWord) {
    holder.activeWord?.node.classList.remove("active");
    w?.node.classList.add("active");
    holder.activeWord = w;
    if (w && scroll) {
      const r = w.node.getBoundingClientRect();
      if (r.top < 80 || r.bottom > window.innerHeight - 140) {
        w.node.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    }
  }
}

// The chord sounding at t in a timeline (chords with a label, by start) and the
// next one; lights up the sounding chord's chips (chipNodes: id -> [nodes]).
function followChord(timeline, chipNodes, t, holder) {
  const ci = findAt(timeline, t);
  const cur = ci >= 0 && t < timeline[ci].end ? timeline[ci] : null;
  if (cur?.id !== holder.activeChord) {
    (chipNodes.get(holder.activeChord) || []).forEach((n) => n.classList.remove("active"));
    (chipNodes.get(cur?.id) || []).forEach((n) => n.classList.add("active"));
    holder.activeChord = cur?.id ?? null;
  }
  return { cur, next: timeline[ci + 1] || null };
}

// chords with a label, in time order: what followChord walks through
const chordTimeline = (song) => song.chords.filter((c) => labelOf(c) !== "N.C.").sort((a, b) => a.start - b.start);
