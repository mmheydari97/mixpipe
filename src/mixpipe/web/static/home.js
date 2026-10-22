"use strict";
// Home page of the hub. Needs common.js (el, api). Each card says what is already there.

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

async function status(id, fn) {
  try { $(`#${id}`).textContent = await fn(); } catch { /* the card still links */ }
}

status("status-chords", async () => {
  const songs = await api("/chords/api/songs");
  return songs.length ? plural(songs.length, "song") : "No songs yet";
});
status("status-splitter", async () => {
  const songs = await api("/splitter/api/songs");
  return songs.length ? plural(songs.length, "song") : "No songs yet";
});
status("status-mixer", async () => {
  const [sessions, ai] = await Promise.all([
    api("/mixer/api/sessions"), api("/mixer/api/ai-edit/status").catch(() => null)]);
  const parts = [sessions.length ? plural(sessions.length, "session") + " open" : "No sessions open"];
  if (ai) parts.push(ai.ok ? "AI edits ready" : "AI edits off (make studio)");
  return parts.join(" · ");
});
