"use strict";
// Shared by the song tools (chord finder, splitter). Load before the app script.

// =========================================================================
// helpers
// =========================================================================

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
};

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return res.json();
}
const jsonBody = (method, data) => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
});

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

// =========================================================================
// job progress
// =========================================================================

// stages: the app's stage ids for an uploaded file; songs from a link get
// "download" in front. labels: {stage id: label}.
function createJobUI(stages, labels) {
  const allLabels = { download: "Download", ...labels };
  const stageLabel = (s) => allLabels[s] || s;

  function renderProgress(container, st, { onRetry } = {}) {
    const list = st.from_url ? ["download", ...stages] : stages;
    const idx = list.indexOf(st.stage);
    const overall = st.state === "done" ? 1 : Math.max(0, (idx + (st.progress || 0)) / list.length);
    const pills = list.map((s, i) => el("span", {
      class: "stage " + (st.state === "done" || i < idx ? "done" : i === idx ? "current" : ""),
    }, stageLabel(s)));
    const parts = [
      el("div", { class: "stages" }, pills),
      el("div", { class: "bar" }, el("div", { style: `width:${(overall * 100).toFixed(1)}%` })),
    ];
    if (st.state === "error") {
      parts.push(el("div", { class: "msg error" }, `Failed during ${stageLabel(st.stage)}: ${st.error}`));
      if (onRetry) parts.push(el("div", { class: "actions" }, el("button", { onclick: onRetry }, "Retry")));
    } else {
      parts.push(el("div", { class: "msg" }, st.state === "queued" ? "Queued…" : st.message || ""));
    }
    container.replaceChildren(el("div", { class: "progress" }, parts));
  }

  async function pollStatus(id, container, { onDone, onRetry } = {}) {
    for (;;) {
      let st;
      try { st = await api(`/api/songs/${id}/status`); } catch (e) {
        container.replaceChildren(el("div", { class: "msg error" }, e.message));
        return;
      }
      renderProgress(container, st, { onRetry });
      if (st.state === "done") { onDone?.(); return; }
      if (st.state === "error") return;
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  return { stageLabel, renderProgress, pollStatus };
}

// =========================================================================
// song list
// =========================================================================

async function refreshSongList(ul, { stageLabel, href }) {
  let songs;
  try { songs = await api("/api/songs"); } catch (e) {
    ul.replaceChildren(el("li", { class: "muted" }, e.message));
    return;
  }
  if (!songs.length) { ul.replaceChildren(el("li", { class: "muted" }, "Nothing here yet.")); return; }
  ul.replaceChildren(...songs.map((s) => {
    const st = s.status || {};
    const badge = st.state === "done" ? el("span", { class: "badge ok" }, "ready")
      : st.state === "error" ? el("span", { class: "badge err" }, "failed")
      : el("span", { class: "badge" }, st.state === "running" ? stageLabel(st.stage) + "…" : st.state);
    const del = el("button", { class: "danger", title: "Delete song" }, "Delete");
    del.addEventListener("click", async () => {
      if (del.dataset.armed !== "1") {
        del.dataset.armed = "1"; del.textContent = "Really delete?";
        setTimeout(() => { del.dataset.armed = ""; del.textContent = "Delete"; }, 3000);
        return;
      }
      await api(`/api/songs/${s.id}`, { method: "DELETE" });
      refreshSongList(ul, { stageLabel, href });
    });
    return el("li", {},
      el("a", { href: href(s) }, s.title),
      badge,
      s.source ? el("span", { class: "badge", title: s.source.page_url }, s.source.site) : null,
      el("span", { class: "badge" }, new Date(s.created * 1000).toLocaleDateString()),
      del);
  }));
}

// =========================================================================
// add a song: drop files or paste a link (either/or)
// =========================================================================

function createSourcePicker(container, { multiple = false, hint = "", onChange = () => {}, onSubmit = () => {} } = {}) {
  const files = [];
  const input = el("input", { type: "file", accept: "audio/*,.m4a,.flac,.aiff,.aif,.opus", multiple, hidden: true });
  const dz = el("label", { class: "dropzone", tabindex: "0" },
    input,
    el("strong", {}, "Drop audio here"), " or click to choose",
    hint ? el("small", {}, hint) : null);
  const list = el("ul", { class: "file-list" });
  const link = el("input", {
    type: "url", inputmode: "url", autocomplete: "off", spellcheck: "false",
    placeholder: "YouTube, SoundCloud, Spotify, Bandcamp… or a direct link to an audio file",
  });
  container.replaceChildren(
    dz, list,
    el("div", { class: "or" }, el("span", {}, "or paste a link")),
    el("label", { class: "field link-field" },
      el("span", { class: "sr-only" }, "Link to a song"),
      link,
      el("small", { class: "muted" }, "Spotify tracks can't be downloaded, so the same song is found on YouTube instead.")),
  );

  const linkValue = () => link.value.trim();
  const picker = {
    get files() { return files; },
    get link() { return files.length ? "" : linkValue(); },
    hasInput: () => files.length > 0 || linkValue() !== "",
    clear() { files.length = 0; link.value = ""; render(); },
    // [path, fetch options] for POST /api/songs (files) or /api/songs/link
    request(options) {
      if (picker.link) return ["/api/songs/link", jsonBody("POST", { url: picker.link, ...options })];
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f, f.name));
      for (const [k, v] of Object.entries(options)) if (v != null) fd.append(k, String(v));
      return ["/api/songs", { method: "POST", body: fd }];
    },
  };

  function render() {
    list.replaceChildren(...files.map((f, i) => el("li", {}, f.name,
      el("button", { title: "Remove", "aria-label": `Remove ${f.name}`,
        onclick: (e) => { e.preventDefault(); files.splice(i, 1); render(); } }, "×"))));
    onChange(picker);
  }
  function add(fileList) {
    for (const f of fileList) {
      if (!multiple) files.length = 0;
      if (!files.some((g) => g.name === f.name && g.size === f.size)) files.push(f);
    }
    if (files.length) link.value = "";
    render();
  }

  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { add(input.files); input.value = ""; });
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => add(e.dataTransfer.files));
  link.addEventListener("input", () => { if (linkValue()) files.length = 0; render(); });
  link.addEventListener("keydown", (e) => { if (e.key === "Enter" && picker.hasInput()) onSubmit(); });
  render();
  return picker;
}
