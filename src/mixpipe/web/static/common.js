"use strict";
// Shared by the web apps (chord finder, splitter, mixer). Load before the app script.

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

  function renderProgress(container, st, { onRetry, onCancel } = {}) {
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
    } else if (st.state === "cancelled") {
      parts.push(el("div", { class: "msg" }, "Cancelled."));
      if (onRetry) parts.push(el("div", { class: "actions" }, el("button", { onclick: onRetry }, "Start again")));
    } else {
      const queued = st.state === "queued"
        ? (st.ahead ? `Queued · ${st.ahead} job${st.ahead === 1 ? "" : "s"} ahead…` : "Queued…")
        : st.message || "";
      parts.push(el("div", { class: "msg" }, queued));
      if (onCancel && (st.state === "queued" || st.state === "running")) {
        parts.push(el("div", { class: "actions" }, el("button", { onclick: onCancel }, "Cancel")));
      }
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
      const cancel = async () => {
        try { await api(`/api/songs/${id}/cancel`, { method: "POST" }); } catch { /* it just finished */ }
      };
      renderProgress(container, st, { onRetry, onCancel: cancel });
      if (st.state === "done") { onDone?.(); return; }
      if (st.state === "error" || st.state === "cancelled") return;
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  return { stageLabel, renderProgress, pollStatus };
}

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

// kind: "chat" or "transcription". The choice is remembered under `storeKey`;
// `value` starts the picker on that model instead (e.g. what a session already
// uses), and `defaultValue` is shown when nothing was ever chosen.
function createModelPicker(container, kind, {
  placeholder, initialQuery = "", onChange, storeKey = `mixpipe.model.${kind}`,
  value = null, defaultValue = null,
} = {}) {
  let selected = value || store.get(storeKey) || defaultValue;
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
      for (const out of m.output_modalities || []) {
        if (out !== "text") badges.push(el("span", { class: "badge" }, `+${out} out`));
      }
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

  return {
    get value() { return selected; },
    set(id) { selected = id; showChosen(); },     // show a model without remembering it
  };
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
      : st.state === "cancelled" ? el("span", { class: "badge" }, "cancelled")
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
