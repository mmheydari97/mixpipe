"use strict";
// The chord finder's Vocals and Guitar sliders: turn either down to sing or play
// it yourself. The first move splits the song into stems on the server
// (chords/stems6.py, the splitter's split, once per song); until they're here the
// full mix plays. Then three stems (vocals, guitar, the rest) play in step with
// the page's own <audio>, which keeps the controls and the clock and goes silent.
// Both sliders at 100% is the original mix again.
// Uses common.js, stem-player.js and app.js (audio, S).

const PA_TRACKS = [
  { name: "vocals", label: "Vocals" }, { name: "guitar", label: "Guitar" },
  { name: "backing", label: "the rest" }];
const PA = { id: null, vol: { vocals: 1, guitar: 1, backing: 1 }, state: "off", msg: "", timer: 0,
             gainDb: 0,       // the stems turned down so their sum can't clip (stems6.gain_db)
             alone: null };   // the guitar tab's "Guitar alone": the levels to go back to
// state: off (full mix) | splitting | loading | on | error

const paActive = () => PA.vol.vocals < 1 || PA.vol.guitar < 1 || PA.vol.backing < 1;
const paKey = (id) => `mixpipe.playalong.${id}`;
const PA_PLAYER = createStemPlayer({ gain: (name) => PA.vol[name] ?? 1 });
PA_PLAYER.follow(audio, () => PA.state === "on" && paActive());

// the page's <audio> is silent while the stems sound, and the other way round
async function paApply() {
  const stems = PA.state === "on" && paActive();
  PA_PLAYER.applyGains();
  if (stems) {
    if (!audio.paused && !PA_PLAYER.playing) await PA_PLAYER.play();
    audio.volume = 0;
  } else {
    audio.volume = 1;
    PA_PLAYER.pause();
  }
}

function paRender() {
  for (const name of ["vocals", "guitar"]) {
    const v = PA.vol[name];
    $(`#pa-${name}`).value = v;
    $(`#pa-${name}-out`).textContent = `${Math.round(v * 100)}%`;
    $(`#pa-${name}`).closest("label").classList.toggle("down", v < 1);
  }
  const msg = $("#pa-msg");
  msg.textContent = PA.msg;
  msg.classList.toggle("error", PA.state === "error");
  msg.hidden = !PA.msg;
  $("#pa-retry").hidden = PA.state !== "error";
}

async function paStart() {
  if (["splitting", "loading", "on"].includes(PA.state)) return paApply();
  const id = PA.id;
  PA.state = "splitting"; PA.msg = "Splitting into stems…"; paRender();
  let st;
  try { st = await api(`/api/songs/${id}/playalong`, { method: "POST" }); } catch (e) {
    if (id === PA.id) { PA.state = "error"; PA.msg = e.message; paRender(); }
    return;
  }
  paFollow(st, id);
}

// poll the split until the stems are there
function paFollow(st, id) {
  if (id !== PA.id) return;
  if (st.ready) { PA.gainDb = st.gain_db || 0; paLoad(id); return; }
  if (st.state === "error" || st.state === "cancelled") {
    PA.state = "error";
    PA.msg = `Couldn't split the song: ${st.error || st.state}`;
    paRender();
    return;
  }
  const pct = st.state === "running" ? ` ${Math.round((st.progress || 0) * 100)}%` : "";
  const wait = st.state === "queued" && st.ahead ? ` (${st.ahead} job${st.ahead === 1 ? "" : "s"} ahead)` : "";
  PA.msg = `Splitting into stems${pct}${wait}, about a minute; full mix until then…`;
  paRender();
  clearTimeout(PA.timer);
  PA.timer = setTimeout(async () => {
    try { paFollow(await api(`/api/songs/${id}/playalong`), id); } catch (e) {
      if (id === PA.id) { PA.state = "error"; PA.msg = e.message; paRender(); }
    }
  }, 1500);
}

async function paLoad(id) {
  PA.state = "loading"; PA.msg = "Loading the stems…"; paRender();
  try {
    const ok = await PA_PLAYER.load(() => PA_PLAYER.makeSet(PA_TRACKS.map((t) => (
      { ...t, url: `/api/songs/${id}/playalong/${t.name}` }))), () => id === PA.id);
    if (!ok) return;
  } catch (e) {
    if (id === PA.id) { PA.state = "error"; PA.msg = e.message; paRender(); }
    return;
  }
  PA.state = "on"; PA.msg = ""; paRender();
  PA_PLAYER.output.gain.value = Math.pow(10, PA.gainDb / 20);
  paApply();
}

const paSave = () => store.set(paKey(PA.id), JSON.stringify({ vocals: PA.vol.vocals, guitar: PA.vol.guitar }));

function paInput(name, value) {
  PA.vol[name] = value;
  if (PA.alone) { PA.alone = null; PA.vol.backing = 1; $("#solo-hear")?.setAttribute("aria-pressed", "false"); }
  paSave();
  paRender();
  if (paActive()) paStart(); else paApply();
}

// a song was opened (app.js openSong)
function openPlayalong(id) {
  if (PA.id === id) return;
  closePlayalong();
  PA.id = id;
  try { Object.assign(PA.vol, JSON.parse(store.get(paKey(id)) || "{}")); } catch { /* bad value */ }
  paRender();
  if (paActive()) paStart();    // turned down last time: the stems are (being) made already
}

// the guitar tab's "Guitar alone": vocals and the rest silent, the guitar up; off puts
// the sliders back where they were
function paGuitarOnly(on) {
  if (!PA.id) return;
  if (on) {
    PA.alone ||= { ...PA.vol };
    Object.assign(PA.vol, { vocals: 0, guitar: 1, backing: 0 });
  } else if (PA.alone) {
    Object.assign(PA.vol, PA.alone);
    PA.alone = null;
  }
  paRender();
  if (paActive()) paStart(); else paApply();
}

function closePlayalong() {
  clearTimeout(PA.timer);
  PA_PLAYER.teardown();
  Object.assign(PA, { id: null, vol: { vocals: 1, guitar: 1, backing: 1 }, state: "off", msg: "", alone: null });
  $("#solo-hear")?.setAttribute("aria-pressed", "false");
  audio.volume = 1;
  paRender();
}

for (const name of ["vocals", "guitar"]) {
  $(`#pa-${name}`).addEventListener("input", (e) => paInput(name, Number(e.target.value)));
  // double-click a slider: back to 100%
  $(`#pa-${name}`).addEventListener("dblclick", () => paInput(name, 1));
}
$("#pa-retry").addEventListener("click", () => { PA.state = "off"; paStart(); });
