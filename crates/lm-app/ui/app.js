// LocalMotive dashboard. No framework, no network: everything goes through
// Tauri IPC (commands + events) and the lmimg:// image protocol.
"use strict";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const img = (path) => "lmimg://localhost/" + path.split("/").map(encodeURIComponent).join("/");
const MASK = "[see image]";

// ---------------------------------------------------------------- boot

const bootList = $("#boot-steps");
function renderBootStep(s) {
  let li = bootList.querySelector(`li[data-id="${s.id}"]`);
  if (!li) {
    li = document.createElement("li");
    li.dataset.id = s.id;
    bootList.appendChild(li);
  }
  li.className = s.status;
  li.innerHTML = `<span class="s"></span><span class="l">${esc(s.label)}</span><span class="d">${esc(s.detail)}</span>`;
}

function bootDone(b) {
  if (!b.ok) {
    const failed = b.steps.find((s) => s.status === "fail");
    const box = $("#boot-error");
    box.hidden = false;
    box.textContent = "STARTUP STOPPED\n\n" + (failed ? `${failed.label}: ${failed.detail}` : "unknown error");
    return;
  }
  setTimeout(() => {
    $("#boot").hidden = true;
    $("#dash").hidden = false;
    $("#pill-model").lastChild.textContent = b.model.toUpperCase();
    loadDtcs();
    log(`boot complete · ${b.chunks} chunks · ${b.vectors} vectors · VIN ${b.vin}`, "pass");
    $("#q").focus();
  }, 700);
}

// ---------------------------------------------------------------- event log

const logBox = $("#log");
let streamLine = null;
function stamp() {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}
function log(text, cls = "") {
  const div = document.createElement("div");
  div.innerHTML = `<time>${stamp()}</time><span class="${cls}">${esc(text)}</span>`;
  logBox.appendChild(div);
  while (logBox.children.length > 40) logBox.firstChild.remove();
  streamLine = null;
  return div;
}
function logStream(tok) {
  if (!streamLine) {
    streamLine = log("model: ", "stream");
    streamLine.dataset.buf = "";
  }
  streamLine.dataset.buf = (streamLine.dataset.buf + tok).slice(-160);
  streamLine.querySelector("span").textContent = "model ▸ " + streamLine.dataset.buf.replace(/\s+/g, " ");
}

function onProgress(p) {
  switch (p.event) {
    case "stage_start":
      if (p.stage !== "draft") log(`${p.stage} …`);
      else streamLine = null;
      break;
    case "stage_end":
      log(`${p.stage} ✓ ${p.ms} ms · ${p.detail}`, p.stage === "verify" && p.detail.startsWith("all") ? "pass" : "");
      break;
    case "retrieved":
      for (const c of p.chunks) log(`chunk ${c.page_id} [${c.kind}${c.ocr ? ",ocr" : ""}] ${c.title}`);
      break;
    case "token":
      logStream(p.text);
      break;
    case "verdicts":
      log(`verifier (attempt ${p.attempt}): ${p.passed} pass, ${p.failed} fail`, p.failed ? "fail" : "pass");
      break;
    case "retry":
      log("retrying with the verifier's reasons", "warn");
      break;
  }
}

// ---------------------------------------------------------------- telemetry

const ARC = Math.PI * 45;
document.querySelectorAll(".gauge").forEach((g) => {
  const bar = g.querySelector(".bar");
  bar.style.stroke = g.dataset.color;
  bar.style.strokeDasharray = ARC;
  bar.style.strokeDashoffset = ARC;
  g.style.color = g.dataset.color;
});
function onTelemetry(s) {
  document.querySelectorAll(".gauge").forEach((g) => {
    const v = s[g.dataset.key];
    const min = +g.dataset.min, max = +g.dataset.max;
    const frac = Math.max(0, Math.min(1, (v - min) / (max - min)));
    g.querySelector(".bar").style.strokeDashoffset = ARC * (1 - frac);
    const unit = g.dataset.unit || "";
    const shown = g.dataset.key === "ltft" ? (v >= 0 ? "+" : "") + v.toFixed(1) : Math.round(v);
    g.querySelector(".val").textContent = shown + unit;
  });
  $("#m-voltage").textContent = s.voltage.toFixed(1);
  $("#m-stft").textContent = (s.stft >= 0 ? "+" : "") + s.stft.toFixed(1);
}

// ---------------------------------------------------------------- DTCs + procedure

let procedure = null;
let active = 0;

async function loadDtcs() {
  const list = $("#dtc-list");
  try {
    const dtcs = await invoke("dtcs");
    const actives = dtcs.filter((d) => d.status === "active").length;
    $("#pill-dtc").lastChild.textContent = `DTC: ${actives} ACTIVE`;
    list.innerHTML = "";
    for (const d of dtcs) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "dtc " + (d.status === "active" ? "" : "history");
      b.innerHTML = `<span class="state">${d.status === "active" ? "ACTIVE" : "HISTORY"}</span><div class="code">${esc(d.code)}</div><div class="desc">${esc(d.description || "see manual")}</div>`;
      b.addEventListener("click", () => {
        list.querySelectorAll(".dtc").forEach((x) => x.classList.remove("sel"));
        b.classList.add("sel");
        openProcedure(d.code);
      });
      list.appendChild(b);
    }
    const first = dtcs.find((d) => d.status === "active");
    if (first) openProcedure(first.code);
  } catch (e) {
    list.innerHTML = `<div class="muted">${esc(e)}</div>`;
  }
}

async function openProcedure(code, focusChunk) {
  return showPage(invoke("procedure", { code }), code, focusChunk);
}

async function openPage(pageId, focusChunk) {
  return showPage(invoke("page", { pageId }), `page ${pageId}`, focusChunk);
}

async function showPage(request, label, focusChunk) {
  try {
    procedure = await request;
  } catch (e) {
    log(`procedure ${label}: ${e}`, "fail");
    return;
  }
  const code = procedure.code || procedure.title;
  $("#proc-title").textContent = `PROCEDURE · ${procedure.title}`;
  $("#proc-src").textContent = `manual p.${procedure.page_id}`;
  const idx = focusChunk ? procedure.sections.findIndex((s) => s.id === focusChunk) : -1;
  active = idx >= 0 ? idx : 0;
  renderProcedure();
  log(`procedure ${code}: ${procedure.sections.length} sections`);
  return procedure;
}

function renderProcedure() {
  const box = $("#proc-steps");
  box.innerHTML = "";
  procedure.sections.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "step" + (i < active ? " done" : "") + (i === active ? " active" : "");
    const heading = s.section_heading || s.title;
    let body = "";
    if (s.ocr) {
      body += s.images.map((p) => `<img src="${img(p)}" alt="${esc(heading)} table">`).join("");
      body += `<div class="note">Table image from the manual. Text below is OCR — check values against the picture.</div>`;
      if (i === active) body += `<pre>${esc(s.text)}</pre>`;
    } else if (i === active) {
      body += `<pre>${esc(s.text)}</pre>`;
      body += s.images.map((p) => `<img src="${img(p)}" alt="">`).join("");
    }
    div.innerHTML = `<span class="n">${String(i + 1).padStart(2, "0")}</span><span class="h">${esc(heading)}</span>${body}`;
    div.addEventListener("click", () => {
      active = i;
      renderProcedure();
    });
    box.appendChild(div);
  });
  box.querySelector(".step.active")?.scrollIntoView({ block: "nearest" });
}
$("#proc-prev").addEventListener("click", () => {
  if (procedure && active > 0) { active--; renderProcedure(); }
});
$("#proc-next").addEventListener("click", () => {
  if (procedure && active < procedure.sections.length - 1) { active++; renderProcedure(); }
});

// ---------------------------------------------------------------- ask

function renderAnswer(r) {
  const byId = Object.fromEntries(r.sources.map((s) => [s.id, s]));
  const box = $("#answer");
  // verified claims, in the order the pipeline put them (manual order)
  const pool = r.verified.claims.filter((c) => c.verdict === "pass");
  const passing = r.answer.claims.map((sc) => {
    const i = pool.findIndex((c) => c.text === sc.text);
    return i >= 0 ? pool.splice(i, 1)[0] : null;
  }).filter(Boolean).concat(pool);
  const best = r.answer.best_match;
  let html = "";
  if (best && (r.answer.weak || !passing.length)) {
    html += `<div class="weak">This may not fully answer the question${r.answer.weak_reason ? ` (${esc(r.answer.weak_reason)})` : ""}. Best match in the manual: <a data-best="1">${esc(best.title)}</a>${best.has_steps ? " — its step tables are open in the procedure panel." : "."}</div>`;
  }
  for (const c of passing) {
    const srcs = c.cites.map((id) => byId[id]).filter(Boolean);
    const ocr = srcs.filter((s) => s.ocr);
    const masked = c.text.includes(MASK);
    html += `<div class="claim">${esc(c.text)}`;
    if (c.needs_image || ocr.length || masked) {
      html += `<span class="badge">CHECK AGAINST THE PICTURE</span>`;
      const imgs = (ocr.length ? ocr : srcs).flatMap((s) => s.images);
      html += [...new Set(imgs)].map((p) => `<img src="${img(p)}" alt="manual table">`).join("");
    }
    html += `</div>`;
  }
  const ns = r.verified.next_step;
  if (ns && ns.verdict === "pass") {
    const s = byId[ns.chunk_id];
    html += `<div class="claim"><span class="lead">Recommendation:</span> continue with ${esc(ns.step_label)}${s ? ` in “${esc(s.section_heading || s.title)}”` : ""}.</div>`;
  }
  if (!passing.length && !best) html += `<div class="muted">No statement could be verified against the manual. The retrieved pages are listed below.</div>`;
  for (const n of r.answer.not_verified) html += `<div class="nv">Not verified: ${esc(n)}</div>`;
  box.innerHTML = html;
  box.querySelector("a[data-best]")?.addEventListener("click", () => openPage(best.page_id, best.chunk_id));

  const cited = new Set(passing.flatMap((c) => c.cites));
  const list = r.sources.filter((s) => cited.has(s.id));
  const shown = list.length ? list : r.sources;
  const sbox = $("#sources");
  sbox.innerHTML = (list.length ? "Sources: " : "Retrieved: ") + shown.map((s) => `<a data-id="${esc(s.id)}">${esc(s.title)}${s.ocr ? " (table image)" : ""}</a>`).join(" · ");
  sbox.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => openSource(a.dataset.id)));

  // Walker: the recommended step if it is on the open page; otherwise open the
  // best-matching page when the answer is weak or that page has step tables.
  if (ns && ns.verdict === "pass" && procedure && procedure.sections.some((s) => s.id === ns.chunk_id)) {
    active = procedure.sections.findIndex((s) => s.id === ns.chunk_id);
    renderProcedure();
  } else if (best && (r.answer.weak || best.has_steps || !passing.length) && (!procedure || procedure.page_id !== best.page_id)) {
    openPage(best.page_id, best.chunk_id);
  }
}

async function openSource(id) {
  try {
    const s = await invoke("source", { id });
    $("#v-title").textContent = s.title + (s.section_heading ? " — " + s.section_heading : "");
    $("#v-crumb").textContent = `${s.breadcrumb} · page ${s.page_id}${s.ocr ? " · OCR of a table image" : ""}`;
    $("#v-images").innerHTML = s.images.map((p) => `<img src="${img(p)}" alt="">`).join("");
    $("#v-text").textContent = s.text;
    $("#viewer").showModal();
  } catch (e) {
    log(`source ${id}: ${e}`, "fail");
  }
}

$("#query").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("#q").value.trim();
  if (!q) return;
  const btn = $("#go");
  btn.disabled = true;
  $("#answer").innerHTML = `<span class="thinking">reading the manual…</span>`;
  $("#sources").textContent = "";
  log(`query: ${q}`);
  try {
    const r = await invoke("ask", { question: q });
    renderAnswer(r);
    log(`answer ready in ${(r.timings.total_ms / 1000).toFixed(1)} s`, r.verified.ok ? "pass" : "warn");
  } catch (e) {
    $("#answer").innerHTML = `<span class="nv">${esc(e)}</span>`;
    log(`error: ${e}`, "fail");
  } finally {
    btn.disabled = false;
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#viewer").open) invoke("set_fullscreen", { on: false });
});

// ---------------------------------------------------------------- start

(async () => {
  await listen("boot", (e) => renderBootStep(e.payload));
  await listen("boot-done", (e) => bootDone(e.payload));
  await listen("telemetry", (e) => onTelemetry(e.payload));
  await listen("progress", (e) => onProgress(e.payload));
  // catch up on steps emitted before the listeners were attached
  const b = await invoke("boot_state");
  b.steps.forEach(renderBootStep);
  if (b.done) bootDone(b);
})();
