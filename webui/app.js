"use strict";
// RWKV-Infer-Next WebUI: chat (conversations in localStorage) + settings / engine status.

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { toast("Could not save (browser storage is full or blocked)"); } },
};
const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
// API key (RWKV_API_KEY on the server): kept in this browser only, asked for on the first 401
function apiKey() { try { return localStorage.getItem("rin.apiKey") || ""; } catch { return ""; } }
async function api(url, opts = {}) {
  const send = () => { const k = apiKey(); const h = Object.assign({}, opts.headers || {}); if (k) h.Authorization = "Bearer " + k; return fetch(url, Object.assign({}, opts, { headers: h })); };
  let r = await send();
  if (r.status === 401) {
    const k = prompt("This server requires an API key:", "");
    if (k) { try { localStorage.setItem("rin.apiKey", k.trim()); } catch {} r = await send(); }
  }
  return r;
}
let convs = store.get("rin.convs", []);
let activeId = store.get("rin.active", null);
let ui = Object.assign({ theme: "", enterSend: true, params: false, side: true }, store.get("rin.ui", {}));
let defaults = { temperature: 1, top_p: 0.3, top_k: 0, min_p: 0, presence_penalty: 0, frequency_penalty: 0, max_tokens: 4096, thinking: "auto" };
let gen = null;          // {ctrl, convId, msg}
let view = "chat";
let status = {};

function saveConvs() { store.set("rin.convs", convs); }
function saveUi() { store.set("rin.ui", ui); }
function toast(t, ms = 2200) { const el = $("toast"); el.textContent = t; el.hidden = false; clearTimeout(toast._t); toast._t = setTimeout(() => el.hidden = true, ms); }
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function conv() { return convs.find((c) => c.id === activeId) || null; }

// ---------------------------------------------------------------- theme / layout
function applyTheme() { if (ui.theme) document.documentElement.dataset.theme = ui.theme; else delete document.documentElement.dataset.theme; }
$("themeBtn").onclick = () => { ui.theme = ui.theme === "" ? "dark" : ui.theme === "dark" ? "light" : ""; applyTheme(); saveUi(); toast(ui.theme ? `Theme: ${ui.theme}` : "Theme: follow the system"); };
function applyLayout() {
  const narrow = window.innerWidth <= 760;
  $("app").classList.toggle("side-closed", !ui.side && !narrow);
  $("app").classList.toggle("side-open", narrow && ui.side);
  $("params").hidden = !ui.params || view !== "chat";
}
$("sideToggle").onclick = () => { ui.side = !ui.side; saveUi(); applyLayout(); };
$("paramsToggle").onclick = () => { ui.params = !ui.params; saveUi(); applyLayout(); };
window.addEventListener("resize", applyLayout);

// ---------------------------------------------------------------- conversations
function newConv() {
  const c = { id: uid(), title: "New chat", created: Date.now(), updated: Date.now(), messages: [], params: null, system: "", think: null };
  convs.unshift(c); activeId = c.id; saveConvs(); store.set("rin.active", activeId);
  setView("chat"); renderAll(); $("input").focus();
}
function groupLabel(t) {
  const d = new Date(t), now = new Date();
  const day = 864e5, diff = (new Date(now.toDateString()) - new Date(d.toDateString())) / day;
  if (diff <= 0) return "Today"; if (diff === 1) return "Yesterday"; if (diff < 7) return "Previous 7 days"; if (diff < 30) return "Previous 30 days";
  return d.toLocaleDateString(undefined, { year: "numeric", month: "long" });
}
function renderConvList() {
  const q = $("search").value.trim().toLowerCase();
  const list = $("convList"); list.innerHTML = "";
  let last = null;
  convs.slice().sort((a, b) => b.updated - a.updated).forEach((c) => {
    if (q && !(c.title.toLowerCase().includes(q) || c.messages.some((m) => (m.content || "").toLowerCase().includes(q)))) return;
    const g = groupLabel(c.updated);
    if (g !== last) { const h = document.createElement("div"); h.className = "conv-group"; h.textContent = g; list.appendChild(h); last = g; }
    const el = document.createElement("div");
    el.className = "conv" + (c.id === activeId ? " active" : ""); el.tabIndex = 0;
    el.innerHTML = `<span class="t">${esc(c.title)}</span><span class="act">
      <button class="icon" data-a="rename" title="Rename" aria-label="Rename"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg></button>
      <button class="icon" data-a="del" title="Delete" aria-label="Delete"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg></button></span>`;
    el.onclick = (e) => {
      const a = e.target.closest("[data-a]")?.dataset.a;
      if (a === "rename") { const t = prompt("Chat name", c.title); if (t && t.trim()) { c.title = t.trim(); saveConvs(); renderConvList(); renderTitle(); } return; }
      if (a === "del") { if (!confirm(`Delete "${c.title}"?`)) return; convs = convs.filter((x) => x !== c); if (activeId === c.id) activeId = convs[0]?.id || null; saveConvs(); store.set("rin.active", activeId); renderAll(); return; }
      activeId = c.id; store.set("rin.active", activeId); setView("chat"); renderAll();
      if (window.innerWidth <= 760) { ui.side = false; applyLayout(); }
    };
    el.onkeydown = (e) => { if (e.key === "Enter") el.click(); };
    list.appendChild(el);
  });
}
$("search").oninput = renderConvList;
$("newChat").onclick = newConv;
$("exportBtn").onclick = () => {
  const blob = new Blob([JSON.stringify({ version: 1, conversations: convs }, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `rwkv-chats-${new Date().toISOString().slice(0, 10)}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
$("importBtn").onclick = () => $("importFile").click();
$("importFile").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const d = JSON.parse(await f.text()); const list = d.conversations || d;
    let n = 0; for (const c of list) { if (!c.messages) continue; if (!convs.some((x) => x.id === c.id)) { convs.push(Object.assign({ params: null, system: "", think: null }, c)); n++; } }
    saveConvs(); renderConvList(); toast(`Imported ${n} chat${n === 1 ? "" : "s"}`);
  } catch (err) { toast("Import failed: " + err.message); }
  e.target.value = "";
};

// ---------------------------------------------------------------- rendering
marked.setOptions({ gfm: true, breaks: true });
function md(text) {
  const html = DOMPurify.sanitize(marked.parse(text || ""));
  const div = document.createElement("div"); div.className = "md"; div.innerHTML = html;
  div.querySelectorAll("pre").forEach((pre) => {
    const code = pre.querySelector("code");
    const lang = (code?.className.match(/language-(\S+)/) || [])[1];
    if (lang) { const l = document.createElement("span"); l.className = "lang"; l.textContent = lang; pre.appendChild(l); }
    const b = document.createElement("button"); b.className = "copy"; b.textContent = "Copy";
    b.onclick = () => { navigator.clipboard.writeText(code ? code.innerText : pre.innerText).then(() => { b.textContent = "Copied"; setTimeout(() => b.textContent = "Copy", 1500); }); };
    pre.appendChild(b);
  });
  div.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
  return div;
}
function splitThink(text) {
  const m = text.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)$/);
  if (!m) return { think: null, rest: text, open: false };
  return { think: m[1].trim(), rest: (m[3] || "").replace(/^\s+/, ""), open: m[2] !== "</think>" };
}
function renderAssistant(bubble, text, streaming, keepOpen, reasoning, toolCalls, reasoningOpen) {
  // reasoning arrives in reasoning_content (REASONING_FORMAT=separate) or inline as <think>...</think>
  let { think, rest, open } = splitThink(text || "");
  if (reasoning != null && reasoning !== "") { think = reasoning; rest = text || ""; open = !!reasoningOpen; }
  bubble.innerHTML = "";
  if (think !== null) {
    const d = document.createElement("details"); d.className = "think";
    d.open = (streaming && open) || !!keepOpen;
    const n = think.length;
    d.innerHTML = `<summary>${open && streaming ? "Thinking…" : "Thought"} <span style="opacity:.7">(${n.toLocaleString()} chars)</span></summary>`;
    const b = document.createElement("div"); b.className = "think-body"; b.textContent = think; d.appendChild(b);
    bubble.appendChild(d);
    if (streaming && open) setTimeout(() => { b.scrollTop = b.scrollHeight; }, 0);
  }
  const body = md(rest);
  if (streaming && !(open && think !== null)) body.classList.add("cursor");
  bubble.appendChild(body);
  if (toolCalls && toolCalls.length) {
    const t = document.createElement("div"); t.className = "md";
    t.innerHTML = toolCalls.map((c) => `<pre><code>${esc(c.function?.name || c.name)}(${esc(c.function?.arguments || c.arguments || "")})</code></pre>`).join("");
    bubble.appendChild(t);
  }
}
function statText(s) {
  if (!s) return "";
  const p = [];
  if (s.completion_tokens != null) p.push(`${s.completion_tokens} tokens`);
  if (s.tok_s) p.push(`${s.tok_s.toFixed(1)} tok/s`);
  if (s.ttft_s != null) p.push(`first token ${(s.ttft_s * 1000).toFixed(0)} ms`);
  return p.join(" · ");
}
const ICON = {
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/></svg>',
  edit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
  regen: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5"/></svg>',
  del: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>',
  cont: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
};
function renderMessages() {
  const c = conv(); const inner = $("msgInner"); inner.innerHTML = "";
  if (!c || !c.messages.length) {
    const ex = ["Translate the following text into natural English:\n", "Proofread this text and explain each correction:\n", "Explain how RWKV differs from a Transformer, in terms of inference speed.", "Write Python that reads a CSV file and prints the mean of every column."];
    inner.innerHTML = `<div class="empty"><h2>How can I help?</h2><div>${esc(status.info?.model || "")}</div><div class="examples">${ex.map((e) => `<button>${esc(e.trim())}</button>`).join("")}</div></div>`;
    inner.querySelectorAll(".examples button").forEach((b, i) => b.onclick = () => { $("input").value = ex[i]; autoGrow(); $("input").focus(); updateTokHint(); });
    return;
  }
  c.messages.forEach((m, i) => inner.appendChild(renderMsg(c, m, i)));
  scrollBottom(true);
}
function renderMsg(c, m, i) {
  const el = document.createElement("div"); el.className = "msg " + m.role + (m.error ? " err" : ""); el.dataset.i = i;
  const bubble = document.createElement("div"); bubble.className = "bubble";
  if (m.role === "user") bubble.textContent = m.content;
  else if (m.error) bubble.textContent = m.content || m.error;
  else renderAssistant(bubble, m.content, false, false, m.reasoning, m.tool_calls);
  el.appendChild(bubble);
  const meta = document.createElement("div"); meta.className = "meta";
  const busy = !!gen;
  if (m.role === "assistant") meta.innerHTML = `<span class="stat">${esc(m.error ? "Error: " + m.error : statText(m.stats))}${m.finish === "length" ? " · stopped at max tokens" : ""}</span>`;
  const btn = (a, ic, t) => `<button class="icon" data-a="${a}" title="${t}" aria-label="${t}" ${busy && a !== "copy" ? "disabled" : ""}>${ic}</button>`;
  meta.innerHTML += btn("copy", ICON.copy, "Copy") + (m.role === "user" ? btn("edit", ICON.edit, "Edit and resend") : btn("regen", ICON.regen, "Regenerate") + (m.finish === "length" ? btn("cont", ICON.cont, "Continue") : "")) + btn("del", ICON.del, "Delete");
  meta.onclick = (e) => {
    const a = e.target.closest("[data-a]")?.dataset.a; if (!a) return;
    if (a === "copy") { navigator.clipboard.writeText(m.content).then(() => toast("Copied")); }
    else if (a === "del") { c.messages.splice(i, 1); touch(c); renderMessages(); }
    else if (a === "regen") { c.messages.splice(i); touch(c); renderMessages(); generate(c); }
    else if (a === "cont") { generate(c, true); }
    else if (a === "edit") { startEdit(c, i, el); }
  };
  el.appendChild(meta);
  return el;
}
function startEdit(c, i, el) {
  const m = c.messages[i];
  const box = document.createElement("div"); box.className = "edit-box";
  box.innerHTML = `<textarea></textarea><div class="row"><button class="btn" data-a="cancel">Cancel</button><button class="btn primary" data-a="ok">Send</button></div>`;
  const ta = box.querySelector("textarea"); ta.value = m.content;
  el.replaceWith(box); ta.focus();
  box.onclick = (e) => {
    const a = e.target.closest("[data-a]")?.dataset.a;
    if (a === "cancel") renderMessages();
    if (a === "ok" && ta.value.trim()) { m.content = ta.value; c.messages.splice(i + 1); touch(c); renderMessages(); generate(c); }
  };
  ta.onkeydown = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) box.querySelector('[data-a="ok"]').click(); if (e.key === "Escape") renderMessages(); };
}
function scrollBottom(force) {
  const s = $("messages");
  if (force || s.scrollHeight - s.scrollTop - s.clientHeight < 160) s.scrollTop = s.scrollHeight;
}
function renderTitle() { const c = conv(); $("title").textContent = view === "settings" ? "Settings & status" : (c ? c.title : "Chat"); }
function touch(c) { c.updated = Date.now(); saveConvs(); renderConvList(); }

// ---------------------------------------------------------------- params panel
const P = [["pTemp", "temperature", "vTemp"], ["pTopP", "top_p", "vTopP"], ["pMinP", "min_p", "vMinP"], ["pPres", "presence_penalty", "vPres"], ["pFreq", "frequency_penalty", "vFreq"]];
function params(c) { return Object.assign({}, defaults, c?.params || {}); }
function renderParams() {
  const c = conv(); const p = params(c);
  P.forEach(([id, k, v]) => { $(id).value = p[k]; $(v).textContent = (+p[k]).toFixed(2); });
  $("pTopK").value = p.top_k; $("pMax").value = p.max_tokens; $("pSystem").value = c?.system || "";
  $("pSystem").placeholder = "Empty: the server default";
  $("pEnter").checked = ui.enterSend;
  const t = c?.think || defaults.thinking || "auto";
  document.querySelectorAll("#thinkSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === t));
  $("input").placeholder = ui.enterSend ? "Message (Enter to send, Shift+Enter for a new line)" : "Message (Ctrl+Enter to send)";
}
function setParam(k, v) { let c = conv(); if (!c) { newConv(); c = conv(); } c.params = Object.assign({}, c.params || {}, { [k]: v }); saveConvs(); }
P.forEach(([id, k, v]) => $(id).addEventListener("input", (e) => { setParam(k, +e.target.value); $(v).textContent = (+e.target.value).toFixed(2); }));
$("pTopK").onchange = (e) => setParam("top_k", parseInt(e.target.value) || 0);
$("pMax").onchange = (e) => setParam("max_tokens", parseInt(e.target.value) || defaults.max_tokens);
$("pSystem").onchange = (e) => { let c = conv(); if (!c) { newConv(); c = conv(); } c.system = e.target.value; saveConvs(); updateTokHint(); };
$("pEnter").onchange = (e) => { ui.enterSend = e.target.checked; saveUi(); renderParams(); };
document.querySelectorAll("[data-preset]").forEach((b) => b.onclick = () => {
  let c = conv(); if (!c) { newConv(); c = conv(); }
  const pr = { precise: { temperature: 0, top_p: 1, top_k: 0, min_p: 0 }, balanced: { temperature: 1, top_p: 0.3, top_k: 0, min_p: 0 },
    creative: { temperature: 1.1, top_p: 0.8, top_k: 0, min_p: 0.02, presence_penalty: 0.3, frequency_penalty: 0.3 }, reset: null }[b.dataset.preset];
  c.params = pr ? Object.assign({}, c.params || {}, pr) : null; saveConvs(); renderParams(); toast(pr ? `Preset: ${b.textContent}` : "Back to the server defaults");
});
document.querySelectorAll("#thinkSeg button").forEach((b) => b.onclick = () => { let c = conv(); if (!c) { newConv(); c = conv(); } c.think = b.dataset.v; saveConvs(); renderParams(); });

// ---------------------------------------------------------------- composer
const input = $("input");
function autoGrow() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, window.innerHeight * 0.4) + "px"; }
input.addEventListener("input", () => { autoGrow(); updateTokHint(); });
input.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.isComposing) return;
  const send = ui.enterSend ? !e.shiftKey : (e.ctrlKey || e.metaKey);
  if (send) { e.preventDefault(); send_(); }
});
$("sendBtn").onclick = send_;
$("stopBtn").onclick = () => gen?.ctrl.abort();
$("attachBtn").onclick = () => $("attachFile").click();
$("attachFile").onchange = async (e) => {
  for (const f of e.target.files) {
    if (f.size > 4 << 20) { toast(`${f.name}: files over 4 MB cannot be attached`); continue; }
    const t = await f.text();
    input.value += (input.value ? "\n\n" : "") + `[${f.name}]\n${t}`;
  }
  e.target.value = ""; autoGrow(); updateTokHint(); input.focus();
};
let tokTimer = null;
function updateTokHint() {
  clearTimeout(tokTimer);
  tokTimer = setTimeout(async () => {
    const c = conv(); const msgs = buildMessages(c, input.value.trim() ? { role: "user", content: input.value } : null);
    if (!msgs.length) { $("tokHint").textContent = ""; return; }
    try {
      const r = await api("/api/tokenize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: msgs, thinking: c?.think || undefined }) });
      const d = await r.json(); $("tokHint").textContent = `prompt: ${d.tokens.toLocaleString()} tokens`;
    } catch { $("tokHint").textContent = ""; }
  }, 350);
}
function buildMessages(c, extra) {
  const out = [];
  if (c?.system?.trim()) out.push({ role: "system", content: c.system });
  (c?.messages || []).forEach((m) => { if (!m.error) out.push({ role: m.role, content: m.content }); });
  if (extra) out.push(extra);
  return out;
}
function send_() {
  if (gen) return;
  const text = input.value; if (!text.trim()) return;
  let c = conv(); if (!c) { newConv(); c = conv(); }
  c.messages.push({ role: "user", content: text });
  if (c.title === "New chat") c.title = text.trim().split("\n")[0].slice(0, 40) || "New chat";
  input.value = ""; autoGrow(); touch(c); renderTitle(); renderMessages(); generate(c);
}

// ---------------------------------------------------------------- generation
async function generate(c, cont = false) {
  if (gen) return;
  if (status.state !== "ready") { toast(status.state === "loading" ? "The engine is still loading, please wait" : "The engine is not running (see Settings & status)"); return; }
  const p = params(c);
  let msg;
  if (cont && c.messages.length && c.messages[c.messages.length - 1].role === "assistant") { msg = c.messages[c.messages.length - 1]; msg.finish = null; }
  else { msg = { role: "assistant", content: "", stats: null }; c.messages.push(msg); }
  const messages = buildMessages(c).filter((m, i, a) => !(i === a.length - 1 && m.role === "assistant" && !m.content));
  renderMessages();
  const el = $("msgInner").lastElementChild; const bubble = el.querySelector(".bubble");
  const ctrl = new AbortController(); gen = { ctrl, convId: c.id, msg };
  $("stopBtn").hidden = false; $("sendBtn").disabled = true;
  const think = c.think === "on" ? true : c.think === "off" ? false : undefined;
  const body = { model: "rwkv", stream: true, messages, thinking: think, temperature: p.temperature, top_p: p.top_p, top_k: p.top_k, min_p: p.min_p,
    presence_penalty: p.presence_penalty, frequency_penalty: p.frequency_penalty, max_tokens: p.max_tokens };
  if (cont && msg.content) body.continue_final_message = true;
  const t0 = performance.now(); let n = 0, tFirst = null, lastDraw = 0;
  const base = msg.content;
  let text = "";
  let reasoning = msg.reasoning || "";
  const calls = [];
  const draw = (final) => { renderAssistant(bubble, base + text, !final, false, reasoning, calls, !final && !text); scrollBottom(); };
  try {
    const r = await api("/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
    const rd = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await rd.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, k).trim(); buf = buf.slice(k + 2);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim(); if (data === "[DONE]") continue;
        const d = JSON.parse(data);
        if (d.error) throw new Error(d.error.message);
        const ch = d.choices?.[0];
        if (ch?.delta?.content) { text += ch.delta.content; n++; if (tFirst == null) tFirst = performance.now(); }
        if (ch?.delta?.reasoning_content) { reasoning += ch.delta.reasoning_content; n++; if (tFirst == null) tFirst = performance.now(); }
        for (const tc of ch?.delta?.tool_calls || []) calls.push(tc);
        if (ch?.finish_reason) msg.finish = ch.finish_reason;
        if (d.usage) msg.stats = Object.assign({}, d.usage, d.rwkv || {});
        const now = performance.now();
        if (now - lastDraw > 60) { draw(false); lastDraw = now; $("speedPill").hidden = false; $("speedPill").textContent = tFirst ? `${(n / ((now - tFirst) / 1000 || 1)).toFixed(0)} chunk/s` : "…"; }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") { msg.error = e.message; }
    else msg.finish = msg.finish || "stopped";
  } finally {
    msg.content = base + text;
    if (reasoning) msg.reasoning = reasoning;
    if (calls.length) msg.tool_calls = calls;
    if (!msg.content && !msg.error && msg.finish === "stopped") c.messages.splice(c.messages.indexOf(msg), 1);
    gen = null; $("stopBtn").hidden = true; $("sendBtn").disabled = false; $("speedPill").hidden = true;
    touch(c);
    if (activeId === c.id) renderMessages();
    updateTokHint();
  }
}

// ---------------------------------------------------------------- status
async function poll() {
  try {
    const r = await api("/api/status"); status = await r.json();
    if (status.defaults) defaults = status.defaults;
    const pill = $("engPill"); pill.className = "pill " + status.state;
    const i = status.info || {};
    const label = { ready: `${status.model_name}${i.tp ? ` · TP${i.tp}` : ""}${i.quant ? ` · ${i.quant}` : ""}`, loading: "Loading…", stopped: "Stopped", error: "Error" }[status.state] || status.state;
    $("engText").textContent = label;
    pill.title = status.error || "";
    const s = status.stats || {};
    if (!gen && status.state === "ready" && s.tok_s > 0.5) { $("speedPill").hidden = false; $("speedPill").textContent = `${s.tok_s.toFixed(0)} tok/s · ${s.active} active`; }
    else if (!gen) $("speedPill").hidden = true;
    if (view === "settings") renderStatus();
  } catch { $("engPill").className = "pill error"; $("engText").textContent = "Server unreachable"; }
  setTimeout(poll, view === "settings" ? 1500 : 2500);
}
function renderStatus() {
  const s = status.stats || {}, i = status.info || {};
  const rows = [["State", { ready: "Running", loading: "Loading", stopped: "Stopped", error: "Error" }[status.state] || status.state],
    ["Model", i.model || "—"], ["Chat template", status.chat_template || "—"], ["MTP", i.mtp || "off"], ["Layout", i.tp ? `TP ${i.tp} · ${i.quant} · L${i.layers} · C${i.n_embd} · ${i.heads} heads` : "—"],
    ["VRAM(rank 0)", i.vram_gb ? `${i.vram_gb.toFixed(2)} GB` : "—"], ["CUDA Graph", i.graph_sizes ? i.graph_sizes.join(", ") : "—"],
    ["Decode", s.tok_s != null ? `${s.tok_s.toFixed(1)} tok/s (${(s.steps_s || 0).toFixed(1)} steps/s${s.mtp_tok_per_step ? `, ${s.mtp_tok_per_step.toFixed(2)} tok/step` : ""})` : "—"],
    ["prefill", s.prefill_tok_s != null ? `${s.prefill_tok_s.toFixed(0)} tok/s` : "—"],
    ["Requests", s.active != null ? `${s.active} / ${s.slots} slots (decoding ${s.decoding}, prefill ${s.prefilling}, queued ${s.waiting})` : "—"],
    ["Uptime", status.uptime_s ? `${Math.floor(status.uptime_s / 60)} min` : "—"]];
  if (status.pending_restart) rows.push(["", "⚠ Saved settings are not applied yet (restart the engine)"]);
  if (status.pending_server_restart) rows.push(["", "⚠ Host / port changes apply when the server process is restarted (python -m rwkv_next)"]);
  $("statusKv").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("statusErr").hidden = !status.error; $("statusErr").textContent = status.error || "";
  $("engLog").textContent = (status.log || []).join("\n");
}
$("restartBtn").onclick = async () => { if (!confirm("Restart the engine? Running generations are cancelled.")) return; const r = await api("/api/engine/restart", { method: "POST" }); if (!r.ok) toast((await r.json()).detail || "Failed"); else toast("Restarting…"); };
$("stopEngineBtn").onclick = async () => { if (!confirm("Stop the engine?")) return; const r = await api("/api/engine/stop", { method: "POST" }); if (!r.ok) toast((await r.json()).detail || "Failed"); };

// ---------------------------------------------------------------- settings form
let schema = null, edits = {};
async function loadSettings() {
  const r = await api("/api/settings"); schema = await r.json(); edits = {}; renderSettings();
}
function renderSettings() {
  const form = $("settingsForm"); form.innerHTML = "";
  for (const g of schema.groups) {
    const items = schema.settings.filter((s) => s.group === g.id); if (!items.length) continue;
    const card = document.createElement("div"); card.className = "card";
    card.innerHTML = `<h2>${esc(g.label)}</h2><div class="grid2"></div>`;
    const grid = card.querySelector(".grid2");
    for (const s of items) grid.appendChild(settingField(s));
    form.appendChild(card);
  }
  const f = document.createElement("div"); f.className = "hint"; f.textContent = `Saved to: ${schema.env_file}`; form.appendChild(f);
  updateSaveBar();
}
function settingField(s) {
  const wrap = document.createElement("div"); wrap.className = "field set"; wrap.dataset.key = s.key;
  if (s.type === "text") wrap.style.gridColumn = "1 / -1";
  const id = "set_" + s.key;
  const badges = (s.restart ? '<span class="badge" title="Applied when the engine restarts">restart</span>' : "") + (s.server ? '<span class="badge" title="Applied when the server process is restarted">server restart</span>' : "") + (s.locked ? '<span class="badge lock" title="Fixed by an environment variable at launch">env</span>' : "");
  wrap.innerHTML = `<label for="${id}"><span>${esc(s.label)} ${badges}</span><span class="k">RWKV_${esc(s.key)}</span></label>`;
  let el;
  if (s.type === "choice") { el = document.createElement("select"); s.choices.forEach((c) => { const o = document.createElement("option"); o.value = c; o.textContent = c; el.appendChild(o); }); el.value = s.value; }
  else if (s.type === "bool") { el = document.createElement("select"); [["true", "On"], ["false", "Off"]].forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; el.appendChild(o); }); el.value = String(!!s.value); }
  else if (s.type === "text") { el = document.createElement("textarea"); el.value = s.value; }
  else { el = document.createElement("input"); el.type = s.secret ? "password" : (s.type === "int" || s.type === "float") ? "number" : "text"; if (s.type === "float") el.step = "any"; el.value = s.secret ? "" : s.value; if (s.secret && s.is_set) el.placeholder = "(set; type only to change it)"; }
  el.id = id; el.disabled = s.locked;
  el.addEventListener("input", () => {
    let v = el.value;
    if (s.type === "bool") v = v === "true"; else if (s.type === "int") v = parseInt(v); else if (s.type === "float") v = parseFloat(v);
    const orig = s.type === "bool" ? !!s.value : s.value;
    if (v === orig || (s.secret && v === "")) delete edits[s.key]; else edits[s.key] = v;
    wrap.classList.toggle("changed", s.key in edits); updateSaveBar();
  });
  el.addEventListener("change", () => el.dispatchEvent(new Event("input")));
  if (s.key === "MODEL_PATH" || s.key === "MTP_PATH") {
    const row = document.createElement("div"); row.style.display = "flex"; row.style.gap = "6px";
    row.appendChild(el); const b = document.createElement("button"); b.className = "btn"; b.textContent = "Browse"; b.type = "button"; b.disabled = s.locked;
    b.onclick = () => openPicker(el.value, (p) => { el.value = p; el.dispatchEvent(new Event("input")); });
    row.appendChild(b); wrap.appendChild(row);
  } else wrap.appendChild(el);
  if (s.help) { const h = document.createElement("div"); h.className = "help"; h.textContent = s.help; wrap.appendChild(h); }
  return wrap;
}
function updateSaveBar() {
  const keys = Object.keys(edits);
  const needRestart = keys.some((k) => schema.settings.find((s) => s.key === k)?.restart);
  $("saveHint").textContent = keys.length ? `${keys.length} change${keys.length === 1 ? "" : "s"}${needRestart ? " (some need an engine restart)" : " (applied immediately)"}` : "No changes";
  ["saveBtn", "revertBtn", "saveRestartBtn"].forEach((id) => $(id).disabled = !keys.length);
  $("saveRestartBtn").hidden = keys.length && !needRestart;
}
async function save(restart) {
  const r = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values: edits, restart }) });
  const d = await r.json();
  if (!r.ok) { toast(d.detail || "Could not save", 4000); return; }
  toast((d.restarting ? "Saved. Restarting the engine…" : d.restart ? "Saved (applied at the next restart)" : "Saved") + (d.server_restart ? " · host / port apply after a server restart" : ""));
  await loadSettings(); poll();
}
$("saveBtn").onclick = () => save(false);
$("saveRestartBtn").onclick = () => { if (confirm("Save and restart the engine? Running generations are cancelled.")) save(true); };
$("revertBtn").onclick = () => { edits = {}; renderSettings(); };

async function openPicker(start, cb) {
  const go = async (dir) => {
    const r = await api("/api/files?dir=" + encodeURIComponent(dir || ""));
    const d = await r.json(); if (!r.ok) { toast(d.detail || "Cannot open"); return; }
    $("pickHead").textContent = d.dir;
    const list = $("pickList"); list.innerHTML = "";
    const up = document.createElement("div"); up.className = "ent"; up.textContent = "↑ Up"; up.onclick = () => go(d.parent); list.appendChild(up);
    d.entries.forEach((e) => {
      const el = document.createElement("div"); el.className = "ent";
      el.innerHTML = `<span>${e.dir ? "📁 " : ""}${esc(e.name)}</span><span class="hint">${e.model ? '<button class="btn" data-use="1">Use this model</button>' : e.dir ? "" : (e.size / 1e9).toFixed(2) + " GB"}</span>`;
      el.onclick = (ev) => { if (ev.target.closest("[data-use]")) { cb(e.path); $("picker").hidden = true; return; } e.dir ? go(e.path) : (cb(e.path), $("picker").hidden = true); };
      list.appendChild(el);
    });
  };
  $("picker").hidden = false;
  go(start ? start.replace(/\/[^/]*$/, "") : "");
}
$("pickClose").onclick = () => $("picker").hidden = true;

// ---------------------------------------------------------------- views
function setView(v) {
  view = v;
  $("viewChat").hidden = v !== "chat"; $("viewSettings").hidden = v !== "settings";
  $("navChat").classList.toggle("active", v === "chat"); $("navSettings").classList.toggle("active", v === "settings");
  $("paramsToggle").hidden = v !== "chat";
  renderTitle(); applyLayout();
  if (v === "settings") { loadSettings(); renderStatus(); }
}
$("navChat").onclick = () => setView("chat");
$("navSettings").onclick = () => setView("settings");
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newConv(); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") { e.preventDefault(); ui.side = !ui.side; saveUi(); applyLayout(); }
  if (e.key === "Escape" && gen) gen.ctrl.abort();
});
function renderAll() { renderConvList(); renderTitle(); renderMessages(); renderParams(); updateTokHint(); }

applyTheme(); applyLayout();
if (!conv() && convs.length) activeId = convs[0].id;
renderAll(); setView("chat"); poll();
