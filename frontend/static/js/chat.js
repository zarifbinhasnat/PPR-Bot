// ============================================================
// PPR-Bot · chat client (vanilla JS, no framework)
// ------------------------------------------------------------
// Drives the KakaoTalk-style messenger UI and streams answers
// from the real backend.
//
// Teaching note on SSE: the browser ships a built-in `EventSource`
// for Server-Sent Events, but it ONLY does GET and can't send a
// JSON body. Our /chat endpoint is a POST (message + session_id in
// the body), so we use `fetch()` with a streaming response reader
// and parse the SSE wire format ourselves. The wire format is
// simple: frames are separated by a blank line, and each frame has
// lines like `event: token` and `data: {...}`.
//
// Backend event sequence (see chat/orchestrator.py):
//   query     -> {query}                     retrieval is starting
//   citations -> {citations:[{breadcrumb, source_pages, rerank_score}]}
//   token     -> {text}                      one streamed answer delta
//   done      -> {}                          turn complete
// ============================================================

const chatEl = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const infoBtn = document.getElementById("info-btn");
const sheetRoot = document.getElementById("sheet-root");
const statusLine = document.getElementById("status-line");

// A stable id so the backend treats this tab as one ongoing conversation.
const sessionId = "web-" + Math.random().toString(36).slice(2);

// Cross-encoder reranking toggle. ON = more precise ordering but slow on CPU;
// OFF = much faster (hybrid search only). Persisted across reloads. Sent to
// the backend with every /chat request.
let rerankEnabled = localStorage.getItem("ppr_rerank") !== "false";

// Starter suggestions shown under the welcome message. Each chip just sends
// its label as a real query.
const QUICK_REPLIES = [
  { label: "Procurement thresholds" },
  { label: "Direct procurement rules" },
  { label: "দরপত্র পদ্ধতি", bangla: true },
];

// ── Helpers ────────────────────────────────────────────────────────────────

// Detect Bengali script so we can switch to the Bangla font + looser leading.
const BANGLA_RE = /[ঀ-৿]/;
function isBangla(text) {
  return BANGLA_RE.test(text || "");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Minimal, safe Markdown -> HTML for bot answers. We escape everything first
// (so model output can't inject HTML), then render the small subset of
// Markdown the model actually emits: bullet lists (`* ` / `- `), **bold**, and
// *italic*. Line structure + indentation is preserved by the bubble's
// `white-space: pre-wrap`.
function formatInline(s) {
  // Bold FIRST so its `**` pairs are consumed before the single-`*` italic
  // pass runs (otherwise italic would chew into bold markers).
  return s
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\*(\S.*?\S|\S)\*/g, "<i>$1</i>");
}

function formatText(text) {
  return escapeHtml(text)
    .split("\n")
    .map((line) => {
      // Bullet line: optional indent, then `*` or `-`, then a space. Turn the
      // marker into a real bullet glyph; keep the indent for nested levels.
      const m = line.match(/^(\s*)[*-]\s+(.*)$/);
      if (m) return m[1] + "• " + formatInline(m[2]);
      return formatInline(line);
    })
    .join("\n");
}

function nowTime() {
  return new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

// bge-reranker returns raw logits, not 0..1. Squash with a sigmoid so the
// green "grounded %" badge is an honest relevance probability.
function scoreToPercent(score) {
  if (score == null || Number.isNaN(score)) return null;
  const p = 1 / (1 + Math.exp(-score));
  return Math.round(p * 100);
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ── Message rendering ────────────────────────────────────────────────────────

function addSystemPill(text) {
  const row = document.createElement("div");
  row.className = "row row--system";
  const pill = document.createElement("span");
  pill.className = "system-pill";
  pill.textContent = text;
  row.appendChild(pill);
  chatEl.appendChild(row);
  scrollToBottom();
  return row;
}

// Create a chat row with a bubble (+ trailing timestamp). Returns the parts so
// callers can mutate the bubble (streaming) or append footers (citations).
function addBubble(from, text, { bangla = false } = {}) {
  const row = document.createElement("div");
  row.className = "row row--" + from;

  const line = document.createElement("div");
  line.className = "bubble-line";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const bn = bangla || isBangla(text);
  if (bn) bubble.setAttribute("lang", "bn");
  if (text) bubble.innerHTML = formatText(text);

  const time = document.createElement("span");
  time.className = "bubble__time";
  time.textContent = nowTime();

  line.appendChild(bubble);
  line.appendChild(time);
  row.appendChild(line);
  chatEl.appendChild(row);
  scrollToBottom();
  return { row, bubble, time };
}

// Set bubble text during streaming, keeping the Bangla font in sync.
function setBubbleText(bubble, text) {
  if (isBangla(text)) bubble.setAttribute("lang", "bn");
  bubble.innerHTML = formatText(text);
}

function addQuickReplies(row, items) {
  const wrap = document.createElement("div");
  wrap.className = "quick-replies";
  items.forEach((it) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = it.label;
    if (it.bangla || isBangla(it.label)) chip.setAttribute("lang", "bn");
    chip.addEventListener("click", () => {
      wrap.remove(); // suggestions are one-shot
      submit(it.label);
    });
    wrap.appendChild(chip);
  });
  row.appendChild(wrap);
  scrollToBottom();
}

// The "N sources from the Gazette" disclosure button inside a bot bubble.
function addCitationToggle(bubble, citations) {
  if (!citations || !citations.length) return;
  const btn = document.createElement("button");
  btn.className = "cite-toggle";
  btn.type = "button";
  btn.innerHTML =
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none">' +
    '<path d="M6 2h9l5 5v15H6V2z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>' +
    '<path d="M14 2v6h6" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>' +
    "<span>" + citations.length + " source" + (citations.length > 1 ? "s" : "") +
    " from the Gazette</span>" +
    '<span class="chev"><svg width="13" height="13" viewBox="0 0 24 24" fill="none">' +
    '<path d="M9 6l6 6-6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></span>';
  btn.style.color = "var(--ink-500)";
  btn.addEventListener("click", () => openSourcesSheet(citations));
  bubble.appendChild(btn);
  scrollToBottom();
}

// ── Typing indicator ─────────────────────────────────────────────────────────

let typingEl = null;
function showTyping(label) {
  hideTyping();
  typingEl = document.createElement("div");
  typingEl.className = "typing";
  typingEl.innerHTML =
    '<div class="typing__dots"><span></span><span></span><span></span></div>' +
    '<span class="typing__label"></span>';
  typingEl.querySelector(".typing__label").textContent = label || "";
  chatEl.appendChild(typingEl);
  scrollToBottom();
}
function setTypingLabel(label) {
  if (typingEl) typingEl.querySelector(".typing__label").textContent = label || "";
}
function hideTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

// ── Bottom sheets ────────────────────────────────────────────────────────────

function openSheet(buildInner) {
  closeSheet();
  const scrim = document.createElement("div");
  scrim.className = "sheet-scrim";
  scrim.addEventListener("click", (e) => { if (e.target === scrim) closeSheet(); });

  const sheet = document.createElement("div");
  sheet.className = "sheet";
  buildInner(sheet);

  scrim.appendChild(sheet);
  sheetRoot.appendChild(scrim);
}
function closeSheet() { sheetRoot.innerHTML = ""; }

function openSourcesSheet(citations) {
  openSheet((sheet) => {
    const grip = document.createElement("div");
    grip.className = "sheet__grip";

    const head = document.createElement("div");
    head.className = "sheet__head";
    head.innerHTML =
      '<div class="sheet__title">Sources from the Gazette</div>' +
      '<button class="icon-btn" aria-label="Close">' +
      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg></button>';
    head.querySelector("button").addEventListener("click", closeSheet);

    const list = document.createElement("div");
    list.className = "sheet__list";
    citations.forEach((c) => list.appendChild(buildCitationCard(c)));

    const note = document.createElement("div");
    note.className = "sheet__note";
    note.innerHTML =
      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M6 2h9l5 5v15H6V2z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M14 2v6h6" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>' +
      "Bangladesh Gazette · Public Procurement Rules 2025";

    sheet.appendChild(grip);
    sheet.appendChild(head);
    sheet.appendChild(list);
    sheet.appendChild(note);
  });
}

function buildCitationCard(c) {
  const crumb = c.breadcrumb || "Gazette";
  const pages = c.source_pages || [];
  const pct = scoreToPercent(c.rerank_score);
  const bn = isBangla(crumb);

  const card = document.createElement("div");
  card.className = "citation";

  const body = document.createElement("div");
  body.className = "citation__body";

  const crumbRow = document.createElement("div");
  crumbRow.className = "citation__crumb";
  crumbRow.innerHTML =
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" style="flex:0 0 13px">' +
    '<path d="M6 3h9l5 5v13H6V3z" stroke="var(--seal-500)" stroke-width="2" stroke-linejoin="round"/>' +
    '<path d="M14 3v6h6" stroke="var(--seal-500)" stroke-width="2" stroke-linejoin="round"/></svg>';
  const crumbText = document.createElement("span");
  crumbText.className = "citation__crumb-text";
  crumbText.textContent = crumb;
  if (bn) crumbText.setAttribute("lang", "bn");
  crumbRow.appendChild(crumbText);

  const meta = document.createElement("div");
  meta.className = "citation__meta";
  const pagesEl = document.createElement("span");
  pagesEl.textContent = pages.length ? "পৃ. " + pages.join(", ") : "Gazette";
  meta.appendChild(pagesEl);
  if (pct != null) {
    const score = document.createElement("span");
    score.className = "citation__score";
    score.innerHTML = '<span class="dot"></span>' + pct + "%";
    meta.appendChild(score);
  }

  body.appendChild(crumbRow);
  body.appendChild(meta);
  card.appendChild(body);
  return card;
}

function openInfoSheet() {
  openSheet((sheet) => {
    sheet.innerHTML =
      '<div class="sheet__grip"></div>' +
      '<div class="info-id">' +
      '<span class="avatar avatar--lg"><svg viewBox="0 0 48 48" fill="none">' +
      '<path d="M11 12h26a6 6 0 0 1 6 6v13a6 6 0 0 1-6 6H23l-10 7v-7h-2a6 6 0 0 1-6-6V18a6 6 0 0 1 6-6z" fill="var(--ink-900)"/>' +
      '<path d="M18 25l5 5 10-12" stroke="var(--brand)" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg></span>' +
      '<div><div class="info-id__name">PPR-Bot</div>' +
      '<div class="info-id__sub">Public Procurement Rules 2025</div></div></div>';

    const rows = [
      ["Knowledge base", "Bangladesh Gazette, September 28 2023"],
      ["Rules version", "PPR 2025 (Bangla)"],
      ["Language", "Bangla + English (bilingual)"],
      ["Retrieval", "Hybrid (dense + BM25) + reranking"],
    ];
    rows.forEach(([k, v]) => {
      const row = document.createElement("div");
      row.className = "info-row";
      row.innerHTML =
        '<span class="info-row__k">' + escapeHtml(k) + "</span>" +
        '<span class="info-row__v">' + escapeHtml(v) + "</span>";
      sheet.appendChild(row);
    });

    // --- Reranking toggle ---------------------------------------------------
    const toggleRow = document.createElement("div");
    toggleRow.className = "toggle-row";
    const txt = document.createElement("div");
    txt.innerHTML =
      '<div class="toggle-row__title">Precise reranking</div>' +
      '<div class="toggle-row__sub">More accurate ordering, but slower on CPU. ' +
      "Turn off for faster answers.</div>";
    const sw = document.createElement("button");
    sw.type = "button";
    sw.className = "switch" + (rerankEnabled ? " is-on" : "");
    sw.setAttribute("role", "switch");
    sw.setAttribute("aria-checked", String(rerankEnabled));
    sw.innerHTML = '<span class="switch__knob"></span>';
    sw.addEventListener("click", () => {
      rerankEnabled = !rerankEnabled;
      localStorage.setItem("ppr_rerank", String(rerankEnabled));
      sw.classList.toggle("is-on", rerankEnabled);
      sw.setAttribute("aria-checked", String(rerankEnabled));
    });
    toggleRow.appendChild(txt);
    toggleRow.appendChild(sw);
    sheet.appendChild(toggleRow);

    const close = document.createElement("button");
    close.className = "sheet__close";
    close.textContent = "Close";
    close.addEventListener("click", closeSheet);
    sheet.appendChild(close);
  });
}

// ── Send / stream ────────────────────────────────────────────────────────────

// Parse a raw SSE buffer into complete {event, data} frames, returning any
// leftover partial text to carry into the next read.
function parseSSE(buffer, onMessage) {
  // Frames are separated by a blank line. Be CRLF-tolerant: the server
  // (sse-starlette) emits "\r\n\r\n" separators, which a plain "\n\n" split
  // would NOT match — silently dropping every event. Split on \r?\n\r?\n,
  // and parse each frame's lines on \r?\n.
  const parts = buffer.split(/\r?\n\r?\n/);
  const remainder = parts.pop(); // last piece may be incomplete
  for (const part of parts) {
    let event = "message";
    let data = "";
    for (const line of part.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (data) onMessage(event, data);
  }
  return remainder;
}

let busy = false;

async function ask(message) {
  addBubble("user", message);
  showTyping("Searching the Gazette…");

  let botRow = null;
  let botBubble = null;
  let answer = "";
  let citations = null;

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, rerank: rerankEnabled }),
  });

  if (!response.ok || !response.body) {
    hideTyping();
    addBubble("bot", "Something went wrong reaching the server (" + response.status + "). Please try again.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSSE(buffer, (event, data) => {
      let payload;
      try { payload = JSON.parse(data); } catch { return; }

      if (event === "query") {
        setTypingLabel("Searching the Gazette…");
      } else if (event === "citations") {
        citations = payload.citations || [];
        setTypingLabel("Generating answer…");
      } else if (event === "token") {
        if (!botBubble) {
          hideTyping();
          const parts = addBubble("bot", "");
          botRow = parts.row;
          botBubble = parts.bubble;
        }
        answer += payload.text;
        setBubbleText(botBubble, answer);
        scrollToBottom();
      } else if (event === "suggestions") {
        // Backend-proposed follow-up questions -> tappable chips under the
        // bot's answer (the design's "quick replies").
        if (!botRow) {
          const parts = addBubble("bot", answer || "…");
          botRow = parts.row;
          botBubble = parts.bubble;
        }
        if (payload.suggestions && payload.suggestions.length) {
          addQuickReplies(botRow, payload.suggestions.map((label) => ({ label })));
        }
      } else if (event === "done") {
        hideTyping();
        if (!botBubble) {
          const parts = addBubble("bot", answer || "…");
          botRow = parts.row;
          botBubble = parts.bubble;
        }
        if (citations && citations.length) addCitationToggle(botBubble, citations);
      }
    });
  }

  // Safety net if the stream ended without an explicit `done`.
  hideTyping();
}

// Single entry point used by both the form and the quick-reply chips.
async function submit(message) {
  const text = (message || "").trim();
  if (!text || busy) return;
  busy = true;
  input.value = "";
  updateSendState();
  try {
    await ask(text);
  } catch (err) {
    hideTyping();
    addBubble("bot", "Network error: " + err.message);
  } finally {
    busy = false;
    input.focus();
  }
}

// ── Composer wiring ──────────────────────────────────────────────────────────

function updateSendState() {
  const hasText = input.value.trim().length > 0;
  sendBtn.classList.toggle("is-active", hasText && !busy);
  // Reflect Bangla typing in the input font.
  if (isBangla(input.value)) input.setAttribute("lang", "bn");
  else input.removeAttribute("lang");
}

input.addEventListener("input", updateSendState);
form.addEventListener("submit", (e) => {
  e.preventDefault();
  submit(input.value);
});
infoBtn.addEventListener("click", openInfoSheet);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSheet(); });

// ── Boot ─────────────────────────────────────────────────────────────────────

function showWelcome() {
  addSystemPill(
    "Today · " +
      new Date().toLocaleDateString(undefined, { month: "short", day: "numeric" })
  );
  const { row, bubble } = addBubble(
    "bot",
    "আমি PPR-Bot — পাবলিক প্রকিউরমেন্ট বিধিমালা, ২০২৫ সম্পর্কে আপনার সহকারী।\n\nAsk me anything about the Public Procurement Rules — in Bangla or English.",
    { bangla: true }
  );
  addQuickReplies(row, QUICK_REPLIES);
}

// Reflect real backend readiness in the header status line.
async function checkReadiness() {
  try {
    const r = await fetch("/readiness");
    const { ready } = await r.json();
    if (!ready) {
      statusLine.textContent = "● Indexing… answers limited";
      statusLine.classList.add("is-offline");
    }
  } catch {
    statusLine.textContent = "● Offline";
    statusLine.classList.add("is-offline");
  }
}

showWelcome();
updateSendState();
checkReadiness();
