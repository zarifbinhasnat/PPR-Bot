// Frontend chat client.
//
// Teaching note on SSE: the browser ships a built-in `EventSource` for SSE,
// but it ONLY supports GET requests and can't send a JSON body. Our /chat
// endpoint is a POST (it needs the message + session_id in the body), so we
// use `fetch()` with a streaming response reader and parse the SSE wire
// format ourselves. The wire format is simple: messages are separated by a
// blank line, and each has lines like `event: token` and `data: {...}`.

const chatEl = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

// A stable id so the backend treats this tab as one ongoing conversation.
const sessionId = "web-" + Math.random().toString(36).slice(2);

function addMessage(cls, text = "") {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.textContent = text;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

function addStatus(text) {
  const el = document.createElement("div");
  el.className = "status";
  el.textContent = text;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

// Parse a raw SSE buffer into complete {event, data} messages, returning any
// leftover partial text to be carried into the next read.
function parseSSE(buffer, onMessage) {
  const parts = buffer.split("\n\n");
  const remainder = parts.pop(); // last piece may be incomplete
  for (const part of parts) {
    let event = "message";
    let data = "";
    for (const line of part.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (data) onMessage(event, data);
  }
  return remainder;
}

async function ask(message) {
  addMessage("user", message);
  const status = addStatus("Thinking…");
  let botEl = null;

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSSE(buffer, (event, data) => {
      const payload = JSON.parse(data);
      if (event === "query") {
        status.textContent = "Searching the rules…";
      } else if (event === "citations") {
        status.remove();
        botEl = addMessage("bot", "");
        botEl._citations = payload.citations;
      } else if (event === "token") {
        if (!botEl) botEl = addMessage("bot", "");
        botEl.textContent += payload.text;
        chatEl.scrollTop = chatEl.scrollHeight;
      } else if (event === "done") {
        if (botEl && botEl._citations) renderCitations(botEl, botEl._citations);
      }
    });
  }
}

function renderCitations(botEl, citations) {
  if (!citations || !citations.length) return;
  const box = document.createElement("div");
  box.className = "citations";
  const lines = citations
    .map((c) => `• ${c.breadcrumb || "?"} (p. ${(c.source_pages || []).join(", ")})`)
    .join("\n");
  box.innerHTML = "<b>Sources</b>\n" + lines;
  box.style.whiteSpace = "pre-wrap";
  botEl.appendChild(box);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  sendBtn.disabled = true;
  try {
    await ask(message);
  } catch (err) {
    addStatus("Error: " + err.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
