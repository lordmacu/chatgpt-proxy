"""
FastAPI proxy — OpenAI-compatible API backed by the ChatGPT Android anonymous flow.

Available endpoints:
  POST /v1/chat/completions   — chat (streaming + non-streaming, multipart content, files)
  GET  /v1/models             — list of models available in anonymous mode
  POST /v1/files              — upload a file (PDF, code, docs) → file_id
  GET  /v1/files              — list uploaded files
  GET  /v1/files/{file_id}    — file info
  DEL  /v1/files/{file_id}    — delete a file
  GET  /v1/session/me         — anonymous session info (user id, device id)
  GET  /health                — server status

Anonymous flow capabilities:
  ✅ Text chat with GPT-5.5 / 5.6 Luna / 5.3 Mini / auto
  ✅ Streaming SSE
  ✅ Multi-turn (within session, ~30 min TTL)
  ✅ System prompt (persistent in session, injected on first turn only)
  ✅ Text attachments: PDF, code, docs (82 MIME types, retrieval mode)
  ✅ Web search — auto, or controlled with web_search: true/false
  ✅ Advanced tools — reservations, shopping, genui widgets (force_use_tools: true)
  ✅ Canvas (collaborative documents) — force_use_canvas: true
  ✅ JSON output — response_format: {"type": "json_object"}
  ✅ Quota exhaustion: auto-recycles device_id on 429/403, transparent retry
  ❌ Image input (vision) — not available in anonymous mode
  ❌ Image generation — not available in anonymous mode
  ❌ Function calling / tool_calls in response — not supported by anonymous backend
  ❌ Voice / TTS — requires an account

OpenAI `tools` field compatibility (input-only):
  The `tools` array is accepted for compatibility, but the anonymous backend never
  returns tool_calls. Function names are mapped to internal backend flags:
    • "web_search" / "search" / ...       → force_use_search = true
    • "chatgpt_tools" / "all_tools" / ... → force_use_tools = true
    • any other name                       → force_use_tools = true
    • tool_choice: "none"                  → disables both modes
  The response is always text in choices[0].message.content, never tool_calls.
"""
import io
import uuid
import json
import time
from typing import Optional, AsyncGenerator, Union, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from chatgpt_client import (
    SessionPool, fetch_anon_models, BASE, _base_headers,
    QuotaExceededError, _extract_json,
)
import httpx as _httpx

# ---------------------------------------------------------------------------
# Per-user stores  {user_id: {file_id: {...}}}  /  {user_id: SessionPool}
# ---------------------------------------------------------------------------
_files: dict[str, dict[str, dict]] = {}
_pools: dict[str, SessionPool] = {}

# Model list cache: (timestamp, list)
_models_cache: tuple[float, list] = (0.0, [])
_MODELS_TTL = 300  # refresh every 5 minutes


def _get_user_id(request: Request) -> str:
    """
    Extract the user identifier from the Authorization header.
    Expected format: 'Bearer <token>'  (same as OpenAI).
    Falls back to 'anonymous' — all share the same namespace.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    return "anonymous"


def _user_files(user_id: str) -> dict[str, dict]:
    """Return (or create) the file dict for a user."""
    if user_id not in _files:
        _files[user_id] = {}
    return _files[user_id]


def _user_pool(user_id: str) -> SessionPool:
    """Return (or create) the SessionPool for a user."""
    if user_id not in _pools:
        _pools[user_id] = SessionPool()
    return _pools[user_id]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ContentPart(BaseModel):
    """A content fragment inside a multipart message."""
    type:      str                    # "text" | "file" | "image_url"
    text:      Optional[str] = None
    file:      Optional[dict] = None  # {"file_id": "file-xxx"} or {"filename": "...", "content": "..."}
    image_url: Optional[dict] = None  # {"url": "..."} — not supported in anonymous mode


class Message(BaseModel):
    role:    str
    content: Union[str, List[ContentPart]] = ""
    name:    Optional[str] = None


# Built-in tool names that this proxy recognises in the OpenAI tools array.
# Any other name activates force_use_tools (the backend's internal tools).
_BUILTIN_TOOL_WEB_SEARCH = {"web_search", "search", "brave_search", "bing_search", "google_search"}
_BUILTIN_TOOL_ALL        = {"chatgpt_tools", "all_tools", "builtin_tools"}


class ChatCompletionRequest(BaseModel):
    model:    str  = "auto"
    messages: List[Message]
    stream:   bool = False

    # ── Standard OpenAI format ─────────────────────────────────────────────────
    # response_format: {"type": "text"} (default) or {"type": "json_object"}
    response_format: Optional[dict] = None

    # tools: list of tools (OpenAI format). Custom functions are ignored by the
    # anonymous backend, but their names control which mode is activated in ChatGPT:
    #   • "web_search" / "search" / ...  → force_use_search = true
    #   • "chatgpt_tools" / "all_tools"  → force_use_tools = true
    #   • any other name                 → force_use_tools = true
    #
    # tool_choice: "none" disables all modes; "auto"/"required" enables them.
    tools:       Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None

    # ── Proxy-specific extensions ──────────────────────────────────────────────
    # web_search: true/false/null — direct override of force_use_search
    web_search:       Optional[bool] = Field(default=None)
    # force_use_tools: true/null — direct override of force_use_tools
    force_use_tools:  Optional[bool] = Field(default=None)
    # force_use_canvas: true/null — enable Canvas mode (collaborative documents)
    force_use_canvas: Optional[bool] = Field(default=None)

    # ── OpenAI compatibility fields (accepted, ignored) ────────────────────────
    temperature:       Optional[float] = None
    max_tokens:        Optional[int]   = None
    top_p:             Optional[float] = None
    presence_penalty:  Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop:              Optional[Union[str, List[str]]] = None
    seed:              Optional[int]   = None
    n:                 Optional[int]   = None

    model_config = {"populate_by_name": True}

    def is_json_mode(self) -> bool:
        """True when the client requested JSON output."""
        return (self.response_format or {}).get("type") == "json_object"

    def resolved_backend_flags(self) -> tuple[Optional[bool], Optional[bool]]:
        """
        Return (force_use_search, force_use_tools) by resolving the OpenAI tools +
        tool_choice semantics on top of the direct web_search / force_use_tools overrides.

        Priority: direct override > inferred from tools array.
        """
        search    = self.web_search
        use_tools = self.force_use_tools

        if self.tools and self.tool_choice != "none":
            names = {
                t.get("function", {}).get("name", "")
                for t in self.tools
                if isinstance(t, dict)
            }
            only_search = names and names.issubset(_BUILTIN_TOOL_WEB_SEARCH)
            has_search  = bool(names & _BUILTIN_TOOL_WEB_SEARCH)

            if only_search:
                if search is None:
                    search = True
            else:
                if use_tools is None:
                    use_tools = True
                if has_search and search is None:
                    search = True

        elif self.tool_choice == "none":
            if search is None:
                search = False
            if use_tools is None:
                use_tools = False

        return search, use_tools

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for p in _pools.values():
        await p.close_all()

app = FastAPI(
    title="ChatGPT Android Proxy",
    description="OpenAI-compatible API backed by the ChatGPT Android anonymous flow",
    version="2.4.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Web UI — served at GET /
# ---------------------------------------------------------------------------

_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChatGPT Free</title>
<style>
:root {
  --bg:#07101E; --surface:#0D1A2C; --surface2:#0A1524; --border:#1A2D44;
  --text:#D8E8F5; --muted:#7D9AB5; --accent:#00B899; --accent2:rgba(0,184,153,.15);
  --user-bg:#0E1E35; --code-bg:#040E1A;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,"Segoe UI",sans-serif;
  height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── header ── */
header{background:var(--surface);border-bottom:1px solid var(--border);
  padding:10px 20px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.logo{display:flex;align-items:center;gap:10px}
.logo-dot{width:9px;height:9px;border-radius:50%;background:var(--accent)}
.logo-name{font-family:monospace;font-weight:700;font-size:15px}
.logo-badge{font-size:11px;background:var(--accent2);color:var(--accent);padding:2px 8px;
  border-radius:20px;font-weight:600}
.status-pill{font-size:12px;color:var(--muted)}

/* ── settings bar ── */
.settings{background:var(--surface2);border-bottom:1px solid var(--border);
  padding:8px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;flex-shrink:0}
.settings select{background:var(--surface);color:var(--text);border:1px solid var(--border);
  border-radius:6px;padding:4px 10px;font-size:13px;cursor:pointer;outline:none}
.settings label{font-size:12px;color:var(--muted);cursor:pointer;
  display:flex;align-items:center;gap:6px;user-select:none}
.settings input[type=checkbox]{accent-color:var(--accent);width:14px;height:14px;cursor:pointer}
.sys-wrap{display:flex;align-items:center;gap:8px;flex:1;min-width:200px}
.sys-wrap span{font-size:12px;color:var(--muted);white-space:nowrap}
.sys-wrap input{flex:1;background:var(--surface);color:var(--text);border:1px solid var(--border);
  border-radius:6px;padding:4px 10px;font-size:13px;outline:none}
.sys-wrap input::placeholder{color:var(--muted)}
.sys-wrap input:focus{border-color:var(--accent)}

/* ── messages ── */
.msgs{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:10px;max-width:820px;width:100%}
.msg.user{align-self:flex-end;flex-direction:row-reverse;max-width:70%}
.msg.assistant{align-self:flex-start}
.avatar{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:13px;font-weight:700;flex-shrink:0}
.user .avatar{background:var(--accent);color:#fff}
.assistant .avatar{background:var(--surface);color:var(--accent);border:1px solid var(--border);font-family:monospace}
.bubble{padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.7;word-break:break-word}
.user .bubble{background:var(--user-bg);border:1px solid var(--border);border-top-right-radius:3px}
.assistant .bubble{background:var(--surface);border:1px solid var(--border);border-top-left-radius:3px}
.bubble p{margin-bottom:8px}.bubble p:last-child{margin-bottom:0}
.bubble pre{background:var(--code-bg);border:1px solid var(--border);border-radius:6px;
  padding:12px;overflow-x:auto;margin:8px 0;font-size:13px}
.bubble code{font-family:'JetBrains Mono','Fira Code',monospace;font-size:13px}
.bubble :not(pre)>code{background:var(--code-bg);padding:2px 6px;border-radius:4px;
  font-size:12px;color:var(--accent)}
.bubble strong{font-weight:600}
.bubble em{font-style:italic}
.bubble ul,.bubble ol{padding-left:20px;margin:6px 0}
.bubble li{margin-bottom:3px}
.bubble h1{font-size:18px;margin:10px 0 6px}
.bubble h2{font-size:16px;margin:10px 0 5px}
.bubble h3{font-size:14px;margin:8px 0 4px}
.bubble a{color:var(--accent);text-decoration:none}
.bubble a:hover{text-decoration:underline}
.cursor{display:inline-block;width:7px;height:14px;background:var(--accent);
  vertical-align:text-bottom;animation:blink .7s step-end infinite}
@keyframes blink{50%{opacity:0}}
.meta-tag{font-size:11px;color:var(--muted);margin-top:4px;font-family:monospace;opacity:.7}

/* ── empty state ── */
.empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:12px;color:var(--muted);pointer-events:none}
.empty-icon{font-size:48px}
.empty-title{font-size:22px;font-weight:700;color:var(--text)}
.empty-sub{font-size:14px;text-align:center;max-width:360px;line-height:1.5}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:4px;pointer-events:auto}
.chip{background:var(--surface);border:1px solid var(--border);border-radius:20px;
  padding:6px 14px;font-size:13px;color:var(--text);cursor:pointer;transition:border-color .15s}
.chip:hover{border-color:var(--accent);color:var(--accent)}

/* ── input area ── */
.input-area{background:var(--surface);border-top:1px solid var(--border);padding:14px 20px;flex-shrink:0}
.input-row{display:flex;gap:10px;align-items:flex-end;max-width:820px;margin:0 auto}
textarea{flex:1;background:var(--surface2);color:var(--text);border:1px solid var(--border);
  border-radius:10px;padding:10px 14px;font-size:14px;resize:none;min-height:44px;max-height:150px;
  line-height:1.5;font-family:inherit;outline:none;transition:border-color .15s}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--muted)}
.btn{border:none;border-radius:8px;padding:10px 16px;font-size:13px;
  font-weight:600;cursor:pointer;transition:opacity .15s;white-space:nowrap}
.btn-send{background:var(--accent);color:#fff}
.btn-send:hover{opacity:.85}
.btn-send:disabled{opacity:.4;cursor:not-allowed}
.btn-stop{background:#C0392B;color:#fff}
.btn-stop:hover{opacity:.85}
.btn-clear{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-clear:hover{color:var(--text);border-color:var(--muted)}
.footer-info{font-size:11px;color:var(--muted);text-align:center;margin-top:6px}

/* ── scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-dot"></div>
    <span class="logo-name">chatgpt-proxy</span>
    <span class="logo-badge">Free</span>
  </div>
  <span class="status-pill" id="hdr-status">🟢 Anonymous · No account needed</span>
</header>

<div class="settings">
  <div style="display:flex;align-items:center;gap:8px">
    <span style="font-size:12px;color:var(--muted)">Model</span>
    <select id="sel-model"><option value="auto">auto</option></select>
  </div>
  <label><input type="checkbox" id="chk-search"> Web Search</label>
  <label><input type="checkbox" id="chk-tools"> Adv. Tools</label>
  <label><input type="checkbox" id="chk-canvas"> Canvas</label>
  <label><input type="checkbox" id="chk-json"> JSON mode</label>
  <div class="sys-wrap">
    <span>System</span>
    <input type="text" id="inp-system" placeholder="System prompt (optional)…">
  </div>
</div>

<div class="msgs" id="msgs">
  <div class="empty" id="empty">
    <div class="empty-icon">🆓</div>
    <div class="empty-title">ChatGPT Free</div>
    <div class="empty-sub">No API key · No account · Powered by the ChatGPT Android anonymous API</div>
    <div class="chips">
      <div class="chip" onclick="prefill('Explain quantum computing in simple terms')">Explain quantum computing</div>
      <div class="chip" onclick="prefill('Write a Python script to rename files in bulk')">Python file renamer script</div>
      <div class="chip" onclick="prefill('What are the best practices for REST API design?')">REST API best practices</div>
      <div class="chip" onclick="prefill('Summarize the history of the internet')">History of the internet</div>
    </div>
  </div>
</div>

<div class="input-area">
  <div class="input-row">
    <button class="btn btn-clear" id="btn-clear">↺ New</button>
    <textarea id="inp" placeholder="Send a message… (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
    <button class="btn btn-send" id="btn-send">Send ↑</button>
  </div>
  <div class="footer-info" id="footer-info"></div>
</div>

<script>
const msgs     = document.getElementById('msgs');
const inp      = document.getElementById('inp');
const btnSend  = document.getElementById('btn-send');
const btnClear = document.getElementById('btn-clear');
const footer   = document.getElementById('footer-info');
const empty    = document.getElementById('empty');
const selModel = document.getElementById('sel-model');

let history = [];
let abort   = null;

// ── Load models ───────────────────────────────────────────────────────────────
fetch('/v1/models').then(r => r.json()).then(data => {
  const ids = (data.data || []).map(m => m.id).filter(Boolean);
  if (!ids.length) return;
  selModel.innerHTML = '';
  ids.forEach(id => {
    const o = document.createElement('option');
    o.value = o.textContent = id;
    if (id === 'auto') o.selected = true;
    selModel.appendChild(o);
  });
}).catch(() => {});

// ── Markdown ──────────────────────────────────────────────────────────────────
function mdToHtml(raw) {
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

  // Extract code blocks first, replace with placeholders
  const blocks = [];
  let text = raw.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(`<pre><code class="lang-${esc(lang)}">${esc(code.trim())}</code></pre>`);
    return `\x00BLOCK${blocks.length-1}\x00`;
  });

  // Inline code
  text = text.replace(/`([^`\n]+)`/g, (_, c) => `<code>${esc(c)}</code>`);

  // Headers
  text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  text = text.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  text = text.replace(/^# (.+)$/gm,   '<h1>$1</h1>');

  // Bold / italic
  text = text.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Lists
  text = text.replace(/^[ \t]*[-*] (.+)$/gm, '<li>$1</li>');
  text = text.replace(/(<li>[\s\S]+?<\/li>)(\n(?!<li>)|$)/g, '<ul>$1</ul>\n');

  // Links
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

  // Paragraphs (double newline)
  const paras = text.split(/\n\n+/);
  text = paras.map(p => {
    p = p.trim();
    if (!p) return '';
    if (/^<[hH\d]|^<ul|^<ol|^<li|^\x00BLOCK/.test(p)) return p;
    return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
  }).join('\n');

  // Restore code blocks
  text = text.replace(/\x00BLOCK(\d+)\x00/g, (_, i) => blocks[+i]);
  return text;
}

// ── Append bubble ─────────────────────────────────────────────────────────────
function addBubble(role, text, live) {
  if (empty.style.display !== 'none') empty.style.display = 'none';
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = role === 'user' ? 'U' : 'AI';
  const bub = document.createElement('div');
  bub.className = 'bubble';
  if (role === 'user') {
    bub.textContent = text;
  } else {
    bub.innerHTML = mdToHtml(text) + (live ? '<span class="cursor"></span>' : '');
  }
  wrap.appendChild(av);
  wrap.appendChild(bub);
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;
  return bub;
}

function updateBubble(bub, text, done) {
  bub.innerHTML = mdToHtml(text) + (done ? '' : '<span class="cursor"></span>');
  msgs.scrollTop = msgs.scrollHeight;
}

// ── Send ──────────────────────────────────────────────────────────────────────
async function send() {
  const text = inp.value.trim();
  if (!text) return;

  const model    = selModel.value || 'auto';
  const search   = document.getElementById('chk-search').checked || null;
  const tools    = document.getElementById('chk-tools').checked  || null;
  const canvas   = document.getElementById('chk-canvas').checked || null;
  const jsonMode = document.getElementById('chk-json').checked;
  const sys      = document.getElementById('inp-system').value.trim();

  const messages = [];
  if (sys) messages.push({role:'system', content:sys});
  history.forEach(m => messages.push(m));
  messages.push({role:'user', content:text});

  history.push({role:'user', content:text});
  addBubble('user', text, false);
  inp.value = '';
  inp.style.height = 'auto';

  btnSend.className = 'btn btn-stop';
  btnSend.textContent = '■ Stop';
  btnSend.onclick = stopStream;
  footer.textContent = 'Connecting…';

  const aiBub = addBubble('assistant', '', true);
  let full = '';
  let usedModel = model;

  abort = new AbortController();

  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        model, messages, stream: true,
        web_search:       search,
        force_use_tools:  tools,
        force_use_canvas: canvas,
        response_format:  jsonMode ? {type:'json_object'} : undefined,
      }),
      signal: abort.signal,
    });

    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      throw new Error(e?.error?.message || `HTTP ${resp.status}`);
    }

    footer.textContent = 'Streaming…';
    const reader = resp.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (raw === '[DONE]') break;
        try {
          const obj = JSON.parse(raw);
          if (obj.error) throw new Error(obj.error.message);
          if (obj.chatgpt_metadata) {
            usedModel = obj.chatgpt_metadata.model || usedModel;
          }
          const delta = obj.choices?.[0]?.delta?.content;
          if (delta) { full += delta; updateBubble(aiBub, full, false); }
          if (obj.model) usedModel = obj.model;
        } catch(e) { if (!e.message.includes('JSON')) throw e; }
      }
    }

    updateBubble(aiBub, full, true);
    history.push({role:'assistant', content:full});
    footer.textContent = `model: ${usedModel} · ${history.length/2|0} turn(s)`;

  } catch(e) {
    if (e.name === 'AbortError') {
      updateBubble(aiBub, full || '_(stopped)_', true);
      footer.textContent = 'Stopped.';
    } else {
      updateBubble(aiBub, `⚠️ **Error:** ${e.message}`, true);
      footer.textContent = 'Error';
    }
  } finally {
    abort = null;
    btnSend.className = 'btn btn-send';
    btnSend.textContent = 'Send ↑';
    btnSend.onclick = send;
    inp.focus();
  }
}

function stopStream() {
  if (abort) { abort.abort(); abort = null; }
}

// ── Clear ─────────────────────────────────────────────────────────────────────
btnClear.onclick = () => {
  history = [];
  msgs.innerHTML = '';
  msgs.appendChild(empty);
  empty.style.display = '';
  footer.textContent = '';
};

// ── Prefill ───────────────────────────────────────────────────────────────────
function prefill(text) {
  inp.value = text;
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 150) + 'px';
  inp.focus();
}

// ── Auto-resize textarea ──────────────────────────────────────────────────────
inp.addEventListener('input', () => {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 150) + 'px';
});

inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

btnSend.onclick = send;
inp.focus();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat_ui():
    """Serve the built-in web chat interface."""
    return HTMLResponse(content=_CHAT_HTML)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(content: str, model: str, completion_id: str, finish: bool = False) -> str:
    chunk = {
        "id":      completion_id,
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "delta":         {} if finish else {"content": content},
            "finish_reason": "stop" if finish else None,
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _make_completion(content: str, model: str, completion_id: str) -> dict:
    words = len(content.split())
    return {
        "id":      completion_id,
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     -1,
            "completion_tokens": words,
            "total_tokens":      words,
        }
    }


def _msg_to_text(m: "Message") -> str:
    """Extract plain text from a message (str or list of ContentPart)."""
    if isinstance(m.content, str):
        return m.content
    if isinstance(m.content, list):
        return " ".join(p.text or "" for p in m.content if p.type == "text")
    return ""


def _resolve_messages(
    messages: List[Message],
    user_files: dict[str, dict],
) -> tuple[str, str, List[str]]:
    """
    Extract:
      - system_prompt: content of the system role message
      - last_user_text: text of the last user message, with prior history
        injected as context (for multi-turn without server-side state)
      - file_texts: contents of file attachments in the last message
    """
    system_prompt = ""
    file_texts: List[str] = []

    for m in messages:
        if m.role == "system":
            system_prompt = _msg_to_text(m)

    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "No user messages found")

    last = user_msgs[-1]

    last_user_text = ""
    if isinstance(last.content, str):
        last_user_text = last.content
    elif isinstance(last.content, list):
        for part in last.content:
            if part.type == "text" and part.text:
                last_user_text += part.text
            elif part.type == "file":
                fdata   = part.file or {}
                file_id = fdata.get("file_id", "")
                if file_id and file_id in user_files:
                    fc = user_files[file_id].get("content", "")
                    if fc:
                        file_texts.append(fc)
                elif fdata.get("content"):
                    file_texts.append(fdata["content"])
            elif part.type == "image_url":
                raise HTTPException(
                    400,
                    "image_url is not available in anonymous mode. "
                    "Use text files (type='file') instead."
                )

    # Inject prior history (user + assistant) as context in the message
    prior_turns = []
    for m in messages:
        if m is last:
            break
        if m.role not in ("user", "assistant"):
            continue
        text = _msg_to_text(m).strip()
        if not text:
            continue
        label = "User" if m.role == "user" else "Assistant"
        prior_turns.append(f"{label}: {text}")

    if prior_turns:
        history_block  = "\n".join(prior_turns)
        last_user_text = (
            f"[Prior conversation — use this as context:\n{history_block}\n]\n\n"
            f"{last_user_text}"
        )

    if not last_user_text:
        raise HTTPException(400, "Last user message is empty")

    return system_prompt, last_user_text, file_texts

# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

_LEGACY_ALIASES = {
    "gpt-4o":        "auto",
    "gpt-4o-mini":   "gpt-5-3-mini",
    "gpt-4":         "gpt-5-5",
    "gpt-3.5-turbo": "gpt-5-3-mini",
}

@app.get("/v1/models")
async def list_models():
    global _models_cache
    now = time.time()
    cached_at, cached_models = _models_cache

    if now - cached_at > _MODELS_TTL or not cached_models:
        try:
            raw            = await fetch_anon_models()
            _models_cache  = (now, raw)
            cached_models  = raw
        except Exception as e:
            if cached_models:
                pass  # serve stale cache before failing
            else:
                raise HTTPException(502, f"Could not fetch model list: {e}")

    data = [
        {
            "id":          m.get("slug", m.get("id", "")),
            "object":      "model",
            "created":     1750000000,
            "owned_by":    "openai",
            "description": m.get("title", ""),
        }
        for m in cached_models
        if m.get("slug") or m.get("id")
    ]

    # Add legacy aliases for OpenAI-compatible clients
    existing_ids = {d["id"] for d in data}
    for alias, target in _LEGACY_ALIASES.items():
        if alias not in existing_ids:
            data.append({
                "id":          alias,
                "object":      "model",
                "created":     1750000000,
                "owned_by":    "openai",
                "description": f"Alias → {target}",
            })

    return {"object": "list", "data": data}

# ---------------------------------------------------------------------------
# /v1/files
# ---------------------------------------------------------------------------

@app.post("/v1/files")
async def upload_file(
    request:  Request,
    file:     UploadFile = File(...),
    purpose:  str        = Form("assistants"),
):
    """
    Upload a text file (PDF, code, docs) and return a file_id.
    Isolated per user — each API key sees only its own files.

    Auth: Authorization: Bearer <your-api-key>
    Compatible with the OpenAI Files API.
    """
    user_id = _get_user_id(request)
    uf      = _user_files(user_id)

    content_type   = file.content_type or "application/octet-stream"
    _IMAGE_EXTS    = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".avif", ".svg"}
    filename_lower = (file.filename or "").lower()
    is_image       = content_type.startswith("image/") or any(filename_lower.endswith(ext) for ext in _IMAGE_EXTS)

    if is_image:
        raise HTTPException(
            415,
            "Images are not available in anonymous mode. "
            "Only text/document files are supported for retrieval."
        )

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "File exceeds the 20 MB limit")

    text_content = ""
    if content_type == "application/pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                text_content = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            text_content = raw.decode("latin-1", errors="ignore")
    else:
        try:
            text_content = raw.decode("utf-8", errors="replace")
        except Exception:
            text_content = raw.decode("latin-1", errors="replace")

    if not text_content.strip():
        raise HTTPException(422, "File contains no extractable text")

    file_id = f"file-{uuid.uuid4().hex[:24]}"
    now     = int(time.time())
    uf[file_id] = {
        "id":         file_id,
        "object":     "file",
        "filename":   file.filename or "upload",
        "purpose":    purpose,
        "size":       len(raw),
        "created_at": now,
        "content":    text_content,
        "mime_type":  content_type,
    }

    return {
        "id":         file_id,
        "object":     "file",
        "filename":   file.filename or "upload",
        "purpose":    purpose,
        "bytes":      len(raw),
        "created_at": now,
        "status":     "processed",
    }


@app.get("/v1/files")
async def list_files(request: Request):
    user_id = _get_user_id(request)
    uf      = _user_files(user_id)
    data    = [
        {k: v for k, v in f.items() if k != "content"}
        for f in uf.values()
    ]
    return {"object": "list", "data": data}


@app.get("/v1/files/{file_id}")
async def get_file(file_id: str, request: Request):
    user_id = _get_user_id(request)
    uf      = _user_files(user_id)
    if file_id not in uf:
        raise HTTPException(404, f"File {file_id!r} not found")
    return {k: v for k, v in uf[file_id].items() if k != "content"}


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, request: Request):
    user_id = _get_user_id(request)
    uf      = _user_files(user_id)
    if file_id not in uf:
        raise HTTPException(404, f"File {file_id!r} not found")
    del uf[file_id]
    return {"id": file_id, "object": "file", "deleted": True}


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

def _build_search_metadata(session) -> dict:
    """
    Build the search_metadata object from the session state.
    Only included when web search was active in the last turn.
    """
    return {
        "queries": session.last_search_queries,
        "sources": session.last_citations,
        "title":   session.last_title,
    }


def _quota_error_payload() -> dict:
    return {
        "error": {
            "message": "Anonymous message limit exhausted. A new device_id will be used on the next request.",
            "type":    "rate_limit_exceeded",
            "code":    "rate_limit_exceeded",
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    user_id   = _get_user_id(request)
    uf        = _user_files(user_id)
    pool      = _user_pool(user_id)
    msgs_raw  = [m.model_dump() for m in req.messages]
    json_mode = req.is_json_mode()

    system_prompt, last_user_text, file_texts = _resolve_messages(req.messages, uf)

    completion_id                   = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    _key, session                   = await pool.get(msgs_raw, system_prompt=system_prompt)
    force_use_search, force_use_tools = req.resolved_backend_flags()

    if req.stream:
        async def event_stream() -> AsyncGenerator[bytes, None]:
            role_chunk = {
                "id":      completion_id,
                "object":  "chat.completion.chunk",
                "created": int(time.time()),
                "model":   req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(role_chunk)}\n\n".encode()

            cur_key     = _key
            cur_session = session

            for attempt in range(2):
                try:
                    async for text_chunk in cur_session.stream_message(
                        last_user_text,
                        model             = req.model,
                        file_texts        = file_texts or None,
                        force_use_search  = force_use_search,
                        force_use_tools   = force_use_tools,
                        force_use_canvas  = req.force_use_canvas,
                        json_mode         = json_mode,
                    ):
                        if text_chunk:
                            yield _make_chunk(text_chunk, req.model, completion_id).encode()
                    break  # success

                except QuotaExceededError:
                    if attempt == 0:
                        # First failure: transparently retry with a fresh device_id
                        pool._pool.pop(cur_key, None)
                        cur_key, cur_session = await pool.get(msgs_raw, system_prompt=system_prompt)
                    else:
                        # Second failure: likely IP-limited, report error to client
                        err = _quota_error_payload()
                        yield f"data: {json.dumps(err)}\n\n".encode()
                        yield b"data: [DONE]\n\n"
                        return

                except Exception as e:
                    err = {"error": {"message": str(e), "type": "api_error"}}
                    yield f"data: {json.dumps(err)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return

            yield _make_chunk("", req.model, completion_id, finish=True).encode()

            if cur_session.last_search_queries or cur_session.last_citations:
                meta = _build_search_metadata(cur_session)
                yield f"event: search_metadata\ndata: {json.dumps(meta)}\n\n".encode()
            if cur_session.last_widgets:
                yield f"event: widgets\ndata: {json.dumps(cur_session.last_widgets)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    else:
        full_text   = ""
        cur_key     = _key
        cur_session = session

        for attempt in range(2):
            full_text = ""
            try:
                async for chunk in cur_session.stream_message(
                    last_user_text,
                    model             = req.model,
                    file_texts        = file_texts or None,
                    force_use_search  = force_use_search,
                    force_use_tools   = force_use_tools,
                    force_use_canvas  = req.force_use_canvas,
                    json_mode         = json_mode,
                ):
                    full_text += chunk
                break  # success

            except QuotaExceededError:
                if attempt == 0:
                    pool._pool.pop(cur_key, None)
                    cur_key, cur_session = await pool.get(msgs_raw, system_prompt=system_prompt)
                else:
                    raise HTTPException(429, detail=_quota_error_payload())

            except Exception as e:
                raise HTTPException(500, str(e))

        # Use clean text (citations preserved, genui removed)
        clean = cur_session.last_clean_text or full_text
        # If JSON mode was requested, extract the JSON from any markdown fencing
        if json_mode:
            clean = _extract_json(clean)

        resp = _make_completion(clean, req.model, completion_id)
        if cur_session.last_search_queries or cur_session.last_citations:
            resp["search_metadata"] = _build_search_metadata(cur_session)
        if cur_session.last_widgets:
            resp["widgets"] = cur_session.last_widgets
        return JSONResponse(resp)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    total_sessions = sum(len(p._pool) for p in _pools.values())
    total_files    = sum(len(uf) for uf in _files.values())
    return {
        "status":               "ok",
        "version":              "2.4.0",
        "active_users":         len(_pools),
        "total_sessions":       total_sessions,
        "total_files_in_memory": total_files,
        "anonymous_models": [
            "gpt-5-6", "gpt-5-5", "gpt-5-6-mini", "gpt-5-5-mini", "gpt-5-3-mini", "auto"
        ],
        "capabilities": {
            "text_chat":        True,
            "streaming":        True,
            "multi_turn":       True,
            "system_prompt":    True,
            "file_attachments": True,
            "multi_user":       True,
            "json_mode":        "response_format: {type: json_object}",
            "quota_handling":   "auto-recycles device_id on 429/403, transparent retry",
            "image_input":      False,
            "image_generation": False,
            "web_search":       "automatic (override with web_search: true/false)",
            "force_use_tools":  "optional (enables advanced tools + genui widgets)",
            "force_use_canvas": "optional (enables Canvas / document mode)",
            "voice":            False,
        }
    }


# ---------------------------------------------------------------------------
# /v1/session/me
# ---------------------------------------------------------------------------

@app.get("/v1/session/me")
async def session_me(request: Request):
    """
    Anonymous user info for the current session.
    Proxies to /backend-anon/me on the ChatGPT API.
    Useful for debugging — returns the anonymous user id (ua-xxx).
    """
    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())

    if sessions:
        session    = sessions[0]
        device_id  = session.device_id
        client     = session.client
        owns_client = False
    else:
        device_id   = str(uuid.uuid4())
        client      = _httpx.AsyncClient(verify=True, timeout=10.0, follow_redirects=True)
        owns_client = True

    try:
        hdrs = {
            **_base_headers(device_id),
            "X-OpenAI-Target-Path": "/backend-anon/me",
        }
        r = await client.get(f"{BASE}/backend-anon/me", headers=hdrs)
        r.raise_for_status()
        return r.json()
    finally:
        if owns_client:
            await client.aclose()
