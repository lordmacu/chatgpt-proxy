"""
FastAPI proxy — OpenAI-compatible API backed by the ChatGPT Android anonymous flow.

Endpoints — [A] works anonymously, [L] needs a logged-in account.
`GET /health` reports which mode is live as auth_mode: "anonymous" | "account".

  [A] POST /v1/chat/completions        — chat (streaming + non-streaming, files)
  [A] POST /v1/tool-calls              — resolve a request into function calls, stateless
  [A] GET  /v1/models                  — models available in the current mode
  [A] GET  /v1/session/me              — session info (user id, device id)
  [A] GET  /v1/limits                  — per-feature remaining counts
  [A] POST /v1/translate               — translate text (spends no chat message)
  [L] POST|GET|DEL /v1/files[/{id}]    — proxy-local file store, gated by capabilities.files
  [A] GET  /health, GET /              — status, built-in web chat UI
  [A] GET  /images/{f}, /audio/{f}     — serve files the proxy already cached locally
  [L] GET  /v1/account                 — plan, id
  [L] GET|POST /v1/custom-instructions — the account's custom instructions
  [L] GET  /v1/gizmos[/{id}]           — custom GPTs (also model: "g-...")
  [A] GET  /v1/conversations           — the account's history; anonymously,
                                         this proxy's own index (conv_store.py)
  [A] GET  /v1/conversations/{id}      — anonymously, any conversation this
                                         proxy created and recorded the device for
  [L] GET  /v1/library[...]            — the account's file library
  [L] GET  /v1/suggestions             — prompt-library starters
  [L] POST /v1/audio/transcriptions    — speech-to-text
  [L] POST /v1/audio/speech            — TTS (/backend-anon/synthesize does not exist), raw audio bytes
  [L] POST /chatgpt/audio/speech       — same TTS, pre-contract JSON shape (url + metadata)
  [L] GET  /v1/audio/from-message      — TTS of a stored message (same backend)
  [L] POST /v1/images/generations      — image generation

Account-only endpoints answer 401 with type "auth_error".

Chat capabilities — anonymous | account:
  ✅|✅ Text chat, streaming SSE, multi-turn (~30 min TTL), system prompt
  ✅|✅ Text attachments: PDF, code, docs (82 MIME types, retrieval mode)
  ✅|✅ Web search (web_search), advanced tools (force_use_tools), Canvas (force_use_canvas)
  ✅|✅ JSON output — response_format: {"type": "json_object"}
  ✅|✅ thinking_effort / reasoning_effort, service_tier
  ✅|—  Quota exhaustion: auto-recycles device_id on 429/403, transparent retry
  34,834 | 52,815 token context window (262,144 on the -t-mini models)
  ❌|✅ Reasoning models (gpt-5-4-t-mini, gpt-5-6-t-mini), research, custom GPTs
  ❌|✅ Image generation, TTS, speech-to-text
  ❌|❌ Image input (vision) — not exposed on this endpoint
  ✅|✅ Function calling / tool_calls — emulated, see below
  ❌|❌ temperature, top_p, max_tokens, penalties, stop, seed, n — see below

OpenAI `tools` field:
  Two different things arrive in one array, and they are handled differently.

  a) Built-in names select a ChatGPT mode. These run server-side and come back
     as text or widgets, never as something the caller executes. The names match
     the `enabled_tools` list upstream reports per model — currently ["tools",
     "tools2", "search", "canvas", "app_pairing", "image_gen_tool_enabled"]:
       • "web_search" / "search" / ...   → force_use_search = true
       • "canvas" / "text_editor"        → force_use_canvas = true
       • "chatgpt_tools" / "tools" / ... → force_use_tools  = true

  b) Anything else is the caller's own function, and the proxy emulates real
     function calling for it: the turn is resolved by a separate stateless
     extraction (tool_calls.py) and comes back as choices[0].message.tool_calls
     with finish_reason "tool_calls", streaming included. Send the results back
     as role:"tool" messages and the next request answers in prose. The official
     OpenAI SDK drives the whole loop unmodified.
       • tool_choice: "none"     → no extraction, no modes, plain chat
       • tool_choice: "required" → the model may not decline to call
       • tool_choice: {...}      → that function or nothing
       • tool_emulation: false   → (b) off; custom names then do nothing
       • tool_verify: true       → +1 message auditing the call for a dropped
                                   condition, worth it on dense requests

  Cost: emulation spends one extra upstream message per turn that declares
  custom functions, because the decision is a request of its own. A turn where
  no function fits therefore costs two messages (decide, then answer). The
  X-Proxy-Tool-Extraction and X-Proxy-Tool-Requests response headers report
  what happened and what it cost.

  What emulation does NOT recover: a required argument the request never states
  is answered with a question rather than a guess (status "need_info"), and a
  request packing many conditions can still drop one — see tool_calls.py for the
  measurements behind both.

  POST /v1/tool-calls exposes the extractor on its own, for a caller that wants
  the decision without a conversation attached.

Sampling parameters:
  The conversation protocol has no temperature / top_p / max_tokens / penalties —
  the field simply does not exist upstream, so these are accepted (for client
  compatibility) and dropped. Requests that carry them get an informational
  `X-Proxy-Ignored-Params` response header so the loss is visible rather than
  silent. What the backend *does* expose is `thinking_effort` (standard |
  extended | max, also reachable through OpenAI's `reasoning_effort`) and
  `service_tier` (standard | priority); both are validated, so a bad value is a
  400 from this proxy instead of a wasted upstream request.
"""
import io
import os
import sys
import uuid
import json
import time
import base64
import asyncio
import random
import pathlib
import ipaddress
from urllib.parse import urlparse
from typing import Optional, AsyncGenerator, Union, List
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse, Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from chatgpt_client import (
    SessionPool, ChatGPTSession, fetch_anon_models, BASE, _base_headers,
    QuotaExceededError, _extract_json, _IMAGE_STORE_DIR, _api,
)
import httpx as _httpx
import chatgpt_client as _chatgpt_client_module

import auth
import capabilities
import conv_store
import tool_calls as _tc

# ---------------------------------------------------------------------------
# capabilities.py wiring: the real vendor call
# ---------------------------------------------------------------------------
# capabilities.py knows the RULES (what a plan means), never the TRANSPORT (how
# to reach ChatGPT) -- this module already owns the HTTP client, the device id
# and the session pool, so the real accounts/check call lives here and gets
# installed into capabilities.py, rather than the other way around.
def _resolve_account_state() -> capabilities.AccountState:
    """Reach accounts/check exactly the way GET /v1/account does, and turn the
    response into an AccountState for capabilities.snapshot().

    Synchronous and blocking on purpose: `capabilities.snapshot()` calls this
    behind a `threading.Lock`, at most once an hour, and the /health handler
    offloads the call to a worker thread (`asyncio.to_thread`) precisely so
    this blocking round trip never stalls the event loop that in-flight chat
    completions and streaming responses depend on.
    """
    if not auth.is_authenticated():
        return capabilities.AccountState(mode="anonymous")

    device_id = str(uuid.uuid4())
    path = "/backend-api/accounts/check/v4-2023-04-27"
    hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path}
    r = _httpx.get(f"{BASE}{path}", params={"timezone_offset_min": "0"},
                   headers=hdrs, timeout=15.0)
    r.raise_for_status()
    data = r.json()

    accts = data.get("accounts", {}) or {}
    acc   = accts.get("default") or next(iter(accts.values()), {}) or {}
    a     = acc.get("account", {}) or {}
    ent   = acc.get("entitlement", {}) or {}
    return capabilities.AccountState(
        mode="account",
        plan=a.get("plan_type"),
        subscription_active=bool(ent.get("has_active_subscription")),
        expires_at=ent.get("expires_at"),
    )


capabilities.set_resolver(_resolve_account_state)


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
    # None is not an oddity to tolerate, it is what every OpenAI SDK sends back:
    # an assistant turn that called a function has content: null, and rejecting
    # it 422s the second leg of every round trip.
    content: Union[str, List[ContentPart], None] = ""
    name:    Optional[str] = None

    # The two fields a function-calling round trip travels on. Without them
    # pydantic drops the assistant's calls and the tool's answer on the floor,
    # and the second request of the loop re-answers the original question as if
    # the tool had never run.
    tool_calls:   Optional[List[dict]] = None   # on role="assistant"
    tool_call_id: Optional[str] = None          # on role="tool"


# Built-in tool names that this proxy recognises in the OpenAI tools array.
# Any other name activates force_use_tools (the backend's internal tools).
# The bare names ("search", "tools", "tools2", "canvas") are the ones upstream
# itself uses in each model's `enabled_tools` list.
_BUILTIN_TOOL_WEB_SEARCH = {"web_search", "search", "brave_search", "bing_search", "google_search"}
_BUILTIN_TOOL_CANVAS     = {"canvas", "text_editor"}
_BUILTIN_TOOL_ALL        = {"chatgpt_tools", "all_tools", "builtin_tools",
                            "tools", "tools2", "app_pairing"}
# Everything above names a mode that runs inside ChatGPT. Anything else in the
# tools array is the caller's own function, which tool_calls.py emulates.
_BUILTIN_TOOL_NAMES      = _BUILTIN_TOOL_WEB_SEARCH | _BUILTIN_TOOL_CANVAS | _BUILTIN_TOOL_ALL

# Sampling fields the OpenAI schema defines and this backend has no equivalent
# for. Accepted so stock clients keep working, reported back in a header so the
# caller can tell they were dropped.
_IGNORED_SAMPLING_FIELDS = (
    "temperature", "max_tokens", "top_p",
    "presence_penalty", "frequency_penalty", "stop", "seed", "n",
)

# OpenAI's reasoning_effort vocabulary folded onto the three values the backend
# accepts for thinking_effort. Native values pass through unchanged.
_EFFORT_ALIASES = {
    "none": "standard", "minimal": "standard", "low": "standard",
    "standard": "standard",
    "medium": "extended", "default": "extended", "extended": "extended",
    "high": "max", "xhigh": "max", "max": "max",
}

# OpenAI's service_tier vocabulary folded onto the two tiers the backend accepts.
_TIER_ALIASES = {
    "auto": "standard", "default": "standard", "standard": "standard",
    "flex": "standard", "scale": "standard",
    "priority": "priority",
}


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
    # tool_emulation: false turns custom function calling off for this request —
    # the tools array then only picks a server-side mode, as it did before the
    # emulation existed. Default follows the TOOL_EMULATION env var.
    tool_emulation:   Optional[bool] = Field(default=None)
    # tool_verify: spend a second upstream message auditing the extracted call
    # against the original request. Worth it when a request packs many
    # conditions, since that is the case where one silently goes missing.
    tool_verify:      bool = Field(default=False)
    # web_search: true/false/null — direct override of force_use_search
    web_search:       Optional[bool] = Field(default=None)
    # force_use_tools: true/null — direct override of force_use_tools
    force_use_tools:  Optional[bool] = Field(default=None)
    # force_use_canvas: true/null — enable Canvas mode (collaborative documents)
    force_use_canvas: Optional[bool] = Field(default=None)

    # ── Generation controls the backend really honours ─────────────────────────
    # thinking_effort: "standard" | "extended" | "max" — how long a reasoning
    # model deliberates. reasoning_effort is OpenAI's spelling of the same idea
    # and is folded onto these three values; thinking_effort wins if both are set.
    thinking_effort:   Optional[str] = None
    reasoning_effort:  Optional[str] = None
    # service_tier: OpenAI's "auto"/"default"/"flex"/"scale" all map to the
    # backend's "standard"; "priority" passes through.
    service_tier:      Optional[str] = None
    # force_disable_features: feature names to switch off for this turn. Not
    # validated upstream — unknown names are ignored rather than rejected.
    force_disable_features: Optional[List[str]] = None

    # ── OpenAI compatibility fields (accepted, dropped) ────────────────────────
    # No equivalent exists in the conversation protocol. Kept so stock OpenAI
    # clients don't break; echoed back in X-Proxy-Ignored-Params so the caller
    # can see they had no effect.
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

    def resolved_backend_flags(
        self,
    ) -> tuple[Optional[bool], Optional[bool], Optional[bool]]:
        """
        Return (force_use_search, force_use_tools, force_use_canvas) by resolving the
        OpenAI tools + tool_choice semantics on top of the direct web_search /
        force_use_tools / force_use_canvas overrides.

        Priority: direct override > inferred from tools array.
        """
        search     = self.web_search
        use_tools  = self.force_use_tools
        use_canvas = self.force_use_canvas

        if self.tools and self.tool_choice != "none":
            names = {
                t.get("function", {}).get("name", "")
                for t in self.tools
                if isinstance(t, dict)
            }
            has_search = bool(names & _BUILTIN_TOOL_WEB_SEARCH)
            has_canvas = bool(names & _BUILTIN_TOOL_CANVAS)
            # Only the names that really mean "ChatGPT's own tools" switch that
            # mode on. A caller's custom function used to land here too, which
            # was actively harmful: with the backend's tools running, the model
            # answers the question itself instead of delegating to the function
            # (measured — weather prompts came back from web search, not as a
            # call). Custom names are emulated by tool_calls.py instead.
            generic    = bool(names & _BUILTIN_TOOL_ALL)

            if has_search and search is None:
                search = True
            if has_canvas and use_canvas is None:
                use_canvas = True
            if generic and use_tools is None:
                use_tools = True

        elif self.tool_choice == "none":
            if search is None:
                search = False
            if use_tools is None:
                use_tools = False
            if use_canvas is None:
                use_canvas = False

        return search, use_tools, use_canvas

    def resolved_thinking_effort(self) -> Optional[str]:
        """Native thinking_effort, or OpenAI's reasoning_effort folded onto it."""
        raw = self.thinking_effort or self.reasoning_effort
        if raw is None:
            return None
        effort = _EFFORT_ALIASES.get(str(raw).strip().lower())
        if effort is None:
            raise HTTPException(400, detail={"error": {
                "message": (
                    f"Unsupported reasoning/thinking effort {raw!r}. "
                    f"Accepted: {', '.join(sorted(set(_EFFORT_ALIASES)))}."
                ),
                "type": "invalid_request_error",
                "param": "reasoning_effort",
                "code": "invalid_value",
            }})
        return effort

    def resolved_service_tier(self) -> Optional[str]:
        """OpenAI service_tier folded onto the two tiers the backend accepts."""
        if self.service_tier is None:
            return None
        tier = _TIER_ALIASES.get(str(self.service_tier).strip().lower())
        if tier is None:
            raise HTTPException(400, detail={"error": {
                "message": (
                    f"Unsupported service_tier {self.service_tier!r}. "
                    f"Accepted: {', '.join(sorted(set(_TIER_ALIASES)))}."
                ),
                "type": "invalid_request_error",
                "param": "service_tier",
                "code": "invalid_value",
            }})
        return tier

    def ignored_params(self) -> list[str]:
        """Sampling fields the caller sent that this backend cannot honour."""
        return [f for f in _IGNORED_SAMPLING_FIELDS if getattr(self, f, None) is not None]

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
# Request tracing
#
# This proxy is the last hop and the slowest one: an image spends thirty to
# fifty seconds here while ChatGPT draws, and from outside that is
# indistinguishable from a slow gateway or a stalled network. The app mints an
# id per turn, the gateway forwards it (llm_libre.tracing), and these lines
# close the chain -- one string grepped in three logs, and the fifty seconds
# land on whichever hop actually spent them.
#
# The alphabet is narrow on purpose: the id is echoed into a response header and
# into a log line, so it is attacker-controlled text and a newline in it would
# forge an entry. Anything outside the pattern is replaced, never reflected.
# ---------------------------------------------------------------------------
import contextvars as _contextvars
import re as _re

TRACE_HEADER = "X-Request-Id"
_TRACE_SAFE = _re.compile(r"\A[A-Za-z0-9_.:-]{1,64}\Z")
_trace_id: "_contextvars.ContextVar[str]" = _contextvars.ContextVar(
    "chatgpt_proxy_trace_id", default="")


def current_trace_id() -> str:
    """The id of the request being served, or "" outside one (a background
    refresh is the proxy acting on its own, not on a caller's behalf, and
    tagging it with a caller's id would make the log lie)."""
    return _trace_id.get()


def _sanitise_trace_id(raw: str | None) -> str:
    if raw and _TRACE_SAFE.match(raw):
        return raw
    # `px-` says the proxy minted it, which is itself information: a trace that
    # starts here has lost the app's and the gateway's half.
    return "px-" + uuid.uuid4().hex[:16]


# Wired here, not imported there: chatgpt_client is imported BY this module, so
# it cannot import back. See its `trace_id_provider`.
_chatgpt_client_module.trace_id_provider = current_trace_id


@app.middleware("http")
async def _trace_requests(request: Request, call_next):
    rid = _sanitise_trace_id(request.headers.get(TRACE_HEADER))
    token = _trace_id.set(rid)
    t0 = time.monotonic()
    try:
        response = await call_next(request)
    finally:
        _trace_id.reset(token)
    ms = (time.monotonic() - t0) * 1000
    response.headers[TRACE_HEADER] = rid
    # Always printed, not gated behind DEBUG like `_log`: the whole point is to
    # be readable in production, where the latency being measured happens.
    print(f"[{rid}] {request.method} {request.url.path} -> "
          f"{response.status_code} in {ms:.0f}ms", flush=True)
    return response


# ---------------------------------------------------------------------------
# Optional API token guard (set API_TOKEN env var to enable)
# When enabled, all /v1/* requests must include: X-API-Token: <token>
# The web UI automatically receives the token and includes it in every request.
# ---------------------------------------------------------------------------
_API_TOKEN = os.getenv("API_TOKEN", "").strip()


@app.middleware("http")
async def _token_guard(request: Request, call_next):
    if _API_TOKEN and request.url.path.startswith("/v1/"):
        token = request.headers.get("X-API-Token", "")
        if token != _API_TOKEN:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Unauthorized: missing or invalid X-API-Token", "type": "auth_error"}},
            )
    return await call_next(request)


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

/* ── hamburger ── */
.btn-hamburger{display:none;background:transparent;border:none;color:var(--text);
  font-size:20px;cursor:pointer;padding:4px 6px;line-height:1;flex-shrink:0}

/* mobile settings panel */
.settings-panel{display:none;position:absolute;top:var(--header-h,49px);left:0;right:0;
  background:var(--surface2);border-bottom:1px solid var(--border);
  flex-direction:column;gap:14px;padding:14px 16px;z-index:50;
  animation:slideDown .18s ease}
.settings-panel.open{display:flex}
@keyframes slideDown{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
.settings-panel .row{display:flex;gap:18px;flex-wrap:wrap}
.settings-panel label{font-size:13px;color:var(--text);cursor:pointer;
  display:flex;align-items:center;gap:7px;user-select:none}
.settings-panel input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px;cursor:pointer}
.settings-panel select{background:var(--surface);color:var(--text);border:1px solid var(--border);
  border-radius:6px;padding:5px 10px;font-size:13px;cursor:pointer;outline:none}
.settings-panel .model-row{display:flex;align-items:center;gap:8px}
.settings-panel .model-row span{font-size:13px;color:var(--muted)}
.settings-panel .sys-row{display:flex;align-items:center;gap:8px}
.settings-panel .sys-row span{font-size:13px;color:var(--muted);white-space:nowrap}
.settings-panel .sys-row input{flex:1;background:var(--surface);color:var(--text);
  border:1px solid var(--border);border-radius:6px;padding:5px 10px;font-size:13px;outline:none}
.settings-panel .sys-row input:focus{border-color:var(--accent)}
.settings-panel .sys-row input::placeholder{color:var(--muted)}

@media(max-width:640px){
  body{height:100svh;height:100dvh}
  header{padding:8px 12px;gap:8px;position:relative}
  .btn-hamburger{display:block}
  .logo-name{font-size:14px}
  .logo-badge{display:none}
  .status-pill{display:none}
  .github-link span{display:none}
  .settings{display:none}
  .msgs{padding:12px 10px;gap:10px}
  .msg.user{max-width:88%}
  .bubble{font-size:13px;padding:9px 12px}
  .input-area{padding:10px 12px 8px}
  .input-row{flex-wrap:wrap;gap:7px}
  textarea{order:0;flex:1 1 100%;min-height:42px;font-size:14px}
  .btn-clear{order:1;flex:0 0 auto;padding:9px 14px;font-size:13px}
  .btn-send{order:2;flex:1;padding:9px 0;font-size:14px;font-weight:700}
  .footer-info{font-size:10px}
  .empty-icon{font-size:36px}
  .empty-title{font-size:18px}
  .chips{gap:6px}
  .chip{font-size:12px;padding:5px 11px}
}
</style>
</head>
<body>
<header>
  <button class="btn-hamburger" id="btn-hamburger" aria-label="Menu">☰</button>
  <div class="logo">
    <div class="logo-dot"></div>
    <span class="logo-name">chatgpt-proxy</span>
    <span class="logo-badge">Free</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <span class="status-pill" id="hdr-status">🟢 Anonymous · No account needed</span>
    <a href="https://github.com/lordmacu/chatgpt-proxy" target="_blank" rel="noopener" class="github-link"
       style="display:flex;align-items:center;gap:5px;color:var(--muted);text-decoration:none;font-size:13px;transition:color .2s"
       onmouseover="this.style.color='var(--accent)'" onmouseout="this.style.color='var(--muted)'">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                 -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                 .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                 -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
                 .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                 .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
      </svg>
      <span>GitHub</span>
    </a>
  </div>
</header>

<!-- Mobile settings panel (hamburger menu) -->
<div class="settings-panel" id="settings-panel">
  <div class="model-row">
    <span>Model</span>
    <select id="sel-model-m"><option value="auto">auto</option></select>
  </div>
  <div class="row">
    <label><input type="checkbox" id="chk-search-m"> Web Search</label>
    <label><input type="checkbox" id="chk-tools-m"> Adv. Tools</label>
    <label><input type="checkbox" id="chk-canvas-m"> Canvas</label>
    <label><input type="checkbox" id="chk-json-m"> JSON mode</label>
  </div>
  <div class="sys-row">
    <span>System</span>
    <input type="text" id="inp-system-m" placeholder="System prompt (optional)…">
  </div>
</div>

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
const API_TOKEN = "__API_TOKEN__";
const _apiHeaders = h => API_TOKEN ? {...h, 'X-API-Token': API_TOKEN} : h;

// ── Load models ───────────────────────────────────────────────────────────────
fetch('/v1/models', {headers: _apiHeaders({})}).then(r => r.json()).then(data => {
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
      headers: _apiHeaders({'Content-Type':'application/json'}),
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

    stream: while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (raw === '[DONE]') break stream;
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
if (window.innerWidth <= 640) inp.placeholder = 'Message…';
inp.focus();

// ── Hamburger menu (mobile) ───────────────────────────────────────────────────
const btnHamburger = document.getElementById('btn-hamburger');
const settingsPanel = document.getElementById('settings-panel');

// Sync mobile ↔ desktop controls
function isMobile() { return window.innerWidth <= 640; }

function getCtrl(id) {
  return document.getElementById(isMobile() ? id + '-m' : id) ||
         document.getElementById(id);
}

// Override the getters used in send() to prefer mobile controls when open
const _origSend = send;
function getChecked(id) {
  const mob = document.getElementById(id + '-m');
  const desk = document.getElementById(id);
  return (mob && settingsPanel.classList.contains('open')) ? mob.checked : (desk ? desk.checked : false);
}

// Populate mobile model selector from desktop one (after fetch)
const desktopModel = document.getElementById('sel-model');
const mobileModel  = document.getElementById('sel-model-m');
const _origFetch = window.fetch;
new MutationObserver(() => {
  if (mobileModel && desktopModel.options.length > 1) {
    mobileModel.innerHTML = desktopModel.innerHTML;
  }
}).observe(desktopModel, {childList: true});

// Sync model selection between panels
desktopModel.addEventListener('change', () => { if (mobileModel) mobileModel.value = desktopModel.value; });
if (mobileModel) mobileModel.addEventListener('change', () => { desktopModel.value = mobileModel.value; });

// Sync checkboxes desktop ↔ mobile
['chk-search','chk-tools','chk-canvas','chk-json'].forEach(id => {
  const d = document.getElementById(id), m = document.getElementById(id + '-m');
  if (d && m) {
    d.addEventListener('change', () => m.checked = d.checked);
    m.addEventListener('change', () => d.checked = m.checked);
  }
});

// Sync system prompt
const dSys = document.getElementById('inp-system'), mSys = document.getElementById('inp-system-m');
if (dSys && mSys) {
  dSys.addEventListener('input', () => mSys.value = dSys.value);
  mSys.addEventListener('input', () => dSys.value = mSys.value);
}

// Hint animation — show settings panel briefly on first visit (mobile only)
if (btnHamburger && settingsPanel && window.innerWidth <= 640 && !localStorage.getItem('menu-hint-seen')) {
  setTimeout(() => {
    settingsPanel.classList.add('open');
    btnHamburger.style.color = 'var(--accent)';
    if (mobileModel && desktopModel.options.length > 1) {
      mobileModel.innerHTML = desktopModel.innerHTML;
      mobileModel.value = desktopModel.value;
    }
    setTimeout(() => {
      settingsPanel.style.transition = 'opacity .4s ease';
      settingsPanel.style.opacity = '0';
      setTimeout(() => {
        settingsPanel.classList.remove('open');
        settingsPanel.style.transition = '';
        settingsPanel.style.opacity = '';
        btnHamburger.style.color = '';
        localStorage.setItem('menu-hint-seen', '1');
      }, 420);
    }, 1600);
  }, 700);
}

// Toggle panel
if (btnHamburger && settingsPanel) {
  btnHamburger.addEventListener('click', e => {
    e.stopPropagation();
    const open = settingsPanel.classList.toggle('open');
    btnHamburger.style.color = open ? 'var(--accent)' : '';
    // sync mobile model list on first open
    if (open && mobileModel && desktopModel.options.length > 1) {
      mobileModel.innerHTML = desktopModel.innerHTML;
      mobileModel.value = desktopModel.value;
    }
  });
  document.addEventListener('click', e => {
    if (!settingsPanel.contains(e.target) && e.target !== btnHamburger) {
      settingsPanel.classList.remove('open');
      btnHamburger.style.color = '';
    }
  });
}
</script>
</body>
</html>"""


def _render_chat_html() -> str:
    return _CHAT_HTML.replace("__API_TOKEN__", _API_TOKEN)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat_ui():
    """Serve the built-in web chat interface."""
    return HTMLResponse(content=_render_chat_html())


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


def _make_completion(content, model: str, completion_id: str) -> dict:
    words = len(content.split()) if isinstance(content, str) else sum(
        len((p.get("text") or "").split()) for p in content if isinstance(p, dict)
    )
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
    """Extract plain text from a message (str, list of ContentPart, or None)."""
    if m.content is None:
        return ""
    if isinstance(m.content, str):
        return m.content
    if isinstance(m.content, list):
        return " ".join(p.text or "" for p in m.content if p.type == "text")
    return ""


def _call_signature(call: dict) -> tuple[str, str]:
    """(name, arguments-as-text) for one OpenAI tool_call."""
    fn   = (call or {}).get("function") or {}
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args or {}, ensure_ascii=False)
    return fn.get("name", "?"), args


def _tool_call_index(messages: List[Message]) -> dict[str, tuple[str, str]]:
    """{tool_call_id: (name, arguments)} — lets a result name what it answers."""
    index = {}
    for m in messages:
        for call in (m.tool_calls or []):
            if isinstance(call, dict) and call.get("id"):
                index[call["id"]] = _call_signature(call)
    return index


def _describe_tool_calls(m: Message) -> str:
    """An assistant turn that called functions, written out for the prompt."""
    parts = [f"{n}({a})" for n, a in (_call_signature(c) for c in (m.tool_calls or []))]
    return "called " + ", ".join(parts) if parts else ""


def _resolve_messages(
    messages: List[Message],
    user_files: dict[str, dict],
) -> tuple[str, str, List[str], List[dict], bool]:
    """
    Extract:
      - system_prompt: content of the system role message
      - last_user_text: text of the last user message, with prior history
        injected as context (for multi-turn without server-side state)
      - file_texts: contents of file attachments in the last message
      - image_parts: OpenAI image_url parts on the last user message (vision
        input); each is the raw {"url": ..., "detail"?: ...} dict, uploaded and
        attached later by the caller. Requires an authenticated account.
      - tool_followup: the array ENDS in tool results, i.e. this request is the
        second leg of a function-calling round trip. The caller has already run
        the functions and wants the answer, so the turn to send is the results,
        not the user message that triggered them.
    """
    system_prompt = ""
    file_texts: List[str] = []
    image_parts: List[dict] = []

    for m in messages:
        if m.role == "system":
            system_prompt = _msg_to_text(m)

    # Tool results are trailing when they close the array: [.., assistant(calls), tool, tool]
    trailing_tools: List[Message] = []
    for m in reversed(messages):
        if m.role != "tool":
            break
        trailing_tools.insert(0, m)

    if trailing_tools:
        return (system_prompt,
                _tool_followup_text(messages, trailing_tools),
                file_texts, image_parts, True)

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
                iu = part.image_url or {}
                if iu.get("url"):
                    image_parts.append(iu)

    # Inject prior history (user + assistant + tool round trips) as context
    prior_turns = _history_lines(messages, stop_at=last)

    if prior_turns:
        history_block  = "\n".join(prior_turns)
        last_user_text = (
            f"[Prior conversation — use this as context:\n{history_block}\n]\n\n"
            f"{last_user_text}"
        )

    if not last_user_text and not image_parts:
        raise HTTPException(400, "Last user message is empty")

    return system_prompt, last_user_text, file_texts, image_parts, False


def _history_lines(messages: List[Message], stop_at: Optional[Message] = None) -> List[str]:
    """Prior turns as prompt lines. Function calls and their results included.

    A round trip that loses the tool leg is worse than one that never called a
    tool: the model sees the original question again with no answer attached and
    simply re-answers it, so the caller's function ran for nothing.
    """
    index = _tool_call_index(messages)
    lines: List[str] = []
    for m in messages:
        if stop_at is not None and m is stop_at:
            break
        if m.role == "tool":
            fn = index.get(m.tool_call_id or "", ("tool", ""))[0]
            lines.append(f"Tool result ({fn}): {_msg_to_text(m).strip()}")
            continue
        if m.role not in ("user", "assistant"):
            continue
        text = _msg_to_text(m).strip()
        if m.role == "assistant" and m.tool_calls:
            described = _describe_tool_calls(m)
            lines.append(f"Assistant: {text} [{described}]".replace(":  [", ": [")
                         if text else f"Assistant: [{described}]")
            continue
        if not text:
            continue
        lines.append(("User: " if m.role == "user" else "Assistant: ") + text)
    return lines


def _tool_followup_text(messages: List[Message], trailing: List[Message]) -> str:
    """The turn to send when the caller has just run the functions.

    The results lead, the conversation that produced them follows as context,
    and the instruction is explicit about not calling the same thing twice --
    otherwise the model happily emits the call it already got an answer for.
    """
    index   = _tool_call_index(messages)
    history = _history_lines(messages, stop_at=trailing[0])
    # Written as plain lines, not as JSON carrying JSON: a result nested inside
    # an envelope comes out double-escaped, which is exactly the shape the model
    # reads worst.
    results = []
    for m in trailing:
        name, args = index.get(m.tool_call_id or "", ("tool", ""))
        results.append(f"{name}({args}) -> {_msg_to_text(m).strip()}")
    block = "\n".join(results)
    parts = []
    if history:
        parts.append("[Prior conversation — use this as context:\n" + "\n".join(history) + "\n]")
    parts.append(
        f"[Function results — you asked for these and they came back:\n{block}\n]\n\n"
        "Answer the user's request using these results. Do not call the same function "
        "again unless something genuinely new is needed, and never invent a result."
    )
    return "\n\n".join(parts)


# --- Vision input (image_url parts) ----------------------------------------

_VISION_MAX_IMAGES = 10
_VISION_MAX_BYTES  = 20 * 1024 * 1024   # 20 MB per image
_VISION_EXT = {"image/png": ".png", "image/jpeg": ".jpg",
               "image/webp": ".webp", "image/gif": ".gif"}


def _decode_data_url(url: str) -> tuple[bytes, str]:
    """Parse a base64 `data:` URL → (bytes, mime)."""
    try:
        head, b64 = url.split(",", 1)
    except ValueError:
        raise HTTPException(400, "malformed data: URL")
    if "base64" not in head:
        raise HTTPException(400, "only base64 data: URLs are supported")
    mime = "image/png"
    meta = head[len("data:"):]
    if meta:
        mime = meta.split(";", 1)[0] or mime
    try:
        data = base64.b64decode(b64, validate=False)
    except Exception:
        raise HTTPException(400, "invalid base64 in data: URL")
    if not data:
        raise HTTPException(400, "empty image data")
    return data, (mime or "image/png")


async def _fetch_remote_image(url: str) -> tuple[bytes, str]:
    """Download a remote image → (bytes, mime), with an SSRF guard.

    This proxy can reach internal services (e.g. the Coolify API on localhost),
    so remote fetches must resolve to a public address and redirects are not
    followed (a 3xx could bounce to an internal target)."""
    host = (urlparse(url).hostname or "").strip("[]")
    if not host:
        raise HTTPException(400, "image URL has no host")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except Exception:
        raise HTTPException(400, "could not resolve image URL host")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise HTTPException(400, "image URL host is not allowed")
    try:
        async with _httpx.AsyncClient(follow_redirects=False, timeout=30.0) as c:
            r = await c.get(url, headers={"User-Agent": "chatgpt-proxy/vision"})
    except Exception as e:
        raise HTTPException(400, f"could not fetch image URL: {e}")
    if r.status_code != 200:
        raise HTTPException(400, f"image URL returned HTTP {r.status_code}")
    mime = (r.headers.get("content-type") or "").split(";")[0].strip()
    if not mime.startswith("image/"):
        mime = "image/png"   # some CDNs mislabel; the backend sniffs the bytes
    return r.content, mime


async def _prepare_images(session, image_parts: List[dict]) -> List[dict]:
    """Load each OpenAI image_url part (data: or http(s) URL) and upload it to
    the backend for vision input. Returns the pointer dicts stream_message()
    consumes. Raises HTTPException on bad/oversized input."""
    if len(image_parts) > _VISION_MAX_IMAGES:
        raise HTTPException(400, f"too many images (max {_VISION_MAX_IMAGES})")
    out: List[dict] = []
    for idx, part in enumerate(image_parts):
        url = (part.get("url") or "").strip()
        if not url:
            continue
        if url.startswith("data:"):
            data, mime = _decode_data_url(url)
        elif url.startswith("http://") or url.startswith("https://"):
            data, mime = await _fetch_remote_image(url)
        else:
            raise HTTPException(400, "image_url.url must be a data: or http(s) URL")
        if len(data) > _VISION_MAX_BYTES:
            raise HTTPException(400, f"image too large (max {_VISION_MAX_BYTES // (1024 * 1024)} MB)")
        ext = _VISION_EXT.get(mime, ".png")
        try:
            meta = await session.upload_image(data, f"image_{idx + 1}{ext}", mime)
        except Exception as e:
            raise HTTPException(502, f"image upload failed: {e}")
        out.append(meta)
    return out

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
            # Capabilities as upstream reports them, so a client can tell what a
            # model actually supports instead of guessing. context_window is the
            # only real token limit in this API — there is no max_tokens *request*
            # parameter, only this per-model ceiling.
            "context_window":  m.get("max_tokens"),
            "reasoning_type":  m.get("reasoning_type"),
            "enabled_tools":   m.get("enabled_tools") or [],
            "configurable_thinking_effort": bool(m.get("configurable_thinking_effort")),
            "thinking_efforts": [
                e.get("thinking_effort") if isinstance(e, dict) else e
                for e in (m.get("thinking_efforts") or [])
            ],
        }
        for m in cached_models
        if m.get("slug") or m.get("id")
    ]

    # Every entry carries the same keys, so a client can read capabilities off any
    # model without special-casing aliases.
    _BLANK_CAPS = {
        "context_window": None, "reasoning_type": None, "enabled_tools": [],
        "configurable_thinking_effort": False, "thinking_efforts": [],
    }
    by_id = {d["id"]: d for d in data}

    # Add legacy aliases for OpenAI-compatible clients
    existing_ids = set(by_id)
    for alias, target in _LEGACY_ALIASES.items():
        if alias not in existing_ids:
            caps = {k: by_id.get(target, {}).get(k, v) for k, v in _BLANK_CAPS.items()}
            data.append({
                "id":          alias,
                "object":      "model",
                "created":     1750000000,
                "owned_by":    "openai",
                "description": f"Alias → {target}",
                **caps,
            })

    # Always expose image-generation models (these go through the chat
    # completions endpoint and return image_url content parts).
    for img_id in sorted(_IMAGE_MODELS):
        if img_id not in existing_ids:
            data.append({
                "id":          img_id,
                "object":      "model",
                "created":     1750000000,
                "owned_by":    "openai",
                "description": "Image generation — returns image_url content parts",
                **_BLANK_CAPS,
            })

    # --- capability contract: per-model metadata -----------------------------
    # The provider-level block on /health cannot say that gpt-image-1 draws and
    # does not chat, or that the two -t-mini models carry 5x the context of the
    # rest. Those are the three things that genuinely vary per model, plus the
    # sizes. A per-model value may only NARROW the provider-level one -- claiming
    # a capability the account does not have would be a lie with a smaller blast
    # radius, not a smaller lie.
    #
    # `context_window` is already set above from the vendor's `max_tokens`, and
    # is None for the aliases and the image models (which get _BLANK_CAPS). The
    # default below is a floor for exactly those, not a replacement for a real
    # value.
    _DEFAULT_CONTEXT = 52815          # what the vendor reports for this family
    _DEFAULT_MAX_OUTPUT = 8192

    # capabilities.snapshot() is synchronous and, on a cache miss, blocks on a
    # vendor round trip behind a threading.Lock -- offloaded to a worker thread
    # for the same reason /health does (see the comment there): awaiting it
    # directly would stall this process's single asyncio event loop.
    state = await asyncio.to_thread(capabilities.snapshot)
    provider_level = capabilities.effective(state)
    for entry in data:
        draws = entry["id"] in _IMAGE_MODELS
        entry["context_window"] = int(entry.get("context_window") or _DEFAULT_CONTEXT)
        entry["max_output_tokens"] = 0 if draws else _DEFAULT_MAX_OUTPUT
        entry["capabilities"] = {
            # `and provider_level[...]` IS the narrowing rule, applied at the
            # source: on an anonymous session or a free plan these come back
            # False no matter what the model would be capable of.
            "tools":  (not draws) and provider_level["tools"],
            "vision": (not draws) and provider_level["vision"],
            "images": draws and provider_level["images"],
        }

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
    await require_capability("files")
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
    await require_capability("files")
    user_id = _get_user_id(request)
    uf      = _user_files(user_id)
    data    = [
        {k: v for k, v in f.items() if k != "content"}
        for f in uf.values()
    ]
    return {"object": "list", "data": data}


@app.get("/v1/files/{file_id}")
async def get_file(file_id: str, request: Request):
    await require_capability("files")
    user_id = _get_user_id(request)
    uf      = _user_files(user_id)
    if file_id not in uf:
        raise HTTPException(404, f"File {file_id!r} not found")
    return {k: v for k, v in uf[file_id].items() if k != "content"}


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, request: Request):
    await require_capability("files")
    user_id = _get_user_id(request)
    uf      = _user_files(user_id)
    if file_id not in uf:
        raise HTTPException(404, f"File {file_id!r} not found")
    del uf[file_id]
    return {"id": file_id, "object": "file", "deleted": True}


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

# Models that route to image generation instead of the normal chat flow.
# Requests to these models via /v1/chat/completions return
# content: [{type: "image_url", image_url: {url: "..."}}] — the same
# format OpenAI's gpt-image-1 uses.
_IMAGE_MODELS: frozenset[str] = frozenset({"gpt-image-1"})


def _extract_image_prompt(messages: list) -> str:
    for m in reversed(messages):
        if m.role != "user":
            continue
        if isinstance(m.content, str):
            return m.content
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
    return ""


async def _image_via_chat_completions(
    req, pool, msgs_raw: list, completion_id: str
):
    """Run image generation and return a chat-completion response.

    The response format mirrors OpenAI's gpt-image-1: each generated image
    becomes a content part of type "image_url".  Streaming is simulated: if
    the caller asked for stream:true we still generate synchronously (image
    generation is inherently blocking) and emit a single content delta.
    """
    # Image generation needs an account -- fail fast so an anonymous call does
    # not spend a message and then 503 with "ensure the account is authenticated".
    if not auth.is_authenticated():
        return JSONResponse(status_code=401, content={"error": {
            "message": "Image generation requires an authenticated account "
                       "(set CHATGPT_ACCESS_TOKEN).", "type": "auth_error"}})
    prompt = _extract_image_prompt(req.messages)
    if not prompt:
        raise HTTPException(400, detail={"error": {"message": "No text prompt found in messages."}})

    _key, session = await pool.get(msgs_raw)
    async for _ in session.stream_message(prompt, model="auto", force_use_tools=True):
        pass

    if session._pending_image_ids:
        await session.resolve_image_urls()

    if not session.last_images:
        raise HTTPException(503, detail={"error": {
            "message": "No image was generated. Ensure the account is authenticated.",
            "type": "generation_failed", "code": "generation_failed",
        }})

    content_parts = [
        {"type": "image_url", "image_url": {"url": img["url"]}}
        for img in session.last_images
        if img.get("url")
    ]

    if req.stream:
        async def _stream():
            role_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n".encode()
            for part in content_parts:
                delta_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": [part]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(delta_chunk)}\n\n".encode()
            done_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        return StreamingResponse(_stream(), media_type="text/event-stream")

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content_parts},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


def _positive_int(raw: Optional[str], default: int) -> int:
    """A query param that must be a non-negative int, or the default.

    Garbage falls back rather than 422ing: these are pagination knobs on a
    listing, and a caller that sends limit=abc wants a page, not a validation
    lecture.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _remember_conversation(user_id: str, session) -> None:
    """Index an anonymous turn so it outlives the session that made it.

    Anonymous only. An account already has server-side history and a listing to
    match, so recording it here would put chat titles on this disk for nothing.

    Never lets a bookkeeping failure break a turn that already succeeded: the
    answer is on its way to the caller by the time this runs, and a locked or
    unwritable database is a reason to lose the index entry, not the reply.
    """
    if auth.is_authenticated() or not getattr(session, "conversation_id", None):
        return
    try:
        conv_store.record(user_id, session.conversation_id, session.device_id,
                          getattr(session, "last_title", None))
    except Exception as e:                                   # noqa: BLE001
        print(f"[conv_store] no se pudo indexar {session.conversation_id}: {e}",
              file=sys.stderr, flush=True)


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


def _tool_calls_completion(calls: list, model: str, completion_id: str) -> dict:
    """A chat.completion whose choice is a set of calls for the caller to run."""
    return {
        "id":      completion_id,
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": None, "tool_calls": calls},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }


def _tool_calls_stream(calls: list, model: str, completion_id: str):
    """The same result as SSE. One delta per call, then finish_reason tool_calls.

    Each delta carries the whole function object rather than dribbling the
    arguments out character by character: the extraction is already complete
    when this runs, so splitting it would fake a progress the proxy does not
    have, and every OpenAI client accumulates by `index` either way.
    """
    def gen():
        head = {"id": completion_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"},
                             "finish_reason": None}]}
        yield f"data: {json.dumps(head)}\n\n".encode()
        for i, call in enumerate(calls):
            delta = {"id": completion_id, "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "finish_reason": None,
                                  "delta": {"tool_calls": [{"index": i, **call}]}}]}
            yield f"data: {json.dumps(delta)}\n\n".encode()
        tail = {"id": completion_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(tail)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    return gen()


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

    # A "g-..." model id means "run this custom GPT" (gizmo). The proxy routes the
    # conversation through it and keeps its own sessions per GPT.
    gizmo_id = req.model if isinstance(req.model, str) and req.model.startswith("g-") else None

    if req.model in _IMAGE_MODELS:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return await _image_via_chat_completions(req, pool, msgs_raw, completion_id)
    json_mode = req.is_json_mode()

    system_prompt, last_user_text, file_texts, image_parts, tool_followup = \
        _resolve_messages(req.messages, uf)

    # ── Custom function calling ────────────────────────────────────────────────
    # The caller declared functions of its own, so this turn may be a call rather
    # than an answer. Resolved by a separate stateless extraction (see
    # tool_calls.py for why it cannot ride along inside the conversation), and
    # only on the FIRST leg: once the results are back, the job is to answer.
    functions = _tc.custom_functions(req.tools, _BUILTIN_TOOL_NAMES)
    emulate   = req.tool_emulation if req.tool_emulation is not None else _tc.EMULATION_ENABLED
    if (functions and emulate and not tool_followup
            and req.tool_choice != "none" and not image_parts):
        try:
            extraction = await _tc.extract(
                functions, last_user_text, req.tool_choice or "auto",
                verify=req.tool_verify,
            )
        except QuotaExceededError:
            raise HTTPException(429, detail=_quota_error_payload())

        head = {"X-Proxy-Tool-Extraction": extraction.status,
                "X-Proxy-Tool-Requests":   str(extraction.requests)}
        if extraction.status == "calls":
            cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            if req.stream:
                return StreamingResponse(
                    _tool_calls_stream(extraction.tool_calls, req.model, cid),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **head},
                )
            return JSONResponse(
                _tool_calls_completion(extraction.tool_calls, req.model, cid), headers=head,
            )
        if extraction.status == "need_info":
            # OpenAI has no wire format for "ask the user for the missing
            # argument", so the turn falls through to a normal answer -- but it
            # has to fall through as a QUESTION. Left alone the model fills the
            # gap itself: asked "what is the temperature?" with no city it
            # answered for Bogotá, from web search. So the gap is named in the
            # prompt and the backend's search is shut off for this turn.
            missing = (extraction.need_info or {}).get("missing") or []
            if missing:
                last_user_text = (
                    "[You cannot answer this yet: the request does not state "
                    + ", ".join(str(x) for x in missing) +
                    ". Ask the user for exactly that, in their language, in one short "
                    "sentence. Do not guess it and do not answer from your own knowledge.]\n\n"
                    + last_user_text
                )
                req.web_search = False
            head["X-Proxy-Tool-Need-Info"] = json.dumps(extraction.need_info or {},
                                                        ensure_ascii=False)[:900]
        # "no_call" falls through too: no declared function fits, so this is an
        # ordinary chat turn and the caller gets prose, exactly as it should.
        _tool_headers = head
    else:
        _tool_headers = {}

    # Vision input needs a real account (the anonymous backend has no file
    # upload). Fail fast before spending a message or uploading anything.
    if image_parts and not auth.is_authenticated():
        return _needs_account()

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    _key, session = await pool.get(msgs_raw, system_prompt=system_prompt, gizmo_id=gizmo_id)

    # Upload any images once, up front. The file ids are account-scoped, so they
    # stay valid even if a quota retry swaps in a fresh device session below.
    uploaded_images = await _prepare_images(session, image_parts) if image_parts else None

    force_use_search, force_use_tools, force_use_canvas = req.resolved_backend_flags()
    # Both raise a 400 before the upstream request is made, so an unsupported
    # value never costs a message from the anonymous quota.
    thinking_effort = req.resolved_thinking_effort()
    service_tier    = req.resolved_service_tier()

    # Advertised on the response so a caller that sent temperature/top_p/... can
    # see they were dropped instead of assuming they took effect.
    ignored = req.ignored_params()
    extra_headers = {"X-Proxy-Ignored-Params": ",".join(ignored)} if ignored else {}
    # An extraction that decided "no call" (or that needs an argument) still
    # reports itself, so a caller can account for the message it cost.
    extra_headers.update(_tool_headers)

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
                        force_use_canvas  = force_use_canvas,
                        json_mode         = json_mode,
                        thinking_effort   = thinking_effort,
                        service_tier      = service_tier,
                        force_disable_features = req.force_disable_features,
                        images            = uploaded_images,
                    ):
                        if text_chunk:
                            yield _make_chunk(text_chunk, req.model, completion_id).encode()
                    break  # success

                except QuotaExceededError:
                    if attempt == 0:
                        # First failure: transparently retry with a fresh device_id
                        pool._pool.pop(cur_key, None)
                        cur_key, cur_session = await pool.get(msgs_raw, system_prompt=system_prompt, gizmo_id=gizmo_id)
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

            _remember_conversation(user_id, cur_session)

            # Resolve DALL-E images and emit them as image_url content delta parts
            if cur_session._pending_image_ids:
                await cur_session.resolve_image_urls()
            for img in cur_session.last_images:
                url = img.get("url", "")
                if not url:
                    continue
                img_delta = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": req.model,
                    "choices": [{"index": 0, "finish_reason": None,
                                 "delta": {"content": [{"type": "image_url",
                                                        "image_url": {"url": url}}]}}],
                }
                yield f"data: {json.dumps(img_delta)}\n\n".encode()

            yield _make_chunk("", req.model, completion_id, finish=True).encode()

            if cur_session.last_search_queries or cur_session.last_citations:
                meta = _build_search_metadata(cur_session)
                yield f"event: search_metadata\ndata: {json.dumps(meta)}\n\n".encode()
            if cur_session.last_widgets:
                yield f"event: widgets\ndata: {json.dumps(cur_session.last_widgets)}\n\n".encode()
            if cur_session.last_images:
                yield f"event: images\ndata: {json.dumps(cur_session.last_images)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     **extra_headers},
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
                    force_use_canvas  = force_use_canvas,
                    json_mode         = json_mode,
                    thinking_effort   = thinking_effort,
                    service_tier      = service_tier,
                    force_disable_features = req.force_disable_features,
                    images            = uploaded_images,
                ):
                    full_text += chunk
                break  # success

            except QuotaExceededError:
                if attempt == 0:
                    pool._pool.pop(cur_key, None)
                    cur_key, cur_session = await pool.get(msgs_raw, system_prompt=system_prompt, gizmo_id=gizmo_id)
                else:
                    raise HTTPException(429, detail=_quota_error_payload())

            except Exception as e:
                raise HTTPException(500, str(e))

        _remember_conversation(user_id, cur_session)

        # Resolve any DALL-E images collected during streaming
        if cur_session._pending_image_ids:
            await cur_session.resolve_image_urls()

        # Use clean text (citations preserved, genui removed)
        clean = cur_session.last_clean_text or full_text
        if json_mode:
            clean = _extract_json(clean)

        image_parts = [
            {"type": "image_url", "image_url": {"url": img["url"]}}
            for img in cur_session.last_images
            if img.get("url")
        ]
        if image_parts:
            # Return content as an array so image_url parts are OpenAI-compatible
            # and llm-libre can rehost them.  A leading text part carries the
            # model's commentary (e.g. "Here is the image you requested").
            content: list | str = (
                [{"type": "text", "text": clean}] + image_parts if clean.strip()
                else image_parts
            )
            resp = _make_completion(content, req.model, completion_id)
        else:
            resp = _make_completion(clean, req.model, completion_id)
        if cur_session.last_search_queries or cur_session.last_citations:
            resp["search_metadata"] = _build_search_metadata(cur_session)
        if cur_session.last_widgets:
            resp["widgets"] = cur_session.last_widgets
        if cur_session.last_images:
            resp["images"] = cur_session.last_images
        if service_tier:
            resp["service_tier"] = service_tier
        return JSONResponse(resp, headers=extra_headers)


# ---------------------------------------------------------------------------
# /v1/tool-calls
# ---------------------------------------------------------------------------

class ToolCallsRequest(BaseModel):
    """Resolve one request into function calls. Stateless: no conversation, no
    history, no session reuse — which is exactly why it is reliable."""
    tools:       List[dict]
    input:       Optional[str] = None
    messages:    Optional[List[Message]] = None
    tool_choice: Optional[Union[str, dict]] = "auto"
    # Defaults to the extractor model rather than "auto": measured at the same
    # accuracy with a much shorter latency tail.
    model:       Optional[str] = None
    verify:      bool = False


@app.post("/v1/tool-calls")
async def tool_calls_endpoint(req: ToolCallsRequest, request: Request):
    functions = _tc.custom_functions(req.tools, _BUILTIN_TOOL_NAMES)
    if not functions:
        raise HTTPException(400, detail={"error": {
            "message": "No custom functions in `tools`. Built-in names "
                       f"({', '.join(sorted(_BUILTIN_TOOL_NAMES))}) select a ChatGPT "
                       "mode on /v1/chat/completions and have nothing to extract.",
            "type": "invalid_request_error", "param": "tools",
        }})

    text = req.input or ""
    if not text and req.messages:
        uf = _user_files(_get_user_id(request))
        _system, text, _files, _images, _followup = _resolve_messages(req.messages, uf)
    if not text.strip():
        raise HTTPException(400, detail={"error": {
            "message": "Provide `input` (a string) or `messages`.",
            "type": "invalid_request_error", "param": "input",
        }})

    try:
        extraction = await _tc.extract(functions, text, req.tool_choice or "auto",
                                       model=req.model, verify=req.verify)
    except QuotaExceededError:
        raise HTTPException(429, detail=_quota_error_payload())

    body = {
        "id":      f"toolcall-{uuid.uuid4().hex[:24]}",
        "object":  "tool_calls",
        "created": int(time.time()),
        "model":   req.model or _tc.EXTRACTOR_MODEL,
        "status":  extraction.status,          # calls | no_call | need_info
        "tool_calls": extraction.tool_calls,
        # Spent upstream messages, not tokens: on this backend the message is
        # the quota unit, so it is the number a caller can actually budget.
        "usage":   {"upstream_requests": extraction.requests},
    }
    if extraction.status == "need_info":
        body["need_info"] = extraction.need_info
    if extraction.notes:
        body["notes"] = [n for n in extraction.notes if isinstance(n, str)]
    return JSONResponse(body)


# ---------------------------------------------------------------------------
# /v1/images/generations
# ---------------------------------------------------------------------------

def _prompt_with_size(prompt: str, size: "str | None") -> str:
    """Turn a `size` into words, because words are the only channel there is.

    The upstream flow has no size field. Verified against the decompiled
    official app (com.openai.chatgpt 1.2026.223): `image_size` and
    `aspect_ratio` appear ONLY in a telemetry event, and no file in the whole
    APK carries both `image_gen` and a size key -- the client reports the shape
    that came back, it never asks for one. The picture is drawn by a tool the
    model drives in natural language, so an aspect ratio can only be requested
    the way a person would request it.

    Which is why this translates ORIENTATION and not pixels: asking a drawing
    tool for "1536x1024" invites it to render the digits into the picture, while
    "horizontal" is an instruction it already understands. Exact pixel sizes were
    never deliverable here and pretending otherwise is the bug being fixed.

    A square size appends nothing. The common path should not pay tokens, and a
    gratuitous "make it square" also steers the composition for no reason.

    Unreadable input changes nothing and never raises: the value arrives verbatim
    from a client, and a size that cannot be parsed is a reason to ignore it, not
    a reason to fail an image the caller is spending a minute of wall clock on.
    """
    if not isinstance(size, str):
        return prompt
    parts = size.strip().lower().split("x")
    if len(parts) != 2:
        return prompt
    try:
        width, height = (int(p) for p in parts)
    except ValueError:
        return prompt
    if width <= 0 or height <= 0 or width == height:
        return prompt
    shape = "vertical (más alta que ancha)" if height > width \
        else "horizontal (más ancha que alta)"
    return f"{prompt}\n\n(Genera la imagen en formato {shape}.)"


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "dall-e-3"
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"


@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest, request: Request):
    await require_capability("images")
    # require_capability above is the real fail-fast gate now (401/501 before
    # any message is spent); this check is redundant defense-in-depth below it.
    if not auth.is_authenticated():
        return JSONResponse(status_code=401, content={"error": {
            "message": "Image generation requires an authenticated account "
                       "(set CHATGPT_ACCESS_TOKEN).", "type": "auth_error"}})
    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    prompt = _prompt_with_size(req.prompt, req.size)
    msgs_raw = [{"role": "user", "content": prompt}]
    _key, session = await pool.get(msgs_raw)

    async for _ in session.stream_message(
        prompt,
        model="auto",
        force_use_tools=True,
    ):
        pass

    if session._pending_image_ids:
        await session.resolve_image_urls()

    if not session.last_images:
        raise HTTPException(503, detail={
            "error": {
                "message": "No image was generated. Ensure the account is authenticated.",
                "type": "generation_failed",
                "code": "generation_failed",
            }
        })

    data = [{"url": img["url"], "revised_prompt": req.prompt}
            for img in session.last_images[: req.n]]

    return JSONResponse({"created": int(time.time()), "data": data})


# ---------------------------------------------------------------------------
# /images/{filename} — serve locally cached image files generated by ChatGPT
# ---------------------------------------------------------------------------

@app.get("/images/{filename}", include_in_schema=False)
async def serve_image(filename: str):
    """Serve a locally downloaded image file from _IMAGE_STORE_DIR."""
    safe = pathlib.Path(filename).name  # strip any path traversal
    path = _IMAGE_STORE_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(path))


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    total_sessions = sum(len(p._pool) for p in _pools.values())
    total_files    = sum(len(uf) for uf in _files.values())
    # Offloaded to a worker thread: capabilities.snapshot() is synchronous and,
    # on a cache miss, blocks on a vendor round trip (up to 15s) behind a
    # threading.Lock. Awaiting it directly would stall THIS process's single
    # asyncio event loop for that whole span -- freezing every in-flight chat
    # completion and streaming response, not merely other /health callers.
    state = await asyncio.to_thread(capabilities.snapshot)
    return {
        "status":  "ok",
        "version": "2.5.0",
        # The capability contract (llm-libre spec 2026-08-20). Everything under
        # `capabilities` is EFFECTIVE: already resolved against this account and
        # its plan, so the gateway reads one boolean instead of learning what a
        # ChatGPT plan is. See capabilities.effective.
        "contract": 1,
        "provider": "chatgpt",
        "auth": {
            "mode":                state.mode,
            "plan":                state.plan,
            "subscription_active": state.subscription_active,
            "expires_at":          state.expires_at,
        },
        "capabilities": capabilities.effective(state),
        # Kept for compatibility: Coolify's container health check and the
        # existing dashboards read these, and the contract is additive.
        # `auth_mode` was the only machine-readable field the old block had.
        "auth_mode":             state.mode,
        "active_users":          len(_pools),
        "total_sessions":        total_sessions,
        "total_files_in_memory": total_files,
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


@app.get("/v1/account")
async def account(request: Request):
    """The authenticated account's plan, subscription and enabled features.

    Proxies /backend-api/accounts/check and returns a clean summary. Requires an
    account (CHATGPT_ACCESS_TOKEN); anonymous sessions have no account to report.
    """
    if not auth.is_authenticated():
        return JSONResponse(
            status_code=401,
            content={"error": {
                "message": "This endpoint needs an authenticated account "
                           "(set CHATGPT_ACCESS_TOKEN).",
                "type": "auth_error"}},
        )

    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())
    if sessions:
        device_id, client, owns_client = sessions[0].device_id, sessions[0].client, False
    else:
        device_id, owns_client = str(uuid.uuid4()), True
        client = _httpx.AsyncClient(verify=True, timeout=15.0, follow_redirects=True)

    path = "/backend-api/accounts/check/v4-2023-04-27"
    try:
        hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path}
        r = await client.get(f"{BASE}{path}",
                             params={"timezone_offset_min": "0"}, headers=hdrs)
        r.raise_for_status()
        data = r.json()
    finally:
        if owns_client:
            await client.aclose()

    accts = data.get("accounts", {}) or {}
    acc   = accts.get("default") or next(iter(accts.values()), {}) or {}
    a     = acc.get("account", {}) or {}
    ent   = acc.get("entitlement", {}) or {}
    feats = acc.get("features", []) or []
    return {
        "account_id":     a.get("account_id"),
        "name":           a.get("name"),
        "plan_type":      a.get("plan_type"),
        "is_deactivated": a.get("is_deactivated"),
        "subscription": {
            "plan":             ent.get("subscription_plan"),
            "active":           ent.get("has_active_subscription"),
            "expires_at":       ent.get("expires_at"),
            "renews_at":        ent.get("renews_at"),
            "cancels_at":       ent.get("cancels_at"),
            "billing_period":   ent.get("billing_period"),
            "billing_currency": ent.get("billing_currency"),
            "is_delinquent":    ent.get("is_delinquent"),
        },
        "features": feats,
    }


@app.get("/v1/limits")
async def limits(request: Request):
    """Current rate-limit / usage state, per feature.

    accounts/check has no limits; they live in `limits_progress` on the
    conversation flow. This calls /conversation/init (cheap, sends no message)
    and returns the per-feature remaining counts + reset times, the model caps
    (populated when you approach one), and any blocked features. Works for both
    the authenticated account and the anonymous session.
    """
    prefix = "/backend-api" if auth.is_authenticated() else "/backend-anon"
    path   = prefix + "/conversation/init"

    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())
    if sessions:
        device_id, client, owns_client = sessions[0].device_id, sessions[0].client, False
    else:
        device_id, owns_client = str(uuid.uuid4()), True
        client = _httpx.AsyncClient(verify=True, timeout=15.0, follow_redirects=True)

    try:
        hdrs = {**_base_headers(device_id),
                "X-OpenAI-Target-Path": path, "Content-Type": "application/json"}
        r = await client.post(f"{BASE}{path}", headers=hdrs,
                             json={"conversation_mode_kind": "primary_assistant"})
        r.raise_for_status()
        d = r.json()
    finally:
        if owns_client:
            await client.aclose()

    return {
        "limits_progress":    d.get("limits_progress", []),
        "model_limits":       d.get("model_limits", []),
        "blocked_features":   d.get("blocked_features", []),
        "default_model_slug": d.get("default_model_slug"),
    }


def _needs_account() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "This endpoint needs an authenticated account "
                                      "(set CHATGPT_ACCESS_TOKEN).", "type": "auth_error"}},
    )


async def require_capability(name: str) -> None:
    """Refuse with 501 when this account cannot do `name`.

    501, not 404 and not 503, and the distinction is load-bearing for the
    gateway. A 404 is indistinguishable from a routing mistake. A 503 says "it
    broke" -- so the gateway retries, accumulates suspicion against the route and
    fails over, spending attempts on something that was never going to work on
    this plan. 501 says: this proxy, deliberately, does not do this right now.

    `capabilities.snapshot()` is blocking (guarded by a threading.Lock, and it
    can make a vendor round trip of up to 15 seconds once per refresh interval),
    so it is offloaded to a thread rather than awaited directly -- calling it
    inline here would freeze the event loop for every concurrent request.
    """
    state = await asyncio.to_thread(capabilities.snapshot)
    if not capabilities.effective(state)[name]:
        raise HTTPException(
            501,
            f"This proxy cannot serve '{name}' with its current account "
            f"(see GET /health, capabilities.{name}).")


async def _backend_get(request: Request, path: str, params: dict = None):
    """GET a /backend-api path, reusing a live session's client or a temp one.
    Returns the httpx.Response (caller checks status)."""
    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())
    if sessions:
        device_id, client, owns = sessions[0].device_id, sessions[0].client, False
    else:
        device_id, owns = str(uuid.uuid4()), True
        client = _httpx.AsyncClient(verify=True, timeout=20.0, follow_redirects=True)
    try:
        hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path}
        return await client.get(f"{BASE}{path}", headers=hdrs, params=params)
    finally:
        if owns:
            await client.aclose()


async def _backend_post(request: Request, path: str, json_body: dict):
    """POST JSON to a /backend-api path (same client handling as _backend_get)."""
    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())
    if sessions:
        device_id, client, owns = sessions[0].device_id, sessions[0].client, False
    else:
        device_id, owns = str(uuid.uuid4()), True
        client = _httpx.AsyncClient(verify=True, timeout=20.0, follow_redirects=True)
    try:
        hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path,
                "Content-Type": "application/json"}
        return await client.post(f"{BASE}{path}", headers=hdrs, json=json_body)
    finally:
        if owns:
            await client.aclose()


async def _backend_request(request: Request, method: str, path: str):
    """Generic bodyless backend call (DELETE, no-body POST) with the usual client."""
    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())
    if sessions:
        device_id, client, owns = sessions[0].device_id, sessions[0].client, False
    else:
        device_id, owns = str(uuid.uuid4()), True
        client = _httpx.AsyncClient(verify=True, timeout=20.0, follow_redirects=True)
    try:
        hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path}
        return await client.request(method, f"{BASE}{path}", headers=hdrs)
    finally:
        if owns:
            await client.aclose()


@app.get("/v1/gizmos")
async def gizmos(request: Request):
    """List the account's custom GPTs (gizmos). Use an id as the chat `model`
    (`"model": "g-..."`) to talk to that GPT."""
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_get(request, "/backend-api/gizmos/bootstrap")
    r.raise_for_status()
    out = []
    for it in r.json().get("gizmos", []):
        g    = (it.get("resource") or {}).get("gizmo") or it.get("gizmo") or it
        disp = g.get("display") or {}
        out.append({
            "id":              g.get("id"),
            "name":            disp.get("name"),
            "description":     disp.get("description"),
            "prompt_starters": disp.get("prompt_starters"),
            "short_url":       g.get("short_url"),
            "voice":           (g.get("voice") or {}).get("id"),
        })
    return {"gizmos": out}


@app.get("/v1/gizmos/{gizmo_id}")
async def gizmo_detail(gizmo_id: str, request: Request):
    """Detail of one custom GPT (instructions, starters, config)."""
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_get(request, "/backend-api/gizmos/" + gizmo_id)
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = {"message": r.text[:200]}
        return JSONResponse(status_code=r.status_code,
                            content={"error": {"type": "gizmo_error", "detail": detail}})
    return r.json()


class CustomInstructions(BaseModel):
    """Writable custom-instruction fields; all optional for partial updates."""
    about_user_message:         Optional[str]  = None  # "what should ChatGPT know about you"
    about_model_message:        Optional[str]  = None  # "how should ChatGPT respond"
    traits_model_message:       Optional[str]  = None  # personality traits
    name_user_message:          Optional[str]  = None  # what to call you
    role_user_message:          Optional[str]  = None  # what you do
    other_user_message:         Optional[str]  = None  # anything else
    personality_type_selection: Optional[str]  = None
    disabled_tools:             Optional[list] = None
    enabled:                    Optional[bool] = None
    traits_enabled:             Optional[bool] = None
    personality_traits:         Optional[dict] = None


@app.get("/v1/custom-instructions")
async def get_custom_instructions(request: Request):
    """The account's custom instructions (about_user, about_model, traits, ...)."""
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_get(request, "/backend-api/user_system_messages")
    if r.status_code != 200 or not r.text.strip():
        return JSONResponse(status_code=502, content={"error": {
            "message": "el backend no devolvió datos (transitorio) -- reintentá",
            "type": "upstream_error"}})
    d = r.json()
    d.pop("object", None)
    return d


@app.post("/v1/custom-instructions")
async def set_custom_instructions(req: CustomInstructions, request: Request):
    """Update custom instructions. Partial: only the fields you send change; the
    rest are preserved (merged onto the current values before saving)."""
    if not auth.is_authenticated():
        return _needs_account()

    cur_r = await _backend_get(request, "/backend-api/user_system_messages")
    if cur_r.status_code != 200 or not cur_r.text.strip():
        return JSONResponse(status_code=502, content={"error": {
            "message": "no se pudo leer el estado actual (transitorio) -- reintentá",
            "type": "upstream_error"}})
    merged = cur_r.json()
    merged.pop("object", None)
    merged.update(req.model_dump(exclude_none=True))

    r = await _backend_post(request, "/backend-api/user_system_messages", merged)
    if r.status_code != 200 or not r.text.strip():
        try:
            detail = r.json()
        except Exception:
            detail = {"message": r.text[:200] or "sin cuerpo"}
        return JSONResponse(status_code=r.status_code if r.status_code != 200 else 502,
                            content={"error": {"type": "custom_instructions_error", "detail": detail}})
    d = r.json()
    d.pop("object", None)
    return d


class TranslateRequest(BaseModel):
    text:   str
    target: str            # target language code, e.g. "en", "es", "fr"
    source: Optional[str] = None  # optional source language code (auto-detect if omitted)


@app.post("/v1/translate")
async def translate(req: TranslateRequest, request: Request):
    """Translate text via ChatGPT's language-learning translate endpoint.

    Lightweight: does NOT spend a chat message (unlike /v1/audio/speech). Give
    `text` and a `target` language code; `source` is optional (auto-detected).
    Returns { text: <translation>, target, source }.
    """
    text = (req.text or "").strip()
    if not text or not req.target:
        return JSONResponse(status_code=400, content={"error": {
            "message": "'text' and 'target' are required", "type": "invalid_request_error"}})

    prefix = "/backend-api" if auth.is_authenticated() else "/backend-anon"
    body = {"text": text, "targetLanguageCode": req.target}
    if req.source:
        body["sourceLanguageCode"] = req.source

    r = await _backend_post(request, prefix + "/language-learning-block/translate", body)
    if r.status_code != 200 or not r.text.strip():
        try:
            detail = r.json()
        except Exception:
            detail = {"message": r.text[:200] or "sin cuerpo"}
        return JSONResponse(status_code=r.status_code if r.status_code != 200 else 502,
                            content={"error": {"type": "translate_error", "detail": detail}})
    d = r.json()
    return {"text": d.get("text"), "target": req.target, "source": req.source}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str = Form(None),
    response_format: str = Form("json"),
):
    """Speech-to-text (OpenAI-compatible). Multipart `file` = the audio.

    Forwards to ChatGPT's /backend-api/transcribe (Whisper). The file's
    content-type must match its bytes (mp3 verified working; m4a/wav/webm too if
    labeled correctly). Returns {"text": ...} (or plain text if
    response_format=text). Authenticated only.
    """
    await require_capability("audio_transcription")
    if not auth.is_authenticated():
        return _needs_account()

    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": {
            "message": "empty audio file", "type": "invalid_request_error"}})
    ctype = file.content_type or "audio/mpeg"
    fname = file.filename or "audio.mp3"

    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())
    if sessions:
        device_id, client, owns = sessions[0].device_id, sessions[0].client, False
    else:
        device_id, owns = str(uuid.uuid4()), True
        client = _httpx.AsyncClient(verify=True, timeout=90.0, follow_redirects=True)

    path = "/backend-api/transcribe"
    try:
        hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path}
        r = await client.post(
            f"{BASE}{path}", headers=hdrs,
            files={"file": (fname, content, ctype)},
            data={"dictation_session_id": str(uuid.uuid4()),
                  "attempt_id":           str(uuid.uuid4())},
        )
    finally:
        if owns:
            await client.aclose()

    if r.status_code != 200 or not r.text.strip():
        try:
            detail = r.json()
        except Exception:
            detail = {"message": r.text[:200] or "sin cuerpo"}
        return JSONResponse(status_code=r.status_code if r.status_code != 200 else 502,
                            content={"error": {"type": "transcription_error", "detail": detail}})

    text = r.json().get("text", "")
    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    return {"text": text}


@app.get("/v1/library")
async def library(request: Request, limit: int = 50, cursor: str = None):
    """List your file library (uploaded + generated files). Authenticated.

    Proxies /backend-api/files/library/nodes. `limit`/`cursor` paginate.
    Each item's `id` is the usable file_id (for /v1/library/{id}/download).
    """
    if not auth.is_authenticated():
        return _needs_account()
    params = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    r = await _backend_get(request, "/backend-api/files/library/nodes", params)
    if r.status_code != 200 or not r.text.strip():
        return JSONResponse(status_code=502, content={"error": {
            "message": "el backend no devolvió datos (transitorio) -- reintentá",
            "type": "upstream_error"}})
    d = r.json()
    items = [{
        "id":            it.get("file_id"),
        "library_id":    it.get("id"),
        "name":          it.get("name"),
        "kind":          it.get("kind"),
        "mime_type":     it.get("mime_type"),
        "size_bytes":    it.get("file_size_bytes"),
        "category":      it.get("library_file_category"),
        "state":         it.get("state"),
        "thumbnail_url": it.get("thumbnail_url"),
        "updated_at":    it.get("updated_at"),
        "directory_id":  it.get("parent_directory_id"),
    } for it in d.get("items", [])]
    return {"items": items, "cursor": d.get("cursor")}


@app.get("/v1/library/usage")
async def library_usage(request: Request):
    """Library storage usage: used/allowed bytes + breakdown by file type."""
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_get(request, "/backend-api/files/library/storage/usage")
    if r.status_code != 200 or not r.text.strip():
        return JSONResponse(status_code=502, content={"error": {
            "message": "el backend no devolvió datos (transitorio) -- reintentá",
            "type": "upstream_error"}})
    return r.json()


@app.get("/v1/library/{file_id}/download")
async def library_download(file_id: str, request: Request, url: bool = False):
    """Download a library file. Takes the `file_id` (not the library id).

    The backend hands back a presigned estuary URL that still requires the
    account's Bearer token, so by default the proxy fetches the bytes itself and
    streams the file back (usable without any auth on the client). Pass `?url=1`
    to instead get the raw (auth-gated) download_url.
    """
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_get(request, "/backend-api/files/download/" + file_id)
    if r.status_code != 200 or not r.text.strip():
        try:
            detail = r.json()
        except Exception:
            detail = {"message": r.text[:200] or "sin cuerpo"}
        return JSONResponse(status_code=r.status_code if r.status_code != 200 else 502,
                            content={"error": {"type": "download_error", "detail": detail}})
    d = r.json()
    durl = d.get("download_url")
    if url:
        return {"file_id": file_id, "download_url": durl, "status": d.get("status")}
    if not durl:
        return JSONResponse(status_code=502, content={"error": {
            "message": "no download_url", "type": "upstream_error"}})

    token = auth.access_token()
    async with _httpx.AsyncClient(timeout=90.0, follow_redirects=True) as c:
        fr = await c.get(durl, headers={"Authorization": "Bearer " + token})
    if fr.status_code != 200:
        return JSONResponse(status_code=fr.status_code, content={"error": {
            "message": "no se pudo descargar el archivo", "type": "upstream_error"}})
    headers = {}
    cd = fr.headers.get("content-disposition")
    if cd:
        headers["Content-Disposition"] = cd
    return Response(content=fr.content,
                    media_type=fr.headers.get("content-type", "application/octet-stream"),
                    headers=headers)


@app.delete("/v1/library/trash")
async def library_empty_trash(request: Request):
    """Empty the library trash -- PERMANENTLY deletes trashed files and frees the
    space. Irreversible. (Soft-delete first with DELETE /v1/library/{library_id}.)"""
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_request(request, "DELETE", "/backend-api/files/library/trash")
    if r.status_code not in (200, 204):
        return JSONResponse(status_code=r.status_code, content={"error": {
            "type": "trash_error", "detail": r.text[:200]}})
    return {"success": True}


@app.delete("/v1/library/{library_id}")
async def library_delete(library_id: str, request: Request):
    """Move a library file to trash (recoverable). Takes the `library_id`
    (libfile_...). Frees space only after emptying the trash."""
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_request(request, "DELETE",
                               "/backend-api/files/library/files/" + library_id)
    if r.status_code not in (200, 204):
        return JSONResponse(status_code=r.status_code, content={"error": {
            "type": "delete_error", "detail": r.text[:200]}})
    return {"success": True, "library_id": library_id, "trashed": True}


@app.post("/v1/library/{library_id}/restore")
async def library_restore(library_id: str, request: Request):
    """Restore a trashed library file. Takes the `library_id` (libfile_...)."""
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_request(request, "POST",
                               "/backend-api/files/library/files/" + library_id + "/restore")
    if r.status_code not in (200, 204) or not r.text.strip():
        return JSONResponse(status_code=r.status_code if r.status_code not in (200, 204) else 502,
                            content={"error": {"type": "restore_error", "detail": r.text[:200]}})
    return r.json()


@app.get("/v1/suggestions")
async def suggestions(request: Request):
    """Suggested prompts (starters) from ChatGPT's prompt library.

    Proxies /backend-api/prompt_library/system_hints and flattens it into a flat
    list of {category, id, title, prompt, description}. Authenticated.
    """
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_get(request, "/backend-api/prompt_library/system_hints")
    if r.status_code != 200 or not r.text.strip():
        return JSONResponse(status_code=502, content={"error": {
            "message": "el backend no devolvió datos (transitorio) -- reintentá",
            "type": "upstream_error"}})
    d = r.json()
    out = []
    for category, items in (d.get("items_by_system_hint") or {}).items():
        for it in (items or []):
            out.append({
                "category":    category,
                "id":          it.get("id"),
                "title":       it.get("title"),
                "prompt":      it.get("prompt"),
                "description": it.get("description"),
            })
    return {"suggestions": out}


class ProjectRequest(BaseModel):
    name:         str
    instructions: str = ""


@app.post("/v1/projects")
async def create_project(req: ProjectRequest, request: Request):
    """Create a ChatGPT Project (a "g-p-..." gizmo that groups chats with shared
    instructions). Authenticated. There is no clean list endpoint upstream, so
    keep the returned `id` -- get it with GET /v1/projects/{id}, delete with
    DELETE /v1/projects/{id}."""
    if not auth.is_authenticated():
        return _needs_account()
    if not (req.name or "").strip():
        return JSONResponse(status_code=400, content={"error": {
            "message": "'name' is required", "type": "invalid_request_error"}})
    r = await _backend_post(request, "/backend-api/projects",
                            {"name": req.name, "instructions": req.instructions})
    if r.status_code != 200 or not r.text.strip():
        try:
            detail = r.json()
        except Exception:
            detail = {"message": r.text[:200] or "sin cuerpo"}
        return JSONResponse(status_code=r.status_code if r.status_code != 200 else 502,
                            content={"error": {"type": "project_error", "detail": detail}})
    g = ((r.json().get("resource") or {}).get("gizmo")) or {}
    disp = g.get("display") or {}
    return {
        "id":           g.get("id"),
        "name":         disp.get("name") or req.name,
        "description":  disp.get("description"),
        "instructions": g.get("instructions"),
        "short_url":    g.get("short_url"),
    }


@app.get("/v1/projects/{project_id}")
async def get_project(project_id: str, request: Request):
    """Detail of a project (a gizmo under the hood -- same as GET /v1/gizmos/{id})."""
    if not auth.is_authenticated():
        return _needs_account()
    r = await _backend_get(request, "/backend-api/gizmos/" + project_id)
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = {"message": r.text[:200]}
        return JSONResponse(status_code=r.status_code,
                            content={"error": {"type": "project_error", "detail": detail}})
    return r.json()


@app.delete("/v1/projects/{project_id}")
async def delete_project(project_id: str, request: Request):
    """Delete a project. Guarded to project ids (g-p-...) so it can't remove a real
    GPT by mistake; projects delete via the gizmo endpoint."""
    if not auth.is_authenticated():
        return _needs_account()
    if not project_id.startswith("g-p-"):
        return JSONResponse(status_code=400, content={"error": {
            "message": "solo se pueden borrar proyectos (id g-p-...). Para GPTs usá el builder de ChatGPT.",
            "type": "invalid_request_error"}})
    r = await _backend_request(request, "DELETE", "/backend-api/gizmos/" + project_id)
    if r.status_code not in (200, 204):
        return JSONResponse(status_code=r.status_code, content={"error": {
            "type": "project_error", "detail": r.text[:200]}})
    return {"deleted": True, "id": project_id}


@app.get("/v1/conversations")
async def conversations(request: Request):
    """List the account's conversation history, most recent first.

    Proxies /backend-api/conversations. Query params: `offset` (default 0),
    `limit` (default 28), `order` (default 'updated').

    Anonymously the vendor has nothing to proxy: /backend-anon/conversations
    answers total=0 even for the device that owns two live conversations
    (measured 2026-08-21). So that case is served from this proxy's own index --
    see conv_store.py -- which lists the conversations THIS proxy created and
    can still open, and nothing else.

    require_capability stays on the authenticated path only, for the same reason
    the detail route below carries none: the `conversations` boolean describes
    the ACCOUNT's server-side history, and a local index of what this proxy
    created is a different thing. The capability set is a byte-for-byte contract
    with the gateway, so it keeps meaning exactly what it meant.
    """
    user_id = _get_user_id(request)
    if not auth.is_authenticated():
        limit  = _positive_int(request.query_params.get("limit"), 28)
        offset = _positive_int(request.query_params.get("offset"), 0)
        rows, total = conv_store.listing(user_id, limit, offset)
        return {
            "items": [{
                "id":          r["conversation_id"],
                "title":       r["title"],
                "snippet":     None,
                "create_time": r["create_time"],
                "update_time": r["update_time"],
                "is_archived": False,
                "is_starred":  False,
                "gizmo_id":    None,
            } for r in rows],
            "total": total, "limit": limit, "offset": offset,
        }

    await require_capability("conversations")

    params = {
        "offset": request.query_params.get("offset", "0"),
        "limit":  request.query_params.get("limit", "28"),
        "order":  request.query_params.get("order", "updated"),
    }
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())
    if sessions:
        device_id, client, owns_client = sessions[0].device_id, sessions[0].client, False
    else:
        device_id, owns_client = str(uuid.uuid4()), True
        client = _httpx.AsyncClient(verify=True, timeout=15.0, follow_redirects=True)

    path = "/backend-api/conversations"
    try:
        hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path}
        r = await client.get(f"{BASE}{path}", params=params, headers=hdrs)
        r.raise_for_status()
        d = r.json()
    finally:
        if owns_client:
            await client.aclose()

    items = [{
        "id":          it.get("id"),
        "title":       it.get("title"),
        "snippet":     it.get("snippet"),
        "create_time": it.get("create_time"),
        "update_time": it.get("update_time"),
        "is_archived": it.get("is_archived"),
        "is_starred":  it.get("is_starred"),
        "gizmo_id":    it.get("gizmo_id"),
    } for it in d.get("items", [])]
    return {"items": items, "total": d.get("total"),
            "limit": d.get("limit"), "offset": d.get("offset")}


@app.get("/v1/conversations/{conversation_id}")
async def conversation_detail(conversation_id: str, request: Request):
    """A single conversation as an ordered message list.

    Flattens the raw ChatGPT `mapping` tree into
    `messages: [{id, role, content, create_time}]`, following creation order and
    dropping empty/system nodes.

    Works anonymously, unlike the LISTING above. Measured 2026-08-20: an
    anonymous conversation is readable at /backend-anon/conversation/{id} from
    the device that created it -- 200, with the title the backend generated and
    the full mapping. From any other device it is 404
    `conversation_inaccessible` ("Log in to view this conversation."), and
    /backend-anon/conversations answers 200 with an empty page no matter what.

    So anonymously this can only return a conversation THIS proxy created: the
    device id is the only credential there is, and a fresh one cannot read
    anything.

    It used to say a conversation whose session had been evicted was gone for
    good. That was wrong, and remeasuring on 2026-08-21 is what showed it: a
    fresh client with no cookies at all reads the conversation back as long as
    it carries the original device id, and still does after the creating session
    is closed. Nothing expires upstream at the 30 minute mark -- the proxy was
    simply throwing away the key along with the session. conv_store.py keeps it,
    so eviction no longer ends the conversation.

    No require_capability("conversations") here: that capability is about the
    account's server-side history, which is what the listing serves. Reading
    back a conversation this proxy itself created is a different thing, and the
    capability set is a contract with the gateway -- adding a twelfth key to
    express the distinction would break it.
    """
    authenticated = auth.is_authenticated()
    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())

    if authenticated:
        await require_capability("conversations")
        if sessions:
            device_id, client, owns_client = sessions[0].device_id, sessions[0].client, False
        else:
            device_id, owns_client = str(uuid.uuid4()), True
            client = _httpx.AsyncClient(verify=True, timeout=15.0, follow_redirects=True)
    else:
        # The creating device, or nothing: any other device gets a 404 from the
        # vendor, so guessing one would turn "we no longer hold that session"
        # into an opaque upstream error.
        owner = next((s for s in sessions if s.conversation_id == conversation_id), None)
        if owner is not None:
            device_id, client, owns_client = owner.device_id, owner.client, False
        else:
            # The session is gone, the conversation is not. Cookies turned out to
            # be irrelevant -- a fresh client carrying the original device id
            # reads it back, and still does after the creating session is closed
            # (measured 2026-08-21) -- so the recorded device is enough.
            known = conv_store.lookup(user_id, conversation_id)
            if known is None:
                raise HTTPException(404, detail={"error": {
                    "message": "This proxy has no record of that anonymous "
                               "conversation. An anonymous conversation is readable "
                               "only from the device that created it, and only this "
                               "proxy knows which device that was.",
                    "type": "not_found_error", "param": "conversation_id",
                }})
            device_id, owns_client = known["device_id"], True
            client = _httpx.AsyncClient(verify=True, timeout=15.0, follow_redirects=True)

    path = _api("/conversation/" + conversation_id)
    try:
        hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path}
        r = await client.get(f"{BASE}{path}", headers=hdrs)
        # An indexed conversation the vendor no longer honours is pruned rather
        # than left to be listed forever: how long an anonymous conversation
        # survives upstream cannot be measured in one sitting, so the index is
        # corrected by what the vendor answers instead of being trusted.
        if r.status_code == 404:
            if not authenticated:
                conv_store.forget(user_id, conversation_id)
            # Answering 404 rather than letting raise_for_status() escape as a
            # 500: the conversation is genuinely not there, and a 500 would tell
            # the caller this proxy is broken instead.
            raise HTTPException(404, detail={"error": {
                "message": "The backend no longer has that conversation.",
                "type": "not_found_error", "param": "conversation_id",
            }})
        r.raise_for_status()
        d = r.json()
    finally:
        if owns_client:
            await client.aclose()

    messages = []
    for node in (d.get("mapping", {}) or {}).values():
        m = node.get("message")
        if not m:
            continue
        role  = (m.get("author") or {}).get("role")
        parts = (m.get("content") or {}).get("parts") or []
        text  = "".join(p for p in parts if isinstance(p, str)).strip()
        if not text or role in (None, "system"):
            continue
        messages.append({
            "id":          m.get("id"),
            "role":        role,
            "content":     text,
            "create_time": m.get("create_time"),
        })
    messages.sort(key=lambda x: x.get("create_time") or 0)

    return {
        "id":          conversation_id,
        "title":       d.get("title"),
        "create_time": d.get("create_time"),
        "update_time": d.get("update_time"),
        "is_archived": d.get("is_archived"),
        "gizmo_id":    d.get("gizmo_id"),
        "messages":    messages,
    }


# Audio (TTS) storage, mirroring the image flow: save the file and serve it via a
# URL rather than streaming raw bytes. Reuses IMAGE_BASE_URL as the proxy's public
# base unless AUDIO_BASE_URL is set.
_AUDIO_STORE_DIR = pathlib.Path(os.environ.get("AUDIO_STORE_DIR", "/tmp/chatgpt_audio"))
_AUDIO_BASE_URL  = (os.environ.get("AUDIO_BASE_URL")
                    or os.environ.get("IMAGE_BASE_URL", "")).rstrip("/")
_AUDIO_EXT = {"mp3": ".mp3", "aac": ".aac", "opus": ".opus", "wav": ".wav", "ogg": ".ogg"}

# Free-text TTS: synthesize only reads back an assistant message, so /v1/audio/speech
# first has the model echo the text verbatim, then synthesizes that reply.
_VERBATIM_PROMPT = (
    "Devolvé ÚNICAMENTE el siguiente texto, palabra por palabra, exactamente igual, "
    "sin agregar, quitar ni cambiar nada, sin comillas, sin prefijos ni explicaciones:\n\n{t}"
)


def _store_audio(content: bytes, key: str, voice: str, fmt: str, media: str) -> dict:
    """Save TTS bytes to _AUDIO_STORE_DIR and return {url, content_type, bytes}."""
    _AUDIO_STORE_DIR.mkdir(parents=True, exist_ok=True)
    ext      = _AUDIO_EXT.get(fmt.lower(), ".mp3")
    filename = f"{key}-{voice}{ext}"
    (_AUDIO_STORE_DIR / filename).write_bytes(content)
    served = (f"{_AUDIO_BASE_URL}/audio/{filename}" if _AUDIO_BASE_URL
              else f"/audio/{filename}")
    return {"url": served, "content_type": media, "bytes": len(content)}


# ChatGPT's own ten voices. They were already named in `_synthesize`'s
# docstring and nowhere else -- not validated, not exposed -- which is how a
# request for OpenAI's default voice reached the backend and took the whole
# endpoint down. See `resolve_voice`.
NATIVE_VOICES: tuple[str, ...] = (
    "juniper", "cove", "ember", "breeze", "maple",
    "vale", "glimmer", "orbit", "fathom", "ridge",
)

# OpenAI's voice names mapped onto ChatGPT's, so a client written against the
# OpenAI API works unchanged -- the same thing perplexity-proxy does with its
# VOICE_MAP.
#
# HONESTY ABOUT THIS TABLE: only `alloy -> juniper` means anything, because
# juniper is this backend's default and alloy is OpenAI's. The other pairings
# are ARBITRARY. They were not chosen by comparing timbre, because nobody has
# listened to all ten and matched them; they exist so that a client asking for
# two different voices gets two different voices instead of the same one. Do
# not read them as "echo sounds like cove".
OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy":   "juniper",
    "echo":    "cove",
    "fable":   "vale",
    "onyx":    "ember",
    "nova":    "breeze",
    "shimmer": "maple",
    "ash":     "ridge",
    "ballad":  "glimmer",
    "coral":   "orbit",
    "sage":    "fathom",
    "verse":   "vale",
    "marin":   "juniper",
    "cedar":   "cove",
}


# ── El contrato de audio, común a los cinco proxies ──────────────────────────
# 4096 es el límite de la API de OpenAI. Acá el tope importa doblemente: con
# 5000 caracteres este backend NO devuelve un error, corta la conexión (medido),
# y la síntesis gasta un mensaje de chat antes de llegar a eso.
MAX_INPUT_CHARS = 4096
SUPPORTED_FORMATS = ("mp3",)


def resolve_voice(voice: str) -> tuple[str, bool]:
    """Returns (voice this backend accepts, whether it was substituted).

    Validating BEFORE the round trip is the whole point. An unknown voice used
    to reach `/backend-api/synthesize`, which does not answer with a clean 4xx
    -- it closes the connection mid-body, so httpx raised RemoteProtocolError
    and the endpoint died with a bare `500 Internal Server Error`. That is the
    worst of both worlds: the client learns nothing, and the gateway in front
    reads 500 as "this route is broken", accumulates suspicion against it and
    fails over -- punishing a healthy provider for a client's typo.

    It also used to cost a chat message before failing: synthesis works by
    making the model echo the text first, so the bad request was paid for and
    then thrown away.
    """
    name = (voice or "").strip().lower()
    if not name:
        return "juniper", False
    if name in NATIVE_VOICES:
        return name, False
    if name in OPENAI_VOICE_MAP:
        return OPENAI_VOICE_MAP[name], False
    # An unknown name picks a real voice at random rather than refusing. That
    # is the operator's call -- audio in some voice beats a rejection -- and it
    # costs determinism: the same request can sound different on a retry. The
    # substitution is reported in a header so that is visible, never silent.
    return random.choice(NATIVE_VOICES), True


class SpeechRequest(BaseModel):
    input:  str
    voice:  str = "juniper"
    format: str = "mp3"
    model:  str = "auto"


@app.get("/v1/audio/from-message")
async def audio_from_message(
    request: Request,
    conversation_id: str,
    message_id: str,
    voice: str = "juniper",
    format: str = "mp3",
    raw: bool = False,
):
    """Text-to-speech of an EXISTING conversation message (ChatGPT `synthesize`).

    The backend only reads back a stored message, so this needs a
    `conversation_id` + `message_id` (get them from /v1/conversations/{id}) --
    it is NOT free-text TTS (`synthesize` rejects arbitrary text). Returns the
    raw audio bytes with the upstream content-type. `voice` defaults to juniper;
    `format` to mp3 (also aac). Authenticated only -- synthesize has no anonymous
    endpoint, and the conversation must belong to the account.
    """
    await require_capability("audio_speech")
    if not auth.is_authenticated():
        return _needs_account()
    path = "/backend-api/synthesize"

    user_id  = _get_user_id(request)
    pool     = _user_pool(user_id)
    sessions = list(pool._pool.values())
    if sessions:
        device_id, client, owns_client = sessions[0].device_id, sessions[0].client, False
    else:
        device_id, owns_client = str(uuid.uuid4()), True
        client = _httpx.AsyncClient(verify=True, timeout=45.0, follow_redirects=True)

    try:
        hdrs = {**_base_headers(device_id), "X-OpenAI-Target-Path": path}
        r = await client.get(f"{BASE}{path}", headers=hdrs, params={
            "conversation_id": conversation_id,
            "message_id":      message_id,
            "voice":           voice,
            "format":          format,
        })
    finally:
        if owns_client:
            await client.aclose()

    if r.status_code != 200:
        # Bubble up the backend's own error (404 no-access, 422 bad params, ...),
        # unwrapping the upstream {"detail": ...} so it is not double-nested.
        try:
            j = r.json()
            detail = j.get("detail", j) if isinstance(j, dict) else j
        except Exception:
            detail = r.text[:200]
        return JSONResponse(status_code=r.status_code,
                            content={"error": {"type": "synthesize_error", "detail": detail}})

    media = r.headers.get("content-type", "audio/mpeg")
    if raw:
        # Opt-out: stream the bytes directly instead of storing + serving a URL.
        return Response(content=r.content, media_type=media)

    # Default: save the file and hand back a URL, exactly like generated images.
    stored = _store_audio(r.content, message_id[:36], voice, format, media)
    return {**stored, "conversation_id": conversation_id, "message_id": message_id,
            "voice": voice, "format": format}


@dataclass(frozen=True)
class Synthesized:
    """One synthesis, before it is shaped for a particular endpoint."""
    audio: bytes
    media_type: str
    text: str            # what was ACTUALLY synthesized
    exact_match: bool    # False when the model altered the text on the way
    voice: str
    format: str
    conversation_id: str
    message_id: str


class _SynthesizeFailed(Exception):
    """A synthesis that failed upstream, carried to whichever endpoint asked.

    It exists so `_synthesize` has ONE way to signal failure to its two callers
    while each of them still renders the error in this proxy's own shape --
    `{"error": {...}}` at the top level, the same as every other endpoint here.
    Raising HTTPException instead would be shorter, but FastAPI's default
    handler nests the body under "detail", and a client that parses our errors
    uniformly would break on exactly these two paths.
    """

    def __init__(self, status_code: int, payload: dict):
        super().__init__(payload)
        self.status_code = status_code
        self.payload = payload


async def _synthesize(text: str, voice: str, fmt: str, model: str) -> Synthesized:
    """The chat-echo + /backend-api/synthesize round trip.

    Extracted so the OpenAI-shaped endpoint and the native one are two thin
    shells over one implementation, instead of one endpoint with a `raw=` flag
    threading a branch through 60 lines.

    synthesize can only read an assistant message, so this first makes the model
    echo `text` verbatim (a throwaway conversation), verifies the reply matches,
    then synthesizes it. Costs one chat message per call. Voices: juniper, cove,
    ember, breeze, maple, vale, glimmer, orbit, fathom, ridge.
    """
    # Resolved here rather than in each endpoint: three call sites share this
    # function, and a validation that lives in only two of them is the kind
    # that drifts.
    voice, _voice_substituted = resolve_voice(voice)

    # Have the model reproduce the text; retry once if it isn't verbatim.
    session = None
    reply = ""
    try:
        for attempt in (1, 2):
            if session:
                await session.close()
            session = ChatGPTSession()
            reply = "".join([frag async for frag in session.stream_message(
                _VERBATIM_PROMPT.format(t=text), model=model)]).strip()
            if reply == text:
                break
        exact = reply == text

        cid, mid = session.conversation_id, session.parent_message_id
        if not cid or not mid:
            raise _SynthesizeFailed(502, {"error": {
                "message": "could not obtain the message to synthesize", "type": "upstream_error"}})

        path = "/backend-api/synthesize"
        try:
            r = await session.client.get(f"{BASE}{path}",
                headers={**_base_headers(session.device_id), "X-OpenAI-Target-Path": path},
                params={"conversation_id": cid, "message_id": mid,
                        "voice": voice, "format": fmt})
        except _httpx.HTTPError as e:
            # This backend answers a bad parameter by closing the connection
            # mid-body rather than with a status, so httpx raises instead of
            # returning. Letting that escape produced a bare 500, which the
            # gateway reads as "the route is broken" -- 502 says the truth:
            # upstream misbehaved on this call.
            raise _SynthesizeFailed(502, {"error": {
                "type": "upstream_error",
                "message": f"synthesize transport failure: {type(e).__name__}: {e}"}})
    finally:
        if session:
            await session.close()

    if r.status_code != 200:
        try:
            j = r.json()
            detail = j.get("detail", j) if isinstance(j, dict) else j
        except Exception:
            detail = r.text[:200]
        raise _SynthesizeFailed(r.status_code, {"error": {
            "type": "synthesize_error", "detail": detail}})

    media = r.headers.get("content-type", "audio/mpeg")
    return Synthesized(audio=r.content, media_type=media, text=reply, exact_match=exact,
                       voice=voice, format=fmt, conversation_id=cid, message_id=mid)


@app.get("/v1/audio/voices")
async def list_voices():
    """The voices this proxy accepts. There was no way to ask before.

    That gap is why the bug existed at all: with nothing to consult, a caller
    reasonably sends OpenAI's default `alloy`, which this backend does not
    have. `native` are ChatGPT's own; `openai_aliases` maps OpenAI's names onto
    them -- see OPENAI_VOICE_MAP for how little those pairings claim.
    """
    return {"default": "juniper",
            "voices": list(NATIVE_VOICES),
            "native": list(NATIVE_VOICES),
            "openai_aliases": OPENAI_VOICE_MAP,
            "selection": "random",
            "max_input_chars": MAX_INPUT_CHARS,
            "formats": list(SUPPORTED_FORMATS),
            "default_format": "mp3"}


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest, request: Request):
    """OpenAI-compatible TTS: raw audio bytes, with the correct Content-Type.

    It used to return JSON carrying an mp3 URL. Every OpenAI client writes the
    response body straight to a file, so that shape needed special-casing this
    one provider -- which is the thing the gateway in front of it exists to
    avoid. The extra facts this flow produces (`exact_match` in particular: it
    makes the model echo the input, and the model sometimes edits it) are real
    and are kept, as response headers, where a client that does not care never
    sees them and one that does can still read them. The JSON form lives on at
    /chatgpt/audio/speech.
    """
    await require_capability("audio_speech")
    text = (req.input or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": {
            "message": "'input' is required", "type": "invalid_request_error"}})
    # Checked BEFORE _synthesize, which makes the model echo the text and so
    # spends a chat message. Measured: at 5000 characters this backend does not
    # answer with an error, it drops the connection.
    if len(text) > MAX_INPUT_CHARS:
        return JSONResponse(status_code=400, content={"error": {
            "type": "invalid_request_error",
            "message": f"input is {len(text)} characters, the limit is {MAX_INPUT_CHARS}",
            "param": "input", "max_input_chars": MAX_INPUT_CHARS}})
    try:
        s = await _synthesize(text, req.voice, req.format, req.model)
    except _SynthesizeFailed as e:
        return JSONResponse(status_code=e.status_code, content=e.payload)
    stored = _store_audio(s.audio, s.message_id[:36], s.voice, s.format, s.media_type)
    return Response(content=s.audio, media_type=s.media_type, headers={
        "X-Audio-Url":       stored["url"],
        "X-Exact-Match":     "true" if s.exact_match else "false",
        "X-Conversation-Id": s.conversation_id,
        "X-Message-Id":      s.message_id,
    })


@app.post("/chatgpt/audio/speech")
async def chatgpt_audio_speech(req: SpeechRequest, request: Request):
    """The pre-contract JSON shape, under this provider's own prefix.

    Anything a provider offers beyond the standard surface lives here; the
    standard path stays the one every OpenAI client already knows.
    """
    await require_capability("audio_speech")
    text = (req.input or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": {
            "message": "'input' is required", "type": "invalid_request_error"}})
    # Checked BEFORE _synthesize, which makes the model echo the text and so
    # spends a chat message. Measured: at 5000 characters this backend does not
    # answer with an error, it drops the connection.
    if len(text) > MAX_INPUT_CHARS:
        return JSONResponse(status_code=400, content={"error": {
            "type": "invalid_request_error",
            "message": f"input is {len(text)} characters, the limit is {MAX_INPUT_CHARS}",
            "param": "input", "max_input_chars": MAX_INPUT_CHARS}})
    try:
        s = await _synthesize(text, req.voice, req.format, req.model)
    except _SynthesizeFailed as e:
        return JSONResponse(status_code=e.status_code, content=e.payload)
    stored = _store_audio(s.audio, s.message_id[:36], s.voice, s.format, s.media_type)
    return {**stored, "text": s.text, "exact_match": s.exact_match,
            "voice": s.voice, "format": s.format,
            "conversation_id": s.conversation_id, "message_id": s.message_id}


@app.get("/audio/{filename}", include_in_schema=False)
async def serve_audio(filename: str):
    """Serve a locally stored TTS file from _AUDIO_STORE_DIR."""
    safe = pathlib.Path(filename).name  # strip any path traversal
    path = _AUDIO_STORE_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(str(path))
