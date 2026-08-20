"""
ChatGPT Android anonymous client — async, FastAPI compatible.
"""
import re
import pathlib
import uuid, json, hashlib, time, sys, os
from typing import AsyncGenerator, Optional
import httpx

import auth

# ---------------------------------------------------------------------------
# Image storage: downloaded locally so URLs don't expire and need no auth.
# IMAGE_BASE_URL   — the URL at which THIS proxy is reachable from outside
#                    (e.g. http://10.0.1.1:8890). Set in Coolify env vars.
#                    When empty the presigned remote URL is used as fallback.
# IMAGE_STORE_DIR  — local directory where images are saved inside the container.
# ---------------------------------------------------------------------------
_IMAGE_BASE_URL  = os.environ.get("IMAGE_BASE_URL", "").rstrip("/")
_IMAGE_STORE_DIR = pathlib.Path(os.environ.get("IMAGE_STORE_DIR", "/tmp/chatgpt_images"))

# Cache: file_id → {url (remote presigned), file_name, size_bytes, local_filename?}
# Persisted to avoid calling /backend-api/files/{id}/download more than once.
_IMAGE_CACHE_FILE = pathlib.Path("/tmp/chatgpt_image_cache.json")
_image_cache_mem: dict = {}


def _image_cache_load() -> dict:
    global _image_cache_mem
    if not _image_cache_mem:
        try:
            _image_cache_mem = json.loads(_IMAGE_CACHE_FILE.read_text())
        except Exception:
            pass
    return _image_cache_mem


def _image_cache_save() -> None:
    try:
        _IMAGE_CACHE_FILE.write_text(json.dumps(_image_cache_mem, indent=2))
    except Exception as e:
        _log(f"[img-cache] save error: {e}")

# Genui/cite widget delimiters (Unicode PUA: U+E200=start, U+E202=sep, U+E201=end)
# Format: <U+E200>{type}<U+E202>{content}<U+E201>  where type='cite'|'genui'
_WIDGET_RE    = re.compile('genui(.*?)', re.DOTALL)
_CITE_RE      = re.compile('cite(.*?)',  re.DOTALL)
_PUA_RE       = re.compile('[-]')
_WIDGET_CHARS = str.maketrans('', '', '')


def _clean_buf(buf: str) -> str:
    """
    Clean accumulated SSE buffer of PUA markers:
      - \\ue200cite\\ue202{ref}\\ue201   ->  cite{ref}  (preserved)
      - \\ue200genui\\ue202{...}\\ue201  ->  ''         (removed)
      - stray PUA chars                  ->  stripped
    """
    text = _CITE_RE.sub(lambda m: f'cite{m.group(1)}', buf)
    text = _WIDGET_RE.sub('', text)
    return _PUA_RE.sub('', text)


def _extract_json(text: str) -> str:
    """
    Strip markdown code fences and return the inner JSON string.
    If text is already valid JSON, return it as-is.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (```json or ```)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Find the outermost JSON object/array if there's surrounding text
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end   = text.rfind(end_char)
        if start != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except ValueError:
                pass
    return text


class QuotaExceededError(Exception):
    """Anonymous device_id has reached its message limit."""
    pass


_DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


def _log(*args):
    if _DEBUG:
        print("[chatgpt]", *args, file=sys.stderr, flush=True)


BASE        = "https://android.chat.openai.com"
APP_VERSION = "1.2026.223"

SENTINEL_PAYLOAD = json.dumps({
    "bot_token": {
        "failure_reason": (
            "-9: Standard Integrity API error (-9): Binding to the service in the "
            "Play Store has failed. This can be due to having an old Play Store "
            "version installed on the device."
        ),
        "failure_detail": "[aft.y(SourceFile:9), wfo.a(SourceFile:85), vfo.invokeSuspend(SourceFile:14)]"
    }
}, separators=(',', ':'))


# The backend is split by whether the caller has an account. Anonymous traffic
# goes to /backend-anon/..., an authenticated session to /backend-api/... -- the
# paths are otherwise identical, which is why one helper covers both and every
# call site stops hard-coding the prefix.
def _api(path: str) -> str:
    """`/models` -> `/backend-anon/models` or `/backend-api/models`."""
    prefix = "/backend-api" if auth.is_authenticated() else "/backend-anon"
    return prefix + path


def _base_headers(device_id: str) -> dict:
    headers = {
        "User-Agent":               f"ChatGPT/{APP_VERSION} (Android 16; sdk_gphone64_arm64; build 2622307)",
        "OAI-Package-Name":         "com.openai.chatgpt",
        "OAI-Client-Type":          "android",
        "OAI-Device-Id":            device_id,
        "Accept-Language":          "en-US,en;q=0.9",
        "X-Device-Tier":            "lower_mid",
        "ChatGPT-Account-ID":       "default",
        "ChatGPT-Residency-Region": "no_constraint",
        "Accept":                   "application/json",
    }
    # Additive on purpose: with nothing configured this is a no-op and every
    # request is byte-for-byte what the anonymous path has always sent.
    token = auth.access_token()
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


class ChatGPTSession:
    """
    Anonymous ChatGPT Android session.
    Maintains cookies + conversation_id for multi-turn conversations.
    """

    _MODEL_MAP = {
        "gpt-4o":        "auto",
        "gpt-4o-mini":   "gpt-5-3-mini",
        "gpt-4":         "gpt-5-5",
        "gpt-3.5-turbo": "gpt-5-3-mini",
    }

    def __init__(self, system_prompt: str = "", gizmo_id: Optional[str] = None):
        self.system_prompt = system_prompt
        # When set (a "g-..." id), every turn runs as that custom GPT and the
        # proxy's own system_prompt is left out -- the GPT's instructions govern.
        self.gizmo_id      = gizmo_id
        self._first_turn   = True
        self.device_id     = str(uuid.uuid4())
        self.client        = httpx.AsyncClient(
            verify=True,
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        )
        self.conversation_id:    Optional[str] = None
        self.parent_message_id:  Optional[str] = None
        self.last_used:  float = time.time()
        self._ready:     bool  = False
        self.quota_exhausted: bool = False  # True when this device_id hit its limit

        # Sentinel chat-requirements token: the backend hands one out (valid a few
        # minutes) and wants it echoed on every conversation request. Captured in
        # _chat_requirements(), refreshed before expiry in _ensure_chat_requirements().
        self._chat_req_token:  Optional[str] = None
        self._chat_req_expire: float = 0.0

        # Metadata from the last turn — reset at the start of each stream_message
        self.last_search_queries: list = []
        self.last_citations:      list = []   # [{url, title, attribution}]
        self.last_title:   Optional[str] = None
        self.last_widgets: list = []           # [{name, data}] — genui widgets
        self.last_clean_text: str = ""         # final text without PUA markers
        self.last_images:  list = []           # [{file_id, url, file_name, size_bytes}]
        self._pending_image_ids: list = []     # collected during streaming, resolved after

    async def initialize(self) -> None:
        await self._get_models()
        await self._ensure_chat_requirements()
        self._ready = True

    async def ensure_ready(self) -> None:
        if not self._ready:
            await self.initialize()

    async def _get_models(self) -> list:
        hdrs = {
            **_base_headers(self.device_id),
            "X-OpenAI-Target-Path": _api("/models"),
        }
        r = await self.client.get(
            f"{BASE}" + _api("/models"),
            params={"iim": "false", "supports_model_picker_upgrade_presets": "true"},
            headers=hdrs,
        )
        r.raise_for_status()
        return r.json().get("models", [])

    async def _chat_requirements(self) -> None:
        """Fetch the sentinel token the backend wants echoed on each conversation
        request (header `openai-sentinel-chat-requirements-token`).

        For this account (and the anon endpoint) the response is just
        `{token, expire_after, expire_at}` -- no proofofwork/turnstile/arkose --
        so capturing the token is enough; there is no proof to compute. If OpenAI
        ever starts returning `proofofwork.required` for this path, the token
        alone would stop being accepted and a PoW step would be needed here.
        """
        hdrs = {
            **_base_headers(self.device_id),
            "X-OpenAI-Target-Path": _api("/sentinel/chat-requirements"),
            "Content-Type": "application/json",
        }
        r = await self.client.post(
            f"{BASE}" + _api("/sentinel/chat-requirements"),
            headers=hdrs,
            content=b"{}",
        )
        r.raise_for_status()
        data = r.json()
        self._chat_req_token = data.get("token")
        # Refresh 30s before the server's expiry so a long turn never rides a dead
        # token; expire_after is in seconds (observed 540 = 9 min).
        ttl = data.get("expire_after") or 300
        self._chat_req_expire = time.time() + max(30, ttl - 30)
        _log(f"chat-requirements token obtained (ttl={ttl}s, "
             f"{'present' if self._chat_req_token else 'MISSING'})")

    async def _ensure_chat_requirements(self) -> None:
        """Hold a live chat-requirements token -- but ONLY for an authenticated
        session. The two flows diverge here:

          * authenticated  -> /backend-api/...: fetch the sentinel token and echo
            it on every conversation request (this method + the header below);
          * anonymous       -> /backend-anon/...: the free endpoint never required
            the token, so we skip the whole dance and leave `_chat_req_token` None,
            which drops the header. This is the flow the proxy always used.
        """
        if not auth.is_authenticated():
            self._chat_req_token = None
            return
        if not self._chat_req_token or time.time() >= self._chat_req_expire:
            await self._chat_requirements()

    @classmethod
    def _resolve_model(cls, requested: str) -> str:
        """Translate legacy aliases to the real backend slug.

        A gizmo id ("g-...") is NOT a model slug: the GPT runs on a base model and
        is selected via conversation_mode, so it maps to "auto" here.
        """
        if isinstance(requested, str) and requested.startswith("g-"):
            return "auto"
        return cls._MODEL_MAP.get(requested, requested)

    async def stream_message(
        self,
        message: str,
        model: str = "auto",
        file_texts: Optional[list[str]] = None,
        force_use_search: Optional[bool] = None,
        force_use_tools:  Optional[bool] = None,
        force_use_canvas: Optional[bool] = None,
        json_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Send a message and yield each new text fragment as a plain string.

        force_use_search:  None=model decides, True=force search, False=disable
        force_use_tools:   True=enable advanced tools (reservations, shopping, widgets)
        force_use_canvas:  True=enable Canvas mode (documents)
        json_mode:         True=instruct the model to respond with valid JSON only
        """
        await self.ensure_ready()
        await self._ensure_chat_requirements()
        self.last_used        = time.time()
        self.last_search_queries = []
        self.last_citations      = []
        self.last_title          = None
        self.last_widgets        = []
        self.last_clean_text     = ""
        self.last_images         = []
        self._pending_image_ids  = []

        real_model = self._resolve_model(model)

        # Build final text: system (first turn only) + files + message
        msg_parts = []
        if self._first_turn and self.system_prompt:
            if not self.gizmo_id:  # a GPT brings its own instructions
                msg_parts.append(f"[System instructions: {self.system_prompt}]")
            self._first_turn = False
        if json_mode:
            msg_parts.append(
                "You must respond with valid JSON only. "
                "No markdown, no explanations — just the raw JSON object or array."
            )
        if file_texts:
            for i, fc in enumerate(file_texts, 1):
                msg_parts.append(f"[Attached file {i}]:\n{fc}")
        msg_parts.append(message)
        final_message = "\n\n".join(msg_parts)

        msg_id     = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        parent_id  = self.parent_message_id or str(uuid.uuid4())

        body: dict = {
            "action":            "next",
            "parent_message_id": parent_id,
            "messages": [{
                "id":     msg_id,
                "author": {"role": "user"},
                "content": {"parts": [final_message], "content_type": "text"},
                "status":    "finished_successfully",
                "recipient": "all",
                "metadata": {
                    "model_slug": real_model, "default_model_slug": real_model,
                    "is_visually_hidden_from_conversation": False,
                    "exclude_after_next_user_message":      False,
                    "content_references": [], "search_result_groups": [],
                    "search_queries": [],     "image_results": [],
                    "real_time_audio_has_video": False, "system_hints": [],
                    "dictation": False, "voice_mode_message": False,
                    "image_gen_async": False, "trigger_async_ux": False,
                    "writing_blocks": {}
                }
            }],
            "model":                         real_model,
            "history_and_training_disabled": False,
            "fork_from_shared_post":         False,
            "enable_message_followups":      True,
            "force_use_sse":                 True,
            "force_use_search":              force_use_search,
            "force_use_tools":               force_use_tools,
            "force_use_canvas":              force_use_canvas,
            "force_paragen":                 False,
            "supported_encodings":           ["v1"],
            "supports_buffering":            True,
            "timezone":                      "America/Bogota",
            "timezone_offset_min":           300,
            "system_hints":                  [],
            "is_onboarding_conversation":    False,
            "no_auth_ad_preferences": {
                "personalization_enabled": True,
                "history_enabled":         True
            },
            "client_prepare_state": "none",
            "stream":               True,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.gizmo_id:
            body["conversation_mode"] = {"kind": "gizmo_interaction",
                                         "gizmo_id": self.gizmo_id}
        if self.conversation_id:
            body["conversation_id"] = self.conversation_id

        hdrs = {
            **_base_headers(self.device_id),
            "Accept":               "text/event-stream, application/json",
            "Cache-Control":        "no-cache",
            "Connection":           "keep-alive",
            "X-Sentinel-Payload":   SENTINEL_PAYLOAD,
            "oai-session-id":       session_id,
            **({"openai-sentinel-chat-requirements-token": self._chat_req_token}
               if self._chat_req_token else {}),
            "x-oai-convo-session-id": session_id,
            "x-oai-turn-trace-id":  str(uuid.uuid4()),
            "OAI-Echo-Logs":        "1,0,0,0",
            "X-OpenAI-Target-Path": _api("/f/conversation"),
            "Content-Type":         "application/json",
        }

        encoded = json.dumps(body, separators=(',', ':')).encode()

        _log(f"POST {_api('/f/conversation')}  model={real_model} json_mode={json_mode}")
        async with self.client.stream(
            "POST",
            f"{BASE}" + _api("/f/conversation"),
            headers=hdrs,
            content=encoded,
        ) as resp:
            _log(f"HTTP {resp.status_code}")

            if resp.status_code == 401:
                # Two different recoveries share this branch, and the order
                # matters. When running with an ACCOUNT, a 401 means the access
                # token expired -- re-minting the anonymous session would not fix
                # it and would silently downgrade an authenticated deployment to
                # the anonymous backend, which is exactly the kind of quiet
                # capability loss the gateway's tools:false declaration exists to
                # prevent. So: refresh the token FIRST, and only fall through to
                # re-initialising the session (the original, anonymous-path
                # behaviour) when there was no token to refresh.
                if auth.is_authenticated():
                    _log("401 → refreshing the access token and retrying")
                    if auth.refresh_access_token():
                        hdrs = {**hdrs, **_base_headers(self.device_id)}
                else:
                    _log("401 → refreshing session and retrying")
                self._ready = False
                await self.initialize()
                async with self.client.stream(
                    "POST",
                    f"{BASE}" + _api("/f/conversation"),
                    headers=hdrs,
                    content=encoded,
                ) as resp2:
                    _log(f"retry HTTP {resp2.status_code}")
                    if resp2.status_code in (429, 403):
                        self.quota_exhausted = True
                        raise QuotaExceededError(
                            f"Anonymous quota exhausted (HTTP {resp2.status_code})"
                        )
                    resp2.raise_for_status()
                    async for chunk in self._parse_sse(resp2):
                        yield chunk
                return

            if resp.status_code in (429, 403):
                self.quota_exhausted = True
                try:
                    body_bytes = await resp.aread()
                    data       = json.loads(body_bytes)
                    msg        = data.get("detail") or data.get("message") or ""
                    if isinstance(msg, dict):
                        msg = msg.get("message") or str(msg)
                    msg = str(msg) or f"Anonymous quota exhausted (HTTP {resp.status_code})"
                except Exception:
                    msg = f"Anonymous quota exhausted (HTTP {resp.status_code})"
                _log(f"QuotaExceededError: {msg}")
                raise QuotaExceededError(msg)

            resp.raise_for_status()
            async for chunk in self._parse_sse(resp):
                yield chunk

    @staticmethod
    def _apply_delta(delta: dict, buf: str) -> str:
        """
        Apply a delta operation to the accumulated text buffer.
        Only processes the visible text channel (/message/content/parts/0).
        """
        op   = (delta.get("operation") or delta.get("o") or "").lower()
        path = (delta.get("path")      or delta.get("p") or "")
        val  = delta.get("value", delta.get("v"))

        is_text = (
            path.endswith("/parts/0") or
            path.endswith("/content/parts") or
            path == ""
        )
        if not is_text:
            return buf

        if op in ("append", "add") and isinstance(val, str):
            return buf + val
        if op == "replace" and isinstance(val, str):
            return val
        if op == "truncate" and isinstance(val, int):
            return buf[:val]
        if op == "patch" and isinstance(val, list):
            for sub in val:
                if isinstance(sub, dict):
                    buf = ChatGPTSession._apply_delta(sub, buf)
            return buf
        # Implicit append: {"v": "text"} without explicit op or path
        if not op and isinstance(val, str):
            return buf + val
        # List of patches without explicit op: {"v": [{p,o,v}, ...]}
        if not op and isinstance(val, list):
            for sub in val:
                if isinstance(sub, dict):
                    buf = ChatGPTSession._apply_delta(sub, buf)
            return buf
        return buf

    async def _parse_sse(self, resp) -> AsyncGenerator[str, None]:
        """
        SSE parser handling the three formats sent by the backend:

          1. event: delta  +  data: {"o":"append","p":"/message/content/parts/0","v":"text"}
             (compact top-level keys — real format from /backend-anon/f/conversation)

          2. data: {"delta":{"operation":"append","path":"...","value":"text"}}
             (nested delta)

          3. data: {"message":{"content":{"parts":["accumulated text"]}},...}
             (Format A legacy — full accumulated snapshot)
        """
        buf        = ""
        emitted    = ""
        event_type: Optional[str] = None
        last_assistant_msg_id: Optional[str] = None
        line_count = 0

        async for line in resp.aiter_lines():
            if not line:
                event_type = None
                continue

            line_count += 1
            if line_count <= 30:
                _log(f"SSE line {line_count}: {line[:120]}")

            if line.startswith("event:"):
                event_type = line[6:].strip()
                continue

            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if data_str == "[DONE]":
                _log(f"[DONE] after {line_count} lines. buf: {buf[:80]!r}")
                # Close the socket immediately — avoids httpx draining the remaining
                # HTTP body (ChatGPT keeps sending after [DONE]) which blocks for seconds.
                try:
                    await resp.aclose()
                except Exception:
                    pass
                break

            try:
                evt = json.loads(data_str)
                if not isinstance(evt, dict):
                    continue

                if evt.get("conversation_id"):
                    self.conversation_id = evt["conversation_id"]

                etype = evt.get("type")

                # ── Error / quota events ────────────────────────────────────
                if event_type == "error" or etype == "error":
                    raw = evt.get("message") or evt.get("detail") or str(evt)
                    if isinstance(raw, dict):
                        raw = raw.get("message") or str(raw)
                    self.quota_exhausted = True
                    _log(f"SSE error event: {raw}")
                    raise QuotaExceededError(str(raw))

                # Plain JSON error (non-SSE format) — usually a quota/auth issue
                if (
                    "detail" in evt and
                    not any(k in evt for k in ("v", "o", "p", "delta", "message", "conversation_id", "type"))
                ):
                    raw = evt["detail"]
                    if isinstance(raw, dict):
                        raw = raw.get("message") or str(raw)
                    self.quota_exhausted = True
                    _log(f"SSE plain error: {raw}")
                    raise QuotaExceededError(str(raw) or "Server error")

                # ── Control events ──────────────────────────────────────────
                if etype in ("delta_encoding", "message_stream_complete", "message_marker"):
                    if evt.get("message_id"):
                        last_assistant_msg_id = evt["message_id"]
                    _log(f"control event type={etype}")
                    if etype == "message_stream_complete":
                        # Text is fully streamed. ChatGPT still sends metadata and
                        # [DONE] 5-10s later — close the connection now instead of waiting.
                        try:
                            await resp.aclose()
                        except Exception:
                            pass
                        break
                    event_type = None
                    continue

                if etype == "title_generation":
                    self.last_title = evt.get("title") or self.last_title
                    event_type = None
                    continue

                if etype == "url_moderation":
                    r = evt.get("url_moderation_result") or {}
                    url = r.get("full_url", "")
                    if url and not r.get("is_blocked") and url not in [c["url"] for c in self.last_citations]:
                        self.last_citations.append({"url": url, "title": "", "attribution": ""})
                    event_type = None
                    continue

                # ── Format 1: event: delta + compact top-level keys ─────────
                if event_type == "delta":
                    v_inner = evt.get("v")
                    if isinstance(v_inner, dict) and "message" in v_inner:
                        inner_msg    = v_inner["message"]
                        inner_author = (inner_msg.get("author") or {}).get("role", "")
                        inner_meta   = inner_msg.get("metadata") or {}
                        if inner_author == "tool":
                            smq = inner_meta.get("search_model_queries") or {}
                            for q in smq.get("queries") or []:
                                if q not in self.last_search_queries:
                                    self.last_search_queries.append(q)
                            self._queue_image_parts(inner_msg.get("content") or {})
                        elif inner_author == "assistant":
                            # Full assistant snapshot — extract text if no prior patches
                            if inner_msg.get("id"):
                                last_assistant_msg_id = inner_msg["id"]
                            parts = (inner_msg.get("content") or {}).get("parts") or []
                            if parts and isinstance(parts[0], str) and parts[0]:
                                new_buf = parts[0]
                                if new_buf != buf:
                                    chunk  = new_buf[len(emitted):]
                                    buf    = new_buf
                                    emitted = buf
                                    clean  = chunk.translate(_WIDGET_CHARS)
                                    if clean:
                                        yield clean

                    # Extract citation titles/attributions from content_references patches
                    elif isinstance(v_inner, list):
                        for sub in v_inner:
                            if not isinstance(sub, dict): continue
                            sub_p = sub.get("p") or ""
                            sub_v = sub.get("v")
                            if (sub_p.startswith("/message/metadata/content_references/") and
                                    not any(sub_p.endswith(s) for s in ["/safe_urls","/alt","/type","/invalid","/items","/attribution","/status","/error"]) and
                                    isinstance(sub_v, dict)):
                                items = sub_v.get("items") or []
                                for item in items:
                                    if not isinstance(item, dict): continue
                                    item_url   = item.get("url","")
                                    item_title = item.get("title","")
                                    item_attr  = item.get("attribution","")
                                    base_url   = item_url.split("?")[0]
                                    for cit in self.last_citations:
                                        if base_url in cit["url"] or cit["url"].split("?")[0] == base_url:
                                            if not cit["title"] and item_title:
                                                cit["title"] = item_title
                                            if not cit["attribution"] and item_attr:
                                                cit["attribution"] = item_attr
                                            break

                    new_buf = self._apply_delta(evt, buf)
                    if new_buf != buf:
                        chunk   = new_buf[len(emitted):]
                        buf     = new_buf
                        emitted = buf
                        clean   = chunk.translate(_WIDGET_CHARS)
                        if clean:
                            yield clean
                    event_type = None
                    continue

                # ── Format 2: nested delta {"delta": {...}} ─────────────────
                delta = evt.get("delta")
                if isinstance(delta, dict):
                    new_buf = self._apply_delta(delta, buf)
                    if new_buf != buf:
                        chunk   = new_buf[len(emitted):]
                        buf     = new_buf
                        emitted = buf
                        clean   = chunk.translate(_WIDGET_CHARS)
                        if clean:
                            yield clean
                    continue

                # ── Format A: accumulated snapshot {"message": {...}} ────────
                msg = evt.get("message")
                if msg and isinstance(msg, dict):
                    last_assistant_msg_id = msg.get("id", last_assistant_msg_id)
                    content = msg.get("content") or {}
                    parts = content.get("parts") or []
                    if parts and isinstance(parts[0], str) and parts[0]:
                        new_buf = parts[0]
                        if new_buf != buf:
                            chunk   = new_buf[len(emitted):]
                            buf     = new_buf
                            emitted = buf
                            clean   = chunk.translate(_WIDGET_CHARS)
                            if clean:
                                yield clean
                    self._queue_image_parts(content)
                    continue

                # ── Format B: full message replace {"v":{message:...}} ───────
                v_val = evt.get("v")
                if isinstance(v_val, dict) and "message" in v_val:
                    inner_msg    = v_val["message"]
                    inner_author = (inner_msg.get("author") or {}).get("role", "")
                    inner_meta   = inner_msg.get("metadata") or {}
                    if inner_author == "tool":
                        smq = inner_meta.get("search_model_queries") or {}
                        for q in smq.get("queries") or []:
                            if q not in self.last_search_queries:
                                self.last_search_queries.append(q)
                        self._queue_image_parts(inner_msg.get("content") or {})
                    if inner_author == "assistant" and inner_msg.get("id"):
                        last_assistant_msg_id = inner_msg["id"]
                    continue

                keys = list(evt.keys())
                _log(f"unknown event event_type={event_type!r} keys={keys}")

            except (json.JSONDecodeError, AttributeError) as e:
                _log(f"parse error: {e}  line={line[:80]}")

        _log(f"stream done. lines={line_count} buf={buf[:80]!r}")

        # Extract genui widgets from the accumulated buffer
        for m in _WIDGET_RE.finditer(buf):
            try:
                data = json.loads(m.group(1).strip())
                if isinstance(data, dict):
                    for name, payload in data.items():
                        self.last_widgets.append({"name": name, "data": payload})
            except (json.JSONDecodeError, ValueError):
                pass

        # Clean text: cite{ref} preserved, genui{...} removed
        self.last_clean_text = _clean_buf(buf).strip()
        if last_assistant_msg_id:
            self.parent_message_id = last_assistant_msg_id

    def _queue_image_parts(self, content: dict) -> None:
        """Collect image_asset_pointer file IDs from a message content dict."""
        if content.get("content_type") != "multimodal_text":
            return
        for part in (content.get("parts") or []):
            if not isinstance(part, dict):
                continue
            if part.get("content_type") != "image_asset_pointer":
                continue
            ptr = part.get("asset_pointer", "")
            fid = ptr.split("://", 1)[-1] if "://" in ptr else ptr
            if fid and fid not in self._pending_image_ids:
                self._pending_image_ids.append(fid)
                _log(f"[img] queued {fid[:28]}")

    async def resolve_image_urls(self) -> None:
        """Resolve pending image file IDs → download bytes locally → stable URL.

        Two-step:
          1. GET /backend-api/files/{id}/download (auth required) → presigned URL.
             Cached in _IMAGE_CACHE_FILE so the metadata endpoint is only called once.
          2. GET presigned URL with auth headers → save PNG/JPEG bytes to
             _IMAGE_STORE_DIR.  Serve via /images/{filename} on this proxy.
             Returns IMAGE_BASE_URL/images/{filename} when IMAGE_BASE_URL is set,
             otherwise falls back to the presigned remote URL.
        """
        if not self._pending_image_ids:
            return
        _IMAGE_STORE_DIR.mkdir(parents=True, exist_ok=True)
        cache = _image_cache_load()
        hdrs  = {**_base_headers(self.device_id)}

        for fid in self._pending_image_ids:
            try:
                # ── Step 1: get the presigned remote URL ──────────────────────
                if fid in cache:
                    remote_url  = cache[fid]["url"]
                    file_name   = cache[fid].get("file_name", "generated_image.png")
                    size_bytes  = cache[fid].get("size_bytes", 0)
                    _log(f"[img-cache] HIT metadata {fid[:28]}")
                else:
                    r = await self.client.get(
                        f"{BASE}/backend-api/files/{fid}/download",
                        headers=hdrs, timeout=20.0,
                    )
                    _log(f"[img] /files/{fid[:20]}.../download → HTTP {r.status_code}")
                    if r.status_code != 200:
                        _log(f"[img] resolve failed: {r.text[:120]}")
                        continue
                    d          = r.json()
                    remote_url = d.get("download_url", "")
                    file_name  = d.get("file_name", "generated_image.png")
                    size_bytes = d.get("file_size_bytes", 0)
                    cache[fid] = {"url": remote_url, "file_name": file_name,
                                  "size_bytes": size_bytes}
                    _image_cache_mem[fid] = cache[fid]
                    _image_cache_save()

                if not remote_url:
                    _log(f"[img] empty remote_url for {fid[:28]}")
                    continue

                # ── Step 2: download bytes and store locally ──────────────────
                local_filename = cache[fid].get("local_filename", "")
                local_path     = (_IMAGE_STORE_DIR / local_filename
                                  if local_filename else None)

                if local_filename and local_path and local_path.exists():
                    _log(f"[img-cache] HIT local {local_filename}")
                else:
                    img_r = await self.client.get(
                        remote_url, headers=hdrs, timeout=60.0,
                        follow_redirects=True,
                    )
                    _log(f"[img] download bytes → HTTP {img_r.status_code} "
                         f"({len(img_r.content)} B)")
                    if img_r.status_code != 200:
                        _log(f"[img] byte download failed: {img_r.text[:80]}")
                        # Fallback: use the remote URL as-is
                        self.last_images.append({
                            "file_id": fid, "url": remote_url,
                            "file_name": file_name, "size_bytes": size_bytes,
                        })
                        continue

                    ct  = img_r.headers.get("content-type", "").split(";")[0].strip()
                    ext = ({"image/png": ".png", "image/jpeg": ".jpg",
                            "image/webp": ".webp", "image/gif": ".gif"}
                           .get(ct) or pathlib.Path(file_name).suffix or ".png")
                    local_filename = f"{fid[:32]}{ext}"
                    local_path     = _IMAGE_STORE_DIR / local_filename
                    local_path.write_bytes(img_r.content)
                    _log(f"[img] saved {len(img_r.content)} B → {local_filename}")
                    # Update cache entry with local filename
                    _image_cache_mem[fid]["local_filename"] = local_filename
                    _image_cache_save()

                # ── Build the URL this proxy will serve ───────────────────────
                if _IMAGE_BASE_URL:
                    served_url = f"{_IMAGE_BASE_URL}/images/{local_filename}"
                else:
                    served_url = remote_url   # fallback when env var not set

                self.last_images.append({
                    "file_id":    fid,
                    "url":        served_url,
                    "file_name":  file_name,
                    "size_bytes": size_bytes,
                })
                _log(f"[img] → {served_url[:90]}")

            except Exception as e:
                _log(f"[img] error for {fid[:28]}: {e}")

        self._pending_image_ids.clear()

    async def close(self):
        await self.client.aclose()


async def fetch_anon_models() -> list:
    """
    Fetch the model list from /backend-anon/models using a temporary client.
    """
    device_id = str(uuid.uuid4())
    hdrs = {
        **_base_headers(device_id),
        "X-OpenAI-Target-Path": _api("/models"),
    }
    async with httpx.AsyncClient(verify=True, timeout=10.0, follow_redirects=True) as client:
        r = await client.get(
            f"{BASE}" + _api("/models"),
            params={"iim": "false", "supports_model_picker_upgrade_presets": "true"},
            headers=hdrs,
        )
        r.raise_for_status()
        return r.json().get("models", [])


class SessionPool:
    """
    Session pool keyed by message history hash.
    Enables transparent multi-turn for OpenAI-compatible clients.
    """
    TTL = 1800  # 30 min idle → close

    def __init__(self):
        self._pool: dict[str, ChatGPTSession] = {}

    @staticmethod
    def _key(messages: list[dict], system_prompt: str = "", gizmo_id: str = "") -> str:
        """
        Key = hash of history EXCEPT the last user message.
        Includes system_prompt AND gizmo_id so different instructions / different
        GPTs use separate sessions (never continue one GPT's thread as another).
        """
        history = [m for m in messages if not (m == messages[-1] and m.get("role") == "user")]
        if not history and not system_prompt and not gizmo_id:
            return "new_" + str(uuid.uuid4())  # always new session with no history
        canonical = (json.dumps(history, sort_keys=True, ensure_ascii=False)
                     + system_prompt + "\x00" + (gizmo_id or ""))
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def get(self, messages: list[dict], system_prompt: str = "",
                  gizmo_id: Optional[str] = None) -> tuple[str, "ChatGPTSession"]:
        key = self._key(messages, system_prompt, gizmo_id or "")
        await self._evict_stale()
        if key.startswith("new_") or key not in self._pool:
            session = ChatGPTSession(system_prompt=system_prompt, gizmo_id=gizmo_id)
            self._pool[key] = session
        elif self._pool[key].quota_exhausted:
            # Previous session hit its limit — create a fresh one with a new device_id
            _log(f"session {key[:8]} quota_exhausted → fresh device_id")
            await self._pool[key].close()
            self._pool[key] = ChatGPTSession(system_prompt=system_prompt, gizmo_id=gizmo_id)
        return key, self._pool[key]

    async def _evict_stale(self):
        now   = time.time()
        stale = [k for k, s in self._pool.items() if now - s.last_used > self.TTL]
        for k in stale:
            await self._pool[k].close()
            del self._pool[k]

    async def close_all(self):
        for s in self._pool.values():
            await s.close()
        self._pool.clear()
