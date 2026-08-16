"""
ChatGPT Android anonymous client — async, compatible con FastAPI.
"""
import uuid, json, hashlib, time, sys, os
from typing import AsyncGenerator, Optional
import httpx

_DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

def _log(*args):
    if _DEBUG:
        print("[chatgpt]", *args, file=sys.stderr, flush=True)

BASE = "https://android.chat.openai.com"
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


def _base_headers(device_id: str) -> dict:
    return {
        "User-Agent": f"ChatGPT/{APP_VERSION} (Android 16; sdk_gphone64_arm64; build 2622307)",
        "OAI-Package-Name": "com.openai.chatgpt",
        "OAI-Client-Type": "android",
        "OAI-Device-Id": device_id,
        "Accept-Language": "en-US,en;q=0.9",
        "X-Device-Tier": "lower_mid",
        "ChatGPT-Account-ID": "default",
        "ChatGPT-Residency-Region": "no_constraint",
        "Accept": "application/json",
    }


class ChatGPTSession:
    """
    Una sesión anónima de ChatGPT Android.
    Mantiene cookies + conversation_id para conversaciones multi-turno.
    """

    _MODEL_MAP = {
        "gpt-4o":        "auto",
        "gpt-4o-mini":   "gpt-5-3-mini",
        "gpt-4":         "gpt-5-5",
        "gpt-3.5-turbo": "gpt-5-3-mini",
    }

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self._first_turn = True
        self.device_id = str(uuid.uuid4())
        self.client = httpx.AsyncClient(
            verify=True,
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        )
        self.conversation_id: Optional[str] = None
        self.parent_message_id: Optional[str] = None
        self.last_used: float = time.time()
        self._ready = False

    async def initialize(self) -> None:
        await self._get_models()
        await self._chat_requirements()
        self._ready = True

    async def ensure_ready(self) -> None:
        if not self._ready:
            await self.initialize()

    async def _get_models(self) -> list:
        hdrs = {
            **_base_headers(self.device_id),
            "X-OpenAI-Target-Path": "/backend-anon/models",
        }
        r = await self.client.get(
            f"{BASE}/backend-anon/models",
            params={"iim": "false", "supports_model_picker_upgrade_presets": "true"},
            headers=hdrs,
        )
        r.raise_for_status()
        return r.json().get("models", [])

    async def _chat_requirements(self) -> None:
        hdrs = {
            **_base_headers(self.device_id),
            "X-OpenAI-Target-Path": "/backend-anon/sentinel/chat-requirements",
            "Content-Type": "application/json",
        }
        r = await self.client.post(
            f"{BASE}/backend-anon/sentinel/chat-requirements",
            headers=hdrs,
            content=b"{}",
        )
        r.raise_for_status()

    @classmethod
    def _resolve_model(cls, requested: str) -> str:
        """Traduce alias legacy al slug real del backend anónimo."""
        return cls._MODEL_MAP.get(requested, requested)

    async def stream_message(
        self,
        message: str,
        model: str = "auto",
        file_texts: Optional[list[str]] = None,
        force_use_search: Optional[bool] = None,
        force_use_tools: Optional[bool] = None,
        force_use_canvas: Optional[bool] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Envía un mensaje y hace yield de cada fragmento de texto nuevo (string puro).

        force_use_search:  None=modelo decide, True=forzar búsqueda, False=desactivar
        force_use_tools:   True=activar tools avanzados (reservas, shopping, widgets)
        force_use_canvas:  True=activar modo Canvas (documentos). La respuesta incluye
                           bloques :::writing{variant="document" id="..." title="..."}
        """
        await self.ensure_ready()
        self.last_used = time.time()

        real_model = self._resolve_model(model)

        # Construir texto final: system (sólo primer turno) + archivos + mensaje
        msg_parts = []
        if self._first_turn and self.system_prompt:
            msg_parts.append(f"[System instructions: {self.system_prompt}]")
            self._first_turn = False
        if file_texts:
            for i, fc in enumerate(file_texts, 1):
                msg_parts.append(f"[Attached file {i}]:\n{fc}")
        msg_parts.append(message)
        final_message = "\n\n".join(msg_parts)

        msg_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        parent_id = self.parent_message_id or str(uuid.uuid4())

        body: dict = {
            "action": "next",
            "parent_message_id": parent_id,
            "messages": [{
                "id": msg_id,
                "author": {"role": "user"},
                "content": {"parts": [final_message], "content_type": "text"},
                "status": "finished_successfully",
                "recipient": "all",
                "metadata": {
                    "model_slug": real_model, "default_model_slug": real_model,
                    "is_visually_hidden_from_conversation": False,
                    "exclude_after_next_user_message": False,
                    "content_references": [], "search_result_groups": [],
                    "search_queries": [], "image_results": [],
                    "real_time_audio_has_video": False, "system_hints": [],
                    "dictation": False, "voice_mode_message": False,
                    "image_gen_async": False, "trigger_async_ux": False,
                    "writing_blocks": {}
                }
            }],
            "model": real_model,
            "history_and_training_disabled": False,
            "fork_from_shared_post": False,
            "enable_message_followups": True,
            "force_use_sse": True,
            "force_use_search": force_use_search,
            "force_use_tools": force_use_tools,
            "force_use_canvas": force_use_canvas,
            "force_paragen": False,
            "supported_encodings": ["v1"],
            "supports_buffering": True,
            "timezone": "America/Bogota",
            "timezone_offset_min": 300,
            "system_hints": [],
            "is_onboarding_conversation": False,
            "no_auth_ad_preferences": {
                "personalization_enabled": True,
                "history_enabled": True
            },
            "client_prepare_state": "none",
            "stream": True,
        }
        if self.conversation_id:
            body["conversation_id"] = self.conversation_id

        hdrs = {
            **_base_headers(self.device_id),
            "Accept": "text/event-stream, application/json",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Sentinel-Payload": SENTINEL_PAYLOAD,
            "oai-session-id": session_id,
            "x-oai-convo-session-id": session_id,
            "x-oai-turn-trace-id": str(uuid.uuid4()),
            "OAI-Echo-Logs": "1,0,0,0",
            "X-OpenAI-Target-Path": "/backend-anon/f/conversation",
            "Content-Type": "application/json",
        }

        encoded = json.dumps(body, separators=(',', ':')).encode()

        _log(f"POST /backend-anon/f/conversation  model={real_model} force_use_search={force_use_search}")
        async with self.client.stream(
            "POST",
            f"{BASE}/backend-anon/f/conversation",
            headers=hdrs,
            content=encoded,
        ) as resp:
            _log(f"HTTP {resp.status_code}")
            if resp.status_code == 401:
                _log("401 → renovando sesión y reintentando")
                self._ready = False
                await self.initialize()
                async with self.client.stream(
                    "POST",
                    f"{BASE}/backend-anon/f/conversation",
                    headers=hdrs,
                    content=encoded,
                ) as resp2:
                    _log(f"retry HTTP {resp2.status_code}")
                    resp2.raise_for_status()
                    async for chunk in self._parse_sse(resp2):
                        yield chunk
                return

            resp.raise_for_status()
            async for chunk in self._parse_sse(resp):
                yield chunk

    @staticmethod
    def _apply_delta(delta: dict, buf: str) -> str:
        """
        Aplica una operación de delta al buffer de texto acumulado.
        Sólo procesa el canal de texto visible (/message/content/parts/0).
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
        # Append implícito: {"v": "texto"} sin op ni path explícito
        if not op and isinstance(val, str):
            return buf + val
        return buf

    async def _parse_sse(self, resp) -> AsyncGenerator[str, None]:
        """
        Parser SSE que maneja los tres formatos que envía el backend:

          1. event: delta  +  data: {"o":"append","p":"/message/content/parts/0","v":"texto"}
             (compact keys al nivel raíz — formato real del servidor /backend-anon/f/conversation)

          2. data: {"delta":{"operation":"append","path":"...","value":"texto"}}
             (delta anidado — documentado en FORMATO-SSE.md)

          3. data: {"message":{"content":{"parts":["texto acumulado"]}},...}
             (Formato A legacy — snapshot completo acumulado)
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
                _log(f"[DONE] después de {line_count} líneas. buf final: {buf[:80]!r}")
                break

            try:
                evt = json.loads(data_str)
                if not isinstance(evt, dict):
                    continue

                if evt.get("conversation_id"):
                    self.conversation_id = evt["conversation_id"]

                etype = evt.get("type")

                if etype in ("delta_encoding", "message_stream_complete",
                             "message_marker", "title_generation"):
                    if evt.get("message_id"):
                        last_assistant_msg_id = evt["message_id"]
                    _log(f"control event type={etype}")
                    event_type = None
                    continue

                # ── Formato 1: event: delta + compact top-level keys ───────
                if event_type == "delta":
                    new_buf = self._apply_delta(evt, buf)
                    if new_buf != buf:
                        chunk = new_buf[len(emitted):]
                        buf = new_buf
                        emitted = buf
                        if chunk:
                            yield chunk
                    event_type = None
                    continue

                # ── Formato 2: delta anidado {"delta": {...}} ───────────────
                delta = evt.get("delta")
                if isinstance(delta, dict):
                    new_buf = self._apply_delta(delta, buf)
                    if new_buf != buf:
                        chunk = new_buf[len(emitted):]
                        buf = new_buf
                        emitted = buf
                        if chunk:
                            yield chunk
                    continue

                # ── Formato A: snapshot acumulado {"message": {...}} ────────
                msg = evt.get("message")
                if msg and isinstance(msg, dict):
                    last_assistant_msg_id = msg.get("id", last_assistant_msg_id)
                    parts = (msg.get("content") or {}).get("parts") or []
                    if parts and isinstance(parts[0], str) and parts[0]:
                        new_buf = parts[0]
                        if new_buf != buf:
                            chunk = new_buf[len(emitted):]
                            buf = new_buf
                            emitted = buf
                            if chunk:
                                yield chunk
                    continue

                keys = list(evt.keys())
                _log(f"evento desconocido event_type={event_type!r} keys={keys}")

            except (json.JSONDecodeError, AttributeError) as e:
                _log(f"parse error: {e}  line={line[:80]}")

        _log(f"stream terminado. líneas={line_count} buf={buf[:80]!r}")
        if last_assistant_msg_id:
            self.parent_message_id = last_assistant_msg_id

    async def close(self):
        await self.client.aclose()


async def fetch_anon_models() -> list:
    """
    Obtiene la lista de modelos del endpoint /backend-anon/models.
    Usa un cliente temporal (sin sesión completa).
    """
    device_id = str(uuid.uuid4())
    hdrs = {
        **_base_headers(device_id),
        "X-OpenAI-Target-Path": "/backend-anon/models",
    }
    async with httpx.AsyncClient(verify=True, timeout=10.0, follow_redirects=True) as client:
        r = await client.get(
            f"{BASE}/backend-anon/models",
            params={"iim": "false", "supports_model_picker_upgrade_presets": "true"},
            headers=hdrs,
        )
        r.raise_for_status()
        return r.json().get("models", [])


class SessionPool:
    """
    Pool de sesiones keyed por hash del historial de mensajes.
    Permite multi-turno transparente para clientes OpenAI-compatible.
    """
    TTL = 1800  # 30 min sin uso → cerrar

    def __init__(self):
        self._pool: dict[str, ChatGPTSession] = {}

    @staticmethod
    def _key(messages: list[dict], system_prompt: str = "") -> str:
        """
        Clave = hash del historial EXCEPTO el último mensaje del usuario.
        Incluye el system_prompt para que distintas instrucciones usen sesiones distintas.
        """
        history = [m for m in messages if not (m == messages[-1] and m.get("role") == "user")]
        if not history and not system_prompt:
            return "new_" + str(uuid.uuid4())  # siempre nueva sesión si no hay historial
        canonical = json.dumps(history, sort_keys=True, ensure_ascii=False) + system_prompt
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def get(self, messages: list[dict], system_prompt: str = "") -> tuple[str, ChatGPTSession]:
        key = self._key(messages, system_prompt)
        await self._evict_stale()
        if key.startswith("new_") or key not in self._pool:
            session = ChatGPTSession(system_prompt=system_prompt)
            self._pool[key] = session
        return key, self._pool[key]

    async def _evict_stale(self):
        now = time.time()
        stale = [k for k, s in self._pool.items() if now - s.last_used > self.TTL]
        for k in stale:
            await self._pool[k].close()
            del self._pool[k]

    async def close_all(self):
        for s in self._pool.values():
            await s.close()
        self._pool.clear()
