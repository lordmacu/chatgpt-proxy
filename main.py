"""
FastAPI proxy — API compatible con OpenAI usando ChatGPT Android anónimo.

Endpoints disponibles:
  POST /v1/chat/completions   — chat (streaming + no-streaming, multipart content, archivos)
  GET  /v1/models             — lista de modelos disponibles en modo anónimo
  POST /v1/files              — subir un archivo (PDF, código, docs) → file_id
  GET  /v1/files              — listar archivos subidos
  GET  /v1/files/{file_id}    — info de un archivo
  DEL  /v1/files/{file_id}    — eliminar un archivo
  GET  /health                — estado del servidor

Capacidades del flujo anónimo:
  ✅ Chat de texto con GPT-5.5 / 5.6 Luna / 5.3 Mini / auto
  ✅ Streaming SSE
  ✅ Multi-turno (dentro de la sesión, ~30 min TTL)
  ✅ System prompt (persistente en la sesión, sólo primer turno)
  ✅ Adjuntos de texto: PDF, código, docs (82 MIME types, tipo retrieval)
  ✅ Web search — auto, o controlable con web_search: true/false
  ✅ Tools avanzados — reservas, shopping, widgets, canvas (force_use_tools: true)
  ❌ Imágenes como input (vision) — no disponible en anónimo
  ❌ Generación de imágenes — no disponible en anónimo
  ❌ Function calling / tool_calls en respuesta — no soportado por el backend anónimo
  ❌ Voice / TTS — requiere cuenta

Compatibilidad con campo `tools` de OpenAI (input-only):
  El array `tools` se acepta para compatibilidad, pero el backend anónimo NO devuelve
  tool_calls. Los nombres de función se mapean a flags internos del backend:
    • "web_search" / "search" / ...       → force_use_search = true
    • "chatgpt_tools" / "all_tools" / ... → force_use_tools = true
    • cualquier otro nombre               → force_use_tools = true
    • tool_choice: "none"                 → desactiva ambos modos
  La respuesta es siempre text en choices[0].message.content, nunca tool_calls.
"""
import io
import uuid
import json
import time
from typing import Optional, AsyncGenerator, Union, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from chatgpt_client import SessionPool, fetch_anon_models

# ---------------------------------------------------------------------------
# Almacenes por usuario  {user_id: {file_id: {...}}}  /  {user_id: SessionPool}
# ---------------------------------------------------------------------------
_files: dict[str, dict[str, dict]] = {}
_pools: dict[str, SessionPool] = {}

# Caché de modelos del servidor: (timestamp, lista)
_models_cache: tuple[float, list] = (0.0, [])
_MODELS_TTL = 300  # refrescar cada 5 minutos


def _get_user_id(request: Request) -> str:
    """
    Extrae el identificador de usuario del header Authorization.
    Formato esperado: 'Bearer <token>'  (igual que OpenAI).
    Si no viene el header, devuelve 'anonymous' — todos comparten el mismo espacio.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    return "anonymous"


def _user_files(user_id: str) -> dict[str, dict]:
    """Devuelve (o crea) el dict de archivos del usuario."""
    if user_id not in _files:
        _files[user_id] = {}
    return _files[user_id]


def _user_pool(user_id: str) -> SessionPool:
    """Devuelve (o crea) el SessionPool del usuario."""
    if user_id not in _pools:
        _pools[user_id] = SessionPool()
    return _pools[user_id]

# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class ContentPart(BaseModel):
    """Un fragmento de contenido en un mensaje multipart."""
    type: str                      # "text" | "file" | "image_url"
    text: Optional[str] = None
    file: Optional[dict] = None    # {"file_id": "file-xxx"} o {"filename": "...", "content": "..."}
    image_url: Optional[dict] = None  # {"url": "..."} — no soportado en anónimo

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]] = ""
    name: Optional[str] = None


# Nombres de herramientas "built-in" que este proxy reconoce en el array tools de OpenAI.
# Cualquier otro nombre activa force_use_tools (las herramientas internas del backend).
_BUILTIN_TOOL_WEB_SEARCH = {"web_search", "search", "brave_search", "bing_search", "google_search"}
_BUILTIN_TOOL_ALL        = {"chatgpt_tools", "all_tools", "builtin_tools"}


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: List[Message]
    stream: bool = False

    # ── Formato OpenAI estándar ────────────────────────────────────────────────
    # tools: lista de herramientas (formato OpenAI). Las funciones personalizadas
    # son ignoradas por el backend anónimo, pero sus nombres controlan qué modo
    # se activa en ChatGPT:
    #
    #   • "web_search" / "search" / "brave_search" / ...  → force_use_search = true
    #   • "chatgpt_tools" / "all_tools" / "builtin_tools" → force_use_tools = true
    #   • cualquier otro nombre                            → force_use_tools = true
    #
    # tool_choice: "none" desactiva todos los modos; "auto"/"required" los activa.
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None

    # ── Extensiones propias (siguen funcionando) ───────────────────────────────
    # web_search: true/false/null — override directo de force_use_search
    web_search: Optional[bool] = Field(default=None)
    # force_use_tools: true/null — override directo de force_use_tools
    force_use_tools: Optional[bool] = Field(default=None)
    # force_use_canvas: true/null — activa modo Canvas (documentos colaborativos)
    # La respuesta llega como texto con bloques :::writing{variant="document"...}
    force_use_canvas: Optional[bool] = Field(default=None)

    # ── Campos de compatibilidad OpenAI (aceptados, ignorados) ────────────────
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    n: Optional[int] = None

    model_config = {"populate_by_name": True}

    def resolved_backend_flags(self) -> tuple[Optional[bool], Optional[bool]]:
        """
        Devuelve (force_use_search, force_use_tools) resolviendo la semántica de
        OpenAI tools + tool_choice sobre los overrides directos web_search /
        force_use_tools.

        Prioridad: override directo > inferido desde tools array.
        """
        search = self.web_search
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
                # Todas las tools son de búsqueda → solo force_use_search
                if search is None:
                    search = True
            else:
                # Hay tools no-search o all_tools → force_use_tools
                if use_tools is None:
                    use_tools = True
                if has_search and search is None:
                    search = True

        elif self.tool_choice == "none":
            # tool_choice: "none" desactiva explícitamente ambos modos
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
    description="API compatible con OpenAI usando el flujo anónimo de ChatGPT Android",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(content: str, model: str, completion_id: str, finish: bool = False) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {} if finish else {"content": content},
            "finish_reason": "stop" if finish else None,
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"

def _make_completion(content: str, model: str, completion_id: str) -> dict:
    words = len(content.split())
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": words,
            "total_tokens": words,
        }
    }

def _msg_to_text(m: "Message") -> str:
    """Extrae el texto plano de un mensaje (str o lista de ContentPart)."""
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
    Extrae:
      - system_prompt: contenido del mensaje con role=system
      - last_user_text: texto del último mensaje del usuario, con el historial
        previo inyectado como contexto (para multi-turno sin estado de sesión)
      - file_texts: contenidos de archivos adjuntos en el último mensaje

    El historial previo (user + assistant) se inyecta directamente en el
    mensaje del usuario porque el backend anónimo no mantiene contexto entre
    requests — cada llamada es nueva para el servidor.
    """
    system_prompt = ""
    file_texts: List[str] = []

    for m in messages:
        if m.role == "system":
            system_prompt = _msg_to_text(m)

    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "No hay mensajes del usuario")

    last = user_msgs[-1]

    # Extraer texto y archivos del último mensaje del usuario
    last_user_text = ""
    if isinstance(last.content, str):
        last_user_text = last.content
    elif isinstance(last.content, list):
        for part in last.content:
            if part.type == "text" and part.text:
                last_user_text += part.text
            elif part.type == "file":
                fdata = part.file or {}
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
                    "image_url no está disponible en el flujo anónimo. "
                    "Usa archivos de texto (type='file') en su lugar."
                )

    # Inyectar historial previo (user + assistant) como contexto en el mensaje
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
        history_block = "\n".join(prior_turns)
        last_user_text = (
            f"[Prior conversation — use this as context:\n{history_block}\n]\n\n"
            f"{last_user_text}"
        )

    if not last_user_text:
        raise HTTPException(400, "El último mensaje del usuario está vacío")

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
            raw = await fetch_anon_models()
            _models_cache = (now, raw)
            cached_models = raw
        except Exception as e:
            if cached_models:
                pass  # usar caché vieja antes de fallar
            else:
                raise HTTPException(502, f"No se pudo obtener la lista de modelos: {e}")

    data = [
        {
            "id": m.get("slug", m.get("id", "")),
            "object": "model",
            "created": 1750000000,
            "owned_by": "openai",
            "description": m.get("title", ""),
        }
        for m in cached_models
        if m.get("slug") or m.get("id")
    ]

    # Agregar aliases legacy para compatibilidad con clientes OpenAI
    existing_ids = {d["id"] for d in data}
    for alias, target in _LEGACY_ALIASES.items():
        if alias not in existing_ids:
            data.append({
                "id": alias,
                "object": "model",
                "created": 1750000000,
                "owned_by": "openai",
                "description": f"Alias → {target}",
            })

    return {"object": "list", "data": data}

# ---------------------------------------------------------------------------
# /v1/files
# ---------------------------------------------------------------------------

@app.post("/v1/files")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
):
    """
    Sube un archivo de texto (PDF, código, docs) y devuelve un file_id.
    Aislado por usuario — cada API key ve solo sus propios archivos.

    Autenticación: Authorization: Bearer <tu-api-key>
    Compatible con la API de OpenAI.
    """
    user_id = _get_user_id(request)
    uf = _user_files(user_id)

    content_type = file.content_type or "application/octet-stream"

    if content_type.startswith("image/"):
        raise HTTPException(
            415,
            "Las imágenes no están disponibles en el flujo anónimo. "
            "Sólo se soportan archivos de texto/documentos para retrieval."
        )

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "El archivo supera el límite de 20 MB")

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
        raise HTTPException(422, "El archivo no contiene texto extraíble")

    file_id = f"file-{uuid.uuid4().hex[:24]}"
    now = int(time.time())
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
    uf = _user_files(user_id)
    data = [
        {k: v for k, v in f.items() if k != "content"}
        for f in uf.values()
    ]
    return {"object": "list", "data": data}


@app.get("/v1/files/{file_id}")
async def get_file(file_id: str, request: Request):
    user_id = _get_user_id(request)
    uf = _user_files(user_id)
    if file_id not in uf:
        raise HTTPException(404, f"File {file_id!r} no encontrado")
    return {k: v for k, v in uf[file_id].items() if k != "content"}


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, request: Request):
    user_id = _get_user_id(request)
    uf = _user_files(user_id)
    if file_id not in uf:
        raise HTTPException(404, f"File {file_id!r} no encontrado")
    del uf[file_id]
    return {"id": file_id, "object": "file", "deleted": True}


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    user_id = _get_user_id(request)
    uf = _user_files(user_id)
    pool = _user_pool(user_id)

    msgs_raw = [m.model_dump() for m in req.messages]
    system_prompt, last_user_text, file_texts = _resolve_messages(req.messages, uf)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    _key, session = await pool.get(msgs_raw, system_prompt=system_prompt)

    force_use_search, force_use_tools = req.resolved_backend_flags()

    if req.stream:
        async def event_stream() -> AsyncGenerator[bytes, None]:
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(role_chunk)}\n\n".encode()

            try:
                async for text_chunk in session.stream_message(
                    last_user_text,
                    model=req.model,
                    file_texts=file_texts or None,
                    force_use_search=force_use_search,
                    force_use_tools=force_use_tools,
                    force_use_canvas=req.force_use_canvas,
                ):
                    if text_chunk:
                        yield _make_chunk(text_chunk, req.model, completion_id).encode()
            except Exception as e:
                err = {"error": {"message": str(e), "type": "api_error"}}
                yield f"data: {json.dumps(err)}\n\n".encode()
                return

            yield _make_chunk("", req.model, completion_id, finish=True).encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    else:
        full_text = ""
        try:
            async for chunk in session.stream_message(
                last_user_text,
                model=req.model,
                file_texts=file_texts or None,
                force_use_search=force_use_search,
                force_use_tools=force_use_tools,
                force_use_canvas=req.force_use_canvas,
            ):
                full_text += chunk
        except Exception as e:
            raise HTTPException(500, str(e))

        return JSONResponse(_make_completion(full_text, req.model, completion_id))


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    total_sessions = sum(len(p._pool) for p in _pools.values())
    total_files = sum(len(uf) for uf in _files.values())
    return {
        "status": "ok",
        "active_users": len(_pools),
        "total_sessions": total_sessions,
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
            "image_input":      False,
            "image_generation": False,
            "web_search":       "automatic (override con web_search: true/false)",
            "force_use_tools":  "optional (activa tools avanzados: reservas, shopping, widgets)",
            "voice":            False,
        }
    }
