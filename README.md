# 🆓 ChatGPT Free

### Use ChatGPT completely free — no API key, no account, no credit card.

> OpenAI-compatible proxy that routes requests through the ChatGPT Android anonymous endpoint.

[![Docs](https://img.shields.io/badge/GitHub%20Pages-Docs-00B899?style=flat-square)](https://lordmacu.github.io/chatgpt-proxy/)
[![Swagger](https://img.shields.io/badge/Swagger-UI-85EA2D?style=flat-square&logo=swagger&logoColor=black)](https://lordmacu.github.io/chatgpt-proxy/swagger/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

---

## How it works

The ChatGPT Android app exposes an anonymous conversation endpoint that requires only a randomly generated device ID — no user token, no API key. This proxy:

1. Manages anonymous device sessions automatically
2. Exposes an OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/models`)
3. Streams responses as Server-Sent Events
4. Auto-rotates the device ID when a session quota is exhausted
5. Includes a built-in web chat UI and a terminal CLI

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/lordmacu/chatgpt-proxy.git
cd chatgpt-proxy
```

**macOS / Linux**
```bash
pip3 install -r requirements.txt
```

**Windows**
```bash
pip install -r requirements.txt
```

### 2. Run — choose your mode

#### Web UI (recommended)

Starts the API server and opens a chat interface in your browser:

**macOS / Linux**
```bash
python3 cli.py web
```

**Windows**
```bash
python cli.py web
```

Optional flags:
```bash
python3 cli.py web 9000          # custom port
python3 cli.py web --no-browser  # don't open browser automatically
```

Then open **http://localhost:8000** in your browser.

---

#### API Server only

**macOS / Linux**
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Windows**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Docker

```bash
docker compose up -d
```

The server listens on port `8888` when running via Docker Compose.

---

## Web UI

`python3 cli.py web` starts both the API server and a built-in chat interface:

- **Model selector** — choose from all available anonymous models
- **Web Search** — toggle real-time web search
- **Advanced Tools** — enable code interpreter, shopping, reservations
- **Canvas** — collaborative document mode
- **JSON mode** — force the model to return valid JSON
- **System prompt** — set a custom system instruction
- **Stop button** — cancel a streaming response mid-flight
- **New conversation** — clear history and start fresh
- **Suggestion chips** — click to prefill quick prompts

---

## Terminal CLI

A standalone terminal client — no server required, talks directly to ChatGPT.

### Single question

**macOS / Linux**
```bash
python3 cli.py "What is the capital of France?"
```

**Windows**
```bash
python cli.py "What is the capital of France?"
```

### All CLI options

```
python3 cli.py [options] [message]

Modes:
  (no message)         Start interactive REPL
  --chat, -c           Start interactive REPL explicitly
  --no-stream          Wait for full response before printing

Model & prompt:
  --model MODEL, -m    Model to use (default: auto)
  --system PROMPT, -s  System prompt

Features:
  --search             Force web search on
  --no-search          Force web search off
  --tools              Enable advanced tools (code interpreter, etc.)
  --canvas             Enable Canvas / document mode
  --effort LEVEL       Thinking effort: standard | extended | max
  --tier TIER          Service tier: standard | priority
  --json, -j           Force JSON output mode

Files:
  --file PATH, -f      Attach a file (PDF or text). Repeatable.

Output:
  --quiet, -q          Suppress decorations (pipe-friendly)

Web UI:
  web [PORT]           Start web server + open browser (default port: 8000)
  web --no-browser     Start server without opening browser
```

### Examples

```bash
# Simple question
python3 cli.py "Explain recursion in one sentence"

# Choose a model
python3 cli.py -m gpt-4o "Write a haiku about Python"

# With a system prompt
python3 cli.py -s "You are a pirate" "Tell me about the sea"

# Force web search
python3 cli.py --search "Latest news about AI"

# Force JSON output
python3 cli.py --json "Return a JSON object with name, age, and city"

# Attach a file
python3 cli.py --file report.pdf "Summarize this document"
python3 cli.py --file data.csv "What trends do you see?"

# Multiple files
python3 cli.py -f file1.txt -f file2.txt "Compare these two documents"

# Quiet mode (pipe-friendly)
python3 cli.py -q "Write a poem about rain" > poem.txt

# Non-streaming (wait for full response)
python3 cli.py --no-stream "Tell me a long story"

# Start the web UI on a custom port
python3 cli.py web 9090
```

### Interactive REPL

Start with `python3 cli.py` or `python3 cli.py --chat`:

```
> python3 cli.py --chat

You: Hello! What can you do?
AI:  I can answer questions, write code, analyze files, search the web...

You: /search on
You: What happened in AI news today?
AI:  [web search results...]

You: /model gpt-4o
You: Now explain that using simpler words
```

**Slash commands in REPL:**

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/model <name>` | Switch model (e.g. `/model gpt-4o`) |
| `/search on\|off` | Toggle web search |
| `/tools on\|off` | Toggle advanced tools |
| `/effort <level>` | Thinking effort: `standard` \| `extended` \| `max` (or `default`) |
| `/json on\|off` | Toggle JSON output mode |
| `/system <text>` | Change the system prompt |
| `/info` | Show current session settings |
| `/clear` | Clear conversation history |
| `/quit` | Exit (also Ctrl+D) |

**File attachments in REPL:**
```bash
python3 cli.py --chat --file report.pdf
```

PDF support requires pdfplumber:
```bash
pip3 install pdfplumber
```

---

## API Reference

Full interactive docs: **[lordmacu.github.io/chatgpt-proxy/swagger/](https://lordmacu.github.io/chatgpt-proxy/swagger/)**

### POST /v1/chat/completions

OpenAI-compatible chat completions.

**Request:**
```json
{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Hello!"}
  ],
  "stream": true
}
```

**All request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `auto` | Model ID |
| `messages` | array | — | Conversation messages |
| `stream` | boolean | `false` | Stream via SSE |
| `response_format` | object | — | `{"type":"json_object"}` for JSON mode. Not an API flag — the backend ignores `response_format`, so it works by instructing the model. Dropping it on a later turn of the same conversation sends an explicit retraction, since the original instruction is still sitting in the history the model reads |
| `web_search` | boolean | `null` | Force web search on/off |
| `force_use_tools` | boolean | `null` | Enable advanced tools |
| `force_use_canvas` | boolean | `null` | Enable Canvas mode |
| `tools` | array | — | OpenAI tools format (see note) |
| `tool_choice` | string/object | — | `"none"` \| `"auto"` \| `"required"` \| `{"type":"function",...}` |
| `tool_emulation` | boolean | `true` | `false` turns custom function calling off |
| `tool_verify` | boolean | `false` | Spend one extra message auditing the call for a dropped condition |
| `reasoning_effort` | string | `null` | `minimal`/`low` → `standard`, `medium` → `extended`, `high` → `max` |
| `thinking_effort` | string | `null` | Native form: `standard` \| `extended` \| `max` |
| `service_tier` | string | `null` | `auto`/`default`/`flex`/`scale` → `standard`; `priority` passes through |
| `force_disable_features` | array | — | Feature names to switch off for this turn |
| `temperature`, `max_tokens`, `top_p`, etc. | — | — | Accepted, dropped (see note) |

> **Note on `tools`:** two different things share this array.
> **Built-in names** switch on a ChatGPT mode that runs server-side, using the same names
> upstream reports in each model's `enabled_tools`: `web_search`/`search` → web search,
> `canvas` → Canvas, `tools`/`chatgpt_tools` → advanced tools.
> **Any other name is your own function**, and the proxy returns real `tool_calls` for it
> (see [Function calling](#function-calling)). `tool_choice: "none"` disables everything.

> **Note on sampling parameters:** `temperature`, `top_p`, `max_tokens`,
> `presence_penalty`, `frequency_penalty`, `stop`, `seed` and `n` have no equivalent
> in the ChatGPT conversation protocol — the fields don't exist upstream. They're
> accepted so stock OpenAI clients keep working, then dropped. A request that carries
> any of them comes back with an `X-Proxy-Ignored-Params` header listing exactly which
> ones were dropped, so the loss is visible instead of silent. The generation controls
> that *are* real are `reasoning_effort`/`thinking_effort` and `service_tier`; an
> unsupported value for either is a `400` from the proxy rather than a wasted upstream
> request. Note that in anonymous mode `/v1/models` reports
> `configurable_thinking_effort: false` for every model, so the effort value is carried
> through but may not change the answer.

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "gpt-4o",
  "choices": [{
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "chatgpt_metadata": {
    "conversation_id": "...",
    "message_id": "...",
    "model": "gpt-4o",
    "sources": []
  }
}
```

**Streaming (SSE):**
```
data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"...","choices":[{"delta":{"content":"!"}}]}
data: {"chatgpt_metadata":{"model":"gpt-4o","sources":[]}}
data: [DONE]
```

---

### GET /v1/models

Returns available models, each with the capabilities upstream reports for it.

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-5-6",
      "object": "model",
      "owned_by": "openai",
      "description": "GPT-5.6 Luna",
      "context_window": 52815,
      "reasoning_type": "auto",
      "enabled_tools": ["tools", "tools2", "search", "canvas", "app_pairing", "image_gen_tool_enabled"],
      "configurable_thinking_effort": false,
      "thinking_efforts": []
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `context_window` | Per-model token ceiling. This is the only token limit in the API — there is no `max_tokens` *request* parameter |
| `reasoning_type` | `auto` \| `reasoning` \| `none` |
| `enabled_tools` | Server-side tools this model may use — the names `tools` accepts |
| `configurable_thinking_effort` | Whether `thinking_effort` is expected to change behaviour on this model |
| `thinking_efforts` | Effort levels offered for this model (empty in anonymous mode) |

---

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "version": "2.4.0"}
```

---

## curl examples

```bash
# Basic chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Tell me a joke"}],"stream":true}'

# JSON mode
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role":"user","content":"List 3 programming languages as JSON"}],
    "response_format": {"type": "json_object"}
  }'

# Web search
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role":"user","content":"What is the weather in New York today?"}],
    "web_search": true
  }'
```

---

## Use with OpenAI SDK

Works with any OpenAI-compatible library — just point `base_url` at the proxy:

**Python**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**JavaScript / Node.js**
```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-needed",
});

const stream = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "Hello!" }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

**LangChain**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="auto",
    streaming=True,
)

response = llm.invoke("Explain machine learning in simple terms")
print(response.content)
```

---

## JSON mode

Force the model to respond with valid JSON:

```bash
python3 cli.py --json "Return a person object with name, age, and hobbies"
```

```json
{
  "name": "Alex",
  "age": 28,
  "hobbies": ["reading", "hiking", "cooking"]
}
```

Via API:
```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "List 5 countries as JSON"}],
  "response_format": {"type": "json_object"}
}
```

---

## Quota handling

Anonymous mode has **two separate ceilings**, and they behave very differently. Both
were measured against the live backend, not read off a doc.

| | What it is | When it hits | What happens | Reset |
|---|---|---|---|---|
| **Model cap** | Messages on the top model (`gpt-5-6`) | message **11** | **Silent downgrade** to `gpt-5-6-mini` — still HTTP 200, no error, no warning | ~5 h |
| **Hourly cap** | Messages per device per hour | ~30–45+ | Real `429`/`403`: *"You've reached our limit of messages per hour"* | 1 h |

The hourly number is a range on purpose: one run did 45 messages on a fresh device
without tripping it, another tripped at 31 — traffic from the same IP earlier in the
window contributes, so it is not a clean per-device constant.

**The hourly cap is not a wall for API callers.** On `429`/`403` the proxy marks the
session exhausted, takes a fresh device ID, and retries the same request transparently
— verified working immediately after a real 429, with no wait. Only a double failure
surfaces a standard `rate_limit_exceeded` error:

1. The session is marked as exhausted
2. The next request automatically gets a fresh device ID
3. The current request is retried transparently
4. On double failure, a standard `rate_limit_exceeded` error is returned

The CLI does the same, with a visible message:
```
Retrying with a fresh session…
```

Multi-turn survives the rotation for API callers, because the proxy rebuilds context
from the `messages` array you send on every request. The CLI keeps state in the session
instead, so a rotation there starts the conversation over.

**The model cap is the one to watch.** Because the downgrade comes back as a normal
`200`, nothing triggers the rotation logic — the proxy keeps using the same device and
the answers keep arriving from the smaller model. Check `model_limits` in
`GET /v1/limits` to see it:

```json
{"model_slug": "gpt-5-6", "using_default_model_slug": "gpt-5-6",
 "resets_after": "2026-08-20T20:46:09Z"}
```

An empty `model_limits` means no model is capped right now.

### Per-feature limits

`GET /v1/limits` reports the rest, and works in both modes:

| Feature | Anonymous | Account (Go plan) |
|---------|-----------|-------------------|
| `file_upload` | 3 / 24 h | 80 / 3 h |
| `paste_text_to_file` | 3 / 24 h | 80 / 3 h |
| `dictation` | 1 / 7 days | — |
| `image_gen` | **0 — blocked** ("Log in or sign up to create images for free") | 106 / ~11 h |
| `reason` | — | 300 / ~1 h |
| `deep_research` | — | 5 / ~1 month |

Account figures are from a `chatgptgoplan` subscription and will differ on other plans;
read the live values rather than hardcoding these. Chat messages themselves have no
counter in `limits_progress` in either mode — the caps only show up as `model_limits`
once you reach one.

---

## Anonymous vs account

The proxy runs in one of two modes, decided by whether a token is configured
(`CHATGPT_ACCESS_TOKEN`, or a saved `tokens.json`). `GET /health` reports which one
is active as `auth_mode: "anonymous" | "account"`.

Anonymous mode needs nothing at all — no API key, no account, no credit card. It
covers full chat, and that is the point of this project. An account only adds the
things that are, by definition, tied to a user: your conversations, your files, your
custom GPTs, TTS, and image generation.

### Chat capabilities

| Capability | Anonymous | Account |
|------------|:---------:|:-------:|
| Text chat + streaming SSE | ✅ | ✅ |
| Multi-turn (session TTL ~30 min) | ✅ | ✅ |
| System prompt | ✅ | ✅ |
| Web search (`web_search`) | ✅ | ✅ |
| Advanced tools (`force_use_tools`) | ✅ | ✅ |
| Canvas (`force_use_canvas`) | ✅ | ✅ |
| JSON output mode | ✅ | ✅ |
| File attachments (PDF, code, docs) | ✅ | ✅ |
| `reasoning_effort` / `thinking_effort` | ✅ | ✅ |
| `service_tier` | ✅ | ✅ |
| Quota auto-rotation (new device ID) | ✅ | n/a |
| Context window | 34,834 tokens | 52,815 (262,144 on `-t-mini`) |
| Reasoning models | ❌ | ✅ `gpt-5-4-t-mini`, `gpt-5-6-t-mini` |
| Deep research model | ❌ | ✅ `research` |
| Custom GPTs (`model: "g-..."`) | ❌ | ✅ |
| Image generation | ❌ | ✅ |
| Text-to-speech | ❌ | ✅ |
| Speech-to-text | ❌ | ✅ |
| Image input (vision, `image_url`) | ❌ | ✅ |
| Function calling (`tool_calls` in the response) | ✅ | ✅ |
| `temperature`, `top_p`, `max_tokens`, penalties, `seed`, `n` | ❌ | ❌ |

Vision accepts OpenAI-style `image_url` parts (`data:` URLs or `http(s)` URLs);
each image is uploaded to the account's file store and attached to the message.
It needs an account, so an image with no token is a fast `401`. The anonymous
backend does have `POST /backend-anon/files` — it answers 200 with a signed
URL and the blob upload succeeds — but finalising the file
(`POST /files/{id}/uploaded`) answers `401`, so the upload never becomes
readable while still spending one of the three uploads allowed per 24 hours.
Anonymous attachments therefore have to be inlined as text. Sampling parameters are ❌ in *both* modes and won't change
with an account: the conversation protocol simply has no such fields, so they are
accepted and then dropped, with an `X-Proxy-Ignored-Params` response header listing
which ones.

### Function calling

The backend has no native function calling — it is emulated, and the official OpenAI
SDK drives the loop unmodified:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

msgs = [{"role": "user", "content": "What's the weather in Bogotá?"}]
r = client.chat.completions.create(model="auto", messages=msgs, tools=TOOLS)
# r.choices[0].finish_reason == "tool_calls"

msgs.append(r.choices[0].message)
for call in r.choices[0].message.tool_calls:
    msgs.append({"role": "tool", "tool_call_id": call.id,
                 "content": '{"temperature": 14}'})
client.chat.completions.create(model="auto", messages=msgs, tools=TOOLS)
# -> "In Bogotá it's 14 °C."
```

Streaming, parallel calls, `tool_choice: "required"` and a pinned function all work.
Two things to know:

- **It costs one extra upstream message** per turn that declares custom functions:
  deciding *whether* to call is a request of its own, so a turn where no function fits
  spends two. `X-Proxy-Tool-Extraction` and `X-Proxy-Tool-Requests` report what happened.
- **A required argument the request never states is answered with a question**, not a
  guess. Dense requests packing many conditions can still drop one — `tool_verify: true`
  spends a message auditing for exactly that.

`POST /v1/tool-calls` exposes the same decision on its own, with no conversation
attached — useful when you want the call and will run the conversation elsewhere:

```bash
curl -X POST http://localhost:8000/v1/tool-calls \
  -H 'Content-Type: application/json' \
  -d '{"tools": [...], "input": "weather in Lima and Quito"}'
# {"status": "calls", "tool_calls": [...], "usage": {"upstream_requests": 1}}
```

`status` is `calls`, `no_call`, or `need_info` (with `need_info.missing` naming the
parameters the request left out).

### Endpoints

| Endpoint | Anonymous | Account | Notes |
|----------|:---------:|:-------:|-------|
| `GET /health` | ✅ | ✅ | Reports the active `auth_mode` |
| `GET /` | ✅ | ✅ | Built-in web chat UI |
| `GET /v1/models` | ✅ | ✅ | Shorter list anonymously — see below |
| `GET /v1/session/me` | ✅ | ✅ | Identity is `ua-...` vs `user-...` |
| `POST /v1/chat/completions` | ✅ | ✅ | Except `model: "g-..."` (custom GPTs) |
| `POST /v1/tool-calls` | ✅ | ✅ | Stateless function-call extraction |
| `POST` / `GET` / `DELETE /v1/files` | ✅ | ✅ | Proxy-local store, never touches the account |
| `GET /v1/limits` | ✅ | ✅ | Per-feature remaining counts |
| `POST /v1/translate` | ✅ | ✅ | Doesn't spend a chat message |
| `GET /v1/account` | ❌ | ✅ | |
| `GET` / `POST /v1/custom-instructions` | ❌ | ✅ | |
| `GET /v1/gizmos`, `GET /v1/gizmos/{id}` | ❌ | ✅ | |
| `GET /v1/conversations` | ✅ | ✅ | Anonymously the vendor answers an empty page even to the owning device, so this serves the proxy's own index instead |
| `GET /v1/conversations/{id}` | ✅ | ✅ | Anonymous turns ARE stored upstream, readable only from the device that created them — which the proxy records, so eviction and restarts no longer lose them |
| `GET /v1/library`, `/usage`, `/{id}/download` | ❌ | ✅ | |
| `DELETE /v1/library/{id}`, `/trash`, `POST /{id}/restore` | ❌ | ✅ | |
| `GET /v1/suggestions` | ❌ | ✅ | Prompt-library starters |
| `POST /v1/projects`, `GET`/`DELETE /v1/projects/{id}` | ❌ | ✅ | Projects = `g-p-...` gizmos; chat inside with `model:"g-p-..."` |
| `POST /v1/audio/transcriptions` | ❌ | ✅ | |
| `POST /v1/audio/speech` | ❌ | ✅ | `/backend-anon/synthesize` doesn't exist |
| `GET /v1/audio/from-message` | ❌ | ✅ | Same `synthesize` backend |
| `POST /v1/images/generations` | ❌ | ✅ | |
| `GET /images/{f}`, `GET /audio/{f}` | ✅ | ✅ | Serve files the proxy already cached locally |

Account-only endpoints answer with a plain `401`:

```json
{"error": {"message": "This endpoint needs an authenticated account (set CHATGPT_ACCESS_TOKEN).", "type": "auth_error"}}
```

---

## Available models

`GET /v1/models` returns what the current mode actually offers, each entry carrying
its real capabilities (`context_window`, `reasoning_type`, `enabled_tools`, …).

| Model ID | Anonymous | Account | Description |
|----------|:---------:|:-------:|-------------|
| `auto` | ✅ | — | Server picks the model. Only listed anonymously, but works in both |
| `gpt-5-6` | ✅ | ✅ | GPT-5.6 Luna — most capable |
| `gpt-5-5` | ✅ | ✅ | GPT-5.5 |
| `gpt-5-6-mini` | ✅ | ✅ | GPT-5.6 Luna Mini |
| `gpt-5-5-mini` | ✅ | ✅ | GPT-5.5 Mini |
| `gpt-5-3-mini` | ✅ | ✅ | GPT-5.3 Mini — fastest |
| `gpt-5-6-t-mini` | ❌ | ✅ | Reasoning, 262k context |
| `gpt-5-4-t-mini` | ❌ | ✅ | Reasoning, 262k context |
| `research` | ❌ | ✅ | Deep research |
| `g-...` | ❌ | ✅ | Any custom GPT you have access to |
| `gpt-image-1` | ❌ | ✅ | Image generation via `/v1/chat/completions` |

Legacy aliases for stock OpenAI clients: `gpt-4o` → `auto`, `gpt-4` → `gpt-5-5`,
`gpt-4o-mini` and `gpt-3.5-turbo` → `gpt-5-3-mini`.

---

## Project structure

```
chatgpt-proxy/
├── main.py                        # FastAPI app — API + web UI
├── chatgpt_client.py              # Core: session management, streaming
├── cli.py                         # CLI + web launcher
├── openapi.json                   # OpenAPI 3.0 spec
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── chatgpt-proxy.postman_collection.json
├── SECURITY_RECOMMENDATIONS.md   # Responsible disclosure for OpenAI
└── docs/                          # GitHub Pages
    ├── index.html
    └── swagger/
        ├── index.html
        └── openapi.json
```

---

## Security

This project is published for **educational and research purposes** to document the anonymous API flow. See [`SECURITY_RECOMMENDATIONS.md`](SECURITY_RECOMMENDATIONS.md) for a responsible-disclosure document with hardening recommendations for OpenAI.

Do not use this proxy to violate OpenAI's [Terms of Service](https://openai.com/policies/terms-of-use/).

---

## Links

- **Web UI:** run `python3 cli.py web` → http://localhost:8000
- **Docs:** [lordmacu.github.io/chatgpt-proxy/](https://lordmacu.github.io/chatgpt-proxy/)
- **Swagger:** [lordmacu.github.io/chatgpt-proxy/swagger/](https://lordmacu.github.io/chatgpt-proxy/swagger/)
- **Security:** [SECURITY_RECOMMENDATIONS.md](SECURITY_RECOMMENDATIONS.md)
