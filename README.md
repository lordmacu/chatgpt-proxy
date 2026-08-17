# chatgpt-proxy

An OpenAI-compatible API proxy that routes requests through the ChatGPT Android anonymous endpoint — no API key, no account required.

[![API Docs](https://img.shields.io/badge/GitHub%20Pages-Docs-00B899?style=flat-square)](https://lordmacu.github.io/chatgpt-proxy/)
[![Swagger](https://img.shields.io/badge/Swagger-UI-85EA2D?style=flat-square&logo=swagger&logoColor=black)](https://lordmacu.github.io/chatgpt-proxy/swagger/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

---

## How it works

The ChatGPT Android app exposes an anonymous conversation endpoint that requires only a randomly generated `oai-device-id` (UUID v4) — no user token, no API key. This proxy:

1. Generates and manages anonymous device sessions
2. Exposes an OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/models`)
3. Streams responses as Server-Sent Events
4. Auto-rotates the device ID when a session quota is exhausted
5. Supports JSON mode, web search, advanced tools, and canvas

> **Security note:** See [`SECURITY_RECOMMENDATIONS.md`](SECURITY_RECOMMENDATIONS.md) for a responsible-disclosure document sent to OpenAI with recommendations for hardening this surface.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/lordmacu/chatgpt-proxy.git
cd chatgpt-proxy
pip install -r requirements.txt
```

### 2. Run the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Make a request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## Docker

```bash
docker compose up -d
```

The server listens on port `8888` when running via Docker Compose.

---

## CLI

A standalone terminal client is included — no server required.

```bash
python cli.py "What is the capital of France?"
```

### Options

```
python cli.py [options] [message]

  -m, --model     Model to use (default: auto)
  -s, --system    System prompt
  --search        Enable web search
  --tools         Enable advanced tools (code interpreter, etc.)
  --canvas        Enable canvas mode
  --json          Force JSON output
  --no-stream     Collect full response before printing
  --file FILE     Attach a file (text or PDF)
  --quiet         Suppress metadata output
```

### Interactive REPL

```bash
python cli.py -i
```

Slash commands available in interactive mode:

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/model <name>` | Switch model |
| `/search on\|off` | Toggle web search |
| `/tools on\|off` | Toggle advanced tools |
| `/json on\|off` | Toggle JSON mode |
| `/system <text>` | Change system prompt |
| `/info` | Show session info |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

### File attachments

```bash
python cli.py --file report.pdf "Summarize this document"
python cli.py --file data.csv "What trends do you see?"
```

PDF support requires `pdfplumber`:

```bash
pip install pdfplumber
```

---

## API Reference

Full Swagger UI: **[lordmacu.github.io/chatgpt-proxy/swagger/](https://lordmacu.github.io/chatgpt-proxy/swagger/)**

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request body:**

```json
{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "response_format": {"type": "json_object"},
  "chatgpt_web_search": true,
  "chatgpt_advanced_tools": false,
  "chatgpt_canvas": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `auto` | Model ID (see `/v1/models`) |
| `messages` | array | — | Conversation messages |
| `stream` | boolean | `false` | Stream via SSE |
| `response_format` | object | — | `{"type": "json_object"}` for JSON mode |
| `chatgpt_web_search` | boolean | `false` | Enable web search |
| `chatgpt_advanced_tools` | boolean | `false` | Enable code interpreter and tools |
| `chatgpt_canvas` | boolean | `false` | Enable canvas |

**Streaming response:** standard OpenAI SSE format (`data: {...}\n\n`).

**Non-streaming response:** standard OpenAI `ChatCompletion` object.

The response also includes a `chatgpt_metadata` field with:

```json
{
  "chatgpt_metadata": {
    "conversation_id": "uuid",
    "message_id": "uuid",
    "model": "gpt-4o",
    "sources": []
  }
}
```

### `GET /v1/models`

Returns available models fetched from the anonymous ChatGPT endpoint.

### `GET /health`

Returns server version and status.

---

## JSON mode

Force the model to return valid JSON:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "List 3 fruits"}],
    "response_format": {"type": "json_object"}
  }'
```

The proxy injects an instruction into the model, passes `response_format` to the backend, and strips any markdown fences from the response.

---

## Quota handling

Each anonymous `oai-device-id` has a message limit. When exhausted:

1. The current session is marked `quota_exhausted = True`
2. The next request automatically gets a fresh device ID
3. The current in-flight request is retried once transparently
4. On double failure, a standard OpenAI `rate_limit_exceeded` error is returned

This is fully automatic — the caller sees no interruption in most cases.

---

## Session pool

The proxy keeps a pool of `ChatGPTSession` objects keyed by conversation history hash. Requests with the same prior messages reuse the existing session (and thus the same ChatGPT conversation thread). Different conversation histories get independent sessions.

---

## Available models

Models are fetched live from the backend. Typical values:

| Model ID | Description |
|----------|-------------|
| `auto` | ChatGPT auto-selects the best model |
| `gpt-4o` | GPT-4o |
| `gpt-4o-mini` | GPT-4o Mini |
| `o1` | o1 reasoning model |
| `o3-mini` | o3 mini reasoning model |

---

## Use with OpenAI-compatible tools

Since the proxy speaks the OpenAI API format, you can point any OpenAI SDK at it:

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

Works with LangChain, LlamaIndex, and any other OpenAI-compatible library.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Uvicorn log level |

No API key or credentials are required or stored.

---

## Project structure

```
chatgpt-proxy/
├── main.py                        # FastAPI app — OpenAI-compatible API
├── chatgpt_client.py              # Core: session management, streaming, quota handling
├── cli.py                         # Terminal client (no server needed)
├── openapi.json                   # OpenAPI 3.0 spec
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── chatgpt-proxy.postman_collection.json
├── SECURITY_RECOMMENDATIONS.md   # Responsible disclosure / hardening guide for OpenAI
└── docs/                          # GitHub Pages site
    ├── index.html                 # Documentation site
    └── swagger/
        ├── index.html             # Swagger UI
        └── openapi.json
```

---

## Responsible use

This project is published for **educational and research purposes** — to document how the anonymous API flow works and to help OpenAI improve its security posture. See [`SECURITY_RECOMMENDATIONS.md`](SECURITY_RECOMMENDATIONS.md) for the full disclosure.

Do not use this proxy to circumvent rate limits at scale, to resell ChatGPT access, or for any purpose that violates OpenAI's [Terms of Service](https://openai.com/policies/terms-of-use/).

---

## Links

- **Docs:** [lordmacu.github.io/chatgpt-proxy/](https://lordmacu.github.io/chatgpt-proxy/)
- **Swagger UI:** [lordmacu.github.io/chatgpt-proxy/swagger/](https://lordmacu.github.io/chatgpt-proxy/swagger/)
- **GitHub:** [github.com/lordmacu/chatgpt-proxy](https://github.com/lordmacu/chatgpt-proxy)
- **Security recommendations:** [SECURITY_RECOMMENDATIONS.md](SECURITY_RECOMMENDATIONS.md)
