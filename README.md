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
| `response_format` | object | — | `{"type":"json_object"}` for JSON mode |
| `web_search` | boolean | `null` | Force web search on/off |
| `force_use_tools` | boolean | `null` | Enable advanced tools |
| `force_use_canvas` | boolean | `null` | Enable Canvas mode |
| `tools` | array | — | OpenAI tools format (see note) |
| `tool_choice` | string | — | `"none"` disables all modes |
| `temperature`, `max_tokens`, `top_p`, etc. | — | — | Accepted, ignored |

> **Note on `tools`:** The anonymous backend doesn't execute functions, but tool names control modes: `"web_search"` → enables search, `"chatgpt_tools"` → enables advanced tools, any other name → enables advanced tools.

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

Returns available models.

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "auto",        "object": "model"},
    {"id": "gpt-4o",      "object": "model"},
    {"id": "gpt-4o-mini", "object": "model"},
    {"id": "o1",          "object": "model"},
    {"id": "o3-mini",     "object": "model"}
  ]
}
```

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

Each anonymous device ID has a message limit. When exhausted:

1. The session is marked as exhausted
2. The next request automatically gets a fresh device ID
3. The current request is retried transparently
4. On double failure, a standard `rate_limit_exceeded` error is returned

The CLI also retries once automatically with a visible message:
```
Retrying with a fresh session…
```

---

## Available models

| Model ID | Description |
|----------|-------------|
| `auto` | ChatGPT picks the best model automatically |
| `gpt-4o` | GPT-4o |
| `gpt-4o-mini` | GPT-4o Mini |
| `o1` | o1 reasoning model |
| `o3-mini` | o3 Mini reasoning model |

---

## What works / doesn't work

| Feature | Status |
|---------|--------|
| Text chat | ✅ |
| Streaming SSE | ✅ |
| Multi-turn conversations | ✅ |
| System prompt | ✅ |
| Web search | ✅ |
| Advanced tools (code interpreter, etc.) | ✅ |
| Canvas (document mode) | ✅ |
| JSON output mode | ✅ |
| File attachments (PDF, text, code) | ✅ |
| Quota auto-rotation | ✅ |
| Image input (vision) | ❌ Anonymous mode only |
| Image generation | ❌ Anonymous mode only |
| Function calling (tool_calls in response) | ❌ |
| Voice / TTS | ❌ Requires account |

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
