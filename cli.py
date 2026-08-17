#!/usr/bin/env python3
"""
chatgpt-proxy CLI — direct access to the ChatGPT anonymous API from the terminal.
No API server required. Uses chatgpt_client.py directly.

Usage:
  python cli.py "Your question"              # stream response (default)
  python cli.py --no-stream "question"       # wait for full response
  python cli.py --chat                       # interactive multi-turn REPL
  python cli.py -m gpt-5-5 "question"       # choose model
  python cli.py --search "news today"        # force web search
  python cli.py --tools "find a hotel"       # advanced tools (shopping, etc.)
  python cli.py --json "return JSON data"    # JSON output mode
  python cli.py --canvas "write a doc"       # Canvas / document mode
  python cli.py -f report.pdf "summarize"    # attach a file
  python cli.py -s "You are a pirate" "hi"  # system prompt
  python cli.py -q "question"                # quiet mode (pipe-friendly)
"""

import argparse
import asyncio
import io
import json as _json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from chatgpt_client import ChatGPTSession, QuotaExceededError, _extract_json

# ── Color support ─────────────────────────────────────────────────────────────

_NO_COLOR = (
    os.environ.get("NO_COLOR") is not None or
    os.environ.get("TERM") == "dumb" or
    not sys.stdout.isatty()
)


class C:
    """ANSI color codes — all empty strings when color is disabled."""
    def __init__(self, enabled: bool):
        if enabled:
            self.RESET  = "\033[0m"
            self.BOLD   = "\033[1m"
            self.DIM    = "\033[2m"
            self.TEAL   = "\033[38;2;0;200;160m"
            self.YELLOW = "\033[33m"
            self.RED    = "\033[31m"
            self.CYAN   = "\033[36m"
            self.GREY   = "\033[90m"
            self.GREEN  = "\033[32m"
            self.WHITE  = "\033[97m"
            self.BLUE   = "\033[34m"
        else:
            self.RESET = self.BOLD = self.DIM = self.TEAL = self.YELLOW = ""
            self.RED = self.CYAN = self.GREY = self.GREEN = self.WHITE = self.BLUE = ""


c = C(enabled=not _NO_COLOR)

# ── Spinner ───────────────────────────────────────────────────────────────────

class Spinner:
    """Thread-based spinner on stdout — same stream as the response, no race condition."""
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "Thinking"):
        self._msg     = message
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._width   = len(message) + 6
        self._cleared = False

    def start(self) -> "Spinner":
        if not _NO_COLOR:
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if not _NO_COLOR and not self._cleared:
            self._cleared = True
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()

    def _spin(self):
        frames = self._FRAMES
        i = 0
        while not self._stop.is_set():
            frame = frames[i % len(frames)]
            sys.stdout.write(f"\r{c.DIM}{frame} {self._msg}...{c.RESET}")
            sys.stdout.flush()
            self._stop.wait(0.08)
            i += 1

# ── File reading ──────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Read a file and return its text content. Handles PDF and plain text."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    raw    = p.read_bytes()

    if suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            print(
                f"{c.YELLOW}Warning: pdfplumber not installed — reading PDF as raw text. "
                f"Install with: pip install pdfplumber{c.RESET}",
                file=sys.stderr,
            )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")

# ── Output helpers ────────────────────────────────────────────────────────────

def print_sources(session: ChatGPTSession, quiet: bool):
    """Print web search sources after the response."""
    if quiet:
        return
    sources = session.last_citations
    queries = session.last_search_queries
    if not sources and not queries:
        return
    print(f"\n{c.DIM}{'─' * 48}{c.RESET}", file=sys.stderr)
    if queries:
        qs = ", ".join(f'"{q}"' for q in queries)
        print(f"{c.GREY}🔍 Searched: {qs}{c.RESET}", file=sys.stderr)
    for src in sources[:5]:
        title = src.get("title") or src.get("attribution") or src["url"]
        print(f"{c.GREY}   · {title}  {c.DIM}{src['url'][:72]}{c.RESET}", file=sys.stderr)


def print_json(text: str):
    """Pretty-print JSON, falling back to raw text if invalid."""
    extracted = _extract_json(text)
    try:
        parsed = _json.loads(extracted)
        print(_json.dumps(parsed, indent=2, ensure_ascii=False))
    except _json.JSONDecodeError:
        print(extracted)


def err(msg: str):
    print(f"{c.RED}Error:{c.RESET} {msg}", file=sys.stderr)

# ── Core: run a single query ──────────────────────────────────────────────────

async def ask(
    session: ChatGPTSession,
    message: str,
    *,
    model: str        = "auto",
    file_texts:  list = (),
    stream:      bool = True,
    json_mode:   bool = False,
    search:      Optional[bool] = None,
    tools:       Optional[bool] = None,
    canvas:      Optional[bool] = None,
    quiet:       bool = False,
) -> str:
    """
    Send a message and handle output.
    Returns the full response text.
    Raises QuotaExceededError on hard quota failure.
    """
    spinner = Spinner().start()
    first   = True
    full    = ""

    try:
        async for chunk in session.stream_message(
            message,
            model             = model,
            file_texts        = list(file_texts) or None,
            force_use_search  = search,
            force_use_tools   = tools,
            force_use_canvas  = canvas,
            json_mode         = json_mode,
        ):
            if first:
                spinner.stop()
                if not quiet and not json_mode:
                    sys.stdout.write(c.TEAL)
                first = False

            if stream and not json_mode:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            full += chunk

    except QuotaExceededError as e:
        spinner.stop()
        raise
    except KeyboardInterrupt:
        spinner.stop()
        if not quiet:
            sys.stdout.write(f"\n{c.YELLOW}[interrupted]{c.RESET}\n")
            sys.stdout.flush()
        return full
    finally:
        spinner.stop()

    if not quiet and not json_mode:
        sys.stdout.write(c.RESET)

    if json_mode:
        clean = session.last_clean_text or full
        print_json(clean)
    elif not stream:
        clean = session.last_clean_text or full
        if not quiet:
            sys.stdout.write(clean)
        else:
            print(clean)

    if stream or not json_mode:
        if not quiet:
            print()  # trailing newline after streamed content

    print_sources(session, quiet)
    return full

# ── Interactive REPL ──────────────────────────────────────────────────────────

CHAT_HELP = """\
{bold}Chat commands{reset}
  /help              Show this message
  /clear             Start a new conversation (new session)
  /model <name>      Switch model  (e.g. /model gpt-5-5)
  /search on|off     Force web search on or off
  /tools on|off      Enable/disable advanced tools
  /json on|off       Toggle JSON output mode
  /system <text>     Set a new system prompt
  /info              Show current settings
  /quit  or  Ctrl-D  Exit

{bold}Available models{reset}
  auto (default)  gpt-5-6  gpt-5-5  gpt-5-6-mini  gpt-5-5-mini  gpt-5-3-mini
"""


async def chat(
    *,
    model:      str           = "auto",
    system:     str           = "",
    search:     Optional[bool] = None,
    tools:      Optional[bool] = None,
    canvas:     Optional[bool] = None,
    json_mode:  bool           = False,
    file_paths: list           = (),
    stream:     bool           = True,
    quiet:      bool           = False,
):
    """Interactive multi-turn chat REPL."""
    # Readline history if available
    try:
        import readline
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass

    session    = ChatGPTSession(system_prompt=system)
    cur_model  = model
    cur_search = search
    cur_tools  = tools
    cur_json   = json_mode

    # Load files specified at startup
    file_texts = []
    for path in file_paths:
        try:
            file_texts.append(read_file(path))
            if not quiet:
                print(f"{c.GREY}Attached: {path}{c.RESET}", file=sys.stderr)
        except Exception as e:
            err(str(e))

    if not quiet:
        print(
            f"\n{c.TEAL}{c.BOLD}chatgpt-proxy{c.RESET}  "
            f"{c.DIM}model={cur_model}  type /help for commands{c.RESET}\n",
            file=sys.stderr,
        )

    loop = asyncio.get_event_loop()

    while True:
        # Prompt
        try:
            if _NO_COLOR or quiet:
                prompt_str = "you> "
            else:
                prompt_str = f"{c.CYAN}you{c.RESET}{c.DIM}>{c.RESET} "

            raw = await loop.run_in_executor(None, lambda: input(prompt_str))
        except (EOFError, KeyboardInterrupt):
            if not quiet:
                print(f"\n{c.DIM}bye{c.RESET}", file=sys.stderr)
            break

        text = raw.strip()
        if not text:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd   = parts[0].lower()
            arg   = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                if not quiet:
                    print(f"{c.DIM}bye{c.RESET}", file=sys.stderr)
                break

            elif cmd == "/help":
                print(
                    CHAT_HELP.format(bold=c.BOLD, reset=c.RESET),
                    file=sys.stderr,
                )

            elif cmd == "/clear":
                await session.close()
                session = ChatGPTSession(system_prompt=system)
                file_texts = []
                if not quiet:
                    print(f"{c.GREY}Conversation cleared.{c.RESET}\n", file=sys.stderr)

            elif cmd == "/model":
                if arg:
                    cur_model = arg
                    if not quiet:
                        print(f"{c.GREY}Model → {cur_model}{c.RESET}\n", file=sys.stderr)
                else:
                    print(f"{c.GREY}Current model: {cur_model}{c.RESET}", file=sys.stderr)

            elif cmd == "/search":
                if arg.lower() in ("on", "true", "1"):
                    cur_search = True
                elif arg.lower() in ("off", "false", "0"):
                    cur_search = False
                else:
                    cur_search = None
                label = {True: "on", False: "off", None: "auto"}[cur_search]
                if not quiet:
                    print(f"{c.GREY}Web search → {label}{c.RESET}\n", file=sys.stderr)

            elif cmd == "/tools":
                if arg.lower() in ("on", "true", "1"):
                    cur_tools = True
                elif arg.lower() in ("off", "false", "0"):
                    cur_tools = False
                else:
                    cur_tools = None
                label = {True: "on", False: "off", None: "auto"}[cur_tools]
                if not quiet:
                    print(f"{c.GREY}Advanced tools → {label}{c.RESET}\n", file=sys.stderr)

            elif cmd == "/json":
                if arg.lower() in ("on", "true", "1"):
                    cur_json = True
                elif arg.lower() in ("off", "false", "0"):
                    cur_json = False
                else:
                    cur_json = not cur_json
                if not quiet:
                    print(f"{c.GREY}JSON mode → {'on' if cur_json else 'off'}{c.RESET}\n", file=sys.stderr)

            elif cmd == "/system":
                if arg:
                    system = arg
                    await session.close()
                    session = ChatGPTSession(system_prompt=system)
                    if not quiet:
                        print(f"{c.GREY}System prompt updated. Conversation reset.{c.RESET}\n", file=sys.stderr)
                else:
                    print(f"{c.GREY}System: {system or '(none)'}{c.RESET}", file=sys.stderr)

            elif cmd == "/info":
                search_label = {True: "on", False: "off", None: "auto"}[cur_search]
                tools_label  = {True: "on", False: "off", None: "auto"}[cur_tools]
                sys_str      = repr(system[:40]) if system else "(none)"
                print(
                    f"{c.GREY}"
                    f"model={cur_model}  search={search_label}  "
                    f"tools={tools_label}  json={cur_json}  "
                    f"system={sys_str}"
                    f"{c.RESET}\n",
                    file=sys.stderr,
                )

            else:
                print(f"{c.YELLOW}Unknown command: {cmd}  (type /help){c.RESET}", file=sys.stderr)

            continue

        # ── Send message ──────────────────────────────────────────────────────
        if not quiet:
            print(f"{c.DIM}──{c.RESET}", file=sys.stderr)

        try:
            await ask(
                session,
                text,
                model      = cur_model,
                file_texts = file_texts,
                stream     = stream,
                json_mode  = cur_json,
                search     = cur_search,
                tools      = cur_tools,
                canvas     = canvas,
                quiet      = quiet,
            )
            file_texts = []  # attachments are one-shot
        except QuotaExceededError as e:
            err(f"Quota exhausted — new device_id will be used next turn. ({e})")
            await session.close()
            session = ChatGPTSession(system_prompt=system)
        except Exception as e:
            err(str(e))

        if not quiet:
            print()

    await session.close()

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="ChatGPT anonymous API — CLI client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python cli.py "What is the capital of France?"
  python cli.py --search "Tech news today"
  python cli.py --json "Return a person object with name and age"
  python cli.py -f report.pdf "Summarize this document"
  python cli.py -s "You are a pirate" "Tell me about the sea"
  python cli.py --chat
  python cli.py -q "Hello" | wc -c
        """,
    )

    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Message to send. Omit to start the interactive REPL.",
    )

    # Mode flags
    mode_group = parser.add_argument_group("modes")
    mode_group.add_argument(
        "--chat", "-c",
        action="store_true",
        help="Start the interactive multi-turn REPL (implied when no message is given).",
    )
    mode_group.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full response before printing (default is streaming).",
    )

    # Model and prompt
    parser.add_argument(
        "--model", "-m",
        default="auto",
        metavar="MODEL",
        help="Model slug. Default: auto. Options: gpt-5-6, gpt-5-5, gpt-5-3-mini, …",
    )
    parser.add_argument(
        "--system", "-s",
        default="",
        metavar="PROMPT",
        help="System prompt injected at the start of the conversation.",
    )

    # Feature flags
    feat_group = parser.add_argument_group("features")
    feat_group.add_argument(
        "--search",
        action="store_true",
        default=None,
        help="Force web search on.",
    )
    feat_group.add_argument(
        "--no-search",
        action="store_true",
        help="Disable web search entirely.",
    )
    feat_group.add_argument(
        "--tools",
        action="store_true",
        help="Enable advanced backend tools (reservations, shopping, widgets).",
    )
    feat_group.add_argument(
        "--canvas",
        action="store_true",
        help="Enable Canvas / collaborative document mode.",
    )
    feat_group.add_argument(
        "--json", "-j",
        action="store_true",
        help="Force JSON output mode (response_format: json_object).",
    )

    # Files
    parser.add_argument(
        "--file", "-f",
        action="append",
        metavar="PATH",
        default=[],
        dest="files",
        help="Attach a file (PDF or text). Can be repeated for multiple files.",
    )

    # Output
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress decorations and spinner. Prints only the response text. Good for piping.",
    )

    args = parser.parse_args()

    # Resolve search flag
    search: Optional[bool] = None
    if getattr(args, "search", False):
        search = True
    if args.no_search:
        search = False

    # Read attached files
    file_texts = []
    for path in args.files:
        try:
            file_texts.append(read_file(path))
            if not args.quiet:
                print(f"{c.GREY}Attached: {path}{c.RESET}", file=sys.stderr)
        except Exception as e:
            err(str(e))
            sys.exit(1)

    stream = not args.no_stream

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.message is None or args.chat:
        # Interactive REPL
        await chat(
            model      = args.model,
            system     = args.system,
            search     = search,
            tools      = True if args.tools else None,
            canvas     = True if args.canvas else None,
            json_mode  = args.json,
            file_paths = args.files,
            stream     = stream,
            quiet      = args.quiet,
        )
    else:
        # Single question
        if not args.quiet:
            pass  # spinner handles the "waiting" state

        ask_kwargs = dict(
            model      = args.model,
            file_texts = file_texts,
            stream     = stream,
            json_mode  = args.json,
            search     = search,
            tools      = True if args.tools else None,
            canvas     = True if args.canvas else None,
            quiet      = args.quiet,
        )

        session = ChatGPTSession(system_prompt=args.system)
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                await ask(session, args.message, **ask_kwargs)
                last_exc = None
                break
            except QuotaExceededError as e:
                err(f"Quota exhausted: {e}")
                sys.exit(1)
            except KeyboardInterrupt:
                print()
                sys.exit(0)
            except Exception as e:
                last_exc = e
                if attempt == 0:
                    if not args.quiet:
                        print(
                            f"{c.YELLOW}Retrying with a fresh session…{c.RESET}",
                            file=sys.stderr,
                        )
                    await session.close()
                    session = ChatGPTSession(system_prompt=args.system)
                    await asyncio.sleep(1)

        if last_exc is not None:
            err(f"{type(last_exc).__name__}: {last_exc}" if str(last_exc)
                else type(last_exc).__name__)
            sys.exit(1)

        await session.close()


def _run_web(port: int = 7842, no_browser: bool = False) -> None:
    """Start the FastAPI server and open the web UI in the browser."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed.\n"
            "Run: pip3 install 'uvicorn[standard]'",
            file=sys.stderr,
        )
        sys.exit(1)

    url = f"http://localhost:{port}"
    print(f"  chatgpt-proxy web UI → {url}", file=sys.stderr)
    print("  Press Ctrl+C to stop.\n", file=sys.stderr)

    if not no_browser:
        import webbrowser, threading
        def _open():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    # Intercept `web` subcommand before argparse so it works synchronously
    if len(sys.argv) >= 2 and sys.argv[1] == "web":
        _port = 7842
        _no_browser = False
        for _arg in sys.argv[2:]:
            if _arg.lstrip("-").isdigit():
                _port = int(_arg.lstrip("-"))
            elif _arg in ("--no-browser", "-n"):
                _no_browser = True
        _run_web(_port, _no_browser)
        sys.exit(0)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
