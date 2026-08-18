"""Optional account authentication for the ChatGPT Android backend.

The proxy has always run ANONYMOUSLY: `chatgpt_client` mints a random
`OAI-Device-Id` per session and talks to `/backend-anon/...`. That still works
and stays the default -- this module is additive, and everything here degrades to
"no token" rather than raising, so a deployment with nothing configured behaves
exactly as it did before.

Where a token comes from, in order:

  1. `CHATGPT_ACCESS_TOKEN` in the environment (what a container gets).
  2. `tokens.json` next to this file, which `login.py` writes.

`login.py` already performed the PKCE login and wrote both tokens; until now
nothing read that file, so its output went nowhere. `refresh_access_token()`
closes that loop: when the backend answers 401 the access token is renewed from
the refresh token, in place, without a human pasting anything.

Both tokens are secrets. Nothing here logs their value -- only a short prefix,
which is enough to tell two tokens apart in a log without being usable.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx

TOKEN_FILE = Path(__file__).parent / "tokens.json"

# Same constants login.py captured from the APK. They are client identifiers,
# not secrets.
AUTH_BASE = "https://auth.openai.com"
CLIENT_ID = "app_xwBKzt04752TTSfXnki17hmB"
APP_VERSION = "1.2026.223"

_HEADERS = {
    "User-Agent": f"ChatGPT/{APP_VERSION} (Android 16; sdk_gphone64_arm64; build 2622307)",
    "OAI-Package-Name": "com.openai.chatgpt",
    "OAI-Client-Type": "android",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

_lock = threading.Lock()
_cache: dict = {}


def _log(*args):
    if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
        print("[auth]", *args, file=sys.stderr, flush=True)


def _read_file() -> dict:
    try:
        return json.loads(TOKEN_FILE.read_text())
    except Exception:
        return {}


def _load() -> dict:
    """The token pair, from the environment first and the file second.

    The environment wins because that is what a container is configured with;
    the file is how a developer running login.py locally gets the same result
    with no extra step.
    """
    env_access = os.environ.get("CHATGPT_ACCESS_TOKEN", "").strip()
    env_refresh = os.environ.get("CHATGPT_REFRESH_TOKEN", "").strip()
    if env_access or env_refresh:
        return {"access_token": env_access, "refresh_token": env_refresh}
    return _read_file()


def access_token() -> str:
    """The current access token, or "" when the proxy is running anonymously.

    An empty string is a normal, supported state -- NOT an error. Callers treat
    it as "stay on the anonymous backend", which is what every deployment did
    before this module existed.
    """
    with _lock:
        if not _cache:
            _cache.update(_load())
        return (_cache.get("access_token") or "").strip()


def is_authenticated() -> bool:
    return bool(access_token())


def _persist(tokens: dict) -> None:
    """Write the refreshed pair back, best effort.

    A read-only filesystem (a hardened container) must not turn a successful
    refresh into a crash: the new token already lives in `_cache` and works for
    this process either way. The only cost of failing to write is that a restart
    goes back to the older token and refreshes again.
    """
    try:
        TOKEN_FILE.write_text(json.dumps(tokens, indent=2))
    except OSError as e:
        _log("could not persist tokens:", e)


def refresh_access_token() -> bool:
    """Renew the access token from the refresh token. True if it now differs.

    Called after a 401. Returns False when there is no refresh token to use, so
    the caller reports the original failure instead of pretending it recovered
    -- the same rule the mistral proxy follows.

    Under the lock, and it re-reads first: two concurrent requests can both get
    a 401, and without this the second would spend a second refresh round-trip
    (and invalidate the token the first one just obtained).
    """
    with _lock:
        if not _cache:
            _cache.update(_load())
        before = (_cache.get("access_token") or "").strip()
        refresh = (_cache.get("refresh_token") or "").strip()
        if not refresh:
            _log("401 with no refresh token configured -- cannot recover")
            return False
        try:
            r = httpx.post(f"{AUTH_BASE}/oauth/token", headers=_HEADERS, timeout=30.0,
                           json={"grant_type": "refresh_token",
                                 "refresh_token": refresh,
                                 "client_id": CLIENT_ID})
        except httpx.HTTPError as e:
            _log("refresh request failed:", type(e).__name__, e)
            return False
        if r.status_code != 200:
            _log(f"refresh rejected: HTTP {r.status_code} {r.text[:200]}")
            return False
        data = r.json()
        new_access = (data.get("access_token") or "").strip()
        if not new_access or new_access == before:
            _log("refresh returned no new access token")
            return False
        _cache["access_token"] = new_access
        # OpenAI may rotate the refresh token too; keep the old one if it did not.
        _cache["refresh_token"] = data.get("refresh_token") or refresh
        _cache["id_token"] = data.get("id_token")
        _cache["token_type"] = data.get("token_type", "Bearer")
        _cache["expires_in"] = data.get("expires_in")
        _cache["refreshed_at"] = time.time()
        _persist({k: v for k, v in _cache.items() if k != "refreshed_at"})
        _log("access token refreshed:", new_access[:12] + "...")
        return True


def reset() -> None:
    """Drop the cache so the next call re-reads env and file. For tests."""
    with _lock:
        _cache.clear()
