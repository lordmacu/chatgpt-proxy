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
import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx

import dpop
import session_web

TOKEN_FILE = Path(__file__).parent / "tokens.json"

# Renew this many seconds BEFORE the access token's `exp`, so a request never
# rides an about-to-die token. And don't retry a proactive renew more often than
# the cooldown, so a dead credential can't turn every request into a round-trip.
REFRESH_SKEW = int(os.environ.get("CHATGPT_REFRESH_SKEW", "600"))
REFRESH_COOLDOWN = int(os.environ.get("CHATGPT_REFRESH_COOLDOWN", "120"))

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

# Reentrant: access_token() may call refresh_access_token() while holding it
# (the proactive-renew path), and both take this lock.
_lock = threading.RLock()
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


def _token_exp(token: str):
    """The `exp` (unix seconds) of a JWT access token, or None if undecodable."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get("exp")
    except Exception:
        return None


def _expiring_soon(token: str) -> bool:
    exp = _token_exp(token)
    return exp is not None and (exp - time.time()) < REFRESH_SKEW


def access_token() -> str:
    """The current access token, or "" when the proxy is running anonymously.

    An empty string is a normal, supported state -- NOT an error. Callers treat
    it as "stay on the anonymous backend", which is what every deployment did
    before this module existed.

    If a renewal method is configured and the token is within REFRESH_SKEW of
    expiry, it is renewed here first -- so a request never leaves on a token
    that is about to die. The cooldown keeps a broken credential from turning
    every call into a failed refresh.
    """
    with _lock:
        if not _cache:
            _cache.update(_load())
        tok = (_cache.get("access_token") or "").strip()
        needs = (not tok) or _expiring_soon(tok)
        if needs and _can_refresh():
            last = _cache.get("last_refresh_attempt", 0)
            if time.time() - last >= REFRESH_COOLDOWN:
                _log("renewing access token (empty or near expiry)")
                refresh_access_token()
                tok = (_cache.get("access_token") or "").strip()
        return tok


def _can_refresh() -> bool:
    """True when some renewal method is configured (web session or refresh token)."""
    if session_web.enabled():
        return True
    return bool((_cache.get("refresh_token") or "").strip())


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


def _post_oauth_token(payload: dict):
    """POST /oauth/token, adding a DPoP proof when enabled.

    Honors one DPoP-Nonce challenge (`hxo.a0` in the app: the server rejects the
    first proof and returns a `DPoP-Nonce` to echo, so we retry once with it).
    When DPoP is off `dpop.proof()` returns None and this is the plain request
    the proxy always sent. Returns the final response, or None on a transport
    error.
    """
    url = f"{AUTH_BASE}/oauth/token"
    r = None
    for attempt in (1, 2):
        headers = dict(_HEADERS)
        proof = dpop.proof(url, "POST")  # no access token: refresh has no `ath`
        if proof:
            headers["DPoP"] = proof
        try:
            r = httpx.post(url, headers=headers, timeout=30.0, json=payload)
        except httpx.HTTPError as e:
            _log("refresh request failed:", type(e).__name__, e)
            return None
        dpop.remember_nonce(r.headers)
        if (attempt == 1 and proof and r.status_code in (400, 401)
                and r.headers.get("DPoP-Nonce")):
            _log("server issued a DPoP nonce; retrying with it")
            continue
        break
    return r


def _save(new_access: str) -> None:
    """Store a freshly obtained access token and persist the pair."""
    _cache["access_token"] = new_access
    _cache["refreshed_at"] = time.time()
    # runtime-only bookkeeping never goes to disk
    _persist({k: v for k, v in _cache.items()
              if k not in ("refreshed_at", "last_refresh_attempt")})


def refresh_access_token() -> bool:
    """Renew the access token. True if it now differs from the previous one.

    Two methods, tried in order:
      1. Web session -- when a chatgpt.com session cookie is configured, GET
         /api/auth/session for a fresh accessToken. Needs no refresh_token; this
         is what a captured browser session gives us, and its cookie lives ~3
         months, renewing the short-lived access token on demand.
      2. OAuth refresh_token -- POST /oauth/token (grant_type=refresh_token),
         the Android app's flow, with a DPoP proof when enabled.

    Returns False when nothing is configured or the renewal produced no new
    token, so the caller reports the original failure instead of pretending it
    recovered.

    Under the lock, and it re-reads first: two concurrent requests can both get
    a 401, and without this the second would spend a second refresh round-trip.
    """
    with _lock:
        if not _cache:
            _cache.update(_load())
        before = (_cache.get("access_token") or "").strip()
        _cache["last_refresh_attempt"] = time.time()

        # 1) Web session -- preferred when cookies are present.
        if session_web.enabled():
            new_access = session_web.fetch_access_token()
            if new_access and new_access != before:
                _save(new_access)
                _log("access token refreshed via web session:", new_access[:12] + "...")
                return True
            if new_access == before and before:
                _log("web session returned the same token (not yet rotated)")
            elif new_access is None:
                _log("web-session refresh failed")
            # Fall through to the OAuth flow only if a refresh_token also exists.
            if not (_cache.get("refresh_token") or "").strip():
                return False

        # 2) OAuth refresh_token flow.
        refresh = (_cache.get("refresh_token") or "").strip()
        if not refresh:
            _log("cannot recover: no session cookie or refresh token configured")
            return False
        r = _post_oauth_token({"grant_type": "refresh_token",
                               "refresh_token": refresh,
                               "client_id": CLIENT_ID})
        if r is None:
            return False
        if r.status_code != 200:
            _log(f"refresh rejected: HTTP {r.status_code} {r.text[:200]}")
            return False
        data = r.json()
        new_access = (data.get("access_token") or "").strip()
        if not new_access or new_access == before:
            _log("refresh returned no new access token")
            return False
        # OpenAI may rotate the refresh token too; keep the old one if it did not.
        _cache["refresh_token"] = data.get("refresh_token") or refresh
        _cache["id_token"] = data.get("id_token")
        _cache["token_type"] = data.get("token_type", "Bearer")
        _cache["expires_in"] = data.get("expires_in")
        _save(new_access)
        _log("access token refreshed:", new_access[:12] + "...")
        return True


def reset() -> None:
    """Drop the cache so the next call re-reads env and file. For tests."""
    with _lock:
        _cache.clear()
