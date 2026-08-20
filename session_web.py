"""Web-session refresh: mint fresh access tokens from a chatgpt.com session.

The Android app renews via `auth.openai.com/oauth/token` (a `refresh_token`).
The chatgpt.com **web** app renews differently: the browser holds a NextAuth
session cookie (`__Secure-next-auth.session-token`, valid ~3 months) and calls

    GET https://chatgpt.com/api/auth/session

which returns the current `accessToken` -- and mints a fresh one as the old one
nears expiry. That is exactly how the web stays logged in without re-entering a
password. This module reproduces it so a proxy seeded with a captured session
can keep itself renewed with no `refresh_token` and no login.

What you must provide (either one; the file wins because it holds rotations):
  * `CHATGPT_SESSION_COOKIE` -- the raw `Cookie:` header from a logged-in
    chatgpt.com request (must include `__Secure-next-auth.session-token` and,
    because chatgpt.com sits behind Cloudflare, `cf_clearance`).
  * `CHATGPT_COOKIE_FILE` -- a JSON `{name: value}` file this module reads AND
    writes (to persist cookies the server rotates via Set-Cookie).

The User-Agent MUST match the one `cf_clearance` was issued for, or Cloudflare
challenges the request. Set `CHATGPT_UA` to the exact browser UA from the same
capture; a sensible default is baked in.

This mode is OFF unless cookies are configured. With nothing set, callers fall
back to the OAuth `refresh_token` flow in auth.py, unchanged.
"""
import json
import os
import sys
import threading
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Optional

import httpx

SESSION_URL = "https://chatgpt.com/api/auth/session"

COOKIE_FILE = Path(os.environ.get("CHATGPT_COOKIE_FILE",
                                  str(Path(__file__).parent / "session_cookies.json")))

# Must match the UA that cf_clearance was minted under. Override with CHATGPT_UA.
DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/151.0.0.0 Safari/537.36")

_lock = threading.Lock()
_cookies: Optional[dict] = None  # name -> value, lazily loaded


def _log(*args):
    if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
        print("[session]", *args, file=sys.stderr, flush=True)


def _parse_cookie_header(header: str) -> dict:
    out = {}
    for part in header.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            if k:
                out[k] = v.strip()
    return out


def _load_cookies() -> dict:
    """Cookies from the persisted file first (has rotations), else the env seed."""
    global _cookies
    if _cookies is not None:
        return _cookies
    if COOKIE_FILE.exists():
        try:
            _cookies = json.loads(COOKIE_FILE.read_text())
            return _cookies
        except Exception as e:
            _log("could not read cookie file:", type(e).__name__, e)
    env = os.environ.get("CHATGPT_SESSION_COOKIE", "").strip()
    _cookies = _parse_cookie_header(env) if env else {}
    return _cookies


def enabled() -> bool:
    """Active when we hold a NextAuth session cookie to renew with."""
    return "__Secure-next-auth.session-token" in _load_cookies()


def _persist(cookies: dict) -> None:
    try:
        COOKIE_FILE.write_text(json.dumps(cookies, indent=2))
        os.chmod(COOKIE_FILE, 0o600)
    except OSError as e:
        # Read-only FS: keep rotations in memory for this process. The session
        # cookie is long-lived, so this still works for a good while.
        _log("could not persist cookies:", e)


def _apply_set_cookie(cookies: dict, response: httpx.Response) -> bool:
    """Merge the response's Set-Cookie into our jar (`hxo.a0`-style rotation)."""
    changed = False
    for raw in response.headers.get_list("set-cookie"):
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except Exception:
            continue
        for name, morsel in jar.items():
            val = morsel.value
            expired = (morsel["max-age"] == "0") or (val in ("", "deleted"))
            if expired:
                if cookies.pop(name, None) is not None:
                    changed = True
            elif cookies.get(name) != val:
                cookies[name] = val
                changed = True
    return changed


def fetch_access_token() -> Optional[str]:
    """Call /api/auth/session and return a fresh access token, or None.

    Rotated cookies (Set-Cookie) are merged back and persisted, so the session
    keeps working across renewals.
    """
    with _lock:
        cookies = _load_cookies()
        if "__Secure-next-auth.session-token" not in cookies:
            _log("no session cookie configured")
            return None
        headers = {
            "User-Agent": os.environ.get("CHATGPT_UA", DEFAULT_UA),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://chatgpt.com/",
            "Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items()),
        }
        try:
            r = httpx.get(SESSION_URL, headers=headers, timeout=30.0, follow_redirects=True)
        except httpx.HTTPError as e:
            _log("session request failed:", type(e).__name__, e)
            return None

        if _apply_set_cookie(cookies, r):
            _persist(cookies)

        if r.status_code != 200:
            _log(f"session rejected: HTTP {r.status_code} {r.text[:160]}")
            return None
        ctype = r.headers.get("content-type", "")
        if "application/json" not in ctype:
            # A Cloudflare challenge (HTML) rather than JSON: cf_clearance/UA stale.
            _log("session did not return JSON (Cloudflare challenge?):", ctype)
            return None
        try:
            data = r.json()
        except Exception as e:
            _log("session JSON parse failed:", e)
            return None
        token = (data.get("accessToken") or "").strip()
        if not token:
            _log("session had no accessToken (logged out?)")
            return None
        return token


def reset() -> None:
    """Drop the in-memory cookie cache so the next call re-reads. For tests."""
    global _cookies
    with _lock:
        _cookies = None
