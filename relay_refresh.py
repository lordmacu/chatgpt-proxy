#!/usr/bin/env python3
"""Relay: renew the proxy's access token from home, push it to Coolify.

Why this exists. The proxy runs on a datacenter host (Coolify on `blog`) where
BOTH automatic renewal paths fail:
  * the web session endpoint chatgpt.com/api/auth/session is Cloudflare-IP-bound
    and 403s from the datacenter;
  * the mobile OAuth login needs Google Play Integrity (a real Android device).
But /api/auth/session works fine from a residential IP. So this script runs on
the Mac (home IP), mints a fresh access token from the captured chatgpt.com
session (session_web + session_cookies.json), and when it changes, updates
CHATGPT_ACCESS_TOKEN in Coolify and redeploys the container.

The session cookie lasts ~3 months and the access token ~10 days, so a real
change (and a redeploy) happens roughly every 10 days; every other run is a
no-op. Schedule via launchd every couple of hours (see the .plist alongside).

The Coolify API is localhost-only on `blog`, so every call is tunnelled over
SSH, reading the API token from ~/.coolify-api-token on `blog`. Nothing secret
lives in this file.
"""
from __future__ import annotations  # macOS system python is 3.9: defer `x | None`

import json
import subprocess
import sys
import time

import session_web

APP_UUID = "rs3okqn9jehjs7k6mj43haxm"
SSH_HOST = "blog"


def _log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def _coolify(method: str, path: str, body: dict | None = None) -> str:
    """One Coolify API call, run on `blog` over SSH (the API is localhost-only)."""
    remote = (
        'T=$(cat ~/.coolify-api-token); '
        f'curl -s -m 25 -X {method} '
        '-H "Authorization: Bearer $T" -H "Content-Type: application/json" '
        + ('--data-binary @- ' if body is not None else '')
        + f'"http://localhost:8000/api/v1{path}"'
    )
    r = subprocess.run(
        ["ssh", SSH_HOST, remote],
        input=(json.dumps(body) if body is not None else None),
        capture_output=True, text=True, timeout=90,
    )
    if r.returncode != 0:
        _log("ssh/coolify error:", r.stderr.strip()[:200])
    return r.stdout


def _current_token() -> str | None:
    out = _coolify("GET", f"/applications/{APP_UUID}/envs")
    try:
        envs = json.loads(out)
    except Exception:
        _log("no pude leer envs de Coolify:", out[:150])
        return None
    for e in envs:
        if e.get("key") == "CHATGPT_ACCESS_TOKEN":
            return e.get("value")
    return None


def main() -> int:
    if not session_web.enabled():
        _log("session_web no está configurado (falta session_cookies.json) -- salgo")
        return 1

    fresh = session_web.fetch_access_token()
    if not fresh:
        _log("no obtuve token de /api/auth/session (cookies o Cloudflare vencidos) -- salgo")
        return 1

    current = _current_token()
    if current is None:
        _log("no pude leer el token actual de Coolify -- salgo sin tocar nada")
        return 1
    if current == fresh:
        _log(f"sin cambios (token vigente {fresh[:12]}..) -- nada que hacer")
        return 0

    _log(f"token CAMBIÓ (nuevo {fresh[:12]}..) -- actualizando Coolify + redeploy")
    _coolify("PATCH", f"/applications/{APP_UUID}/envs",
             {"key": "CHATGPT_ACCESS_TOKEN", "value": fresh, "is_preview": False})
    dep = _coolify("POST", f"/deploy?uuid={APP_UUID}")
    _log("deploy encolado:", dep[:160])
    return 0


if __name__ == "__main__":
    sys.exit(main())
