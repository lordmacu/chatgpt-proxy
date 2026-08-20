"""App-faithful ChatGPT login (email/password), reverse-engineered from the APK.

The decompiled Android app does NOT have a native email/password API. Its only
auth endpoints are `api/accounts/authorize` (GET, to build the URL) and
`api/accounts/authorize/native` (POST, the SOCIAL token-exchange with a
`subject_token` -- Google/Apple/Microsoft). For email/password the app opens the
system browser in an **Auth Tab** (`androidx.browser.auth.AuthTabIntent`,
class `x1w` launching the `ho4` authorize URL), lets you authenticate on the web
pages at auth.openai.com, and catches the `com.openai.chatgpt://...callback?code`
redirect -- then exchanges the code at `/oauth/token`.

So the faithful way to reproduce it in Python is a real browser, exactly like the
app's Auth Tab. This does that, but presents as the app rather than a desktop:

  * the authorize URL carries the app's own params (`android_device_id`,
    `requester_metadata_app_version`, `ccaps`, `hydra_flow=condense`), matching a
    live capture of the app's GET /api/accounts/authorize;
  * the browser context emulates Android Chrome (what the Auth Tab runs);
  * the token exchange goes out with the app's exact headers (User-Agent
    ChatGPT/..., OAI-Package-Name, OAI-Client-Type, OAI-Device-Id) and an
    optional DPoP proof -- reused from login.py so the two never drift.

It leaves login.py and login_playwright.py untouched, and writes the same
tokens.json (access + refresh + id) that auth.py reads.

Run:  python login_app.py          # opens a visible browser; you type creds/code
The account may be passwordless (emailed code): the page shows whatever step is
needed and you complete it in the window; nothing is typed for you.
"""
import asyncio
import base64
import hashlib
import json
import secrets
from urllib.parse import urlencode, urlparse, parse_qs, quote

import httpx

# Reuse the app constants, headers, DPoP-aware exchange and PKCE from login.py so
# this script and that one stay byte-identical where it matters. Importing login
# does not prompt or perform any network call at import time.
import login
from login import (AUTH_BASE, CLIENT_ID, REDIRECT_URI, SCOPE, AUDIENCE,
                   APP_VERSION, DEVICE_ID, HEADERS_BASE, TOKEN_FILE)

# The Auth Tab runs the phone's Chrome. Emulate a current Android Chrome so the
# pages render and behave as they do inside the app.
ANDROID_UA = ("Mozilla/5.0 (Linux; Android 14; Pixel 7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/140.0.0.0 Mobile Safari/537.36")


def _authorize_url() -> tuple[str, str]:
    """The exact URL the app's Auth Tab opens, plus the PKCE verifier to keep."""
    verifier, challenge = login.pkce_pair()
    params = {
        "scope":                          SCOPE,
        "code_challenge":                 challenge,
        "code_challenge_method":          "S256",
        "client_id":                      CLIENT_ID,
        "redirect_uri":                   REDIRECT_URI,
        "state":                          base64.urlsafe_b64encode(secrets.token_bytes(24)).rstrip(b"=").decode(),
        "nonce":                          base64.urlsafe_b64encode(secrets.token_bytes(24)).rstrip(b"=").decode(),
        "audience":                       AUDIENCE,
        "issuer":                         AUTH_BASE,
        "ccaps":                          "default_otp_v2 login_methods",
        "android_device_id":              DEVICE_ID,
        "requester_metadata_app_version": APP_VERSION,
        "screen_hint":                    "login_or_signup",
        "response_type":                  "code",
        "hydra_flow":                     "condense",
    }
    # quote (not quote_plus) so spaces become %20, as the app sends them.
    return f"{AUTH_BASE}/api/accounts/authorize?{urlencode(params, quote_via=quote)}", verifier


async def _capture_code() -> tuple[str, str]:
    """Open the Auth Tab equivalent and return (authorization_code, verifier)."""
    from playwright.async_api import async_playwright

    url, verifier = _authorize_url()
    loop = asyncio.get_event_loop()
    got_code: asyncio.Future = loop.create_future()

    def _code_from(u: str):
        if u.startswith("com.openai.chatgpt://"):
            qs = parse_qs(urlparse(u).query)
            return qs.get("code", [None])[0]
        return None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False, slow_mo=150,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = await browser.new_context(
            user_agent=ANDROID_UA,
            viewport={"width": 412, "height": 915},
            device_scale_factor=3.0, is_mobile=True, has_touch=True,
            locale="en-US",
        )
        page = await ctx.new_page()
        try:  # best-effort anti-bot, harmless if the package is absent
            from playwright_stealth import stealth_async
            await stealth_async(page)
        except Exception:
            pass

        # The redirect to the app's private scheme never loads -- intercept it and
        # lift the code out, exactly as the app catches its callback.
        async def route(r):
            code = _code_from(r.request.url)
            if code and not got_code.done():
                got_code.set_result(code)
            if r.request.url.startswith("com.openai.chatgpt://"):
                await r.abort()
            else:
                await r.continue_()
        await page.route("**/*", route)

        def on_nav(frame):
            code = _code_from(frame.url)
            if code and not got_code.done():
                got_code.set_result(code)
        page.on("framenavigated", on_nav)

        print("[1] Abriendo el login de la app (Auth Tab)…")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            print(f"    (nav: {e})")

        print("\n    >>> Completá el login EN LA VENTANA: correo, contraseña y/o el")
        print("    >>> código que te llegue por correo. Tenés hasta 5 minutos.\n")
        try:
            code = await asyncio.wait_for(got_code, timeout=300.0)
        except asyncio.TimeoutError:
            code = _code_from(page.url)
        await browser.close()

    if not code:
        raise RuntimeError(
            "No se capturó el authorization code. Si la página pidió un paso que no "
            "completaste a tiempo, corré de nuevo y terminá el login en la ventana.")
    return code, verifier


async def login_app() -> dict:
    code, verifier = await _capture_code()
    print(f"[2] Code capturado: {code[:24]}…  intercambiando por tokens (headers del app + DPoP)…")

    client = httpx.AsyncClient(headers=HEADERS_BASE, timeout=30.0)
    r = await login._post_token(client, {
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  REDIRECT_URI,
        "client_id":     CLIENT_ID,
        "code_verifier": verifier,
    })
    await client.aclose()
    if r.status_code != 200:
        raise RuntimeError(f"Token exchange falló: {r.status_code} {r.text[:300]}")

    tokens = r.json()
    result = {
        "access_token":  tokens.get("access_token"),
        "refresh_token": tokens.get("refresh_token"),
        "id_token":      tokens.get("id_token"),
        "token_type":    tokens.get("token_type", "Bearer"),
        "expires_in":    tokens.get("expires_in"),
        "scope":         tokens.get("scope"),
    }
    TOKEN_FILE.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Tokens guardados en {TOKEN_FILE}")
    print(f"  access_token:  {(result['access_token'] or '')[:40]}…")
    rt = result["refresh_token"] or ""
    print(f"  refresh_token: {(rt[:30] + '…') if rt else 'NONE (la cuenta no dio refresh_token)'}")
    return result


if __name__ == "__main__":
    asyncio.run(login_app())
