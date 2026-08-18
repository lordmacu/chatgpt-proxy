"""
ChatGPT login con Playwright — maneja CSRF/WASM automáticamente.
Lee EMAIL/PASSWORD de .env, guarda tokens en tokens.json.
"""
import os, json, asyncio, hashlib, base64, secrets
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import getpass

EMAIL    = os.environ.get("EMAIL", "").strip()
PASSWORD = os.environ.get("PASSWORD", "")

_PLACEHOLDERS = {"tu-correo", "tu-clave", "tu-contraseña", "tu-contrasena",
                 "your-email", "your-password", "changeme"}
if EMAIL.lower() in _PLACEHOLDERS:
    EMAIL = ""
if PASSWORD.strip().lower() in _PLACEHOLDERS:
    PASSWORD = ""

if not EMAIL:
    EMAIL = input("Correo de la cuenta ChatGPT: ").strip()
# PASSWORD is intentionally NOT prompted here: this account may be passwordless
# (login by emailed code only). The browser flow asks for whatever the page
# actually shows, and the password is filled only if a password field appears.

AUTH_BASE    = "https://auth.openai.com"
CLIENT_ID    = "app_xwBKzt04752TTSfXnki17hmB"
REDIRECT_URI = "com.openai.chatgpt://auth.openai.com/android/com.openai.chatgpt/callback"
SCOPE        = "openid email profile offline_access model.request model.read organization.read organization.write"
AUDIENCE     = "https://api.openai.com/v1"
TOKEN_FILE   = Path(__file__).parent / "tokens.json"

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def pkce_pair():
    verifier  = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge

async def login():
    from playwright.async_api import async_playwright

    verifier, challenge = pkce_pair()
    state = _b64url(secrets.token_bytes(24))
    nonce = _b64url(secrets.token_bytes(24))

    auth_url = (
        f"{AUTH_BASE}/api/accounts/authorize"
        f"?scope={SCOPE.replace(' ', '%20')}"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=S256"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI.replace(':', '%3A').replace('/', '%2F')}"
        f"&state={state}"
        f"&nonce={nonce}"
        f"&audience={AUDIENCE.replace(':', '%3A').replace('/', '%2F')}"
        f"&issuer={AUTH_BASE.replace(':', '%3A').replace('/', '%2F')}"
        f"&ccaps=default_otp_v2%20login_methods"
        f"&screen_hint=login_or_signup"
        f"&response_type=code"
        f"&hydra_flow=condense"
    )

    # Alternativa: URL más simple de inicio de sesión
    login_url = "https://auth.openai.com/log-in-or-create-account"

    print(f"[*] Iniciando Playwright login...")
    code = None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,
            slow_mo=300,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = await ctx.new_page()
        # Aplicar stealth para evadir detección de Cloudflare
        try:
            from playwright_stealth import stealth_async
            await stealth_async(page)
            print("    stealth aplicado")
        except Exception as e:
            print(f"    stealth no disponible: {e}")

        # Interceptar redirects al custom scheme para capturar el code
        redirect_future = asyncio.get_event_loop().create_future()

        async def handle_route(route):
            url = route.request.url
            if url.startswith("com.openai.chatgpt://"):
                parsed = urlparse(url)
                qs = parse_qs(parsed.query)
                if "code" in qs and not redirect_future.done():
                    redirect_future.set_result(qs["code"][0])
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", handle_route)

        # Navegar directamente a la URL PKCE — el servidor redirige al login
        # y al final hará redirect a com.openai.chatgpt://... que interceptamos
        print(f"[1] Navegando a auth_url PKCE...")
        try:
            await page.goto(auth_url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"    nav warning: {e}")
        await asyncio.sleep(2)
        print(f"    URL post-navigate: {page.url}")

        # Puede aparecer pantalla "sesión terminada" o "use sin cuenta" — buscar botón de login
        text0 = await page.evaluate("document.body.innerText")
        print(f"    Texto inicial: {text0[:200]}")
        for btn in ['a:has-text("Iniciar sesión")', 'button:has-text("Iniciar sesión")',
                    'a:has-text("Log in")', 'button:has-text("Log in")',
                    '[data-testid="login-button"]']:
            els = page.locator(btn)
            if await els.count() > 0:
                await els.first.click()
                print(f"    Click: {btn}")
                await asyncio.sleep(2)
                break

        print(f"[2] Buscando campo de email...")
        email_sel = 'input[type="email"], input[name="email"], input[name="username"], input[autocomplete="email"]'
        await page.wait_for_selector(email_sel, timeout=20000)
        await page.screenshot(path="/tmp/login_step2_email.png")
        await page.fill(email_sel, EMAIL)
        print(f"    email llenado")

        # Presionar Enter — más robusto que click cuando React re-renderiza
        await asyncio.sleep(1)
        await page.keyboard.press("Enter")
        print(f"    Enter → submit email")

        print(f"[3] Esperando siguiente pantalla post-email...")
        await asyncio.sleep(3)
        await page.screenshot(path="/tmp/login_after_email.png")
        print(f"    URL: {page.url}")
        text = await page.evaluate("document.body.innerText")
        print(f"    Texto: {text[:400]}")

        # Si hay verificación por email, buscar "Continuar con contraseña"
        for link_text in ["Continuar con contraseña", "Continue with password",
                          "Use password instead", "Try another method"]:
            els = page.locator(f'button:has-text("{link_text}"), a:has-text("{link_text}")')
            if await els.count() > 0:
                await els.first.click()
                print(f"    Click: '{link_text}'")
                await asyncio.sleep(2)
                break

        # Llenar password
        # Password step is OPTIONAL. A passwordless account (login by emailed code)
        # never shows a password field, and forcing one here is exactly what broke
        # this for such accounts. If a field appears and we have a password, fill
        # it; otherwise leave it for you to complete in the visible window.
        try:
            await page.wait_for_selector('input[type="password"]', timeout=8000)
            if PASSWORD:
                await page.fill('input[type="password"]', PASSWORD)
                await page.keyboard.press("Enter")
                print("    password llenado y enviado")
            else:
                print("    *** Esta cuenta pide contraseña — escríbela en la ventana ***")
        except Exception:
            print("    (no hay campo de contraseña — cuenta passwordless, seguimos)")

        # Whatever comes next -- an emailed code, an MFA code -- happens in the
        # BROWSER. This waits up to 5 minutes for you to finish, watching for the
        # OAuth redirect that carries the authorization code. Nothing is typed for
        # you here; you complete the human steps and the script catches the result.
        print("\n    >>> Completa lo que falte EN LA VENTANA (código del correo, etc.)")
        print("    >>> Tienes hasta 5 minutos. Esperando el redirect de OAuth...\n")

        # Esperar redirect al custom scheme o a timeout
        print(f"[4] Esperando redirect OAuth...")

        # Si hay MFA, el browser está visible — esperar hasta 3 minutos a que el user lo complete
        if "mfa" in page.url or "challenge" in page.url or "otp" in page.url:
            print(f"    MFA detectado en: {page.url}")
            print(f"    *** Ingresa el código MFA en el browser — esperando hasta 3 min ***")
            # Esperar hasta que la URL ya no sea de MFA
            try:
                await page.wait_for_url(
                    lambda u: "mfa" not in u and "challenge" not in u and "otp" not in u,
                    timeout=180000
                )
                print(f"    MFA completado — URL: {page.url}")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"    Timeout esperando MFA: {e}")

        try:
            # 5 minutes, not 30 seconds: reading an emailed code and typing it back
            # is a human step, and 30s guaranteed a timeout for exactly the
            # passwordless flow this is meant to support. The redirect fires the
            # instant the browser finishes, so a fast login still returns fast --
            # this only raises the ceiling for the slow, human case.
            code = await asyncio.wait_for(redirect_future, timeout=300.0)
            print(f"    code capturado: {code[:30]}...")
        except asyncio.TimeoutError:
            current = page.url
            print(f"    timeout — URL actual: {current[:200]}")
            if "code=" in current:
                parsed = urlparse(current)
                qs = parse_qs(parsed.query)
                code = qs.get("code", [None])[0]

        await browser.close()

    if not code:
        raise RuntimeError("No se obtuvo el authorization code.")

    print(f"[5] Intercambiando code por tokens...")
    import httpx
    r = await httpx.AsyncClient().post(
        f"{AUTH_BASE}/oauth/token",
        json={
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
            "client_id":     CLIENT_ID,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/json"},
    )
    print(f"    → {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"Token exchange falló: {r.status_code} {r.text[:300]}")

    tokens = r.json()
    result = {
        "access_token":  tokens.get("access_token"),
        "refresh_token": tokens.get("refresh_token"),
        "id_token":      tokens.get("id_token"),
        "token_type":    tokens.get("token_type", "Bearer"),
        "expires_in":    tokens.get("expires_in"),
    }
    TOKEN_FILE.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Tokens guardados en {TOKEN_FILE}")
    at = result["access_token"] or ""
    rt = result["refresh_token"] or ""
    print(f"  access_token:  {at[:40]}...")
    print(f"  refresh_token: {rt[:30]}...")
    return result

if __name__ == "__main__":
    asyncio.run(login())
