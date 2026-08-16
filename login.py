"""
ChatGPT Android — login con email/password via PKCE + first_party_authorize.
Obtiene access_token, refresh_token, id_token y los guarda en tokens.json.
"""
import os, json, hashlib, base64, secrets, re, asyncio
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, quote
import httpx

# ── Credenciales ─────────────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

EMAIL    = os.environ["EMAIL"]
PASSWORD = os.environ["PASSWORD"]

# ── Constantes (capturadas con Frida del APK) ─────────────────────────────────
AUTH_BASE    = "https://auth.openai.com"
CLIENT_ID    = "app_xwBKzt04752TTSfXnki17hmB"
REDIRECT_URI = "com.openai.chatgpt://auth.openai.com/android/com.openai.chatgpt/callback"
SCOPE        = "openid email profile offline_access model.request model.read organization.read organization.write"
AUDIENCE     = "https://api.openai.com/v1"
APP_VERSION  = "1.2026.223"
DEVICE_ID    = secrets.token_hex(16)[:32]   # nuevo por sesión

HEADERS_BASE = {
    "User-Agent": f"ChatGPT/{APP_VERSION} (Android 16; sdk_gphone64_arm64; build 2622307)",
    "OAI-Package-Name": "com.openai.chatgpt",
    "OAI-Client-Type": "android",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

TOKEN_FILE = Path(__file__).parent / "tokens.json"

# ── PKCE helpers ──────────────────────────────────────────────────────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def pkce_pair() -> tuple[str, str]:
    verifier  = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge

# ── Login ─────────────────────────────────────────────────────────────────────

async def login(email: str = EMAIL, password: str = PASSWORD) -> dict:
    verifier, challenge = pkce_pair()
    state = _b64url(secrets.token_bytes(24))
    nonce = _b64url(secrets.token_bytes(24))

    client = httpx.AsyncClient(
        headers=HEADERS_BASE,
        follow_redirects=True,
        timeout=httpx.Timeout(30.0),
    )

    # ── Paso 1: iniciar flujo PKCE ────────────────────────────────────────────
    print("[1] Iniciando flujo PKCE...")
    params = {
        "scope":                          SCOPE,
        "code_challenge":                 challenge,
        "code_challenge_method":          "S256",
        "client_id":                      CLIENT_ID,
        "redirect_uri":                   REDIRECT_URI,
        "state":                          state,
        "nonce":                          nonce,
        "audience":                       AUDIENCE,
        "issuer":                         AUTH_BASE,
        "ccaps":                          "default_otp_v2 login_methods",
        "android_device_id":              DEVICE_ID,
        "requester_metadata_app_version": APP_VERSION,
        "screen_hint":                    "login_or_signup",
        "response_type":                  "code",
        "hydra_flow":                     "condense",
    }
    r1 = await client.get(f"{AUTH_BASE}/api/accounts/authorize", params=params)
    print(f"    → {r1.status_code}")
    if r1.status_code not in (200, 302):
        raise RuntimeError(f"Paso 1 falló: {r1.status_code} {r1.text[:300]}")

    # Extraer session/state del body o headers para el paso siguiente
    data1 = {}
    try:
        data1 = r1.json()
    except Exception:
        pass
    print(f"    resp1: {json.dumps(data1, indent=2)[:800]}")

    session_id  = data1.get("oai-client-auth-session", {}).get("session_id", "")
    page_type1  = data1.get("page", {}).get("type", "login_or_signup_start")
    continue_url = data1.get("continue_url", "")

    # Navegar al continue_url para que el servidor establezca cookies de sesión y CSRF
    csrf_token = None
    if continue_url:
        print(f"[1b] Navegando continue_url: {continue_url}")
        rc = await client.get(continue_url)
        print(f"    → {rc.status_code}")
        # Extraer CSRF token del cookie hydra_redirect (Flask signed cookie)
        hydra_raw = client.cookies.get("hydra_redirect", "")
        if hydra_raw:
            try:
                payload_b64 = hydra_raw.split(".")[0]
                padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
                payload = json.loads(base64.urlsafe_b64decode(padded))
                # El único key tiene el CSRF token base64-encoded
                for v in payload.values():
                    csrf_inner_b64 = v
                    padded2 = csrf_inner_b64 + "=" * (4 - len(csrf_inner_b64) % 4)
                    csrf_raw = base64.urlsafe_b64decode(padded2)
                    # Formato: timestamp|token|hmac — extraer la parte del token
                    csrf_parts = csrf_raw.split(b"|")
                    if len(csrf_parts) >= 2:
                        csrf_token = csrf_parts[1].decode("ascii", errors="ignore")
                        print(f"    csrf_token (extraído): {csrf_token[:40]}...")
                    break
            except Exception as ce:
                print(f"    CSRF decode error: {ce}")
        login_sess = client.cookies.get("login_session", "")
        print(f"    login_session present: {bool(login_sess)}")
        print(f"    hydra_redirect present: {bool(hydra_raw)}")

    # ── Paso 2: enviar email ──────────────────────────────────────────────────
    print("[2] Enviando email...")
    body2 = {
        "origin_page_type": page_type1,
        "session_id":       session_id,
        "identifier":       email,
    }

    extra_headers = {
        "Origin": AUTH_BASE,
        "Referer": (continue_url or AUTH_BASE) + "/",
    }
    if csrf_token:
        extra_headers["X-CSRF-Token"] = csrf_token
    print(f"    body2: {body2}")
    print(f"    extra_headers: {extra_headers}")
    r2 = await client.post(
        f"{AUTH_BASE}/api/first_party_authorize/next",
        json=body2,
        headers=extra_headers,
    )
    print(f"    → {r2.status_code}")
    print(f"    resp-body: {r2.text[:800]}")
    data2 = {}
    try:
        data2 = r2.json()
        print(f"    keys: {list(data2.keys())}")
        page_node = data2.get("page", {})
        if page_node:
            print(f"    page.type: {page_node.get('type')}")
    except Exception:
        pass

    if r2.status_code not in (200, 201):
        raise RuntimeError(f"Paso 2 (email) falló: {r2.status_code}")

    # ── Paso 3: enviar password ───────────────────────────────────────────────
    print("[3] Enviando password...")
    page_type2 = data2.get("page", {}).get("type", "login_enter_password")
    body3 = {
        "origin_page_type": page_type2,
        "session_id":       session_id,
        "identifier":       email,
        "password":         password,
    }

    r3 = await client.post(
        f"{AUTH_BASE}/api/first_party_authorize/next",
        json=body3,
    )
    print(f"    → {r3.status_code}")
    data3 = {}
    try:
        data3 = r3.json()
        print(f"    keys: {list(data3.keys())}")
        if "type" in data3:
            print(f"    type: {data3['type']}")
    except Exception:
        print(f"    body: {r3.text[:300]}")

    if r3.status_code not in (200, 201):
        raise RuntimeError(f"Paso 3 (password) falló: {r3.status_code} {r3.text[:300]}")

    # ── Paso 4: extraer code del redirect URL ─────────────────────────────────
    # La respuesta puede incluir un redirect_uri con el code, o un campo "code"
    code = None

    # Opción A: campo "code" directo en JSON
    if "code" in data3:
        code = data3["code"]

    # Opción B: redirect_to con el code en query params
    for field in ("redirect_to", "redirect_uri", "location", "url"):
        if field in data3 and not code:
            parsed = urlparse(data3[field])
            qs = parse_qs(parsed.query)
            if "code" in qs:
                code = qs["code"][0]

    # Opción C: Location header
    if not code and "location" in r3.headers:
        parsed = urlparse(r3.headers["location"])
        qs = parse_qs(parsed.query)
        code = qs.get("code", [None])[0]

    if not code:
        # Puede necesitar un paso extra (e.g. MFA o captcha)
        print("    No se obtuvo 'code'. Respuesta completa:")
        print(f"    {json.dumps(data3, indent=2)[:800]}")
        raise RuntimeError("No se pudo obtener el authorization code. Revisar flujo.")

    print(f"[4] Code obtenido: {code[:30]}...")

    # ── Paso 5: intercambiar code por tokens ──────────────────────────────────
    print("[5] Intercambiando code por tokens...")
    r4 = await client.post(
        f"{AUTH_BASE}/oauth/token",
        json={
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
            "client_id":     CLIENT_ID,
            "code_verifier": verifier,
        },
    )
    print(f"    → {r4.status_code}")
    if r4.status_code != 200:
        raise RuntimeError(f"Token exchange falló: {r4.status_code} {r4.text[:300]}")

    tokens = r4.json()
    await client.aclose()

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
    print(f"  access_token:  {result['access_token'][:40]}...")
    print(f"  refresh_token: {result['refresh_token'][:30] if result['refresh_token'] else 'None'}...")
    return result


async def refresh(refresh_token: str) -> dict:
    """Renueva el access_token usando el refresh_token."""
    client = httpx.AsyncClient(headers=HEADERS_BASE, timeout=30.0)
    r = await client.post(
        f"{AUTH_BASE}/oauth/token",
        json={
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
            "client_id":     CLIENT_ID,
        },
    )
    await client.aclose()
    r.raise_for_status()
    tokens = r.json()
    result = {
        "access_token":  tokens.get("access_token"),
        "refresh_token": tokens.get("refresh_token", refresh_token),
        "id_token":      tokens.get("id_token"),
        "token_type":    tokens.get("token_type", "Bearer"),
        "expires_in":    tokens.get("expires_in"),
    }
    TOKEN_FILE.write_text(json.dumps(result, indent=2))
    print(f"✓ Token renovado. access_token: {result['access_token'][:40]}...")
    return result


def load_tokens():
    if TOKEN_FILE.exists():
        return json.loads(TOKEN_FILE.read_text())
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        t = load_tokens()
        if not t or not t.get("refresh_token"):
            print("No hay refresh_token guardado. Corre sin argumentos para hacer login.")
            sys.exit(1)
        asyncio.run(refresh(t["refresh_token"]))
    else:
        asyncio.run(login())
