"""DPoP (RFC 9449) proof support for the ChatGPT Android auth endpoints.

Reconstructed from the decompiled app (`defpackage/zih.java`, `mih.java`,
`hxo.java`). The Android client sender-constrains its tokens with DPoP: every
`/oauth/token` call carries a `DPoP:` header, an ES256 JWT signed by a P-256 key
that lives in the phone's hardware KeyStore.

Two facts drive this module:

  * The proof for a REFRESH does not need the current access token. The access
    token only feeds the optional `ath` claim, and `mih.a` adds that claim solely
    when the request carries an `Authorization: Bearer` header -- which the
    `/oauth/token` refresh does not. So a refresh proof is just:
        key + htu (url) + htm (method) + jti + iat  (+ nonce, if asked).

  * DPoP binds tokens to the KEY (its JWK thumbprint), not to any token. The
    refresh token must have been ISSUED under this same key. You cannot bolt DPoP
    onto a refresh token minted without it -- the server sees a key it never
    bound and answers `invalid_dpop_proof`. Hence one persistent key, used at
    BOTH login (token exchange) and refresh.

DPoP is OFF unless you opt in: set `CHATGPT_DPOP=1`, or just let the key file
exist. With nothing configured the proxy behaves exactly as before.
"""
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

# The private key lives next to tokens.json. Override with DPOP_KEY_FILE, e.g.
# to mount it read-only into a container.
KEY_FILE = Path(os.environ.get("DPOP_KEY_FILE",
                               str(Path(__file__).parent / "dpop_key.pem")))

# The server hands out a nonce via the `DPoP-Nonce` response header and expects
# it echoed in the next proof. The app keeps it in an in-memory AtomicReference
# (`mih.b`); one module global mirrors that -- good enough for a long-lived proxy.
_nonce: Optional[str] = None


def _log(*args):
    if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
        print("[dpop]", *args, file=sys.stderr, flush=True)


def enabled() -> bool:
    """DPoP is active when explicitly turned on, or when a key file exists.

    Mirrors the app's own gate (`zih.e()` -- "is there a key?"): once you have a
    key you are committed to it, because your tokens are bound to it.
    """
    if os.environ.get("CHATGPT_DPOP", "").lower() in ("1", "true", "yes"):
        return True
    return KEY_FILE.exists()


def _b64url(raw: bytes) -> str:
    """URL-safe base64, no padding, no wrap -- Android's Base64 flag 11."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _load_or_create_key() -> Optional[ec.EllipticCurvePrivateKey]:
    """The persistent P-256 key, created on first use when DPoP is enabled.

    Returns None (rather than raising) when DPoP is off and no key exists, so
    `proof()` degrades to "no header" and the caller stays on the plain flow.
    """
    if KEY_FILE.exists():
        try:
            return serialization.load_pem_private_key(KEY_FILE.read_bytes(), password=None)
        except Exception as e:  # corrupt/unreadable key: fail loud, don't mint a new one
            _log("could not read key file:", type(e).__name__, e)
            raise
    if not enabled():
        return None
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    try:
        KEY_FILE.write_bytes(pem)
        os.chmod(KEY_FILE, 0o600)
        _log("generated new P-256 DPoP key at", KEY_FILE)
    except OSError as e:
        # Read-only FS: keep the key in memory for this process. It will not
        # survive a restart, which for a bound token means re-login -- surface it.
        _log("could NOT persist DPoP key (in-memory only this run):", e)
    return key


def _jwk(public_key: ec.EllipticCurvePublicKey) -> dict:
    """The public JWK exactly as `zih.b` builds it: crv, kty, x, y (32-byte)."""
    nums = public_key.public_numbers()
    x = nums.x.to_bytes(32, "big")
    y = nums.y.to_bytes(32, "big")
    return {"crv": "P-256", "kty": "EC", "x": _b64url(x), "y": _b64url(y)}


def _der_to_raw(der: bytes) -> bytes:
    """DER ECDSA signature -> raw R||S, 32 bytes each (what `zih.a` does)."""
    r, s = decode_dss_signature(der)
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def current_nonce() -> Optional[str]:
    return _nonce


def remember_nonce(headers) -> None:
    """Store the `DPoP-Nonce` from a response, if present (`hxo.a0`)."""
    global _nonce
    try:
        n = headers.get("DPoP-Nonce")
    except Exception:
        n = None
    if n:
        _nonce = n
        _log("stored DPoP-Nonce")


def proof(url: str, method: str,
          nonce: Optional[str] = None,
          access_token: Optional[str] = None) -> Optional[str]:
    """Build a DPoP proof JWT for `method url`, or None when DPoP is inactive.

    `access_token` is only used for the `ath` claim on resource requests; leave
    it None for `/oauth/token` calls (login and refresh) -- see the module docs.
    `nonce` defaults to the last one the server handed us.
    """
    key = _load_or_create_key()
    if key is None:
        return None

    header = {"typ": "dpop+jwt", "alg": "ES256", "jwk": _jwk(key.public_key())}

    # htu is the URL with query and fragment stripped (`new URI(scheme, host, path)`).
    parts = urlsplit(url)
    htu = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    payload = {
        "jti": str(uuid.uuid4()),
        "htm": method.upper(),
        "htu": htu,
        "iat": int(time.time()),
    }
    if access_token:
        digest = hashes.Hash(hashes.SHA256())
        digest.update(access_token.encode("utf-8"))
        payload["ath"] = _b64url(digest.finalize())
    n = nonce if nonce is not None else _nonce
    if n:
        payload["nonce"] = n

    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    )
    der = key.sign(signing_input.encode("ascii"), ec.ECDSA(hashes.SHA256()))
    return signing_input + "." + _b64url(_der_to_raw(der))
