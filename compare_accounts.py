#!/usr/bin/env python3
"""Print a ChatGPT account's profile (plan, limits, storage, models) from its token.

Use it to compare accounts -- e.g. a free account vs a paid (Go/Plus) one:

    python compare_accounts.py <ACCESS_TOKEN>            # one account
    python compare_accounts.py <GO_TOKEN> <FREE_TOKEN>   # side-by-side diff
    CHATGPT_ACCESS_TOKEN=<token> python compare_accounts.py   # from env

The ACCESS_TOKEN is the account's Bearer token (a JWT, aud api.openai.com/v1).
You get it the same way as the Go account: from chatgpt.com while logged in with
that account, `GET /api/auth/session` returns { accessToken }, or copy the
`Authorization: Bearer ...` from any backend-api request in devtools.

It hits the same Android backend the proxy uses, so no proxy/deploy needed.
"""
import json
import os
import secrets
import sys
import time

import httpx

BASE = "https://android.chat.openai.com/backend-api"


def _headers(token: str) -> dict:
    return {
        "User-Agent": "ChatGPT/1.2026.223 (Android 16; sdk_gphone64_arm64; build 2622307)",
        "OAI-Package-Name": "com.openai.chatgpt",
        "OAI-Client-Type": "android",
        "OAI-Device-Id": secrets.token_hex(16)[:32],
        "Authorization": "Bearer " + token,
    }


def _get(client, headers, path, params=None):
    """GET with a few retries (the backend returns transient empty bodies)."""
    r = None
    for _ in range(4):
        r = client.get(BASE + path, headers={**headers, "X-OpenAI-Target-Path": "/backend-api" + path.split("?")[0]},
                       params=params, timeout=25)
        if r.status_code != 200 or r.text.strip():
            return r
        time.sleep(2)
    return r


def _post(client, headers, path, body):
    for _ in range(4):
        r = client.post(BASE + path,
                        headers={**headers, "X-OpenAI-Target-Path": "/backend-api" + path, "Content-Type": "application/json"},
                        json=body, timeout=25)
        if r.status_code != 200 or r.text.strip():
            return r
        time.sleep(2)
    return r


def profile(token: str) -> dict:
    out = {}
    with httpx.Client(timeout=30, follow_redirects=True) as c:
        h = _headers(token)

        # plan / subscription
        try:
            acc = _get(c, h, "/accounts/check/v4-2023-04-27", {"timezone_offset_min": "0"}).json()
            a = (acc.get("accounts", {}).get("default") or next(iter(acc.get("accounts", {}).values()), {}))
            out["plan_type"] = (a.get("account") or {}).get("plan_type")
            ent = a.get("entitlement") or {}
            out["subscription"] = {
                "plan": ent.get("subscription_plan"),
                "active": ent.get("has_active_subscription"),
                "expires_at": ent.get("expires_at"),
            }
            out["features_count"] = len(a.get("features") or [])
            out["features"] = a.get("features") or []
        except Exception as e:
            out["plan_error"] = str(e)[:80]

        # limits (per-feature) + model_limits, via conversation/init
        try:
            init = _post(c, h, "/conversation/init", {"conversation_mode_kind": "primary_assistant"}).json()
            out["limits"] = {x["feature_name"]: x.get("remaining") for x in init.get("limits_progress", [])}
            out["model_limits"] = init.get("model_limits")
            out["default_model_slug"] = init.get("default_model_slug")
            out["blocked_features"] = init.get("blocked_features")
        except Exception as e:
            out["limits_error"] = str(e)[:80]

        # storage
        try:
            u = _get(c, h, "/files/library/storage/usage").json()
            out["storage"] = {
                "limit_tier": u.get("limit_tier"),
                "used_mb": round((u.get("used_bytes") or 0) / 1e6, 1),
                "allowed_gb": round((u.get("allowed_bytes") or 0) / 1e9, 2),
            }
        except Exception as e:
            out["storage_error"] = str(e)[:80]

        # available models
        try:
            models = _get(c, h, "/models", {"iim": "false"}).json().get("models", [])
            out["models"] = [m.get("slug") for m in models]
        except Exception as e:
            out["models_error"] = str(e)[:80]
    return out


def _print(label, p):
    print(f"\n==================== {label} ====================")
    print("plan_type        :", p.get("plan_type"))
    print("subscription     :", p.get("subscription"))
    print("storage          :", p.get("storage"))
    print("default_model    :", p.get("default_model_slug"))
    print("models           :", p.get("models"))
    print("features (count) :", p.get("features_count"))
    print("model_limits     :", p.get("model_limits"))
    print("blocked_features :", p.get("blocked_features"))
    print("LIMITS (per feature):")
    for k, v in (p.get("limits") or {}).items():
        print(f"    {k:22} {v}")
    for k in ("plan_error", "limits_error", "storage_error", "models_error"):
        if p.get(k):
            print(f"  [{k}] {p[k]}")


def main():
    tokens = [t for t in sys.argv[1:] if t]
    if not tokens and os.environ.get("CHATGPT_ACCESS_TOKEN"):
        tokens = [os.environ["CHATGPT_ACCESS_TOKEN"]]
    if not tokens:
        print("uso: python compare_accounts.py <TOKEN> [<TOKEN2>]", file=sys.stderr)
        sys.exit(1)

    profs = [profile(t) for t in tokens]
    labels = ["ACCOUNT A", "ACCOUNT B"] if len(profs) == 2 else ["ACCOUNT"]
    for label, p in zip(labels, profs):
        _print(label, p)

    if len(profs) == 2:
        a, b = profs
        print("\n==================== DIFERENCIAS A vs B ====================")
        print(f"plan   : {a.get('plan_type')}  vs  {b.get('plan_type')}")
        print(f"storage: {a.get('storage')}  vs  {b.get('storage')}")
        la, lb = a.get("limits") or {}, b.get("limits") or {}
        for k in sorted(set(la) | set(lb)):
            if la.get(k) != lb.get(k):
                print(f"  limit {k:22}: {la.get(k)}  vs  {lb.get(k)}")
        onlya = set(a.get("models") or []) - set(b.get("models") or [])
        onlyb = set(b.get("models") or []) - set(a.get("models") or [])
        if onlya: print("  modelos solo en A:", sorted(onlya))
        if onlyb: print("  modelos solo en B:", sorted(onlyb))


if __name__ == "__main__":
    main()
