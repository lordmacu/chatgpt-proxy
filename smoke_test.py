#!/usr/bin/env python3
"""Hit every proxy endpoint with a given account and report what's accessible.

Run against a locally-running proxy (started with the account's token):

    CHATGPT_ACCESS_TOKEN=<token> python -m uvicorn main:app --port 8899 &
    python smoke_test.py                 # defaults to http://127.0.0.1:8899

Or point it elsewhere:  python smoke_test.py http://127.0.0.1:8890

Read-only endpoints are always exercised. The ones that COST a message
(chat, image, speech) are only run with --spend, so a free account's tiny quota
isn't burned by accident.
"""
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8899"
SPEND = "--spend" in sys.argv
for a in sys.argv[1:]:
    if a.startswith("http"):
        BASE = a.rstrip("/")


def _wait():
    for _ in range(30):
        try:
            if httpx.get(BASE + "/health", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def call(method, path, **kw):
    try:
        r = httpx.request(method, BASE + path, timeout=120, **kw)
    except Exception as e:
        return "ERR", str(e)[:60]
    body = ""
    try:
        body = json.dumps(r.json())
    except Exception:
        body = r.text[:80]
    return r.status_code, body


def main():
    if not _wait():
        print("proxy no responde en", BASE); sys.exit(1)
    h = httpx.get(BASE + "/health", timeout=5).json()
    print("auth_mode:", h.get("auth_mode"))
    print("=" * 60)

    results = []

    def row(label, sc, note=""):
        ok = "OK " if str(sc).startswith("2") else ("---" if sc in (401, 403) else "!! ")
        print(f"  [{ok}] {label:32} -> {sc}  {note[:70]}")
        results.append((label, sc))

    # --- read-only ---
    sc, b = call("GET", "/v1/account"); row("GET /v1/account", sc, b[:60])
    sc, b = call("GET", "/v1/limits")
    lim = ""
    try:
        d = json.loads(b); lim = str({x["feature_name"]: x["remaining"] for x in d.get("limits_progress", [])})
    except Exception:
        pass
    row("GET /v1/limits", sc, lim)
    sc, b = call("GET", "/v1/models"); row("GET /v1/models", sc)
    sc, b = call("GET", "/v1/session/me"); row("GET /v1/session/me", sc)
    sc, b = call("GET", "/v1/custom-instructions"); row("GET /v1/custom-instructions", sc)
    sc, b = call("GET", "/v1/suggestions"); row("GET /v1/suggestions", sc)
    sc, b = call("GET", "/v1/gizmos")
    gid = None
    try:
        gz = json.loads(b).get("gizmos", []); gid = gz[0]["id"] if gz else None
    except Exception:
        pass
    row("GET /v1/gizmos", sc, f"({len(json.loads(b).get('gizmos', []))} GPTs)" if sc == 200 else b[:50])
    if gid:
        sc, b = call("GET", "/v1/gizmos/" + gid); row("GET /v1/gizmos/{id}", sc)

    # conversations / library (resolve ids)
    sc, b = call("GET", "/v1/conversations?limit=1")
    cid = None
    try:
        items = json.loads(b).get("items", []); cid = items[0]["id"] if items else None
    except Exception:
        pass
    row("GET /v1/conversations", sc, f"(total {json.loads(b).get('total')})" if sc == 200 else b[:50])
    if cid:
        sc, b = call("GET", "/v1/conversations/" + cid); row("GET /v1/conversations/{id}", sc)

    sc, b = call("GET", "/v1/library?limit=1")
    fid = None
    try:
        items = json.loads(b).get("items", []); fid = items[0]["id"] if items else None
    except Exception:
        pass
    row("GET /v1/library", sc, f"({len(json.loads(b).get('items', []))} archivos)" if sc == 200 else b[:50])
    sc, b = call("GET", "/v1/library/usage"); row("GET /v1/library/usage", sc, b[:60])
    if fid:
        sc, b = call("GET", "/v1/library/" + fid + "/download?url=1"); row("GET /v1/library/{id}/download", sc)

    # translate (no cuesta chat)
    sc, b = call("POST", "/v1/translate", json={"text": "hola mundo", "target": "en"}); row("POST /v1/translate", sc, b[:60])

    # --- costly (solo con --spend) ---
    if SPEND:
        sc, b = call("POST", "/v1/chat/completions", json={"model": "auto", "stream": False,
                     "messages": [{"role": "user", "content": "di solo: ok"}]})
        row("POST /v1/chat/completions", sc, b[:50])
        sc, b = call("POST", "/v1/audio/speech", json={"input": "hola", "voice": "juniper"})
        row("POST /v1/audio/speech", sc, b[:50])
        sc, b = call("POST", "/v1/images/generations", json={"prompt": "a small red dot"})
        row("POST /v1/images/generations", sc, b[:50])
    else:
        print("  (chat/speech/images: omitidos -- pasá --spend para probarlos, gastan cuota)")

    print("=" * 60)
    okc = sum(1 for _, s in results if str(s).startswith("2"))
    print(f"accesibles: {okc}/{len(results)}")


if __name__ == "__main__":
    main()
