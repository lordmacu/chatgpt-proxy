import time

from fastapi.testclient import TestClient

import capabilities as cap
import main

PER_MODEL = {"tools", "vision", "images"}

# A live paid plan, matching the GO fixture in tests/test_capabilities.py:
# capabilities._paid() requires subscription_active=True as well as
# plan != "free" -- the dataclass default (False) makes a bare plan="go"
# read as unpaid, so every "go" fixture in this file must set it explicitly.
GO = cap.AccountState(mode="account", plan="go", subscription_active=True)

# Seeded straight into the handler's cache so no test reaches the network. The
# shape is the vendor's, as fetch_anon_models returns it: `max_tokens` is what
# becomes `context_window`, and it is absent on the aliases.
UPSTREAM = [
    {"slug": "gpt-5-6", "title": "GPT-5.6 Luna", "max_tokens": 52815,
     "reasoning_type": "auto", "enabled_tools": ["tools", "search"]},
    {"slug": "gpt-5-6-t-mini", "title": "GPT-5.6 T Mini", "max_tokens": 262144,
     "reasoning_type": "reasoning", "enabled_tools": ["tools", "search"]},
]


def _models(monkeypatch, state):
    cap.reset()
    monkeypatch.setattr(cap, "snapshot", lambda **kw: state)
    monkeypatch.setattr(main, "_models_cache", (time.time(), UPSTREAM))
    with TestClient(main.app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200
    return r.json()["data"]


def test_every_model_carries_a_context_window(monkeypatch):
    # The gateway declared 128000 by hand for every id while the real value is
    # 52815 (and 262144 for the two -t-mini). Publishing it is what ends that.
    for m in _models(monkeypatch, GO):
        assert isinstance(m["context_window"], int)
        assert m["context_window"] > 0


def test_every_model_carries_per_model_capabilities(monkeypatch):
    for m in _models(monkeypatch, GO):
        assert set(m["capabilities"]) == PER_MODEL
        assert all(isinstance(v, bool) for v in m["capabilities"].values())


def test_a_per_model_capability_is_never_wider_than_the_provider_level_one(monkeypatch):
    # The contract's narrowing rule: on a free plan nothing may claim images.
    state = cap.AccountState(mode="account", plan="free")
    provider_level = cap.effective(state)
    for m in _models(monkeypatch, state):
        for name in PER_MODEL:
            assert not (m["capabilities"][name] and not provider_level[name]), \
                f"{m['id']} claims {name} while the provider reports it false"


def test_only_the_image_models_claim_images(monkeypatch):
    models = _models(monkeypatch, GO)
    drawing = {m["id"] for m in models if m["capabilities"]["images"]}
    assert drawing == set(main._IMAGE_MODELS)


def test_an_image_model_reports_no_output_ceiling(monkeypatch):
    # It returns a picture, not tokens. 0 is what fixed_models already declares
    # for dall-e-3 in the gateway's providers.yaml.
    models = _models(monkeypatch, GO)
    for m in models:
        if m["id"] in main._IMAGE_MODELS:
            assert m["max_output_tokens"] == 0
