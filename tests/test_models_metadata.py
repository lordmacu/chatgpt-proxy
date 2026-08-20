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


def _assert_narrowing_holds(monkeypatch, state):
    provider_level = cap.effective(state)
    for m in _models(monkeypatch, state):
        for name in PER_MODEL:
            assert not (m["capabilities"][name] and not provider_level[name]), \
                f"{m['id']} claims {name} while the provider reports it false"


def test_a_per_model_capability_is_never_wider_than_the_provider_level_one__on_a_free_plan(monkeypatch):
    # The contract's narrowing rule: on a free plan nothing may claim images.
    # `images` is the only key this state can catch a violation on -- `vision`
    # and `tools` are both True for a free account, so an entry claiming
    # either would not trip `not provider_level[name]` here. See the two
    # cases below for those.
    _assert_narrowing_holds(monkeypatch, cap.AccountState(mode="account", plan="free"))


def test_a_per_model_capability_is_never_wider_than_the_provider_level_one__anonymous(monkeypatch):
    # `vision` is gated on `mode == "account"` (capabilities.effective), so it
    # is False for an anonymous session -- this is the state that lets the
    # invariant catch a violation on that key specifically.
    _assert_narrowing_holds(monkeypatch, cap.AccountState(mode="anonymous"))


def test_a_per_model_capability_is_never_wider_than_the_provider_level_one__emulation_off(monkeypatch):
    # `tools` tracks tool_calls.EMULATION_ENABLED, not AccountState at all (see
    # capabilities.effective) -- no plan or mode can make it False, so this is
    # the only state that lets the invariant catch a violation on that key.
    monkeypatch.setattr(cap.tool_calls, "EMULATION_ENABLED", False)
    _assert_narrowing_holds(monkeypatch, GO)


def test_an_image_model_never_reports_tools_or_vision(monkeypatch):
    # Stated directly rather than inferred from the narrowing inequality above:
    # on a paid account with emulation on, provider-level `tools` and `vision`
    # are both True, so a non-drawing model legitimately inherits them. An
    # image model must still report False for both -- that is the
    # `(not draws) and` guard in main.py, asserted as itself instead of via an
    # inequality that only bites when the provider-level value is already
    # False. This is the test that catches an edit that drops that guard.
    monkeypatch.setattr(cap.tool_calls, "EMULATION_ENABLED", True)
    provider_level = cap.effective(GO)
    assert provider_level["tools"] is True
    assert provider_level["vision"] is True
    for m in _models(monkeypatch, GO):
        if m["id"] in main._IMAGE_MODELS:
            assert m["capabilities"]["tools"] is False
            assert m["capabilities"]["vision"] is False


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
