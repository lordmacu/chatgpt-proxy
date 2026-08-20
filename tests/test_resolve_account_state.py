"""Tests for main._resolve_account_state(): the vendor-shape parsing that
capabilities.py's resolver hook delegates to.

Every other test bypasses this function entirely -- test_capabilities.py
injects a fake `_resolve` into `snapshot()`, and test_health_contract.py
monkeypatches `capabilities.snapshot` outright -- so this file is the only
place a wrong field name or a wrong nesting level in the real accounts/check
response would actually be caught. `.get()` chains fail silently (None/False,
not an exception), so this is the correctness-critical path.
"""
import capabilities as cap
import main


class _FakeResponse:
    """Just enough of an httpx.Response to satisfy _resolve_account_state."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


# The real shape captured live from this deployment. Note expires_at carries
# "+00:00", not "Z" -- _resolve_account_state passes it through verbatim, so
# the test must not "normalize" it either.
_LIVE_PAYLOAD_ACTIVE = {"accounts": {"default": {
    "account": {"account_id": "092dcce0-1111-2222-3333-444444444444",
                "plan_type": "go"},
    "entitlement": {"subscription_plan": "chatgptgoplan",
                    "has_active_subscription": True,
                    "expires_at": "2026-09-06T00:28:46+00:00"}}}}

_LIVE_PAYLOAD_INACTIVE = {"accounts": {"default": {
    "account": {"account_id": "092dcce0-1111-2222-3333-444444444444",
                "plan_type": "go"},
    "entitlement": {"subscription_plan": "chatgptgoplan",
                    "has_active_subscription": False,
                    "expires_at": "2026-09-06T00:28:46+00:00"}}}}


def test_resolve_account_state_parses_an_active_subscription(monkeypatch):
    monkeypatch.setattr(main.auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(main._httpx, "get",
                        lambda *a, **kw: _FakeResponse(_LIVE_PAYLOAD_ACTIVE))
    state = main._resolve_account_state()
    assert state == cap.AccountState(mode="account", plan="go",
                                     subscription_active=True,
                                     expires_at="2026-09-06T00:28:46+00:00")


def test_resolve_account_state_parses_an_inactive_subscription(monkeypatch):
    # has_active_subscription: False must survive as subscription_active=False,
    # and that must turn `images` off downstream -- the event this contract
    # exists for: a lapsed plan self-reports through the real vendor call.
    monkeypatch.setattr(main.auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(main._httpx, "get",
                        lambda *a, **kw: _FakeResponse(_LIVE_PAYLOAD_INACTIVE))
    state = main._resolve_account_state()
    assert state.subscription_active is False
    assert cap.effective(state)["images"] is False


def test_resolve_account_state_is_anonymous_without_auth(monkeypatch):
    # Must short-circuit before any HTTP call -- no _httpx.get patch here on
    # purpose, so a real network attempt would fail the test loudly.
    monkeypatch.setattr(main.auth, "is_authenticated", lambda: False)
    state = main._resolve_account_state()
    assert state == cap.AccountState(mode="anonymous")
