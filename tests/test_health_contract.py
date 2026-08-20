from fastapi.testclient import TestClient

import capabilities as cap
import main


REQUIRED = set(cap.REQUIRED_CAPABILITIES)
AUTH_MODES = {"anonymous", "account", "unknown"}


def _health(monkeypatch, state):
    cap.reset()
    monkeypatch.setattr(cap, "snapshot", lambda **kw: state)
    with TestClient(main.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    return r.json()


def test_health_declares_the_contract_version(monkeypatch):
    body = _health(monkeypatch, cap.AccountState(mode="anonymous"))
    assert body["contract"] == 1
    assert body["provider"] == "chatgpt"


def test_health_capabilities_are_exactly_the_required_booleans(monkeypatch):
    body = _health(monkeypatch, cap.AccountState(mode="anonymous"))
    assert set(body["capabilities"]) == REQUIRED
    assert all(isinstance(v, bool) for v in body["capabilities"].values())


def test_health_auth_block_reports_the_account(monkeypatch):
    state = cap.AccountState(mode="account", plan="go", subscription_active=True,
                             expires_at="2026-09-06T00:28:46Z")
    body = _health(monkeypatch, state)
    assert body["auth"]["mode"] in AUTH_MODES
    assert body["auth"] == {"mode": "account", "plan": "go",
                            "subscription_active": True,
                            "expires_at": "2026-09-06T00:28:46Z"}


def test_images_is_false_without_a_paid_plan(monkeypatch):
    body = _health(monkeypatch, cap.AccountState(mode="account", plan="free"))
    assert body["capabilities"]["images"] is False


def test_the_legacy_status_and_version_fields_survive(monkeypatch):
    # Coolify's container health check and every existing dashboard read these.
    body = _health(monkeypatch, cap.AccountState(mode="anonymous"))
    assert body["status"] == "ok"
    assert isinstance(body["version"], str)
    assert body["auth_mode"] in AUTH_MODES
