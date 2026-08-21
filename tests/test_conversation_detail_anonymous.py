"""Reading one conversation back works anonymously; listing them does not.

Measured against the live backend on 2026-08-20:
  - /backend-anon/conversation/{id} from the device that created it -> 200,
    with the generated title and the full mapping.
  - the same id from any other device -> 404 conversation_inaccessible,
    "Log in to view this conversation."
  - /backend-anon/conversations -> 200 with an empty page, always.

So the device id is the only credential, and this proxy can serve a
conversation only while it still holds the session that created it.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

import auth
import capabilities as cap
import main


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    """Records the path asked for, so the test can prove which prefix was used."""

    def __init__(self, payload):
        self._payload = payload
        self.paths = []

    async def get(self, url, headers=None, **kw):
        self.paths.append(url)
        return _FakeResponse(self._payload)


class _FakeSession:
    def __init__(self, conversation_id, device_id, client):
        self.conversation_id = conversation_id
        self.device_id = device_id
        self.client = client

    async def close(self):
        # The app's shutdown hook calls SessionPool.close_all().
        pass


_MAPPING = {
    "title": "Decir zanahoria",
    "create_time": 1,
    "mapping": {
        "a": {"message": {"id": "m1", "author": {"role": "user"},
                          "content": {"parts": ["di zanahoria"]}, "create_time": 1}},
        "b": {"message": {"id": "m2", "author": {"role": "assistant"},
                          "content": {"parts": ["zanahoria"]}, "create_time": 2}},
        "c": {"message": {"id": "m3", "author": {"role": "system"},
                          "content": {"parts": [""]}, "create_time": 0}},
    },
}


@pytest.fixture
def anonymous(monkeypatch):
    cap.reset()
    monkeypatch.setattr(cap, "snapshot", lambda **kw: cap.AccountState(mode="anonymous"))
    monkeypatch.setattr(auth, "is_authenticated", lambda: False)
    yield
    main._pools.clear()


def _seed_pool(conversation_id, payload=_MAPPING):
    """Puts one session in the default user's pool, as a real turn would."""
    fake = _FakeClient(payload)
    pool = main._user_pool("anonymous")
    pool._pool["k"] = _FakeSession(conversation_id, str(uuid.uuid4()), fake)
    return fake


def test_a_conversation_this_proxy_created_is_readable_anonymously(anonymous):
    cid = "6a87a81e-5058-83ea-955b-7fd4ba745956"
    fake = _seed_pool(cid)

    with TestClient(main.app) as c:
        r = c.get(f"/v1/conversations/{cid}")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title"] == "Decir zanahoria"
    # Ordered by create_time, with the empty system node dropped.
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    # The anonymous prefix, not the account one: /backend-api would 401 here.
    assert fake.paths and "/backend-anon/conversation/" in fake.paths[0]


def test_it_uses_the_creating_device_not_just_any_pooled_session(anonymous):
    cid = "the-one-we-want"
    other = _FakeClient({"mapping": {}})
    pool = main._user_pool("anonymous")
    pool._pool["other"] = _FakeSession("a-different-conversation", "wrong-device", other)
    mine = _FakeClient(_MAPPING)
    pool._pool["mine"] = _FakeSession(cid, "right-device", mine)

    with TestClient(main.app) as c:
        r = c.get(f"/v1/conversations/{cid}")

    assert r.status_code == 200
    # Reading it through the wrong device is a 404 from the vendor, so picking
    # the first pooled session would have silently broken this.
    assert mine.paths and not other.paths


def test_an_unknown_conversation_says_why_instead_of_asking_for_an_account(anonymous):
    _seed_pool("some-other-conversation")

    with TestClient(main.app) as c:
        r = c.get("/v1/conversations/never-seen-this-one")

    assert r.status_code == 404, r.text
    # Not 401: an account would not have helped, and saying so would send the
    # caller off to fix the wrong thing.
    assert "device that created it" in r.text


def test_listing_still_needs_an_account(anonymous):
    # The vendor answers 200 with an empty page anonymously, so there is
    # genuinely nothing to serve here.
    with TestClient(main.app) as c:
        r = c.get("/v1/conversations")

    assert r.status_code in (401, 501), r.text
