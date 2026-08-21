"""Anonymous conversations: read back by id, listed from this proxy's own index.

Measured against the live backend on 2026-08-20:
  - /backend-anon/conversation/{id} from the device that created it -> 200,
    with the generated title and the full mapping.
  - the same id from any other device -> 404 conversation_inaccessible,
    "Log in to view this conversation."
  - /backend-anon/conversations -> 200 with an empty page, always.

Remeasured 2026-08-21, which changed two of the conclusions drawn from that:
  - a fresh client with NO cookies reads the conversation back as long as it
    carries the original device id, and still does after the creating session
    is closed. The device id is the whole credential, so holding the session is
    not required -- only remembering the device is (conv_store.py).
  - the empty listing is not "there is nothing", it is "the vendor will not
    tell you": a device owning two live conversations still gets total=0. So
    the listing is served locally, or not at all.
"""
import sqlite3
import uuid

import httpx as _httpx
import pytest
from fastapi.testclient import TestClient

import auth
import capabilities as cap
import conv_store
import main


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self)

    def json(self):
        return self._payload


class _FakeClient:
    """Records the path and headers, so a test can prove which device was used."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self._status_code = status_code
        self.paths = []
        self.headers_seen = []

    async def get(self, url, headers=None, **kw):
        self.paths.append(url)
        self.headers_seen.append({k.lower(): v for k, v in (headers or {}).items()})
        return _FakeResponse(self._payload, self._status_code)

    async def aclose(self):
        # The handler closes a client it opened itself.
        pass


class _FakeSession:
    def __init__(self, conversation_id, device_id, client):
        self.conversation_id = conversation_id
        self.device_id = device_id
        self.client = client
        self.last_title = None

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
    # is_authenticated alone is not enough: _base_headers asks access_token()
    # directly, so a dev checkout with a tokens.json was putting a real Bearer
    # on requests these tests call anonymous.
    monkeypatch.setattr(auth, "access_token", lambda: "")
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


def test_the_listing_is_served_from_this_proxys_own_index(anonymous):
    # This used to assert 401/501, on the grounds that the vendor answers an
    # empty page anonymously. It still does -- remeasured 2026-08-21, total=0
    # even for the device that owns two live conversations -- which is precisely
    # why the index has to be local. There IS something to serve; it just was
    # never going to come from upstream.
    conv_store.record("anonymous", "c-old", "device-1", "Lo viejo", now=100.0)
    conv_store.record("anonymous", "c-new", "device-2", "Lo nuevo", now=200.0)

    with TestClient(main.app) as c:
        r = c.get("/v1/conversations")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2
    # Most recently used first, like the account listing it stands in for.
    assert [i["id"] for i in body["items"]] == ["c-new", "c-old"]
    assert body["items"][0]["title"] == "Lo nuevo"


def test_one_callers_index_is_not_another_callers(anonymous):
    conv_store.record("anonymous", "c-shared", "device-1", "Sin token", now=100.0)
    conv_store.record("token-abc", "c-mine", "device-2", "Con token", now=100.0)

    with TestClient(main.app) as c:
        untokened = c.get("/v1/conversations").json()
        tokened = c.get("/v1/conversations",
                        headers={"Authorization": "Bearer token-abc"}).json()

    assert [i["id"] for i in untokened["items"]] == ["c-shared"]
    assert [i["id"] for i in tokened["items"]] == ["c-mine"]


def test_the_listing_paginates(anonymous):
    for i in range(5):
        conv_store.record("anonymous", f"c{i}", "d", f"t{i}", now=float(i))

    with TestClient(main.app) as c:
        page = c.get("/v1/conversations?limit=2&offset=1").json()

    assert page["total"] == 5
    assert [i["id"] for i in page["items"]] == ["c3", "c2"]


def test_a_nonsense_limit_falls_back_instead_of_422ing(anonymous):
    with TestClient(main.app) as c:
        r = c.get("/v1/conversations?limit=abc&offset=-4")

    assert r.status_code == 200
    assert r.json()["limit"] == 28 and r.json()["offset"] == 0


def test_an_evicted_session_no_longer_ends_the_conversation(anonymous, monkeypatch):
    # The whole point: the pool is empty (TTL expired) but the device that can
    # read this conversation was recorded, and a fresh client carrying it works.
    cid = "6a884412-b944-83ea-8f94-0205f72a910e"
    conv_store.record("anonymous", cid, "the-creating-device", "Di hola")
    fake = _FakeClient(_MAPPING)
    monkeypatch.setattr(main._httpx, "AsyncClient", lambda **kw: fake)

    with TestClient(main.app) as c:
        r = c.get(f"/v1/conversations/{cid}")

    assert r.status_code == 200, r.text
    assert r.json()["title"] == "Decir zanahoria"
    assert fake.paths and "/backend-anon/conversation/" in fake.paths[0]
    assert fake.headers_seen and fake.headers_seen[0].get("oai-device-id") == "the-creating-device"


def test_a_conversation_the_vendor_forgot_is_dropped_from_the_index(anonymous, monkeypatch):
    cid = "long-gone"
    conv_store.record("anonymous", cid, "some-device", "Fantasma")
    gone = _FakeClient({}, status_code=404)
    monkeypatch.setattr(main._httpx, "AsyncClient", lambda **kw: gone)

    with TestClient(main.app) as c:
        r = c.get(f"/v1/conversations/{cid}")

    # A vendor 404 is a 404 to the caller, not the 500 an unhandled
    # raise_for_status() used to produce.
    assert r.status_code == 404, r.text
    # Listing a row that resolves to nothing forever is worse than not listing it.
    assert conv_store.lookup("anonymous", cid) is None


def test_an_account_turn_is_not_written_to_the_anonymous_index(monkeypatch):
    monkeypatch.setattr(auth, "is_authenticated", lambda: True)
    session = _FakeSession("c-account", "device", None)
    main._remember_conversation("anonymous", session)
    # An account has server-side history and a listing of its own; indexing it
    # here would put chat titles on this disk for nothing.
    assert conv_store.lookup("anonymous", "c-account") is None


def test_an_anonymous_turn_is_indexed(monkeypatch):
    monkeypatch.setattr(auth, "is_authenticated", lambda: False)
    session = _FakeSession("c-anon", "device-9", None)
    session.last_title = "Título"
    main._remember_conversation("anonymous", session)
    row = conv_store.lookup("anonymous", "c-anon")
    assert row["device_id"] == "device-9" and row["title"] == "Título"


def test_a_broken_index_does_not_break_a_turn_that_already_answered(monkeypatch):
    monkeypatch.setattr(auth, "is_authenticated", lambda: False)

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(conv_store, "record", boom)
    # The reply is already on its way to the caller by the time this runs.
    main._remember_conversation("anonymous", _FakeSession("c", "d", None))
