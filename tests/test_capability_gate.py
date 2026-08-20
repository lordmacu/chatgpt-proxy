from fastapi.testclient import TestClient

import capabilities as cap
import main


def _client(monkeypatch, state):
    cap.reset()
    monkeypatch.setattr(cap, "snapshot", lambda **kw: state)
    return TestClient(main.app)


def test_image_generation_on_a_free_plan_is_501_not_503(monkeypatch):
    # 501 says "this proxy, deliberately, does not do this". A 503 said "it
    # broke", so the gateway retried, accumulated suspicion and failed over --
    # for a capability that was never going to work on this plan.
    with _client(monkeypatch, cap.AccountState(mode="account", plan="free")) as c:
        r = c.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 501


def test_audio_speech_on_an_anonymous_session_is_501(monkeypatch):
    with _client(monkeypatch, cap.AccountState(mode="anonymous")) as c:
        r = c.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 501


def test_the_501_body_names_the_capability(monkeypatch):
    with _client(monkeypatch, cap.AccountState(mode="anonymous")) as c:
        r = c.post("/v1/audio/transcriptions", files={"file": ("a.mp3", b"x", "audio/mpeg")})
    assert r.status_code == 501
    assert "audio_transcription" in r.text


def test_translate_is_never_gated(monkeypatch):
    # It works anonymously and spends no chat message; gating it would be wrong.
    #
    # /v1/translate is intentionally ungated, which means -- unlike the gated
    # endpoints above, where require_capability short-circuits before any
    # backend call -- this handler runs all the way through to
    # main._backend_post. Left unmocked that would place a real request
    # against the live ChatGPT backend using whatever local credentials this
    # machine has, which is exactly what "no test may reach the network"
    # rules out. Stubbing _backend_post keeps the assertion (never 501) while
    # keeping the test hermetic.
    class _FakeResponse:
        status_code = 200
        text = '{"text": "hola"}'

        def json(self):
            return {"text": "hola"}

    async def fake_backend_post(request, path, json_body):
        return _FakeResponse()

    monkeypatch.setattr(main, "_backend_post", fake_backend_post)
    with _client(monkeypatch, cap.AccountState(mode="anonymous")) as c:
        r = c.post("/v1/translate", json={"text": "hi", "target": "es"})
    assert r.status_code != 501
