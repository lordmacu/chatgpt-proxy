import httpx
import pytest
from fastapi.testclient import TestClient

import capabilities as cap
import main

MP3 = b"ID3\x03\x00\x00\x00fake-mp3-bytes"


@pytest.fixture
def synthesized(monkeypatch):
    """Stand in for the chat turn + /backend-api/synthesize round trip."""
    cap.reset()
    monkeypatch.setattr(cap, "snapshot",
                        lambda **kw: cap.AccountState(mode="account", plan="go"))

    async def fake_synthesize(text, voice, fmt, model):
        return main.Synthesized(audio=MP3, media_type="audio/mpeg", text=text,
                                exact_match=True, voice=voice, format=fmt,
                                conversation_id="conv-1", message_id="msg-1")

    monkeypatch.setattr(main, "_synthesize", fake_synthesize)
    return MP3


def test_audio_speech_returns_raw_bytes(synthesized):
    with TestClient(main.app) as c:
        r = c.post("/v1/audio/speech", json={"input": "hola", "voice": "juniper"})
    assert r.status_code == 200
    assert r.content == MP3
    assert r.headers["content-type"].startswith("audio/mpeg")


def test_the_metadata_travels_in_headers_not_in_the_body(synthesized):
    # `exact_match` is a real signal -- this flow makes the model echo the input,
    # and it sometimes alters it -- so it is kept, just out of the body, where
    # the OpenAI contract says only audio goes.
    with TestClient(main.app) as c:
        r = c.post("/v1/audio/speech", json={"input": "hola"})
    assert r.headers["X-Exact-Match"] == "true"
    assert r.headers["X-Conversation-Id"] == "conv-1"
    assert r.headers["X-Message-Id"] == "msg-1"
    assert r.headers["X-Audio-Url"].endswith(".mp3")


def test_the_native_endpoint_still_returns_the_json_form(synthesized):
    with TestClient(main.app) as c:
        r = c.post("/chatgpt/audio/speech", json={"input": "hola"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hola"
    assert body["exact_match"] is True
    assert body["url"].endswith(".mp3")
