"""Turning json_mode off has to say so.

json_mode is a prompt instruction, not an API flag -- response_format is inert
on this backend -- so the instruction stays in the conversation history the
model reads. Measured against the live backend on 2026-08-20, before the fix:

    turn 1, json_mode=True  -> {"frutas":["Manzana","Banano","Mango"]}
    turn 2, json_mode=False -> {"verduras":["Zanahoria","Brócoli","Espinaca"]}

Flipping the flag retracted nothing, so the toggle looked broken. After the
fix, turn 2 answered "Zanahoria, brócoli y espinaca."

These run stream_message for real and read the body it was about to POST, so
they pin the shipped assembly rather than a copy of it. No quota is spent.
"""
import json

import pytest

import auth
import chatgpt_client as cc

RETRACTION = "Stop answering in JSON"
INSTRUCTION = "valid JSON only"


class _FakeStream:
    """Enough of an httpx streaming response for stream_message to finish."""

    status_code = 200
    headers = {"content-type": "text/event-stream"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        yield "data: [DONE]"

    async def aread(self):
        return b""

    def raise_for_status(self):
        pass


class _RecordingClient:
    """Captures every body posted, and answers with an empty stream."""

    def __init__(self):
        self.bodies = []

    def stream(self, method, url, **kw):
        self.bodies.append(json.loads(kw["content"]))
        return _FakeStream()

    async def aclose(self):
        pass


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(auth, "is_authenticated", lambda: False)
    s = cc.ChatGPTSession()
    recorder = _RecordingClient()
    s.client = recorder
    # ensure_ready reaches the vendor for a sentinel token; nothing here needs one.
    async def ready(self=None):
        return None
    monkeypatch.setattr(cc.ChatGPTSession, "ensure_ready", ready)
    s._recorder = recorder
    return s


async def _turn(session, message, json_mode):
    async for _ in session.stream_message(message, json_mode=json_mode):
        pass
    body = session._recorder.bodies[-1]
    return body["messages"][0]["content"]["parts"][0]


@pytest.mark.asyncio
async def test_a_fresh_session_has_nothing_to_retract(session):
    sent = await _turn(session, "hola", json_mode=False)

    assert sent == "hola"
    assert session._json_mode_used is False


@pytest.mark.asyncio
async def test_asking_for_json_sends_the_instruction(session):
    sent = await _turn(session, "tres frutas", json_mode=True)

    assert INSTRUCTION in sent
    assert session._json_mode_used is True


@pytest.mark.asyncio
async def test_turning_it_off_retracts_it_explicitly(session):
    await _turn(session, "tres frutas", json_mode=True)
    sent = await _turn(session, "tres verduras", json_mode=False)

    assert RETRACTION in sent
    assert "tres verduras" in sent
    assert INSTRUCTION not in sent


@pytest.mark.asyncio
async def test_the_retraction_is_sent_once_not_on_every_later_turn(session):
    # Repeating it would spend context on a conversation that already moved on.
    await _turn(session, "tres frutas", json_mode=True)
    await _turn(session, "tres verduras", json_mode=False)
    sent = await _turn(session, "tres granos", json_mode=False)

    assert RETRACTION not in sent
    assert sent == "tres granos"


@pytest.mark.asyncio
async def test_json_can_be_asked_for_again_after_a_retraction(session):
    await _turn(session, "tres frutas", json_mode=True)
    await _turn(session, "tres verduras", json_mode=False)
    sent = await _turn(session, "tres granos", json_mode=True)

    assert INSTRUCTION in sent
    assert RETRACTION not in sent
