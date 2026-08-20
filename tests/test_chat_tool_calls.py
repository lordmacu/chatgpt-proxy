import json

import pytest
from fastapi.testclient import TestClient

import main
import tool_calls as tc
from main import Message, _resolve_messages

client = TestClient(main.app)

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}

CALL = {"id": "call_x1", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Bogot\\u00e1"}'}}


# --- the round trip that used to be dropped on the floor ---------------------

def test_a_trailing_tool_result_is_recognised_as_the_second_leg():
    msgs = [Message(role="user", content="¿Clima en Bogotá?"),
            Message(role="assistant", content="", tool_calls=[CALL]),
            Message(role="tool", tool_call_id="call_x1", content='{"temp":14}')]
    _sys, text, _f, _i, followup = _resolve_messages(msgs, {})
    assert followup is True
    assert '"temp":14' in text
    assert "get_weather" in text                 # the result says what it answers
    assert "¿Clima en Bogotá?" in text           # and the question is still context


def test_the_assistants_call_survives_into_the_history():
    msgs = [Message(role="user", content="¿Clima en Lima?"),
            Message(role="assistant", content="", tool_calls=[CALL]),
            Message(role="tool", tool_call_id="call_x1", content='{"temp":19}'),
            Message(role="assistant", content="Hace 19°."),
            Message(role="user", content="¿Y en Quito?")]
    _sys, text, _f, _i, followup = _resolve_messages(msgs, {})
    assert followup is False
    assert "get_weather" in text and "Tool result" in text
    assert text.endswith("¿Y en Quito?")


def test_an_ordinary_conversation_is_untouched():
    msgs = [Message(role="system", content="Sé breve."),
            Message(role="user", content="Hola"),
            Message(role="assistant", content="¿Qué tal?"),
            Message(role="user", content="Bien")]
    system, text, _f, _i, followup = _resolve_messages(msgs, {})
    assert system == "Sé breve." and followup is False
    assert "User: Hola" in text and text.endswith("Bien")


# --- /v1/chat/completions routing -------------------------------------------

@pytest.fixture
def extraction(monkeypatch):
    """Capture what the endpoint asks the extractor for, and script the answer."""
    seen = {}

    async def fake_extract(functions, text, tool_choice="auto", model=None, verify=False):
        seen.update(functions=functions, text=text, tool_choice=tool_choice, verify=verify)
        return seen.get("result") or tc.ToolExtraction("calls", tool_calls=[CALL])

    monkeypatch.setattr(main._tc, "extract", fake_extract)
    return seen


@pytest.fixture
def no_upstream(monkeypatch):
    """Make the normal chat path answer without touching the network.

    Records what was actually sent upstream, so a test can assert on the prompt
    and not just on the reply.
    """
    sent = {}

    class FakeSession:
        last_clean_text = "respuesta normal"
        last_search_queries: list = []
        last_citations: list = []
        last_widgets: list = []
        last_images: list = []
        _pending_image_ids: list = []

        async def stream_message(self, message, *a, **k):
            sent["message"] = message
            sent["kwargs"] = k
            yield "respuesta normal"

    class FakePool:
        _pool: dict = {}

        async def get(self, *a, **k):
            return "k", FakeSession()

    monkeypatch.setattr(main, "_user_pool", lambda uid: FakePool())
    return sent


def test_a_custom_function_comes_back_as_tool_calls(extraction):
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content": "¿Clima en Bogotá?"}]})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Bogotá"}
    assert r.headers["X-Proxy-Tool-Extraction"] == "calls"
    assert extraction["functions"][0]["name"] == "get_weather"


def test_streaming_emits_tool_call_deltas_and_the_right_finish_reason(extraction):
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "stream": True, "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content": "¿Clima en Bogotá?"}]})
    chunks = [json.loads(l[6:]) for l in r.text.splitlines()
              if l.startswith("data: ") and not l.endswith("[DONE]")]
    deltas = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert deltas and deltas[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_no_call_falls_through_to_a_normal_answer(extraction, no_upstream):
    extraction["result"] = tc.ToolExtraction("no_call", requests=1)
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content": "¿Capital de Francia?"}]})
    assert r.json()["choices"][0]["message"]["content"] == "respuesta normal"
    assert r.headers["X-Proxy-Tool-Extraction"] == "no_call"


def test_need_info_answers_in_prose_and_names_the_missing_parameter(extraction, no_upstream):
    extraction["result"] = tc.ToolExtraction(
        "need_info", need_info={"function": "get_weather", "missing": ["city"]})
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content": "¿Qué clima hace?"}]})
    assert r.json()["choices"][0]["message"]["content"] == "respuesta normal"
    assert json.loads(r.headers["X-Proxy-Tool-Need-Info"])["missing"] == ["city"]
    # Left to itself the model invents the missing argument -- measured: asked
    # for "the temperature" with no city it answered for Bogotá, off web search.
    assert "does not state city" in no_upstream["message"]
    assert "Ask the user" in no_upstream["message"]
    assert no_upstream["kwargs"]["force_use_search"] is False


def test_tool_choice_none_skips_extraction_entirely(extraction, no_upstream):
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "tools": [WEATHER_TOOL], "tool_choice": "none",
        "messages": [{"role": "user", "content": "¿Clima en Bogotá?"}]})
    assert r.json()["choices"][0]["message"]["content"] == "respuesta normal"
    assert "functions" not in extraction


def test_tool_emulation_false_restores_the_old_behaviour(extraction, no_upstream):
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "tools": [WEATHER_TOOL], "tool_emulation": False,
        "messages": [{"role": "user", "content": "¿Clima en Bogotá?"}]})
    assert r.json()["choices"][0]["message"]["content"] == "respuesta normal"
    assert "functions" not in extraction


def test_a_builtin_tool_name_is_not_extracted_but_still_flips_its_mode(extraction, no_upstream):
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "tools": [{"type": "function", "function": {"name": "web_search"}}],
        "messages": [{"role": "user", "content": "noticias de hoy"}]})
    assert r.status_code == 200
    assert "functions" not in extraction


def test_the_second_leg_answers_instead_of_calling_again(extraction, no_upstream):
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "tools": [WEATHER_TOOL], "messages": [
            {"role": "user", "content": "¿Clima en Bogotá?"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "call_x1", "content": '{"temp":14}'}]})
    assert r.json()["choices"][0]["message"]["content"] == "respuesta normal"
    assert "functions" not in extraction, "the results are in; extracting again would loop"


# --- /v1/tool-calls ---------------------------------------------------------

def test_the_dedicated_endpoint_returns_calls(extraction):
    r = client.post("/v1/tool-calls", json={
        "tools": [WEATHER_TOOL], "input": "¿Clima en Bogotá?"})
    body = r.json()
    assert body["status"] == "calls"
    assert body["tool_calls"][0]["function"]["name"] == "get_weather"
    assert body["usage"]["upstream_requests"] == 1


def test_the_dedicated_endpoint_reports_what_is_missing(extraction):
    extraction["result"] = tc.ToolExtraction(
        "need_info", need_info={"function": "get_weather", "missing": ["city"]})
    body = client.post("/v1/tool-calls", json={
        "tools": [WEATHER_TOOL], "input": "¿Qué clima hace?"}).json()
    assert body["status"] == "need_info"
    assert body["need_info"]["missing"] == ["city"]


def test_the_dedicated_endpoint_accepts_messages_instead_of_input(extraction):
    client.post("/v1/tool-calls", json={
        "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content": "¿Clima en Quito?"}]})
    assert "Quito" in extraction["text"]


def test_verify_is_passed_through(extraction):
    client.post("/v1/tool-calls", json={
        "tools": [WEATHER_TOOL], "input": "algo denso", "verify": True})
    assert extraction["verify"] is True


def test_builtin_only_tools_are_rejected_rather_than_silently_doing_nothing():
    r = client.post("/v1/tool-calls", json={
        "tools": [{"type": "function", "function": {"name": "web_search"}}],
        "input": "noticias"})
    assert r.status_code == 400


def test_an_empty_request_is_rejected():
    assert client.post("/v1/tool-calls", json={"tools": [WEATHER_TOOL], "input": "  "}).status_code == 400


def test_an_assistant_turn_with_null_content_is_accepted():
    # What every OpenAI SDK echoes back on the second leg. Rejecting it 422s the
    # round trip before the tool result is ever read.
    msgs = [Message(role="user", content="¿Clima?"),
            Message(role="assistant", content=None, tool_calls=[CALL]),
            Message(role="tool", tool_call_id="call_x1", content='{"temp":14}')]
    _sys, text, _f, _i, followup = _resolve_messages(msgs, {})
    assert followup is True and "get_weather" in text


def test_the_round_trip_survives_a_null_content_assistant_over_http(monkeypatch):
    async def never(*a, **k):
        raise AssertionError("the second leg must not extract again")
    monkeypatch.setattr(main._tc, "extract", never)

    class FakeSession:
        last_clean_text = "En Bogotá hace 14 °C."
        last_search_queries: list = []
        last_citations: list = []
        last_widgets: list = []
        last_images: list = []
        _pending_image_ids: list = []

        async def stream_message(self, *a, **k):
            yield "En Bogotá hace 14 °C."

    class FakePool:
        _pool: dict = {}

        async def get(self, *a, **k):
            return "k", FakeSession()

    monkeypatch.setattr(main, "_user_pool", lambda uid: FakePool())
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "tools": [WEATHER_TOOL], "messages": [
            {"role": "user", "content": "¿Clima en Bogotá?"},
            {"role": "assistant", "content": None, "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "call_x1", "content": '{"temperature":14}'}]})
    assert r.status_code == 200
    assert "14" in r.json()["choices"][0]["message"]["content"]
