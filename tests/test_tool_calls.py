import json

import tool_calls as tc

WEATHER = {"name": "get_weather", "description": "Weather for a city.",
           "parameters": {"type": "object", "additionalProperties": False,
                          "properties": {"city": {"type": "string"},
                                         "unit": {"type": "string",
                                                  "enum": ["celsius", "fahrenheit"]}},
                          "required": ["city"]}}
ORDER = {"name": "create_order", "description": "Create an order.",
         "parameters": {"type": "object", "properties": {
             "items": {"type": "array", "items": {"type": "object", "properties": {
                 "sku": {"type": "string"}, "qty": {"type": "integer"}},
                 "required": ["sku", "qty"]}}},
             "required": ["items"]}}
FUNCS = [WEATHER, ORDER]

# The allow-list detection is gated on: the functions declared in THIS
# request, after tool_choice has narrowed them.
NAMES = {"get_weather", "create_order"}


# --- which tools this proxy has to emulate ---------------------------------

def test_builtin_names_are_left_to_the_backend_flags():
    tools = [{"type": "function", "function": {"name": "web_search"}},
             {"type": "function", "function": {"name": "canvas"}}]
    assert tc.custom_functions(tools, {"web_search", "canvas"}) == []


def test_a_caller_function_is_picked_up_with_its_schema():
    tools = [{"type": "function", "function": WEATHER}]
    out = tc.custom_functions(tools, {"web_search"})
    assert [f["name"] for f in out] == ["get_weather"]
    assert out[0]["parameters"]["required"] == ["city"]


def test_a_function_without_a_schema_still_yields_an_object_schema():
    out = tc.custom_functions([{"type": "function", "function": {"name": "ping"}}], set())
    assert out[0]["parameters"] == {"type": "object", "properties": {}}


# --- the prompt contract ----------------------------------------------------

def test_auto_offers_the_no_tool_escape_hatch():
    p = tc.build_prompt(FUNCS, "hola", "auto")
    assert tc.NO_CALL in p and tc.NEED_INFO in p


def test_required_removes_the_no_tool_option_entirely():
    # Leaving option B listed and merely forbidding it below was measured at 0/2:
    # the model took the escape hatch anyway.
    p = tc.build_prompt(FUNCS, "hola", "required")
    assert f"B) {tc.NO_CALL}" not in p
    assert "forbidden" in p


def test_a_named_tool_choice_pins_that_function():
    p = tc.build_prompt(FUNCS, "Bogotá",
                        {"type": "function", "function": {"name": "get_weather"}})
    assert 'You MUST call the function "get_weather"' in p
    assert f"B) {tc.NO_CALL}" not in p


def test_the_prompt_denies_the_model_its_own_knowledge():
    # The single line that took "weather in Lima and Quito" from 0/5 to 4/4.
    assert "NO knowledge and NO live data" in tc.build_prompt(FUNCS, "x", "auto")


# --- parsing ----------------------------------------------------------------

def test_a_clean_envelope_parses():
    raw = tc.SENTINEL + '{"calls":[{"name":"get_weather","arguments":{"city":"Bogotá"}}]}'
    calls, notes = tc.parse_envelope(raw, NAMES)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Bogotá"}}]
    assert notes == []


def test_markdown_fences_are_survivable():
    raw = "```json\n" + tc.SENTINEL + '{"calls":[{"name":"get_weather","arguments":{"city":"Lima"}}]}\n```'
    calls, notes = tc.parse_envelope(raw, NAMES)
    assert calls[0]["arguments"]["city"] == "Lima"
    assert "fenced" in notes


def test_a_repeated_marker_takes_the_first_envelope_and_flags_it():
    body = '{"calls":[{"name":"get_weather","arguments":{"city":"Quito"}}]}'
    calls, notes = tc.parse_envelope(tc.SENTINEL + body + tc.SENTINEL + body, NAMES)
    assert len(calls) == 1
    assert "duplicate-marker" in notes


def test_prose_before_the_marker_is_recovered_and_flagged():
    raw = "Claro, aquí tienes:\n" + tc.SENTINEL + '{"calls":[{"name":"get_weather","arguments":{"city":"Cali"}}]}'
    calls, notes = tc.parse_envelope(raw, NAMES)
    assert calls[0]["arguments"]["city"] == "Cali"
    assert "prose-before" in notes


def test_no_tool_is_an_empty_call_list_not_a_failure():
    assert tc.parse_envelope(tc.NO_CALL, NAMES) == ([], [])


def test_need_info_carries_the_missing_parameters():
    raw = tc.NEED_INFO + '{"function":"search_flights","missing":["origin","date"]}'
    status, notes = tc.parse_envelope(raw, NAMES)
    assert status == "NEED_INFO"
    assert notes[0]["missing"] == ["origin", "date"]


def test_plain_prose_is_not_mistaken_for_a_call():
    calls, notes = tc.parse_envelope("La capital de Francia es París.", NAMES)
    assert calls is None and "no-marker" in notes


def test_a_json_answer_without_the_marker_is_not_a_call():
    # A user asking for a JSON example must not trip the extractor.
    calls, _ = tc.parse_envelope('{"calls":[{"name":"whatever"}]}', NAMES)
    assert calls is None


def test_broken_json_after_the_marker_reports_invalid():
    calls, notes = tc.parse_envelope(tc.SENTINEL + '{"calls":[{"name":', NAMES)
    assert calls is None and "invalid-json" in notes


# --- validation -------------------------------------------------------------

def test_a_correct_call_validates():
    assert tc.validate_calls([{"name": "get_weather",
                               "arguments": {"city": "Bogotá", "unit": "celsius"}}], FUNCS) == []


def test_a_missing_required_parameter_is_caught():
    errs = tc.validate_calls([{"name": "get_weather", "arguments": {}}], FUNCS)
    assert errs and "city" in errs[0]


def test_a_value_outside_the_enum_is_caught():
    errs = tc.validate_calls([{"name": "get_weather",
                               "arguments": {"city": "Lima", "unit": "kelvin"}}], FUNCS)
    assert errs


def test_an_invented_parameter_is_caught():
    errs = tc.validate_calls([{"name": "get_weather",
                               "arguments": {"city": "Lima", "wind": 5}}], FUNCS)
    assert errs


def test_validation_reaches_inside_arrays_of_objects():
    errs = tc.validate_calls([{"name": "create_order",
                               "arguments": {"items": [{"sku": "A", "qty": "dos"}]}}], FUNCS)
    assert errs and "qty" in errs[0]


def test_an_unknown_function_is_rejected():
    assert tc.validate_calls([{"name": "launch_missiles", "arguments": {}}], FUNCS)


def test_the_fallback_walker_matches_on_the_nested_case():
    # jsonschema is optional; the degraded path must still catch the same bug.
    assert tc._walk({"items": [{"sku": "A", "qty": "dos"}]}, ORDER["parameters"])


# --- OpenAI shape -----------------------------------------------------------

def test_arguments_are_serialised_as_a_json_string():
    out = tc.to_openai_tool_calls([{"name": "get_weather", "arguments": {"city": "Bogotá"}}])
    assert out[0]["type"] == "function"
    assert out[0]["id"].startswith("call_")
    assert isinstance(out[0]["function"]["arguments"], str)
    assert json.loads(out[0]["function"]["arguments"]) == {"city": "Bogotá"}


def test_each_call_gets_its_own_id():
    out = tc.to_openai_tool_calls([{"name": "get_weather", "arguments": {"city": "A"}},
                                   {"name": "get_weather", "arguments": {"city": "B"}}])
    assert out[0]["id"] != out[1]["id"]
