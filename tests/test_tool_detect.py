"""Detection and repair, ported from llm-libre's tool_emulator.

Every case here used to cost a repair round trip -- one more message off an
anonymous hourly allowance -- or, worse, produced the wrong call.
"""
import json

import pytest

import tool_calls as tc
import tool_detect as td

WEATHER = {"name": "get_weather",
           "parameters": {"type": "object",
                          "properties": {"city": {"type": "string"},
                                         "days": {"type": "integer"},
                                         "unit": {"type": "string",
                                                  "enum": ["celsius", "fahrenheit"]}},
                          "required": ["city"]}}
EMAIL = {"name": "send_email",
         "parameters": {"type": "object",
                        "properties": {"to": {"type": "string"}},
                        "required": ["to"]}}
FUNCS = [WEATHER, EMAIL]
NAMES = {"get_weather", "send_email"}


def parse(text, names=NAMES, funcs=FUNCS):
    return td.parse_tool_calls(text, names, funcs)


# ---------------------------------------------------------------------------
# 1. Soundness: the allow-list is the whole defence against a false positive
# ---------------------------------------------------------------------------

def test_a_name_nobody_declared_stays_text():
    assert parse('{"name": "launch_missiles", "arguments": {}}') is None


def test_no_declared_functions_can_never_produce_a_call():
    assert parse('{"name": "get_weather", "arguments": {}}', names=set()) is None


def test_prose_about_json_is_not_a_call():
    assert parse("The config looks like a name/arguments pair in most setups.") is None


# ---------------------------------------------------------------------------
# 2. tool_choice is enforced, not merely prompted
# ---------------------------------------------------------------------------

def test_allowed_names_narrows_to_the_forced_function():
    forced = {"type": "function", "function": {"name": "get_weather"}}
    assert td.allowed_names(FUNCS, forced) == {"get_weather"}


def test_allowed_names_reads_the_flat_forced_spelling_too():
    assert td.allowed_names(FUNCS, {"type": "function", "name": "send_email"}) == {"send_email"}


def test_allowed_names_reads_an_allowed_tools_subset():
    choice = {"type": "allowed_tools",
              "tools": [{"type": "function", "function": {"name": "send_email"}}]}
    assert td.allowed_names(FUNCS, choice) == {"send_email"}


def test_tool_choice_none_authorises_nothing():
    assert td.allowed_names(FUNCS, "none") == set()


def test_a_forced_choice_rejects_a_call_to_another_declared_function():
    # The hole this closes: before, the forced name lived only in the prompt.
    # A model that ignored it and called a different DECLARED function produced
    # a call that validated cleanly, and the caller ran the wrong function.
    forced = {"type": "function", "function": {"name": "get_weather"}}
    names = td.allowed_names(FUNCS, forced)

    assert parse('{"name": "send_email", "arguments": {"to": "a@b.c"}}', names) is None
    assert parse('{"name": "get_weather", "arguments": {"city": "Lima"}}', names)


def test_demands_call_variants():
    assert td.demands_call("required")
    assert td.demands_call({"type": "function", "function": {"name": "x"}})
    assert td.demands_call({"type": "allowed_tools", "mode": "required", "tools": []})
    assert not td.demands_call("auto")
    assert not td.demands_call("none")
    assert not td.demands_call(None)


# ---------------------------------------------------------------------------
# 3. Coverage: the dialects a prompted model actually emits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,city", [
    ('{"name":"get_weather","arguments":{"city":"Lima"}}', "Lima"),
    ('```json\n{"name":"get_weather","arguments":{"city":"Lima"}}\n```', "Lima"),
    ('<tool_call>{"name":"get_weather","arguments":{"city":"Lima"}}</tool_call>', "Lima"),
    ('[TOOL_CALLS][{"name":"get_weather","arguments":{"city":"Lima"}}]', "Lima"),
    ("{'name': 'get_weather', 'arguments': {'city': 'Lima'}}", "Lima"),
    ('{"name":"get_weather","input":{"city":"Lima"}}', "Lima"),
    ('{"name":"get_weather","parameters":{"city":"Lima"}}', "Lima"),
    ('{"action":"get_weather","action_input":{"city":"Lima"}}', "Lima"),
    ('{"tool":"get_weather","tool_input":{"city":"Lima"}}', "Lima"),
    ('{"function_call":{"name":"get_weather","arguments":"{\\"city\\":\\"Lima\\"}"}}', "Lima"),
    ('{"tool_use":{"name":"get_weather","arguments":{"city":"Lima"}}}', "Lima"),
    ('{"tool_calls":[{"name":"get_weather","arguments":{"city":"Lima"}}]}', "Lima"),
    ('{"name":"functions.get_weather","arguments":{"city":"Lima"}}', "Lima"),
    ('{"name":"GET_WEATHER","arguments":{"city":"Lima"}}', "Lima"),
    ('{"name":"get_weather","arguments":{"city":"Lima",}}', "Lima"),
])
def test_every_dialect_reads_as_the_same_call(text, city):
    calls = parse(text)
    assert calls == [{"name": "get_weather", "arguments": {"city": city}}]


def test_a_zero_argument_call_needs_no_arguments_key():
    assert parse('{"name":"get_weather"}') == [{"name": "get_weather", "arguments": {}}]


def test_malformed_argument_json_keeps_the_call_with_empty_arguments():
    # The name is valid and the model clearly meant to call. A call the caller
    # can reject beats prose it will misread as an answer.
    assert parse('{"name":"get_weather","arguments":"{not json"}') == [
        {"name": "get_weather", "arguments": {}}]


def test_one_tag_per_parallel_call():
    text = ('<tool_call>{"name":"get_weather","arguments":{"city":"Lima"}}</tool_call>'
            '<tool_call>{"name":"get_weather","arguments":{"city":"Quito"}}</tool_call>')
    assert [c["arguments"]["city"] for c in parse(text)] == ["Lima", "Quito"]


def test_jsonl_style_adjacent_objects_batch():
    text = ('{"name":"get_weather","arguments":{"city":"Lima"}}\n'
            '{"name":"get_weather","arguments":{"city":"Quito"}}')
    assert [c["arguments"]["city"] for c in parse(text)] == ["Lima", "Quito"]


def test_a_case_ambiguous_name_names_neither():
    # Two declared names differing only by case make a third spelling ambiguous.
    ambiguous = {"Run", "run"}
    assert td.parse_tool_calls('{"name":"RUN","arguments":{}}', ambiguous) is None


# ---------------------------------------------------------------------------
# 4. Where the false positives live
# ---------------------------------------------------------------------------

def test_a_mixed_array_of_calls_and_data_is_data():
    text = ('[{"name":"get_weather","arguments":{"city":"Lima"}},'
            ' {"temperature": 21}]')
    assert parse(text) is None


def test_a_schema_quoted_inside_a_clarifying_question_is_not_a_call():
    text = ('Its schema is {"name": "get_weather"}, but which city did you '
            'mean? I can look up several if you tell me which ones you care '
            'about, since the function takes one city at a time.')
    assert parse(text) is None


def test_the_real_call_wins_over_an_illustrative_example():
    text = ('Here is how I would call it:\n'
            '```json\n{"name":"get_weather","arguments":{"city":"EXAMPLE"}}\n```\n'
            'Now the real call: {"name":"get_weather","arguments":{"city":"Lima"}}')
    assert parse(text)[0]["arguments"]["city"] == "Lima"


def test_a_draft_inside_a_reasoning_block_is_not_the_answer():
    text = ('<think>{"name":"send_email","arguments":{"to":"draft@x.com"}}</think>'
            '{"name":"get_weather","arguments":{"city":"Lima"}}')
    assert parse(text) == [{"name": "get_weather", "arguments": {"city": "Lima"}}]


def test_an_unclosed_reasoning_block_is_scratchpad_to_the_end():
    text = '<think>maybe {"name":"get_weather","arguments":{"city":"Lima"}}'
    assert parse(text) is None


def test_braces_inside_string_values_do_not_close_the_object():
    text = '{"name":"get_weather","arguments":{"city":"Lima } Peru"}}'
    assert parse(text)[0]["arguments"]["city"] == "Lima } Peru"


# ---------------------------------------------------------------------------
# 5. Argument repair: lossless or identity
# ---------------------------------------------------------------------------

def test_a_stringified_integer_is_re_read_as_one():
    # This is the round trip the repair pass used to cost.
    calls = parse('{"name":"get_weather","arguments":{"city":"Lima","days":"5"}}')
    assert calls[0]["arguments"]["days"] == 5


def test_python_only_numeric_spellings_are_not_json_numbers():
    # int("1_000") succeeds in Python; "1_000" is not a JSON number, so
    # accepting it would not be a lossless re-reading of what the model wrote.
    calls = parse('{"name":"get_weather","arguments":{"city":"Lima","days":"1_000"}}')
    assert calls[0]["arguments"]["days"] == "1_000"


def test_a_whole_float_folds_to_the_canonical_integer():
    assert td.coerce_value(5.0, {"type": "integer"}) == 5


def test_a_union_leaves_an_already_satisfying_value_alone():
    # "5" under ["integer", "string"] IS the string.
    assert td.coerce_value("5", {"type": ["integer", "string"]}) == "5"


def test_a_unique_case_insensitive_enum_value_is_repaired():
    calls = parse('{"name":"get_weather","arguments":{"city":"Lima","unit":"Celsius"}}')
    assert calls[0]["arguments"]["unit"] == "celsius"


def test_coercion_recurses_into_nested_objects_and_array_items():
    schema = {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "qty": {"type": "integer"}}}}}}
    out = td.coerce_value({"items": [{"qty": "2"}, {"qty": "3"}]}, schema)
    assert out == {"items": [{"qty": 2}, {"qty": 3}]}


def test_an_undeclared_parameter_is_never_touched():
    calls = parse('{"name":"get_weather","arguments":{"city":"Lima","extra":"7"}}')
    assert calls[0]["arguments"]["extra"] == "7"


def test_a_boolean_is_not_folded_into_a_number():
    # JSON Schema draws the line Python does not, and folding True into 1 would
    # rewrite a flag the model may have meant literally.
    assert td.coerce_value(True, {"type": "integer"}) is True


def test_coercion_is_idempotent():
    schema = {"type": "object", "properties": {"days": {"type": "integer"}}}
    once = td.coerce_value({"days": "5"}, schema)
    assert td.coerce_value(once, schema) == once


# ---------------------------------------------------------------------------
# 6. Totality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junk", [
    "", "   ", "{", "}" * 50, "[" * 200, '{"name":', "\x00\x01",
    '{"name":"get_weather","arguments":' + "{" * 100,
    "<think>" * 40, "```" * 30, "[TOOL_CALLS]" * 20,
])
def test_the_parser_never_raises_and_never_escapes_the_allow_list(junk):
    out = parse(junk)
    assert out is None or all(c["name"] in NAMES for c in out)


def test_a_storm_of_unclosed_openers_gives_up_in_the_safe_direction():
    # Bounded work: a missed call, never an invented one, and never O(n²).
    text = "{" * 10_000 + '{"name":"get_weather","arguments":{"city":"Lima"}}'
    out = parse(text)
    assert out is None or out[0]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# 7. How it composes with our own markers
# ---------------------------------------------------------------------------

def test_our_markers_still_decide_outright():
    assert tc.parse_envelope(tc.NO_CALL, NAMES) == ([], [])
    status, notes = tc.parse_envelope(
        tc.NEED_INFO + '{"function":"send_email","missing":["to"]}', NAMES)
    assert status == "NEED_INFO"
    assert notes[0]["missing"] == ["to"]


def test_a_reply_in_another_dialect_no_longer_costs_a_repair():
    # No marker, no {"calls": …} envelope -- just the call, in a shape the
    # model was never asked for. This used to come back None and spend a
    # second upstream message.
    calls, notes = tc.parse_envelope(
        '```json\n{"name":"get_weather","arguments":{"city":"Lima"}}\n```',
        NAMES, FUNCS)
    assert calls[0]["arguments"]["city"] == "Lima"
    assert "dialect" in notes


def test_the_documented_envelope_is_still_the_fast_path():
    raw = tc.SENTINEL + '{"calls":[{"name":"get_weather","arguments":{"city":"Lima"}}]}'
    calls, notes = tc.parse_envelope(raw, NAMES, FUNCS)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Lima"}}]
    assert "dialect" not in notes


def test_genuine_prose_is_still_unreadable_rather_than_guessed_at():
    calls, notes = tc.parse_envelope("La capital de Francia es París.", NAMES, FUNCS)
    assert calls is None
    assert "invalid-json" in notes
