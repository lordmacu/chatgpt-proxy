"""The Python detector must agree with the Dart port, case for case.

The two implementations are ports of each other, and a port drifts silently:
each side keeps passing its own tests while quietly answering differently.
tests/fixtures/detector_conformance.json is the shared truth — the same file
lives in the chatgpt_free repo and pins its detector too — so either side
drifting fails here without the other language being installed.
"""
import json
import pathlib

import pytest

import tool_detect as td

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "detector_conformance.json"
_SPEC = json.loads(_FIXTURE.read_text())
_NAMES = set(_SPEC["names"])
_FUNCS = _SPEC["functions"]


@pytest.mark.parametrize(
    "case", _SPEC["cases"], ids=[f"case{i}" for i in range(len(_SPEC["cases"]))])
def test_matches_the_shared_expectation(case):
    assert td.parse_tool_calls(case["input"], _NAMES, _FUNCS) == case["expected"]


def test_the_corpus_actually_exercises_detection():
    # A corpus that detected nothing would pass vacuously forever.
    detected = sum(1 for c in _SPEC["cases"] if c["expected"])
    assert detected >= 30, f"only {detected} of {len(_SPEC['cases'])} produce calls"
