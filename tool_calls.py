"""Custom function calling, emulated on a backend that has none.

The conversation protocol has no place to DECLARE a caller's functions: the only
tool flags it understands (force_use_search / force_use_tools / force_use_canvas)
switch on tools that run inside OpenAI and come back as prose. So `tool_calls`
are produced the same way `json_object` already is -- by prompt -- but through a
dedicated, STATELESS request instead of the conversation itself.

Why a separate request and not the chat turn (measured 2026-08-20, anonymous):
  - Manifest in the system prompt, inside a live conversation: "weather in Lima
    and Quito" produced the envelope 0/5 times. The model answered from its own
    web search instead (the reply carried search citations), and
    force_use_search=False did not reliably stop it -- search still ran ~1/5.
  - Manifest in the USER turn of a throwaway session, plus the rule that the
    model has no knowledge of anything a declared function covers: 4/4 on that
    same prompt, 25/28 over a 14-case battery, 0 false positives on 8 prompts
    that needed no function at all.
The difference is not the wording of the schema, it is WHERE the manifest sits
and whether the model believes it is allowed to answer by itself.

Reliability, measured with strict jsonschema validation over 42 extractions:
  - Structure is a solved problem: 42/42 schema-valid. Three levels of nesting,
    arrays of objects, enums, booleans and a zero-parameter function all survive.
  - Declaring 20 functions instead of 5 does not degrade selection (16/18 vs
    15/18), so the manifest can hold a real toolbox.
  - The one real failure mode is a DROPPED CONSTRAINT: a request packing ~6
    conditions loses one, and the result still validates. That is what `verify`
    exists for -- it recovered the lost filter (2/4 -> 3/4) and costs a second
    upstream message, so it is opt-in.
  - Parsing itself almost never fails (1 repair in 30), so the repair pass is a
    safety net, not part of the happy path.
"""
import json
import os
import re
import uuid
from typing import Any, Optional

import tool_detect as _detect
from chatgpt_client import ChatGPTSession, QuotaExceededError, _extract_json

# The three things the model is allowed to emit. A marker rather than "reply in
# JSON" for two reasons: a normal answer that happens to contain a JSON block is
# not mistaken for a call, and the marker lands in the FIRST streamed chunk, so a
# caller can tell a call from prose after a handful of characters.
SENTINEL  = "<<<TOOL_CALL>>>"
NO_CALL   = "<<<NO_TOOL>>>"
NEED_INFO = "<<<NEED_INFO>>>"

# ANONYMOUS MODE IGNORES THIS. Measured 2026-08-20: asking for gpt-5-3-mini and
# asking for gpt-5-5 both come back with resolved_model_slug=gpt-5-6 -- the only
# model_slug that echoes the request is the one on the user message we sent. So
# the timings this default was once chosen for (gpt-5-3-mini p50 2.5s vs
# gpt-5-5's 30.7s tail) can only have been an authenticated measurement, and
# changing this will not change an anonymous extraction. Kept because an
# account-backed session does honour it.
EXTRACTOR_MODEL = os.environ.get("TOOL_EXTRACTOR_MODEL", "gpt-5-3-mini")

# Switch the whole feature off without touching a caller: the proxy falls back to
# the previous behaviour (tool names only pick a server-side mode).
EMULATION_ENABLED = os.environ.get("TOOL_EMULATION", "1").lower() not in ("0", "false", "no")


class ToolExtraction:
    """What one extraction produced.

    status is the caller-facing outcome:
      "calls"     -- tool_calls holds OpenAI-shaped calls to execute
      "no_call"   -- no declared function fits; answer the user normally
      "need_info" -- a REQUIRED parameter is absent from the request. Emitted
                     instead of a guess: without this option the model invented
                     an origin airport and a date that the user never gave, and
                     the invention validated cleanly against the schema.
    """

    __slots__ = ("status", "tool_calls", "need_info", "requests", "notes", "errors", "raw")

    def __init__(self, status, tool_calls=None, need_info=None, requests=1,
                 notes=None, errors=None, raw=""):
        self.status     = status
        self.tool_calls = tool_calls or []
        self.need_info  = need_info
        self.requests   = requests          # upstream messages spent
        self.notes      = notes or []
        self.errors     = errors or []
        self.raw        = raw


def custom_functions(tools: Optional[list], builtin_names: set) -> list[dict]:
    """The caller's own functions -- everything that is not a server-side mode.

    A name in builtin_names ("web_search", "canvas", ...) still means "turn that
    ChatGPT mode on" and is left to resolved_backend_flags(); only what is left
    over is something this proxy has to emulate.
    """
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" else t.get("function", t)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name", "")
        if not name or name in builtin_names:
            continue
        out.append({
            "name":        name,
            "description": fn.get("description", ""),
            "parameters":  fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def build_prompt(functions: list[dict], user_text: str, tool_choice: Any = "auto") -> str:
    """The extractor prompt. Every line here was earned by a failing measurement."""
    forced = isinstance(tool_choice, dict)
    forced_name = (tool_choice.get("function") or {}).get("name") if forced else None
    must = forced or tool_choice == "required"

    options = [f'  A) {SENTINEL}{{"calls":[{{"name":"<function name>","arguments":{{...}}}}]}}']
    if not must:
        options.append(f"  B) {NO_CALL}   (no declared function fits the request)")
    options.append(f'  C) {NEED_INFO}{{"function":"<name>","missing":["<param>"]}}   '
                   f"(a REQUIRED parameter is not stated in the request)")

    lines = [
        "You are a function-call extractor. You do not chat and you do not answer questions.",
        "",
        "AVAILABLE FUNCTIONS (JSON Schema):",
        json.dumps(functions, ensure_ascii=False),
        "",
        ("You MUST call a function. Output ONLY one of these:" if must
         else "Read the USER REQUEST below and output ONLY one of these:"),
        *options,
        "",
        "RULES:",
        "- Nothing before or after the output. No prose, no markdown fences, no explanation.",
        "- Output the marker exactly once.",
        # Without this the model answers the request itself whenever it believes it
        # knows the answer -- weather being the worst offender.
        "- You have NO knowledge and NO live data about anything a function covers. If a",
        "  function covers the request you MUST emit the call, even if you think you know",
        "  the answer. Answering it yourself is an ERROR.",
        '- "arguments" must satisfy that function\'s JSON Schema exactly. Never invent parameters.',
        "- One entry per distinct set of arguments: two cities means two calls.",
        "- Only omit an OPTIONAL parameter when the request does not specify it.",
        "- NEVER guess, infer or default a REQUIRED parameter the request does not state",
        "  (no made-up cities, dates, ids or amounts). If one is missing, output option C.",
        "- Never invent or simulate the RESULT of a function.",
        "- Capture EVERY condition stated in the request. Dropping one is an error.",
    ]
    if forced_name:
        lines.append(f'- You MUST call the function "{forced_name}" and no other.')
    elif must:
        lines.append(f"- Emitting {NO_CALL} is forbidden for this request.")
    lines += ["", "---", "USER REQUEST:", user_text]
    return "\n".join(lines)


def parse_envelope(text: str, valid_names: set, functions=None) -> tuple[Any, list]:
    """(calls | "NEED_INFO" | None, notes).

    Two layers, in this order.

    The MARKERS are ours and explicit: the extractor prompt asks for exactly
    one of three, so NEED_INFO and NO_TOOL are read first and decide outright.
    They have no equivalent anywhere else -- "no function fits" and "a required
    parameter was never stated" are answers this design asks for, not dialects
    a model happens to emit.

    Everything else goes through tool_detect, ported from llm-libre, which
    reads a call out of any dialect a prompted model actually produces: fenced
    or bare JSON, <tool_call> tags, Mistral's [TOOL_CALLS], JSONL runs, Python
    literals, ReAct action/action_input, wrapper envelopes, leaked `functions.`
    namespaces -- all gated on `valid_names`, so a JSON object naming anything
    the caller did not declare stays text.

    Before that second layer, every one of those cost a repair round trip, and
    a repair round trip is one more message off an anonymous hourly allowance.

    `valid_names` is the allow-list AFTER tool_choice has narrowed it (see
    tool_detect.allowed_names) -- passing the raw declared set here would let a
    forced choice be ignored by the model and still validate.
    """
    t = (text or "").strip()
    notes: list = []

    # The markers survive fences, so this pre-strip is only for finding them;
    # detection below reads fences properly on its own.
    unfenced = re.sub(r"```[a-zA-Z]*\n?", "", t).replace("```", "").strip()
    if "```" in t:
        notes.append("fenced")

    if NEED_INFO in unfenced and SENTINEL not in unfenced:
        payload = unfenced.split(NEED_INFO, 1)[1].strip()
        try:
            return "NEED_INFO", notes + [json.loads(_extract_json(payload))]
        except (ValueError, TypeError):
            return "NEED_INFO", notes

    if unfenced.startswith(NO_CALL) or (NO_CALL in unfenced and SENTINEL not in unfenced):
        return [], notes

    if SENTINEL in unfenced:
        if unfenced.count(SENTINEL) > 1:
            notes.append("duplicate-marker")
        if not unfenced.startswith(SENTINEL):
            notes.append("prose-before")
        payload = unfenced.split(SENTINEL, 1)[1].split(SENTINEL)[0].strip()
        calls = _detect.parse_tool_calls(payload, valid_names, functions)
        if calls is not None:
            return calls, notes
        notes.append("marker-payload-unreadable")
    else:
        notes.append("no-marker")

    # No usable marker payload. The model may still have called, just not in
    # the shape it was asked for -- read the whole reply as any dialect.
    calls = _detect.parse_tool_calls(t, valid_names, functions)
    if calls is not None:
        return calls, notes + ["dialect"]
    return None, notes + ["invalid-json"]


def validate_calls(calls: Any, functions: list[dict]) -> list[str]:
    """Schema-check the arguments. Recursive: nested objects and arrays included.

    jsonschema is used when it is installed and a hand-rolled walk otherwise, so
    a missing optional dependency degrades the check instead of the endpoint.
    """
    by_name = {f["name"]: f for f in functions}
    errors: list[str] = []
    if not isinstance(calls, list):
        return ["calls is not a list"]

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        Draft202012Validator = None

    for call in calls:
        if not isinstance(call, dict):
            errors.append("call is not an object")
            continue
        fn = by_name.get(call.get("name"))
        if fn is None:
            errors.append(f"unknown function {call.get('name')!r}")
            continue
        args = call.get("arguments")
        if not isinstance(args, dict):
            errors.append(f"{fn['name']}: arguments is not an object")
            continue
        schema = fn.get("parameters") or {}
        if Draft202012Validator is not None:
            for e in Draft202012Validator(schema).iter_errors(args):
                where = "/".join(str(p) for p in e.path) or "<root>"
                errors.append(f"{fn['name']}: {where}: {e.message}")
        else:
            errors.extend(f"{fn['name']}: {m}" for m in _walk(args, schema))
    return errors


def _walk(value: Any, schema: dict, path: str = "") -> list[str]:
    """Minimal recursive schema check -- the fallback when jsonschema is absent."""
    errs: list[str] = []
    want = schema.get("type")
    py = {"string": str, "integer": int, "number": (int, float), "array": list,
          "object": dict, "boolean": bool}.get(want)
    # bool is a subclass of int in Python, so True satisfies isinstance() for
    # both "integer" and "number" unless it is excluded by hand. Without the
    # second clause a model that answered `true` where a quantity belonged
    # passed this check and the bad call went out.
    wrong_bool = want in ("integer", "number") and isinstance(value, bool)
    if (py and not isinstance(value, py)) or wrong_bool:
        return [f"{path or '<root>'}: expected {want}"]
    if schema.get("enum") and value not in schema["enum"]:
        errs.append(f"{path or '<root>'}: {value!r} not in enum")
    if want == "object" and isinstance(value, dict):
        props = schema.get("properties") or {}
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"{path}/{req}".lstrip("/") + ": missing required")
        for k, v in value.items():
            if k in props:
                errs.extend(_walk(v, props[k], f"{path}/{k}".lstrip("/")))
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}/{k}".lstrip("/") + ": not allowed")
    if want == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errs.extend(_walk(item, schema["items"], f"{path}/{i}".lstrip("/")))
    return errs


def to_openai_tool_calls(calls: list) -> list[dict]:
    """OpenAI shape: arguments is a JSON *string*, not an object."""
    return [{
        "id":   "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {
            "name":      c["name"],
            "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False),
        },
    } for c in calls if isinstance(c, dict) and c.get("name")]


_VERIFY_PROMPT = (
    "You are a completeness auditor for an extracted function call.\n\n"
    "ORIGINAL REQUEST:\n{request}\n\n"
    "EXTRACTED CALL:\n{call}\n\n"
    "FUNCTION SCHEMAS:\n{schemas}\n\n"
    "Find every condition stated in the ORIGINAL REQUEST that is NOT represented in the\n"
    "EXTRACTED CALL, then output the corrected call. If nothing is missing, output the\n"
    "extracted call unchanged.\n"
    "Output ONLY:\n" + SENTINEL + '{{"calls":[{{"name":"...","arguments":{{...}}}}]}}'
)

_REPAIR_SUFFIX = (
    "\n\n---\nYour previous output was rejected:\n{previous}\n"
    "Errors: {errors}\n"
    "Output the corrected marker line and nothing else."
)


async def _ask(prompt: str, model: str) -> str:
    """One throwaway session, one message, no history. Retries once on quota.

    A fresh ChatGPTSession per extraction is the point, not an oversight: the
    session pool keys on conversation history, and an extraction has none. It
    also means a device that hits its limit is simply replaced, the same way
    SessionPool.get() replaces an exhausted one.
    """
    for attempt in range(2):
        session = ChatGPTSession()
        out = ""
        try:
            async for chunk in session.stream_message(
                prompt,
                model            = model,
                # The server-side tools are the competition, not the helpers: with
                # search on, the model answers the question itself instead of
                # delegating to the caller's function.
                force_use_search = False,
                force_use_tools  = False,
            ):
                out += chunk
            return (session.last_clean_text or out).strip()
        except QuotaExceededError:
            if attempt == 1:
                raise
        finally:
            await session.close()
    return ""


async def extract(
    functions:   list[dict],
    user_text:   str,
    tool_choice: Any = "auto",
    model:       Optional[str] = None,
    verify:      bool = False,
) -> ToolExtraction:
    """Turn a request into tool_calls with one upstream message (two if repaired).

    verify=True adds a second message that re-reads the original request looking
    for a condition the first pass dropped. Worth it for dense, multi-condition
    requests; wasteful for "what is the weather in Bogota".
    """
    model  = model or EXTRACTOR_MODEL
    prompt = build_prompt(functions, user_text, tool_choice)

    # tool_choice narrows what the model is ALLOWED to have called, not just
    # what it was asked to call. A forced function authorises only itself, so a
    # model that ignores the instruction and calls a different declared
    # function now produces nothing instead of a call that validated cleanly
    # and ran the wrong thing.
    valid = _detect.allowed_names(functions, tool_choice)

    raw    = await _ask(prompt, model)
    calls, notes = parse_envelope(raw, valid, functions)
    spent  = 1

    if calls == "NEED_INFO":
        missing = next((n for n in notes if isinstance(n, dict)), None)
        return ToolExtraction("need_info", need_info=missing, requests=spent,
                              notes=[n for n in notes if not isinstance(n, dict)], raw=raw)

    errors = validate_calls(calls, functions) if calls else []

    # Repair: only when the first pass produced something unusable. Measured at
    # 1 in 30, so this is the exception path and not a second leg of the flow.
    if calls is None or errors:
        spent += 1
        repair = prompt + _REPAIR_SUFFIX.format(
            previous = (raw or "")[:600],
            errors   = "; ".join(errors) if errors else "output did not match the required format",
        )
        raw2 = await _ask(repair, model)
        calls2, notes2 = parse_envelope(raw2, valid, functions)
        if calls2 == "NEED_INFO":
            missing = next((n for n in notes2 if isinstance(n, dict)), None)
            return ToolExtraction("need_info", need_info=missing, requests=spent,
                                  notes=notes + ["repaired"], raw=raw2)
        errors2 = validate_calls(calls2, functions) if calls2 else []
        if isinstance(calls2, list) and not errors2:
            calls, notes, errors, raw = calls2, notes + ["repaired"], [], raw2
        else:
            calls  = calls2 if isinstance(calls2, list) else calls
            notes  = notes + notes2 + ["repair-failed"]
            errors = errors2 or errors

    if not isinstance(calls, list):
        return ToolExtraction("no_call", requests=spent, notes=notes, errors=errors, raw=raw)

    if calls and verify:
        spent += 1
        audited = await _ask(_VERIFY_PROMPT.format(
            request = user_text,
            call    = json.dumps(calls, ensure_ascii=False),
            schemas = json.dumps(functions, ensure_ascii=False),
        ), model)
        calls3, _ = parse_envelope(audited, valid, functions)
        # Only accept the audit when it is at least as valid as what it replaces.
        if isinstance(calls3, list) and calls3 and not validate_calls(calls3, functions):
            calls = calls3
            notes = notes + ["verified"]

    if not calls:
        return ToolExtraction("no_call", requests=spent, notes=notes, errors=errors, raw=raw)
    return ToolExtraction("calls", tool_calls=to_openai_tool_calls(calls),
                          requests=spent, notes=notes, errors=errors, raw=raw)
