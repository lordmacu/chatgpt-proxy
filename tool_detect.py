"""Reading a tool call back out of a model's prose.

Ported from llm-libre's tool_emulator (src/llm_libre/tool_emulator.py), which
solves the harder half of the same problem: that module has to read calls out of
an ORDINARY conversation turn, from any of a dozen providers, in whatever
dialect each one's fine-tune emits. Here the model was asked for one exact
format in a dedicated request, so the documented envelope is the fast path --
but a model that answers in a neighbouring dialect used to cost a whole repair
round trip, and a repair round trip is one more message out of an anonymous
hourly allowance. Everything below turns those into a first-pass parse.

**The central risk is a false positive.** Converting a genuine text answer into
a tool call is worse than missing one: the caller runs a function the user never
asked for. So every heuristic here is gated on `valid_names` -- the functions
the caller actually declared in THIS request, narrowed by its own tool_choice.
A JSON object naming anything else stays text.

Invariants, each pinned by tests:

1. SOUNDNESS -- every call named here is in `valid_names`. No input (model
   output, schema, argument data) can widen that set.
2. tool_choice is ENFORCED, not merely prompted. A forced function authorises
   only itself, so a model that calls a different declared function produces
   nothing rather than the wrong call.
3. COVERAGE -- bare JSON, fenced JSON, <tool_call> tags, Mistral's
   [TOOL_CALLS] marker, JSONL runs, Python-literal dicts, ReAct
   action/action_input, wrapper envelopes, leaked `functions.` namespaces.
4. Argument repair is LOSSLESS OR IDENTITY, and idempotent: values are re-read
   against the declared JSON Schema by JSON's own grammar ("5" -> 5 for an
   integer) and anything that does not convert cleanly travels exactly as the
   model sent it.
5. TOTALITY -- the parser never raises on any string, and its work is bounded
   linearly in the response size. When a bound trips, the failure direction is
   a missed call, never an invented one.
"""
import ast
import json
import math
import re

# ---------------------------------------------------------------------------
# What has to come off before any brace scanning
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)

# A scratchpad cut off before its closing tag (a truncated stream, a model that
# ran out of tokens mid-thought) is reasoning to the end of the text. Left in
# place, a draft call inside it is indistinguishable from an answer.
_THINK_UNCLOSED = re.compile(
    r"<(think|thinking|reasoning)>(?:(?!</\1>).)*$", re.DOTALL | re.IGNORECASE)

# Mistral fine-tunes prefix their calls with a literal control token; what
# follows it is the ordinary JSON array the rest of the parser already reads.
# Anchored to the start on purpose: appearing anywhere else, the same bytes are
# far more likely to be argument DATA (a string value quoting the token) than a
# marker.
_TOOL_CALLS_MARKER = re.compile(r"^\s*\[/?TOOL_CALLS\]\s*")

# An embedded JSON object surrounded by much more prose is far more likely to
# be the model TALKING ABOUT a call ("its schema is {...}, but which city?")
# than making one. A call the model actually intends is the bulk of its reply.
_EMBEDDED_MIN_SHARE = 0.5

# Some open-weights models wrap calls in an XML-ish tag instead of bare JSON.
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>",
                            re.DOTALL | re.IGNORECASE)

# Only fence labels that MARK a call are explicit markers. A ```python fence
# stays a bare-scan region -- code examples full of dict literals must keep
# having to earn conversion through the prose heuristics.
_CODE_FENCE = re.compile(r"```(?:json|JSON|tool_call|tool_code)?\s*(.*?)\s*```",
                         re.DOTALL)

# Keys under which different vendors put the arguments; first present wins.
_ARGUMENT_KEYS = ("arguments", "input", "parameters", "args")

# Name-key spellings with the argument keys that accompany each. The first
# name-ish key PRESENT decides the shape; if its value fails the allow-list the
# object is data, and no further digging is allowed -- an object that names one
# thing and wraps another did not clearly call anything.
_NAME_KEY_TABLE = (
    ("name", _ARGUMENT_KEYS),
    ("action", ("action_input",) + _ARGUMENT_KEYS),
    ("tool", ("tool_input",) + _ARGUMENT_KEYS),
)

# Envelope keys some models wrap a call in, e.g. {"function_call": {...}}.
_CALL_WRAPPER_KEYS = ("function_call", "tool_call", "tool_use", "function")

# Namespace prefixes OpenAI-tuned models leak from their training format
# ("functions.get_weather"). Stripping one never invents a match: the tail
# still has to clear the same allow-list.
_NAMESPACE_PREFIXES = ("functions", "tools")

# The bare scan's work bound: each unclosed opener buys one O(n) walk, so this
# caps the scan at O(_MAX_UNCLOSED_SCANS * n) on any input.
_MAX_UNCLOSED_SCANS = 8


# ---------------------------------------------------------------------------
# The allow-list: tool_choice as a contract, not a hint
# ---------------------------------------------------------------------------

def function_names(functions) -> set:
    """Names in a normalised function list (see tool_calls.custom_functions)."""
    return {f["name"] for f in functions or []
            if isinstance(f, dict) and isinstance(f.get("name"), str) and f["name"]}


def forced_function_name(tool_choice):
    """The single function a dict-shaped tool_choice forces, or None.

    Reads both spellings in the wild: the Chat Completions nested form
    {"type": "function", "function": {"name": X}} and the Responses-style flat
    form {"type": "function", "name": X}. This is THE definition of "forced" --
    allowed_names, demands_call and the prompt all read it here, so the three
    can never disagree about what was forced.
    """
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") == "allowed_tools":
        return None
    fn = tool_choice.get("function")
    name = fn.get("name") if isinstance(fn, dict) else None
    if isinstance(name, str) and name:
        return name
    name = tool_choice.get("name")
    return name if isinstance(name, str) and name else None


def _allowed_tools_names(tool_choice):
    """The subset an `allowed_tools` choice authorises, or None if it is not
    one. An empty set is a meaningful answer: the caller authorised nothing."""
    if not (isinstance(tool_choice, dict)
            and tool_choice.get("type") == "allowed_tools"):
        return None
    names = set()
    for entry in tool_choice.get("tools") if isinstance(tool_choice.get("tools"), list) else []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if not (isinstance(name, str) and name):
            name = entry.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def allowed_names(functions, tool_choice) -> set:
    """The names detection may produce once tool_choice has had its say.

    tool_choice does not only demand calls, it NARROWS them: a caller that
    forced one specific function has not authorised any other, an
    `allowed_tools` subset authorises exactly its members, and "none"
    authorises nothing at all.

    Gating detection on this instead of the raw function list is what turns
    tool_choice from a hint the prompt makes into a contract the parser keeps.
    Before this, a forced choice lived only in the prompt: a model that called
    a different DECLARED function produced a call that passed validation, and
    the caller ran the wrong function.
    """
    names = function_names(functions)
    if tool_choice == "none":
        return set()
    subset = _allowed_tools_names(tool_choice)
    if subset is not None:
        return names & subset
    forced = forced_function_name(tool_choice)
    if forced is not None:
        return names & {forced}
    return names


def demands_call(tool_choice) -> bool:
    """Whether tool_choice PROMISES a call.

    OpenAI's contract: "required", a forced specific function, and an
    `allowed_tools` choice with mode "required" all guarantee tool_calls.
    auto/none/absent promise nothing.
    """
    if tool_choice == "required":
        return True
    if (isinstance(tool_choice, dict)
            and tool_choice.get("type") == "allowed_tools"):
        return tool_choice.get("mode") == "required"
    return forced_function_name(tool_choice) is not None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def parse_tool_calls(text, valid_names, functions=None):
    """Read model output as tool calls, accepting only `valid_names`.

    Returns a non-empty list of {"name", "arguments"} dicts, or None. Never
    raises: every input is either a call list or None.

    `functions`, when given, repairs argument types against each function's
    declared schema -- losslessly or not at all.
    """
    if not valid_names or not isinstance(text, str):
        return None

    # Strip reasoning scratchpads first (closed AND cut-off ones): their braces
    # would otherwise anchor the balanced scan on content that is not the
    # answer. The Mistral marker comes off next -- what it prefixes is the
    # ordinary array format.
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    cleaned = _TOOL_CALLS_MARKER.sub("", cleaned).strip()
    if not cleaned:
        return None

    # Every <tool_call> tag is a deliberate, marked invocation, so ALL of them
    # together are the answer: models trained on that format emit one tag per
    # parallel call. A tag whose payload fails the allow-list is dropped, not
    # fatal.
    tag_calls = []
    for m in _TOOL_CALL_TAG.finditer(cleaned):
        calls = _parse_candidate(m.group(1).strip(), valid_names)
        if calls:
            tag_calls.extend(calls)
    if tag_calls:
        return apply_schemas(tag_calls, functions)

    # A response that is EXACTLY a JSON array is authoritative: the model chose
    # that structure deliberately, so the all-or-nothing rule decides it
    # outright rather than letting the single-object fallback resurrect one
    # call from what is much more likely a list of data.
    if cleaned.startswith("[") and isinstance(_loads_tolerant(cleaned), list):
        return apply_schemas(_parse_candidate(cleaned, valid_names), functions)

    # Among several parseable candidates, the LAST one wins. A model that
    # illustrates the format before committing ("here is how I would call it:
    # ```…``` -- now the real call: {…}") puts the demo first and the real call
    # last, so taking the first match hands back the example's arguments.
    best = None
    for position, candidate, qualifies in _json_candidates(cleaned):
        if not candidate or not qualifies:
            continue
        calls = _parse_candidate(candidate, valid_names)
        if calls and (best is None or position >= best[0]):
            best = (position, calls)
    return apply_schemas(best[1], functions) if best else None


def _blank_out(text, pattern):
    """Replace every match with spaces, preserving offsets, so the bare scan
    skips regions already claimed by an explicit marker."""
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def _json_candidates(text):
    """Every substring of `text` that might be a JSON call.

    Returns (position, candidate, qualifies) triples. `qualifies` is the
    prose-vs-invocation verdict: candidates carrying a deliberate marker (a tag
    or a fence) always qualify; a bare object loose in a paragraph has to earn
    it.

    Bare scanning walks EVERY balanced span, not just the first, so a schema
    the model quoted early must not shadow the real call at the end. It runs
    over a copy with fences and tags blanked out, so an illustrative example
    inside a fence is not re-discovered as if it were loose text.

    Runs of bare spans separated by nothing but whitespace are also offered
    joined into one synthetic array -- the JSONL habit of models that emit one
    object per line.
    """
    found = []
    for m in _TOOL_CALL_TAG.finditer(text):
        found.append((m.start(), m.group(1).strip(), True))
    for m in _CODE_FENCE.finditer(text):
        found.append((m.start(), m.group(1).strip(), True))

    residual = _blank_out(_blank_out(text, _TOOL_CALL_TAG), _CODE_FENCE)
    spans = []
    # Every unclosed opener costs one walk to the end of the text before the
    # scan can step past it. Unbudgeted, a storm of bare openers makes the scan
    # quadratic in the response size. The budget keeps total work at O(k·n) and
    # gives up in the SAFE direction: a missed call, never an invented one.
    unclosed_budget = _MAX_UNCLOSED_SCANS
    for opener, closer in (("[", "]"), ("{", "}")):
        pos = 0
        while unclosed_budget > 0:
            hit = _balanced_span(residual, opener, closer, pos)
            if hit is None:
                break
            span, start = hit
            if span is None:
                unclosed_budget -= 1
                pos = start + 1
                continue
            spans.append((start, start + len(span), span))
            pos = start + len(span)
    spans.sort(key=lambda s: s[0])

    for start, end, span in spans:
        found.append((start, span, _looks_like_an_invocation(span, residual, start)))

    for run in _whitespace_runs(spans, residual):
        if len(run) < 2:
            continue
        first_start, last_end = run[0][0], run[-1][1]
        joined = "[" + ", ".join(s[2] for s in run) + "]"
        qualifies = (first_start <= 0
                     or not residual[last_end:].strip()
                     or (last_end - first_start) / len(residual) >= _EMBEDDED_MIN_SHARE)
        found.append((run[-1][0], joined, qualifies))
    return found


def _whitespace_runs(spans, text):
    """Group sorted spans into runs separated by whitespace only.

    Overlapping spans (an object already inside a scanned array) and spans with
    prose between them each start a new run: only true side-by-side emission
    reads as one batch.
    """
    runs = []
    for span in spans:
        if runs:
            prev_end = runs[-1][-1][1]
            if span[0] >= prev_end and not text[prev_end:span[0]].strip():
                runs[-1].append(span)
                continue
        runs.append([span])
    return runs


def _balanced_span(text, opener, closer, begin=0):
    """Scan for the next balanced opener/closer span from `begin`.

    Returns (span, start); span is None when an opener was found but never
    closed (the caller may resume past it), and the whole result is None when
    no opener remains.

    Brace-counting rather than regex, and string-aware for BOTH quote styles: a
    brace inside a JSON string value must not close the object, the same brace
    inside a Python-literal string must not either, and a backslash-escaped
    quote must not end the string.
    """
    start = text.find(opener, begin)
    if start == -1:
        return None
    depth = 0
    quote = None
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1], start
    return None, start


def _looks_like_an_invocation(span, whole, position) -> bool:
    """Whether a bare JSON object is a call or just prose.

    A model that is actually calling opens with the JSON or works up to it and
    ends there. A model quoting the tool's own schema to ask a clarifying
    question ("its schema is {...}, but which city?") leaves a small object
    stranded mid-sentence with the question after it, and converting that
    destroys the question and issues a call the model never intended.

    So a bare span qualifies when it opens the message, closes it, or is most
    of it. Explicitly marked candidates skip this check entirely.
    """
    if position <= 0:
        return True
    if not whole:
        return False
    if not whole[position + len(span):].strip():
        return True
    return len(span) / len(whole) >= _EMBEDDED_MIN_SHARE


def _loads_tolerant(text):
    """Read JSON, falling back to a GUARDED Python-literal read.

    Weaker models emit repr output as often as JSON: single quotes, trailing
    commas, True/None. ast.literal_eval parses exactly that dialect and
    evaluates nothing (literals only, no names, no calls), so the fallback adds
    no execution surface. The result is round-tripped through json so
    downstream code only ever sees JSON-compatible values.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return _as_jsonable(ast.literal_eval(text))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _as_jsonable(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return None


def _parse_candidate(text, valid_names):
    obj = _loads_tolerant(text)
    if obj is None:
        return None
    return _calls_from_object(obj, valid_names)


def _calls_from_object(obj, valid_names):
    if isinstance(obj, list):
        calls = [c for c in (_extract_call(i, valid_names) for i in obj) if c]
        # Every element must be a valid call. A list where only some entries
        # qualify is far more likely to be data the model returned than a batch
        # of calls, and converting it would invent calls it never made.
        return calls if calls and len(calls) == len(obj) else None

    if isinstance(obj, dict):
        call = _extract_call(obj, valid_names)
        if call:
            return [call]
        # {"tool_calls": [...]} mirrors the RESPONSE shape these models were
        # fine-tuned on. Same all-or-nothing rule as any other batch.
        batch = obj.get("tool_calls")
        if isinstance(batch, list) and batch:
            return _calls_from_object(batch, valid_names)
        return None

    return None


def _extract_call(obj, valid_names):
    """Normalise one JSON object into a call, across known vendor shapes.

        {"name": ..., "arguments": {...}}                   documented format
        {"name": ..., "input": {...}}                       Anthropic
        {"name": ..., "parameters"|"args": {...}}           open-weights
        {"action": ..., "action_input": {...}}              LangChain/ReAct
        {"tool": ..., "tool_input": {...}}                  ReAct sibling
        {"function_call": {"name": ..., "arguments": ...}}  OpenAI legacy
        {"tool_call"|"tool_use"|"function": {...}}          wrappers
    """
    if not isinstance(obj, dict):
        return None

    for name_key, argument_keys in _NAME_KEY_TABLE:
        name = obj.get(name_key)
        if isinstance(name, str) and name:
            for key in argument_keys:
                if key in obj:
                    return _normalize(name, obj[key], valid_names)
            # A name with no arguments key at all is a legitimate
            # zero-argument call.
            return _normalize(name, {}, valid_names)

    for key in _CALL_WRAPPER_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict):
            call = _extract_call(inner, valid_names)
            if call:
                return call

    return None


def _resolve_name(name, valid_names):
    """Map an emitted name onto the declared one it clearly means, or None.

    Exact match first. Then the same name with a leaked vendor namespace
    stripped ("functions.get_weather"), then a case-insensitive match --
    accepted only when it is UNIQUE: if two declared names differ only by case,
    a third spelling names neither and stays text. Every path still ends inside
    valid_names; nothing here can invent a function nobody declared.
    """
    candidates = [name]
    head, dot, tail = name.partition(".")
    if dot and head in _NAMESPACE_PREFIXES and tail:
        candidates.append(tail)
    for candidate in candidates:
        if candidate in valid_names:
            return candidate
    for candidate in candidates:
        matches = [n for n in valid_names if n.lower() == candidate.lower()]
        if len(matches) == 1:
            return matches[0]
    return None


def _normalize(name, arguments, valid_names):
    """Validate a name/arguments pair and coerce arguments to a dict."""
    if not isinstance(name, str):
        return None
    resolved = _resolve_name(name, valid_names)
    if resolved is None:
        return None

    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            arguments = {}
        else:
            parsed = _loads_tolerant(text)
            # Malformed argument JSON degrades to {}: the name is valid and the
            # model clearly meant to call, so the call is kept with empty
            # arguments rather than dropped -- the caller gets a call it can
            # reject, instead of prose it will misread as an answer.
            arguments = parsed if isinstance(parsed, dict) else {}

    if not isinstance(arguments, dict):
        arguments = {}

    return {"name": resolved, "arguments": arguments}


# ---------------------------------------------------------------------------
# Argument repair: lossless or identity
# ---------------------------------------------------------------------------

def apply_schemas(calls, functions):
    """Repair argument types against each function's declared schema.

    A prompted model returns scalars as strings far more often than a native
    tool-caller does ("5" where the schema says integer). Every coercion here
    is lossless or skipped: a value that does not convert CLEANLY travels
    exactly as the model sent it, and parameters the schema does not declare
    are never touched.

    This is what turns a type mismatch from a repair round trip -- one more
    message off an anonymous hourly allowance -- into a first-pass parse.
    """
    if not calls or not isinstance(functions, (list, tuple)):
        return calls
    params_by_name = {f["name"]: f.get("parameters")
                      for f in functions
                      if isinstance(f, dict) and isinstance(f.get("name"), str)}
    repaired = []
    for call in calls:
        params = params_by_name.get(call["name"])
        props = params.get("properties") if isinstance(params, dict) else None
        if isinstance(props, dict):
            call = {**call, "arguments": {
                k: coerce_value(v, props.get(k))
                for k, v in call["arguments"].items()}}
        repaired.append(call)
    return repaired


# Coercion reads numbers by JSON's own grammar, not Python's: int("1_000") and
# float("1_2.5") succeed in Python but "1_000" is not a JSON number, so
# accepting them would not be a lossless re-reading of what the model wrote.
_JSON_INT = re.compile(r"[+-]?[0-9]+\Z")
_JSON_NUMBER = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")

# Bounds recursion through nested structures. Schemas arrive as parsed JSON
# (acyclic by construction), so this is belt: a depth past it stops repairing,
# never fails.
_COERCE_MAX_DEPTH = 8


def _satisfies(value, kind) -> bool:
    """Whether `value` already inhabits JSON-Schema type `kind`.

    bool is deliberately NOT an integer here, although Python says it is: JSON
    Schema draws the same line, and folding True into 1 would rewrite a flag
    the model may have meant literally. A whole float (5.0) is deliberately NOT
    accepted as `integer` either, even though JSON Schema would validate it:
    rejecting it here routes it through coercion, which folds it to the
    canonical int that strict consumers expect.
    """
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return False


def _coerce_scalar(value, kind, kinds):
    """One lossless conversion attempt toward `kind`; the input on failure."""
    if kind == "integer":
        if isinstance(value, str) and _JSON_INT.match(value.strip()):
            return int(value.strip())
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
    elif kind == "number":
        if isinstance(value, str) and _JSON_NUMBER.match(value.strip()):
            return float(value.strip())
    elif kind == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
    elif kind == "string":
        # bool is excluded: json.dumps(True) is "true", which silently rewrites
        # a flag the model may have meant literally.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return json.dumps(value)
    elif kind in ("array", "object"):
        if isinstance(value, str):
            parsed = _loads_tolerant(value.strip())
            if kind == "array" and isinstance(parsed, list):
                return parsed
            if kind == "object" and isinstance(parsed, dict):
                return parsed
    elif kind == "null":
        # Only when the union has no "string" member: with one, the literal
        # text "null" is at least as plausibly the string as the null.
        if (isinstance(value, str) and value.strip().lower() == "null"
                and "string" not in kinds):
            return None
    return value


def coerce_value(value, schema, depth=0):
    """Repair one value against its schema: losslessly, or not at all.

    In order:

    1. Enum repair: a string differing from exactly ONE enum member only by
       case/padding becomes that member. Two members colliding
       case-insensitively make a third spelling ambiguous -- untouched.
    2. Identity: a value already satisfying ANY declared type (`type` may be a
       union list) is never converted -- "5" under ["integer", "string"] IS the
       string. The one exception is a whole float under `integer`, folded to
       the canonical int.
    3. Otherwise the declared types are tried in order and the first clean
       conversion wins; no clean conversion, no change.
    4. Structure recursion: dicts repair their declared properties, lists
       repair every element against `items` -- bounded by _COERCE_MAX_DEPTH.

    Idempotent: every repaired value satisfies its type, and step 2 makes
    satisfying values fixed points.
    """
    if not isinstance(schema, dict) or depth > _COERCE_MAX_DEPTH:
        return value

    enum = schema.get("enum")
    if isinstance(enum, list) and isinstance(value, str) and value not in enum:
        matches = [e for e in enum
                   if isinstance(e, str) and e.lower() == value.strip().lower()]
        if len(matches) == 1:
            value = matches[0]

    declared = schema.get("type")
    if isinstance(declared, str):
        kinds = [declared]
    elif isinstance(declared, list):
        kinds = [k for k in declared if isinstance(k, str)]
    else:
        kinds = []

    if not any(_satisfies(value, k) for k in kinds):
        for kind in kinds:
            converted = _coerce_scalar(value, kind, kinds)
            if converted is not value:
                value = converted
                break
    elif ("integer" in kinds and isinstance(value, float)
            and math.isfinite(value) and value.is_integer()):
        value = int(value)

    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            value = {k: coerce_value(v, props.get(k), depth + 1)
                     for k, v in value.items()}
    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            value = [coerce_value(v, items, depth + 1) for v in value]
    return value
