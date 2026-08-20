"""What this proxy can actually do right now, resolved against the account.

Spec: the proxy capability contract, llm-libre
docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md

THE RULE THIS MODULE EXISTS TO ENFORCE: a capability boolean says what a request
sent right now would ACHIEVE, not what this codebase implements. On a free plan
/v1/images/generations exists, accepts the request, invokes the image tool, and
returns nothing -- so `images` is False. To the gateway "the endpoint exists" and
"a request to it produces an image" are the same question, and only the second
one is worth answering.

Where the rule STOPS: a boolean tracks entitlement, not the meter. `images` is
False for anonymous, free, expired or revoked -- never for "today's 106
generations are spent". Exhaustion is a 429 the gateway already handles with a
cooldown and the vendor's own Retry-After, and it recovers by itself; flipping
a boolean for it would flap between sweeps and replace self-healing with a wait
for the next one. The dividing line is durability: if a fresh request TOMORROW
would still be refused for the same reason, it belongs in the boolean.

This is also the ONLY place that knows what "go" or "free" mean. The gateway
never sees a plan name -- it reads booleans, which is what keeps it ignorant of
every vendor's billing.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass

import auth
import tool_calls

log = logging.getLogger(__name__)

# The eleven keys the contract requires, byte-for-byte the same set the gateway
# validates against (llm_libre.contract.REQUIRED_CAPABILITIES). Duplicated here
# rather than imported because the two live in different repos and deploy
# independently; tests/test_health_contract.py is what keeps them honest.
REQUIRED_CAPABILITIES = (
    "chat", "streaming", "tools", "vision", "images",
    "audio_speech", "audio_transcription", "translate",
    "search", "files", "conversations",
)

# /health is called by the gateway on every catalogue sweep and by Coolify as a
# container health check, so it must NOT reach the vendor per hit: that would
# make both depend on OpenAI being up, which is the opposite of what a health
# check is for. The account state is resolved at most this often and served from
# cache in between. An hour is far shorter than anything that can change here
# (a plan lapses on a date, a token is revoked once) and far longer than the
# sweep interval.
REFRESH_INTERVAL_S = float(os.environ.get("CAPABILITY_REFRESH_S", "3600"))

_lock = threading.Lock()
_state: "AccountState | None" = None
_resolved_at: float = 0.0


@dataclass(frozen=True)
class AccountState:
    """The vendor-side facts the capability booleans are derived from."""
    mode: str                          # "anonymous" | "account" | "unknown"
    plan: str | None = None            # vendor string: "go", "free", ...
    subscription_active: bool = False
    expires_at: str | None = None      # ISO 8601 UTC


UNKNOWN = AccountState(mode="unknown")


def _paid(state: AccountState) -> bool:
    """Whether this account holds a live PAID plan.

    Both halves are needed. `plan` alone is stale the moment a subscription
    lapses -- the vendor keeps reporting plan_type "go" with
    subscription.active false -- and `subscription_active` alone cannot tell a
    paid plan from a free tier that reports itself as an active subscription.
    """
    return state.mode == "account" and state.subscription_active \
        and (state.plan or "free") != "free"


def effective(state: AccountState) -> dict:
    """The eleven booleans, for this account state. Measured, not guessed.

    Every False below was observed, and CAPABILITIES.md records how:
      - anonymous reaches 5 of 14 endpoints; `synthesize`, `library`, `gizmos`
        and the file APIs have no /backend-anon variant at all, so everything
        that needs one is False.
      - `tools` follows `tool_calls.EMULATION_ENABLED`. Native function calling
        does not exist on any backend -- with tool_choice:"required" the
        backend returns tool_calls:None and prose, measured 0/3, twice -- but
        /v1/chat/completions emulates it (see tool_calls.py) and returns real
        `tool_calls` with finish_reason "tool_calls", streaming included. The
        contract promises what a request ACHIEVES, not how, so this reports
        True whenever that emulation is on and False only when the operator
        disables it with TOOL_EMULATION=0.
      - `images` needs a paid plan. On free the tool IS invoked and returns
        empty; that is a plan block, not a transient failure.
      - `translate` and `search` work anonymously: /v1/translate does not even
        spend a chat message, and search is a flag on the chat request.
    """
    account = state.mode == "account"
    return {
        "chat":                True,
        "streaming":           True,
        # Custom function calling is EMULATED (see tool_calls.py) rather than
        # native, and the contract does not care: it promises what a request
        # achieves, not how. What matters is that /v1/chat/completions returns
        # real `tool_calls` with finish_reason "tool_calls" -- and it stops
        # doing so when the operator sets TOOL_EMULATION=0, which is exactly
        # when this must report False.
        "tools":               tool_calls.EMULATION_ENABLED,
        "vision":              account,
        "images":              _paid(state),
        "audio_speech":        account,
        "audio_transcription": account,
        "translate":           True,
        "search":              True,
        "files":               account,
        "conversations":       account,
    }


def snapshot(_resolve=None, _now=None) -> AccountState:
    """The cached account state, refreshed at most every REFRESH_INTERVAL_S.

    A failed refresh KEEPS the previous value rather than degrading to unknown:
    the last known state is far better evidence than "we could not ask just now",
    and turning capabilities off because the vendor blinked would take routes out
    of the gateway's rotation for no reason. Only a failure with nothing cached
    yet -- a cold start while the vendor is down -- reports unknown, where every
    account-gated capability is False. That direction is the safe one: claiming
    a capability we cannot confirm sends real traffic into a wall.

    `_resolve` and `_now` are injection points for tests; production passes
    neither.
    """
    global _state, _resolved_at
    resolve = _resolve or _resolve_from_vendor
    now = time.time() if _now is None else _now
    with _lock:
        if _state is not None and (now - _resolved_at) < REFRESH_INTERVAL_S:
            return _state
        try:
            _state = resolve()
        except Exception as e:                       # noqa: BLE001 -- see docstring
            log.warning("capabilities: could not resolve the account state "
                        "(%s: %s); keeping %s.", type(e).__name__, e,
                        "the previous value" if _state else "unknown")
            if _state is None:
                _state = UNKNOWN
        _resolved_at = now
        return _state


def reset() -> None:
    """Drop the cache. For tests, and for a token change at runtime."""
    global _state, _resolved_at
    with _lock:
        _state, _resolved_at = None, 0.0


def _resolve_from_vendor() -> AccountState:
    """Ask ChatGPT what this token's account is.

    Synchronous and blocking on purpose: it runs at most once an hour, behind
    `snapshot`'s lock, and making it async would push an event loop requirement
    into every caller of a function that is meant to be trivially callable.

    Kept separate from `snapshot` so the caching rules can be tested without a
    network, and so the vendor call has exactly one home. Carries its own
    timeout: this runs under `snapshot`'s lock, so a hung request here would
    otherwise block every concurrent /health caller indefinitely.
    """
    if not auth.is_authenticated():
        return AccountState(mode="anonymous")
    import httpx
    r = httpx.get(
        "https://chatgpt.com/backend-api/accounts/check/v4-2023-04-27",
        params={"timezone_offset_min": "0"},
        headers={"Authorization": "Bearer " + auth.access_token(),
                 "User-Agent": "chatgpt-proxy/capabilities",
                 "Accept": "application/json"},
        timeout=15.0,
    )
    r.raise_for_status()
    accounts = (r.json().get("accounts") or {})
    account = accounts.get("default") or next(iter(accounts.values()), {}) or {}
    entitlement = account.get("entitlement") or {}
    inner = account.get("account") or {}
    return AccountState(
        mode="account",
        plan=inner.get("plan_type"),
        subscription_active=bool(entitlement.get("has_active_subscription")),
        expires_at=entitlement.get("expires_at"),
    )
