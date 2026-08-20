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
      - `tools` is False on EVERY plan: with tool_choice:"required" the backend
        returns tool_calls:None and prose. Measured 0/3, twice. This is the one
        boolean that does not vary, and claiming it would make the gateway route
        agentic traffic here and receive prose -- a silent failure, where a
        refusal would at least fail over.
      - `images` needs a paid plan. On free the tool IS invoked and returns
        empty; that is a plan block, not a transient failure.
      - `translate` and `search` work anonymously: /v1/translate does not even
        spend a chat message, and search is a flag on the chat request.
    """
    account = state.mode == "account"
    return {
        "chat":                True,
        "streaming":           True,
        "tools":               False,
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
    """Ask ChatGPT what this token's account is. Wired in Task 3.

    Kept separate from `snapshot` so the caching rules can be tested without a
    network, and so the vendor call has exactly one home.
    """
    if not auth.is_authenticated():
        return AccountState(mode="anonymous")
    return AccountState(mode="account")
