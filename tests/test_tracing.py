"""The proxy is the last hop, and the slowest one.

An image request spends thirty to fifty seconds here while ChatGPT draws, and
from outside that is indistinguishable from the gateway being slow or the
network stalling. These tests pin the one property that makes the difference
readable: the id the app minted survives to THIS log, so the same string can be
grepped in three places and the fifty seconds attributed to whichever hop
actually spent them.
"""
from fastapi.testclient import TestClient

import main


def test_the_callers_request_id_comes_back_on_the_response():
    with TestClient(main.app) as c:
        r = c.get("/health", headers={"X-Request-Id": "t-abc123def456"})
    assert r.headers["X-Request-Id"] == "t-abc123def456"


def test_a_request_that_brings_no_id_is_given_one():
    with TestClient(main.app) as c:
        r = c.get("/health")
    assert r.headers.get("X-Request-Id")


def test_an_unusable_id_is_replaced_rather_than_reflected():
    # The id lands in a log line and in a response header, so it is
    # attacker-controlled text: a newline in it would forge a log entry.
    with TestClient(main.app) as c:
        r = c.get("/health", headers={"X-Request-Id": "bad id\twith space"})
    got = r.headers.get("X-Request-Id")
    assert got and got != "bad id\twith space"
