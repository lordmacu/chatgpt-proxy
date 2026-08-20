import capabilities as cap


GO = cap.AccountState(mode="account", plan="go", subscription_active=True,
                      expires_at="2026-09-06T00:28:46Z")
FREE = cap.AccountState(mode="account", plan="free", subscription_active=False,
                        expires_at=None)
ANON = cap.AccountState(mode="anonymous", plan=None, subscription_active=False,
                        expires_at=None)


def test_every_required_key_is_present_in_all_three_modes():
    for state in (GO, FREE, ANON):
        assert set(cap.effective(state)) == set(cap.REQUIRED_CAPABILITIES)
        assert all(isinstance(v, bool) for v in cap.effective(state).values())


def test_anonymous_gets_only_chat_translate_and_search():
    e = cap.effective(ANON)
    assert e["chat"] and e["streaming"] and e["translate"] and e["search"]
    assert not e["vision"]
    assert not e["images"]
    assert not e["audio_speech"]
    assert not e["audio_transcription"]
    assert not e["files"]
    assert not e["conversations"]


def test_a_free_account_gets_everything_except_images():
    # Measured, and recorded in CAPABILITIES.md: on free the model DOES invoke
    # the image tool, the generation returns empty, and the proxy answers
    # "no image was generated". That is a plan block, not a transient failure.
    e = cap.effective(FREE)
    assert e["vision"] and e["audio_speech"] and e["audio_transcription"]
    assert e["files"] and e["conversations"]
    assert not e["images"]


def test_a_paid_plan_gets_images():
    assert cap.effective(GO)["images"] is True


def test_an_expired_paid_plan_loses_images():
    # The event this whole contract exists for: the subscription lapses and the
    # boolean turns itself off, with nobody editing YAML.
    lapsed = cap.AccountState(mode="account", plan="go",
                              subscription_active=False, expires_at="2026-09-06T00:28:46Z")
    assert cap.effective(lapsed)["images"] is False


def test_tools_is_false_on_every_plan():
    # No function calling on any backend: measured 0/3, twice, with
    # tool_choice:"required" returning tool_calls:None and prose.
    for state in (GO, FREE, ANON):
        assert cap.effective(state)["tools"] is False


def test_snapshot_is_cached_and_does_not_refetch_within_the_interval():
    calls = []

    def fake_resolve():
        calls.append(1)
        return GO

    cap.reset()
    assert cap.snapshot(_resolve=fake_resolve, _now=1000.0) == GO
    assert cap.snapshot(_resolve=fake_resolve, _now=1000.0 + cap.REFRESH_INTERVAL_S - 1) == GO
    assert len(calls) == 1


def test_snapshot_refetches_once_the_interval_has_passed():
    calls = []

    def fake_resolve():
        calls.append(1)
        return GO

    cap.reset()
    cap.snapshot(_resolve=fake_resolve, _now=1000.0)
    cap.snapshot(_resolve=fake_resolve, _now=1000.0 + cap.REFRESH_INTERVAL_S + 1)
    assert len(calls) == 2


def test_a_failing_resolve_keeps_the_last_known_state():
    # /health must not start lying because the vendor had a bad minute.
    def boom():
        raise RuntimeError("upstream down")

    cap.reset()
    cap.snapshot(_resolve=lambda: GO, _now=1000.0)
    assert cap.snapshot(_resolve=boom, _now=1000.0 + cap.REFRESH_INTERVAL_S + 1) == GO


def test_a_failing_resolve_with_no_previous_state_reports_unknown():
    def boom():
        raise RuntimeError("upstream down")

    cap.reset()
    state = cap.snapshot(_resolve=boom, _now=1000.0)
    assert state.mode == "unknown"
    assert cap.effective(state)["images"] is False
