"""The index holds one thing: which device can reopen which conversation."""
import conv_store


def test_a_recorded_conversation_can_be_looked_up():
    conv_store.record("u1", "c1", "d1", "Título")
    row = conv_store.lookup("u1", "c1")
    assert row["device_id"] == "d1" and row["title"] == "Título"


def test_an_unknown_conversation_is_none():
    assert conv_store.lookup("u1", "nunca-visto") is None


def test_the_device_is_overwritten_because_the_pool_rotates_it():
    # A device that exhausts its quota is replaced, and it is the CURRENT owner
    # that can still read the thread.
    conv_store.record("u1", "c1", "device-viejo")
    conv_store.record("u1", "c1", "device-nuevo")
    assert conv_store.lookup("u1", "c1")["device_id"] == "device-nuevo"


def test_a_later_turn_does_not_blank_out_the_title():
    # The vendor generates the title on the first turn and never resends it, so
    # a second turn arrives with title=None.
    conv_store.record("u1", "c1", "d1", "El título")
    conv_store.record("u1", "c1", "d1", None)
    assert conv_store.lookup("u1", "c1")["title"] == "El título"


def test_a_first_turn_without_a_title_can_still_get_one_later():
    conv_store.record("u1", "c1", "d1", None)
    conv_store.record("u1", "c1", "d1", "Llegó después")
    assert conv_store.lookup("u1", "c1")["title"] == "Llegó después"


def test_one_callers_rows_are_invisible_to_another():
    conv_store.record("u1", "c1", "d1")
    conv_store.record("u2", "c2", "d2")
    assert conv_store.lookup("u2", "c1") is None
    assert conv_store.listing("u1")[1] == 1


def test_the_listing_is_most_recent_first():
    conv_store.record("u1", "a", "d", None, now=1.0)
    conv_store.record("u1", "b", "d", None, now=3.0)
    conv_store.record("u1", "c", "d", None, now=2.0)
    rows, total = conv_store.listing("u1")
    assert total == 3
    assert [r["conversation_id"] for r in rows] == ["b", "c", "a"]


def test_the_listing_paginates_without_losing_the_total():
    for i in range(5):
        conv_store.record("u1", f"c{i}", "d", None, now=float(i))
    rows, total = conv_store.listing("u1", limit=2, offset=2)
    assert total == 5 and [r["conversation_id"] for r in rows] == ["c2", "c1"]


def test_forget_removes_only_that_row():
    conv_store.record("u1", "c1", "d")
    conv_store.record("u1", "c2", "d")
    conv_store.forget("u1", "c1")
    assert conv_store.lookup("u1", "c1") is None
    assert conv_store.lookup("u1", "c2") is not None


def test_a_turn_with_no_conversation_id_yet_is_not_recorded():
    # A turn that failed before the backend assigned one has nothing to index.
    conv_store.record("u1", "", "d1")
    conv_store.record("u1", "c1", "")
    assert conv_store.listing("u1")[1] == 0
