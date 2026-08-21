"""Nothing in the suite is allowed to touch the real conversation index.

conv_store defaults to conversations.db next to the module, which in a dev
checkout is the developer's own anonymous history. A test that recorded into it
would pollute it, and one that read from it would pass or fail depending on
whose machine it ran on. Every test gets its own empty database instead.
"""
import pytest

import conv_store


@pytest.fixture(autouse=True)
def isolated_conv_store(tmp_path):
    conv_store.reset(str(tmp_path / "conversations.db"))
    yield
    conv_store.reset()
