"""`size` was declared on the request and then dropped on the floor.

A caller sending `size: "1024x1536"` got a square image and no warning, which is
worse than a 400: the field's presence in the schema is a promise. And it cannot
be forwarded, because the upstream flow has no size field to forward it TO --
checked against the decompiled official app (com.openai.chatgpt 1.2026.223),
whose only mention of `image_size`/`aspect_ratio` is a telemetry event, with no
file in the whole APK carrying both `image_gen` and a size key.

The image is drawn by a tool driven by natural language, so the prompt is the
only channel there is. These tests pin the translation.
"""
import pytest

from main import _prompt_with_size

PROMPT = "un gato naranja"


def test_a_square_size_asks_for_a_square_image():
    # It has to ask, and this was MEASURED, not assumed. The first version of
    # this function appended nothing for a square size, on the theory that
    # square was the upstream default. Six generations later: of the three that
    # carried no hint, two came back 1024x1536 and one 1402x1122. There is no
    # square default -- unhinted, the tool picks a shape -- so "1024x1024" only
    # means anything if it is said out loud.
    for square in ("1024x1024", "512x512", "2048x2048"):
        out = _prompt_with_size(PROMPT, square)
        assert out.startswith(PROMPT)
        assert "cuadrada" in out.lower(), out


def test_a_taller_size_asks_for_a_vertical_image():
    out = _prompt_with_size(PROMPT, "1024x1536")
    assert out.startswith(PROMPT)
    assert out != PROMPT
    assert "vertical" in out.lower()


def test_a_wider_size_asks_for_a_horizontal_image():
    out = _prompt_with_size(PROMPT, "1536x1024")
    assert out.startswith(PROMPT)
    assert "horizontal" in out.lower()


@pytest.mark.parametrize("junk", ["", "   ", "big", "1024", "axb", "1024x", "x1024",
                                  "0x0", "-1x5", "1024x1024x1024", None])
def test_an_unreadable_size_changes_nothing_and_never_raises(junk):
    # The value arrives verbatim from a client. A size that cannot be read is a
    # reason to ignore it, never a reason to fail an image the caller is paying
    # a minute of wall clock for.
    assert _prompt_with_size(PROMPT, junk) == PROMPT


def test_the_prompt_itself_is_never_lost():
    # The instruction is appended, not substituted: a translation that dropped
    # the subject would return a correctly-shaped picture of the wrong thing.
    out = _prompt_with_size("un dragón sobre Bogotá", "1024x1792")
    assert "un dragón sobre Bogotá" in out
