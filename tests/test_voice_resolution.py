"""Resolución de voces: el defecto que un cliente OpenAI encontraba de entrada.

Mandar `voice="alloy"` -- el valor por defecto de la API de OpenAI, y por lo
tanto lo primero que manda cualquier cliente -- tumbaba el endpoint con un
`500 Internal Server Error` pelado. El backend no responde 4xx a una voz
inválida: cierra la conexión a mitad del cuerpo, httpx levanta
RemoteProtocolError y la excepción escapaba.
"""
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


def test_alloy_no_longer_reaches_the_backend():
    """El caso exacto que producía el 500."""
    assert main.resolve_voice("alloy") == "juniper"


@pytest.mark.parametrize("name", main.NATIVE_VOICES)
def test_every_native_voice_passes_through(name):
    assert main.resolve_voice(name) == name


@pytest.mark.parametrize("alias", sorted(main.OPENAI_VOICE_MAP))
def test_every_openai_alias_resolves_to_a_real_voice(alias):
    """Un alias que apunte a una voz inexistente reintroduce el bug entero."""
    assert main.resolve_voice(alias) in main.NATIVE_VOICES


def test_different_openai_names_give_different_voices():
    """Mapear todo a juniper "funcionaría" y haría inútil elegir voz."""
    picked = {main.resolve_voice(a) for a in main.OPENAI_VOICE_MAP}
    assert len(picked) >= 8


def test_case_and_whitespace_do_not_matter():
    assert main.resolve_voice("  ALLOY  ") == "juniper"


def test_an_empty_voice_is_the_default_not_an_error():
    assert main.resolve_voice("") == "juniper"
    assert main.resolve_voice(None) == "juniper"


def test_an_unknown_voice_is_a_400_that_says_what_is_valid():
    """400, no 500: es error del cliente, y el gateway no debe castigar la ruta
    por un typo ajeno."""
    with pytest.raises(HTTPException) as exc:
        main.resolve_voice("no-existe")
    assert exc.value.status_code == 400
    detail = exc.value.detail["error"]
    assert detail["param"] == "voice"
    assert "juniper" in detail["supported"]
    assert "alloy" in detail["openai_aliases"]


def test_the_supported_list_is_not_empty():
    """Sin esto el 400 sería tan inútil como el 500 que reemplaza."""
    assert len(main.NATIVE_VOICES) == 10
