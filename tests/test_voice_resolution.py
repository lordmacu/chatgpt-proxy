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
    assert main.resolve_voice("alloy") == ("juniper", False)


@pytest.mark.parametrize("name", main.NATIVE_VOICES)
def test_every_native_voice_passes_through(name):
    assert main.resolve_voice(name) == (name, False)


@pytest.mark.parametrize("alias", sorted(main.OPENAI_VOICE_MAP))
def test_every_openai_alias_resolves_to_a_real_voice(alias):
    """Un alias que apunte a una voz inexistente reintroduce el bug entero."""
    assert main.resolve_voice(alias)[0] in main.NATIVE_VOICES


def test_different_openai_names_give_different_voices():
    """Mapear todo a juniper "funcionaría" y haría inútil elegir voz."""
    picked = {main.resolve_voice(a)[0] for a in main.OPENAI_VOICE_MAP}
    assert len(picked) >= 8


def test_case_and_whitespace_do_not_matter():
    assert main.resolve_voice("  ALLOY  ") == ("juniper", False)


def test_an_empty_voice_is_the_default_not_an_error():
    assert main.resolve_voice("") == ("juniper", False)
    assert main.resolve_voice(None) == ("juniper", False)


def test_an_unknown_voice_picks_a_real_one_at_random():
    """Decisión del operador: audio en una voz cualquiera antes que un rechazo.
    Lo que NUNCA puede pasar es que llegue al backend, que responde cortando la
    conexión y tumbaba el endpoint con un 500."""
    for _ in range(20):
        chosen, substituted = main.resolve_voice("no-existe")
        assert chosen in main.NATIVE_VOICES
        assert substituted is True


def test_the_substitution_is_reported_never_silent():
    """Elegir al azar cuesta determinismo: la misma petición puede sonar
    distinta, así que la sustitución tiene que ser visible."""
    assert main.resolve_voice("no-existe")[1] is True
    assert main.resolve_voice("alloy")[1] is False


def test_the_limits_are_published():
    assert main.MAX_INPUT_CHARS == 4096


def test_the_supported_list_is_not_empty():
    """Sin esto el 400 sería tan inútil como el 500 que reemplaza."""
    assert len(main.NATIVE_VOICES) == 10
