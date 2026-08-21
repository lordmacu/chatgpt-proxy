"""El Dockerfile lista módulos por nombre, así que agregar un .py al repo NO lo
mete en la imagen. Este test es el que faltaba las tres veces que pasó.

Historial: `capabilities` (tumbó el puerto 8890 diez minutos), `auth` (commit
35aa724) y `tool_detect` -- este último llegó con el detector de herramientas y
no se notó hasta que un push posterior forzó la reconstrucción.
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _copied_modules() -> set[str]:
    """Los .py nombrados en las instrucciones COPY, continuaciones incluidas.

    Sólo dentro de un COPY: este Dockerfile menciona `.py` en un comentario que
    justamente advierte de este fallo, y contarlo daba un nombre vacío.
    """
    names, in_copy = set(), False
    for raw in (REPO / "Dockerfile").read_text().splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line.upper().startswith("COPY"):
            in_copy = True
        elif not in_copy:
            continue
        for token in line.replace("\\", " ").split():
            if token.endswith(".py") and len(token) > 3:
                names.add(token[:-3])
        in_copy = raw.rstrip().endswith("\\")
    return names


def _local_modules() -> set[str]:
    return {p.stem for p in REPO.glob("*.py")}


def _imported_by_shipped_code() -> set[str]:
    local, needed = _local_modules(), set()
    for name in _copied_modules():
        source = REPO / f"{name}.py"
        if not source.exists():
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in local:
                        needed.add(root)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in local:
                    needed.add(root)
    return needed


def test_every_imported_module_is_in_the_image():
    """Lo que importa el código embarcado tiene que estar embarcado.

    Sin esto el fallo aparece en el ARRANQUE del contenedor, en producción, y
    con el puerto a oscuras -- nunca en los tests.
    """
    missing = _imported_by_shipped_code() - _copied_modules()
    assert not missing, (
        f"estos módulos se importan pero no están en el COPY del Dockerfile: "
        f"{sorted(missing)}. El contenedor va a hacer crash-loop con "
        f"ModuleNotFoundError.")


def test_the_manifest_only_names_files_that_exist():
    """Un nombre mal escrito en el COPY rompe el build, no el arranque, pero
    igual conviene cazarlo acá."""
    for name in _copied_modules():
        if name == "requirements":
            continue
        assert (REPO / f"{name}.py").exists(), f"el Dockerfile copia {name}.py y no existe"


def test_tool_detect_is_shipped():
    """Regresión explícita: este módulo tumbó el arranque el 2026-08-21."""
    assert "tool_detect" in _copied_modules()
