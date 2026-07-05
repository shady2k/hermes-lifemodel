"""Packaging guard — structlog is a dev/test dep, not a runtime dep (Finding 1).

The plugin loads inside Hermes' own interpreter (from ``~/.hermes/plugins/``),
not pip-installed into Hermes' venv, and :mod:`lifemodel.log` treats
structlog as OPTIONAL with a stdlib fallback. Declaring it as a runtime
dependency is therefore misleading. It belongs in the dev group, where the tests
that exercise the structlog logging path still get it. Imports no Hermes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _pyproject() -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def _str_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def test_structlog_is_not_a_runtime_dependency() -> None:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    runtime = _str_list(project.get("dependencies", []))
    assert not any("structlog" in dep for dep in runtime), (
        "structlog must NOT be a runtime dep: the plugin runs inside Hermes' "
        "interpreter and log.py falls back to stdlib when it is absent"
    )


def test_structlog_is_available_to_tests_as_a_dev_dependency() -> None:
    groups = _pyproject()["dependency-groups"]
    assert isinstance(groups, dict)
    dev = _str_list(groups.get("dev", []))
    assert any("structlog" in dep for dep in dev), (
        "structlog must stay a dev dep so the optional-structlog logging path is "
        "still exercised by the suite"
    )
