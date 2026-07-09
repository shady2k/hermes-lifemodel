"""Packaging guard — the plugin is stdlib-only, no structlog/loguru anywhere.

The plugin loads inside Hermes' own interpreter (from ``~/.hermes/plugins/``),
not pip-installed into Hermes' venv, so runtime deps must stay empty. Runtime
logging is stdlib only (spec §v1.2 / lm-edu): :class:`~lifemodel.log.SpanLogger`
for the tick path and ``logging.getLogger`` for lifecycle/boundary code —
structlog/loguru were removed outright, so they must appear in NO dependency
group (runtime OR dev), not just be demoted. Imports no Hermes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_BANNED = ("structlog", "loguru")


def _pyproject() -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def _str_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def test_runtime_dependencies_are_empty() -> None:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    assert _str_list(project.get("dependencies", [])) == [], (
        "runtime deps must stay empty: the plugin runs inside Hermes' interpreter"
    )


def test_no_structlog_or_loguru_in_any_dependency_group() -> None:
    proj = _pyproject()
    project = proj["project"]
    assert isinstance(project, dict)
    groups = proj.get("dependency-groups", {})
    assert isinstance(groups, dict)

    all_deps = _str_list(project.get("dependencies", []))
    for group in groups.values():
        all_deps.extend(_str_list(group))

    for banned in _BANNED:
        assert not any(banned in dep for dep in all_deps), (
            f"{banned!r} must not appear in any dependency group: runtime logging "
            "is stdlib-only (SpanLogger + logging.getLogger), nothing imports it"
        )
