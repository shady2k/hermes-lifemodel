"""AST enforcement of the observability invariant (spec §4.5, bead lm-edu.4).

Types + API close the STATUTORY path (a tick component gets only a
``SpanBoundLogger``; an async intent needs ``origin_traceparent``). The
escape-hatches — a bare ``logging.getLogger``/``print`` inside a tick component,
an ``import structlog``, a resurrected ``EventTee``, an async span that forgets
to thread the origin trace — are NOT closeable by types, so this stdlib ``ast``
guard closes them and runs inside ``make check``.

Reflecting the CLEAN END STATE after the Phase-4 removal:

* **stdlib-only** — no runtime module imports ``structlog``/``loguru``, calls
  ``print``, or references the deleted ``EventTee``/``EventSink``/``get_logger``/
  ``_StdlibEventLogger``/``EventLogger``/``EVENTS_FILENAME``/``events.jsonl``.
* **two logging surfaces** — ``logging.getLogger`` is ALLOWLISTED to the
  lifecycle/registration/boundary/adapter modules (spec §4.5); every ``core/``
  tick component logs only through its ``SpanBoundLogger`` and so may not import
  ``logging`` or call ``print`` at all.
* **origin threading** — every ``LaunchProactive`` construction and every
  ``open_correlated_span`` call threads ``origin_traceparent`` (an async launch/
  outcome can never fall onto a fresh/foreign trace), and the ``post_llm``
  handler rebinds the origin from the state anchor.

Negative fixtures below prove a deliberately-violating snippet is caught, so the
guard cannot silently rot into a no-op.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

#: The lifemodel package root (this file is tests/architecture/…).
_PKG = Path(__file__).resolve().parents[2]

#: Directories that are NOT the runtime tree (dev/test support, caches, docs).
_EXCLUDE_PARTS = {"tests", "testing", "__pycache__", "docs", ".git", ".beads", ".venv"}

#: The ONLY runtime modules permitted to touch stdlib ``logging`` directly
#: (spec §4.5 allowlist): the log surface itself, lifecycle/registration, the
#: reach-in boundary, the being platform, the disposable-store infra, the
#: fail-loud wiring backbone (spec §4.3 — ``wire``/``BrainHealth`` log at wiring
#: boundaries with NO ambient span), the brain-liveness status renderer (spec §4.4 —
#: logs a fail-soft WARNING on a degraded health/state read at the ``/lifemodel status``
#: boundary, again with no ambient span), the afferent hook boundary (a frame-machinery
#: failure is observed before any span exists, spec §4.3/MAJOR-4), and the adapters
#: that have no ambient span. Everything else — every ``core/`` tick component above
#: all — must log ONLY through a ``SpanBoundLogger``.
_LOGGING_ALLOWLIST: frozenset[str] = frozenset(
    {
        "log.py",
        "__init__.py",
        "gateway_core.py",
        "hooks.py",
        "adapters/being_platform.py",
        "adapters/delivery.py",
        "state/sqlite_store.py",
        "state/trace_store.py",
        "state/metrics_store.py",
        "state/brain_health.py",
        "state/brain_liveness.py",
        "state/wiring.py",
    }
)

#: Identifiers whose mere appearance means the old machinery came back.
_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "EventTee",
        "EventSink",
        "_StdlibEventLogger",
        "EventLogger",
        "get_logger",
        "EVENTS_FILENAME",
    }
)

#: Import roots that must never appear in the runtime tree (stdlib-only, §v1.2).
_FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset({"structlog", "loguru"})

#: String literals that betray a deleted durable sink.
_FORBIDDEN_STRINGS: frozenset[str] = frozenset({"events.jsonl"})


# --------------------------------------------------------------------------- #
# Runtime-tree discovery
# --------------------------------------------------------------------------- #


def _runtime_files() -> list[Path]:
    files: list[Path] = []
    for path in _PKG.rglob("*.py"):
        rel_parts = path.relative_to(_PKG).parts
        if any(part in _EXCLUDE_PARTS for part in rel_parts):
            continue
        if path.name == "conftest.py":
            continue
        files.append(path)
    return files


def _rel(path: Path) -> str:
    return path.relative_to(_PKG).as_posix()


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# --------------------------------------------------------------------------- #
# Small AST helpers
# --------------------------------------------------------------------------- #


def _call_func_name(node: ast.Call) -> str | None:
    """The simple callee name of a Call (``foo(...)`` or ``bar.foo(...)``)."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _has_kw(node: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in node.keywords)


def _imports_root(tree: ast.Module, roots: frozenset[str]) -> set[str]:
    """Root module names imported by *tree* that are in *roots*."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in roots:
                    found.add(alias.name.split(".")[0])
        elif (
            isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in roots
        ):
            found.add(node.module.split(".")[0])
    return found


def _uses_logging(tree: ast.Module) -> bool:
    """True if the module imports ``logging`` or calls ``*.getLogger(...)``."""
    if _imports_root(tree, frozenset({"logging"})):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_func_name(node) == "getLogger":
            return True
    return False


# --------------------------------------------------------------------------- #
# Checks (pure: (rel_path, tree) -> list[violation strings])
# --------------------------------------------------------------------------- #


def check_no_forbidden_imports_names_or_print(rel: str, tree: ast.Module) -> list[str]:
    out: list[str] = []
    for root in sorted(_imports_root(tree, _FORBIDDEN_IMPORT_ROOTS)):
        out.append(f"{rel}: imports forbidden runtime dependency {root!r} (stdlib-only, §v1.2)")
    for node in ast.walk(tree):
        # Deleted identifiers (as a Name, an Attribute tail, or an imported name).
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            out.append(f"{rel}: references deleted logging identifier {node.id!r}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            out.append(f"{rel}: references deleted logging identifier {node.attr!r}")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _FORBIDDEN_NAMES:
                    out.append(f"{rel}: imports deleted logging identifier {alias.name!r}")
        elif isinstance(node, ast.Call) and _call_func_name(node) == "print":
            out.append(f"{rel}: calls print() — use SpanLogger or logging (spec §4.5)")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in _FORBIDDEN_STRINGS
        ):
            out.append(f"{rel}: references deleted durable sink {node.value!r}")
    return out


def check_logging_allowlist(rel: str, tree: ast.Module) -> list[str]:
    if _uses_logging(tree) and rel not in _LOGGING_ALLOWLIST:
        return [
            f"{rel}: uses stdlib logging outside the allowlist — a tick component "
            "must log ONLY through its SpanBoundLogger (spec §4.5)"
        ]
    return []


def check_launch_proactive_threads_origin(rel: str, tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _call_func_name(node) == "LaunchProactive"
            and not _has_kw(node, "origin_traceparent")
        ):
            out.append(
                f"{rel}: constructs LaunchProactive without origin_traceparent — an "
                "async launch must carry its origin trace (spec §4.4)"
            )
    return out


def check_correlated_span_threads_origin(rel: str, tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _call_func_name(node) == "open_correlated_span"
            and not _has_kw(node, "origin_traceparent")
        ):
            out.append(
                f"{rel}: open_correlated_span without origin_traceparent — an async "
                "outcome span must rebind the origin, never fall onto a fresh trace "
                "(spec §4.4)"
            )
    return out


#: The async-boundary handlers that observe an outcome across the Hermes seam
#: MUST rebind the origin from the durable state anchor (spec §4.4 miss policy).
_ORIGIN_ANCHOR = "pending_proactive_origin_traceparent"


def check_post_llm_rebinds_origin(rel: str, tree: ast.Module) -> list[str]:
    if rel != "hooks.py":
        return []
    referenced = any(
        (isinstance(n, ast.Name) and n.id == _ORIGIN_ANCHOR)
        or (isinstance(n, ast.Attribute) and n.attr == _ORIGIN_ANCHOR)
        for n in ast.walk(tree)
    )
    if not referenced:
        return [
            f"{rel}: the post_llm handler must read {_ORIGIN_ANCHOR!r} to rebind the "
            "async outcome onto its origin trace (spec §4.4)"
        ]
    return []


_ALL_CHECKS = (
    check_no_forbidden_imports_names_or_print,
    check_logging_allowlist,
    check_launch_proactive_threads_origin,
    check_correlated_span_threads_origin,
    check_post_llm_rebinds_origin,
)


def _all_violations() -> Iterator[str]:
    for path in _runtime_files():
        rel = _rel(path)
        tree = _parse(path)
        for check in _ALL_CHECKS:
            yield from check(rel, tree)


# --------------------------------------------------------------------------- #
# The guard, run over the REAL runtime tree
# --------------------------------------------------------------------------- #


def test_runtime_tree_has_no_trace_invariant_violations() -> None:
    violations = list(_all_violations())
    assert violations == [], "trace-invariant violations:\n" + "\n".join(violations)


def test_guard_actually_scanned_the_runtime_tree() -> None:
    # A guard that scanned nothing is a green lie — prove it saw the real modules.
    rels = {_rel(p) for p in _runtime_files()}
    assert "core/coreloop.py" in rels
    assert "hooks.py" in rels
    assert "log.py" in rels
    # And it must NOT be scanning tests/testing (they legitimately use print/caplog).
    assert not any(r.startswith(("tests/", "testing/")) for r in rels)


# --------------------------------------------------------------------------- #
# Negative fixtures — a deliberately-violating snippet MUST be caught
# --------------------------------------------------------------------------- #


def _snippet(src: str) -> ast.Module:
    return ast.parse(src)


def test_negative_core_component_using_logging_is_caught() -> None:
    tree = _snippet("import logging\n_log = logging.getLogger('x')\n")
    assert check_logging_allowlist("core/rogue_component.py", tree)
    # …and an allowlisted module is NOT flagged for the same code.
    assert not check_logging_allowlist("state/sqlite_store.py", tree)


def test_negative_print_in_runtime_is_caught() -> None:
    tree = _snippet("def f():\n    print('debug')\n")
    assert check_no_forbidden_imports_names_or_print("core/x.py", tree)


def test_negative_structlog_import_is_caught() -> None:
    assert check_no_forbidden_imports_names_or_print("core/x.py", _snippet("import structlog\n"))
    assert check_no_forbidden_imports_names_or_print(
        "core/x.py", _snippet("from loguru import logger\n")
    )


def test_negative_resurrected_event_machinery_is_caught() -> None:
    for src in (
        "from .events import EventSink\n",
        "logger = EventTee(base, sink)\n",
        "x = get_logger('lifemodel')\n",
        "p = sdir / EVENTS_FILENAME\n",
        "PATH = 'events.jsonl'\n",
    ):
        assert check_no_forbidden_imports_names_or_print("__init__.py", _snippet(src)), src


def test_negative_launch_without_origin_is_caught() -> None:
    caught = check_launch_proactive_threads_origin(
        "core/cognition.py", _snippet("LaunchProactive(prompt=p, correlation_id=c)\n")
    )
    assert caught
    ok = check_launch_proactive_threads_origin(
        "core/cognition.py",
        _snippet("LaunchProactive(prompt=p, origin_traceparent=o)\n"),
    )
    assert not ok


def test_negative_correlated_span_without_origin_is_caught() -> None:
    caught = check_correlated_span_threads_origin(
        "hooks.py", _snippet("open_correlated_span(tracer=t, writer=w, ring=r)\n")
    )
    assert caught
    ok = check_correlated_span_threads_origin(
        "hooks.py", _snippet("open_correlated_span(tracer=t, origin_traceparent=o)\n")
    )
    assert not ok


def test_negative_post_llm_without_origin_rebind_is_caught() -> None:
    caught = check_post_llm_rebinds_origin("hooks.py", _snippet("x = state.pending_proactive_id\n"))
    assert caught
    ok = check_post_llm_rebinds_origin(
        "hooks.py", _snippet("o = state.pending_proactive_origin_traceparent\n")
    )
    assert not ok
