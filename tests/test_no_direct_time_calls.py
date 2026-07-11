"""Static guard: runtime code reads/serializes time ONLY through the helpers (spec §5).

The unified-time invariant (lm-fib.10) rests on a single source of "now" (the
``ClockPort`` → ``adapters/clock.py:SystemClock``) and a single (de)serializer
(``core/timeutil.py``: ``to_iso``/``from_iso``/``to_epoch_seconds``/``to_display``).
If any runtime module reaches past those — a stray ``datetime.now()``, a raw
``dt.isoformat()`` writing an un-normalized string — the ordering/expiry
correctness proven on ``to_iso``'s fixed-width normalization silently rots (a
whole-second instant serializes to ``...T12:00:00+00:00`` with no µs and misorders
lexically against normalized text). This AST linter, a sibling of
``test_no_absolute_self_imports.py``, closes that gap.

Precise by construction (codex #5): it flags the constructors/readers
(``datetime.now(`` / ``datetime.utcnow(`` / ``datetime.fromtimestamp(`` /
``datetime.timestamp(``) and a time (de)serialize — ``.isoformat(`` / ``.timestamp(``
— only when the *receiver* is time-named (``now``/``dt``/``ts``/``started``/``ended``/
``when``/``instant`` or an ``*_at`` name). It does NOT flag a legitimate
``to_epoch_seconds(...)`` epoch-VALUED metric, a non-datetime ``.isoformat()``
(e.g. a ``date``), nor a chained ``x.clock.now().isoformat()`` (a Call receiver —
e.g. the deliberate ``schema_migrations.applied_at`` audit column). The ONLY
sanctioned homes are ``adapters/clock.py`` (the one ``datetime.now(UTC)`` read) and
``core/timeutil.py`` (the one place isoformat/fromisoformat/timestamp live).

Stdlib-only (``ast``, ``pathlib``) so it runs inside Hermes' own venv too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# The package root is the directory that contains the runtime dirs (this test
# lives in ``<pkg>/tests/``).
_PKG_ROOT = Path(__file__).resolve().parent.parent

#: Runtime source dirs — the Hermes-free engine + the adapter boundary. Scanned
#: recursively (``domain/objects/`` etc. count).
_RUNTIME_DIRS = ("core", "domain", "state", "adapters", "ports", "sim")

#: Files under the package root that are runtime code, scanned non-recursively.
#: ``conftest.py`` is a pytest harness file, not runtime, so it is exempt.
_ROOT_EXCLUDE = {"conftest.py"}

#: The ONLY sanctioned homes (spec §5 allowlist), relative to the package root:
#: ``adapters/clock.py`` is the one system-time read; ``core/timeutil.py`` is the
#: one place time strings are (de)serialized. Both are exempt from the scan.
_ALLOWLIST = {"adapters/clock.py", "core/timeutil.py"}

#: ``datetime.<attr>(...)`` constructors/readers banned everywhere else — reading
#: "now" is the ClockPort's job, epoch is ``to_epoch_seconds``'s.
_BANNED_DATETIME_ATTRS = frozenset({"now", "utcnow", "fromtimestamp", "timestamp"})

#: Method calls that serialize an instant to a string / epoch value.
_TIME_METHODS = frozenset({"isoformat", "timestamp"})

#: Receiver identifiers whose ``.isoformat()``/``.timestamp()`` is a *time*
#: (de)serialize (as opposed to a ``date``/``Decimal``/… that also has those
#: methods). ``*_at`` names (``created_at``, ``expires_at``, …) are time by
#: convention and matched by suffix.
_RECEIVER_NAMES = frozenset({"now", "dt", "ts", "started", "ended", "when", "instant"})


def _receiver_terminal_name(node: ast.expr) -> str | None:
    """The identifier a method receiver ultimately reads as — ``foo`` for the Name
    ``foo`` and ``x.foo`` for the Attribute ``x.foo``. A Call (``x.foo()``) or any
    other expression has no terminal *name*, so ``.isoformat()`` on it is not
    receiver-name-flagged (that is how the deliberate ``self._clock.now().isoformat()``
    audit column and helper calls stay clean)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_time_named(name: str | None) -> bool:
    return name is not None and (name in _RECEIVER_NAMES or name.endswith("_at"))


def _offenders_in(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, offending-snippet)`` for every banned direct time call in
    the file's AST — at module level, inside functions, or under ``TYPE_CHECKING``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        func = node.func
        receiver = func.value
        # `datetime.now(` / `datetime.utcnow(` / `datetime.fromtimestamp(` /
        # `datetime.timestamp(` — a call on the `datetime` class itself.
        if (
            isinstance(receiver, ast.Name)
            and receiver.id == "datetime"
            and func.attr in _BANNED_DATETIME_ATTRS
        ):
            found.append((node.lineno, f"datetime.{func.attr}(...)"))
            continue
        # `<time-named>.isoformat(` / `<time-named>.timestamp(` — a raw (de)serialize
        # of an instant that must go through timeutil instead.
        if func.attr in _TIME_METHODS:
            name = _receiver_terminal_name(receiver)
            if _is_time_named(name):
                found.append((node.lineno, f"{name}.{func.attr}(...)"))
    return found


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for d in _RUNTIME_DIRS:
        files.extend(sorted((_PKG_ROOT / d).rglob("*.py")))
    for f in sorted(_PKG_ROOT.glob("*.py")):
        if f.name not in _ROOT_EXCLUDE:
            files.append(f)
    return [f for f in files if f.relative_to(_PKG_ROOT).as_posix() not in _ALLOWLIST]


def test_no_direct_time_calls_in_runtime_code() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        rel = path.relative_to(_PKG_ROOT)
        for lineno, snippet in _offenders_in(path):
            offenders.append(f"{rel}:{lineno}: {snippet}")

    assert not offenders, (
        "Runtime code must read time via the injected ClockPort and (de)serialize it "
        "via core/timeutil (to_iso/from_iso/to_epoch_seconds/to_display) — never a "
        "raw datetime.now()/utcnow()/fromtimestamp() nor a time-named .isoformat()/"
        ".timestamp(). The ONLY sanctioned homes are adapters/clock.py and "
        "core/timeutil.py. Offenders:\n  " + "\n  ".join(offenders)
    )


# --- The linter's own detection branches ---------------------------------
# The real tree, once migrated, exercises NONE of the banned forms, so pin every
# branch (and the negatives that must NOT trip) against synthetic source — else a
# regression could quietly blind the linter to a whole class (codex #5).

_BANNED = [
    "datetime.now()",
    "datetime.now(UTC)",
    "datetime.utcnow()",
    "datetime.fromtimestamp(0)",
    "datetime.timestamp(dt)",
    "x = now.isoformat()",
    "x = ctx.now.isoformat()",  # attribute receiver named `now`
    "x = created_at.isoformat()",  # *_at suffix
    "x = expires_at.isoformat()",
    "x = dt.isoformat()",
    "x = ts.timestamp()",
    "x = started.isoformat()",
    "x = ended.isoformat()",
    "x = when.isoformat()",
    "x = instant.timestamp()",
    "x = datetime.now(UTC).isoformat()",  # caught via the inner datetime.now call
    # Anywhere in the AST — inside a function and under TYPE_CHECKING.
    "def f():\n    return now.isoformat()",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    y = datetime.utcnow()",
]

_ALLOWED = [
    "x = to_iso(now)",  # the canonical serializer
    "x = from_iso(s)",  # the canonical parser
    "x = to_epoch_seconds(now)",  # the epoch-VALUED metric helper — NOT a false positive
    "x = to_display(value, tz)",
    "x = datetime.fromisoformat(s)",  # not in the banned datetime-attr set
    "x = self._clock.now().isoformat()",  # Call receiver — the deliberate audit column
    "x = lm.clock.now().isoformat()",  # chained Call receiver, likewise not name-flagged
    "x = some_date.isoformat()",  # a `date`/other object, receiver not time-named
    "x = record.isoformat()",  # receiver not time-named
    "x = payload.timestamp()",  # a Kafka-ish field, receiver not time-named
    "x = message.timestamp",  # attribute access, not a call
    "x = dt.astimezone(UTC)",  # a datetime op that is not a banned (de)serialize
]


@pytest.mark.parametrize("src", _BANNED)
def test_linter_flags_every_banned_form(tmp_path: Path, src: str) -> None:
    f = tmp_path / "sample.py"
    f.write_text(src, encoding="utf-8")
    assert _offenders_in(f), f"linter missed a banned direct time call: {src!r}"


@pytest.mark.parametrize("src", _ALLOWED)
def test_linter_ignores_legitimate_time_code(tmp_path: Path, src: str) -> None:
    f = tmp_path / "sample.py"
    f.write_text(src, encoding="utf-8")
    assert not _offenders_in(f), f"linter false-positived on legitimate code: {src!r}"
