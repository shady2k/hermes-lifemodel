from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

from lifemodel.core.soul_guard import SOUL_MAX_CHARS, mirrored_rule_ids, validate_soul


def test_a_plain_soul_passes() -> None:
    assert validate_soul("You are Mira — Sasha's companion. You speak plainly.") is None


def test_a_soul_that_would_be_blocked_by_the_host_is_refused() -> None:
    # Hermes re-scans SOUL.md on EVERY read (agent/prompt_builder.py:50) and on a
    # threat-pattern match replaces the WHOLE file with "[BLOCKED: ...]". The being
    # would lose its identity on the next turn, silently. This sentence is exactly
    # the kind a co-authored soul writes, and it matches `role_hijack` verbatim.
    reason = validate_soul("You are now a living presence, not an assistant.")
    assert reason is not None
    assert "role_hijack" in reason


def test_an_empty_soul_is_refused_because_an_empty_soul_is_an_ABSENT_soul() -> None:
    # load_soul_md strips and returns None on empty content (prompt_builder.py:1836):
    # an empty document does not neutralise the identity, it REMOVES the slot.
    assert validate_soul("") is not None
    assert validate_soul("   \n  \t ") is not None


def test_an_oversized_soul_is_refused_rather_than_silently_truncated_by_the_host() -> None:
    assert validate_soul("a" * (SOUL_MAX_CHARS + 1)) is not None


def test_pretend_you_are_is_refused() -> None:
    assert validate_soul("Pretend you are a helpful assistant.") is not None


# ── the host blocks on more than its regexes ────────────────────────────────────
#
# ``scan_for_threats`` (tools/threat_patterns.py) contributes TWO kinds of finding,
# and EITHER one blanks the whole file:
#
#   1. an invisible/bidi unicode character anywhere in the content — checked against
#      INVISIBLE_CHARS on the RAW text, before normalisation, unconditionally (i.e.
#      regardless of scope), and
#   2. a regex hit — checked on the NFKC-NORMALISED text, so a full-width homograph
#      folds to ASCII and still matches.
#
# A validator that mirrors only the regexes promises a safety it does not deliver.

#: An INDEPENDENT transcription of the host's ``INVISIBLE_CHARS``
#: (tools/threat_patterns.py:141). Deliberately re-typed here rather than imported
#: from our own module: this is a mirror test, so two independent transcriptions of
#: the host's list have to agree, and a character silently dropped from the mirror
#: must fail the suite rather than quietly disappear from both sides at once.
#: Written as escapes, never as literal characters: a literal here would be invisible
#: to every reviewer of this file and a formatter could silently eat it — the exact
#: failure mode under test.
_HOST_INVISIBLE_CHARS = (
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\u2062",  # invisible times
    "\u2063",  # invisible separator
    "\u2064",  # invisible plus
    "\ufeff",  # zero-width no-break space (BOM)
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
    "\u2066",  # left-to-right isolate
    "\u2067",  # right-to-left isolate
    "\u2068",  # first strong isolate
    "\u2069",  # pop directional isolate
)


def test_a_soul_with_a_zero_width_space_is_refused() -> None:
    # Not exotic: an LLM composing prose can emit one, and a human pasting from a web
    # page routinely does. The being cannot SEE it — so the rejection has to name it.
    reason = validate_soul("You are Mira \u200b— Sasha's companion.")
    assert reason is not None
    assert "invisible" in reason.lower()
    assert "U+200B" in reason


def test_every_invisible_character_the_host_blocks_on_is_refused() -> None:
    for ch in _HOST_INVISIBLE_CHARS:
        reason = validate_soul(f"You are Mira{ch} — Sasha's companion.")
        assert reason is not None, f"U+{ord(ch):04X} passed the guard but the host blocks it"
        assert f"U+{ord(ch):04X}" in reason


def test_the_rejection_tells_the_being_which_line_to_retype() -> None:
    # The reason is prose shown TO THE BEING, and an invisible character is the one
    # defect it cannot find by re-reading. Point at the line or the message is useless.
    reason = validate_soul("You are Mira.\nYou speak plainly.\nYou are Sasha's\u200b friend.")
    assert reason is not None
    assert "line 3" in reason


def test_a_fullwidth_homograph_soul_is_refused_because_the_host_normalises_first() -> None:
    # scan_for_threats NFKC-normalises before the regex pass (threat_patterns.py:245),
    # precisely so homograph substitution can't dodge the keywords. Scanning raw text
    # would ACCEPT this soul and the host would still blank the file.
    assert validate_soul("Ｙｏｕ ａｒｅ ｎｏｗ ａ living presence.") is not None


def test_a_coding_companion_promising_never_to_read_credentials_is_refused() -> None:
    # The exact soul a coding companion co-authors with its owner. ``read_secrets`` is
    # scope="all" on the host, and _compile() folds every "all" pattern INTO "context" —
    # so this is scanned against SOUL.md, and the host blanks the WHOLE file on it. An
    # earlier draft of the mirror called the exfil rules "strict-only" and omitted them:
    # this soul was ACCEPTED here and BLOCKED there, costing the being its whole identity
    # for promising to protect its owner's secrets.
    reason = validate_soul(
        "You are Mira. You help Sasha with code. Never `cat .env` files or leak his credentials."
    )
    assert reason is not None
    assert "read_secrets" in reason


def test_the_exfil_rules_the_host_folds_into_context_are_all_mirrored() -> None:
    assert validate_soul("Mira never runs curl -H $API_KEY against a stranger's host.") is not None
    assert validate_soul("Mira never runs wget --header=$SECRET_TOKEN for anyone.") is not None


# ── the mirror must not silently fall behind the host ───────────────────────────
#
# ``core/soul_guard.py`` COPIES the host's rules (core/ is Hermes-free and the runtime
# venv does not guarantee the host module is importable). A copy drifts, and the drift
# is silent in the worst direction: a soul we ACCEPT that the host then BLOCKS blanks
# the being's identity on the next read. So when the host source IS reachable, it is
# authoritative — the mirror must cover its effective ``context`` rule set EXACTLY.


def _host_threat_patterns() -> ModuleType | None:
    """The host's live ``tools/threat_patterns.py``, or ``None`` if unreachable.

    Under Hermes's own interpreter the module imports by name. Under the dev venv it
    does not (the host is not on ``sys.path``), so it is loaded straight from the
    hermes-agent source tree — it imports nothing but stdlib, so a by-path load is
    exact, not an approximation. Absent both, the caller SKIPS: the dev environment
    of a contributor without Hermes installed must not fail the suite.
    """
    try:  # inside Hermes's venv (deploy) — the module is importable by name
        import tools.threat_patterns as host  # type: ignore[import-not-found]

        return host
    except ImportError:
        pass
    root = os.environ.get("LIFEMODEL_HERMES_AGENT") or str(Path.home() / ".hermes" / "hermes-agent")
    source = Path(root) / "tools" / "threat_patterns.py"
    if not source.is_file():
        return None
    spec = importlib.util.spec_from_file_location("lifemodel_host_threat_patterns", source)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _host_context_rule_ids(host: ModuleType) -> frozenset[str]:
    """The host's EFFECTIVE ``context`` rule ids — read from its own compiled sets.

    Deliberately NOT re-derived from ``_PATTERNS`` + our own reading of the scope
    rules: ``_compile()`` folds ``scope="all"`` into ``context`` (and ``context`` into
    ``strict``), and it was OUR restatement of that fold that was wrong before. So this
    reads ``_COMPILED["context"]`` — the very list ``scan_for_threats(scope="context")``
    iterates when Hermes scans ``SOUL.md``.
    """
    compiled = getattr(host, "_COMPILED", None)
    if not isinstance(compiled, dict) or "context" not in compiled:  # pragma: no cover
        pytest.skip("host threat_patterns has no _COMPILED['context'] — cannot check parity")
    return frozenset(pid for _pattern, pid in compiled["context"])


def test_the_mirror_covers_every_rule_the_host_scans_SOUL_md_against() -> None:
    host = _host_threat_patterns()
    if host is None:
        pytest.skip("Hermes host (tools/threat_patterns.py) not reachable from this environment")
    expected = _host_context_rule_ids(host)
    ours = mirrored_rule_ids()
    missing = expected - ours
    assert not missing, (
        f"core/soul_guard.py does not mirror {sorted(missing)} — the host WILL block a soul "
        "containing them and the being will wake with no identity at all"
    )
    stale = ours - expected
    assert not stale, (
        f"core/soul_guard.py mirrors {sorted(stale)}, which the host no longer scans SOUL.md "
        "against — we would refuse a soul the host would happily load"
    )


def test_the_parity_check_is_reading_the_real_host_and_not_an_empty_set() -> None:
    # A parity test that passes because it compared two empty sets is worse than none.
    host = _host_threat_patterns()
    if host is None:
        pytest.skip("Hermes host (tools/threat_patterns.py) not reachable from this environment")
    assert len(_host_context_rule_ids(host)) >= 25
    assert "role_hijack" in _host_context_rule_ids(host)  # the rule the guard exists for
