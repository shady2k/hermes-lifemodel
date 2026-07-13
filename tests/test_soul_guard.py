from __future__ import annotations

from lifemodel.core.soul_guard import SOUL_MAX_CHARS, validate_soul


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
