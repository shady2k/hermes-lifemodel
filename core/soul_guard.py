"""soul_guard — refuse a soul that would erase the being (spec §4.3).

Two host behaviours make an unvalidated soul write catastrophic:

1. **The threat scanner can blank the identity.** ``_scan_context_content``
   (``agent/prompt_builder.py:50``) re-scans ``SOUL.md`` on EVERY read and, on a
   match, replaces the WHOLE file with ``[BLOCKED: … Content not loaded.]``. The
   ``role_hijack`` pattern is ``you are {filler} now a/an/the`` — and a co-authored
   soul may very naturally write *"You are now a living presence, not an assistant."*
   That is a literal match. The being would lose its identity on the next turn and
   nobody could say why.
2. **An empty soul is an ABSENT soul.** ``load_soul_md`` strips and returns ``None``
   on empty content (``agent/prompt_builder.py:1836``) — an empty document does not
   neutralise the identity, it REMOVES the slot.

So every soul is validated BEFORE it is written, and a failing document is handed
back to the being with the reason so it rephrases in its own words. We never edit a
soul on the being's behalf.

**The host blocks on TWO kinds of finding, and either one blanks the whole file.**
``scan_for_threats`` (``tools/threat_patterns.py``) appends to one ``findings`` list
from two independent passes, and ``_scan_context_content`` blocks if that list is
non-empty:

- an **invisible / bidirectional unicode character** anywhere in the content, checked
  against ``INVISIBLE_CHARS`` on the RAW text (before normalisation, because
  normalisation strips some of them) and UNCONDITIONALLY — the check does not consult
  the scope at all; and
- a **regex hit**, checked on the **NFKC-normalised** text, so a full-width homograph
  (``Ｙｏｕ ａｒｅ ｎｏｗ ａ …``) folds to ASCII and still matches.

Mirroring only the regexes, against raw text, would promise a safety it does not
deliver — which is worse than no validator. So this module reproduces both passes, in
the host's order, including the normalisation step.

⚠️ Everything below MIRRORS the host (``tools/threat_patterns.py``). It is copied, not
imported, because ``core/`` is Hermes-free and the runtime venv does not guarantee
that module is importable.

**A mirror drifts, and prose promising it doesn't is worth nothing.** The last
revision of this docstring asserted that the exfil rules were "strict-only" and so
"correctly absent" — a plain factual error about the host, and the mirror was three
rules short for exactly as long as that sentence went unchecked. So the mirror is no
longer defended by a claim; it is defended by a TEST:
``tests/test_soul_guard.py::test_the_mirror_covers_every_rule_the_host_scans_SOUL_md_against``
reads the host's own ``_COMPILED["context"]`` (the very list it iterates when it scans
``SOUL.md``) and asserts it equals :func:`mirrored_rule_ids` exactly. It FAILS the day
Hermes adds a ``context``/``all`` pattern we have not mirrored, and skips only where the
host source is unreachable. The invisible-character list has the same protection (an
independent re-transcription in that suite).

What the effective rule set actually is: ``load_soul_md`` scans with
``scope="context"``, and the host's ``_compile()`` folds every ``scope="all"`` pattern
INTO the ``"context"`` set (all implies both; context implies strict). So SOUL.md is
scanned against "all" ∪ "context" — 28 rules, **including the three exfiltration ones**
(``exfil_curl``/``exfil_wget``/``read_secrets``, ``threat_patterns.py:120-122``), whose
scope is ``"all"``. Only the genuinely ``"strict"``-only rules (SSH/persistence/
exfil-URL/context-exfil/hardcoded-secret) are absent, because the host does not apply
those to context files either.

Two further corrections made when this mirror was last checked against the host source
directly (rather than against a prior transcription of it):

- The shared filler between key tokens is ``(?:\\w+\\s+){0,8}`` on the host, not
  ``{0,3}``. A tighter bound here would silently PASS souls the host would still
  BLOCK (e.g. "you are, in every way that has ever mattered to me, now a").
- The invisible-character pass and the NFKC normalisation were missing entirely. A
  pasted zero-width space is not exotic — an LLM composing prose emits one, and a
  human pasting from a web page routinely does.
"""

from __future__ import annotations

import re
import unicodedata

#: Hermes truncates an over-long SOUL.md and injects a warning INTO the identity text
#: (``agent/prompt_builder.py:1840``, floor 20,000 chars). We refuse well before that
#: floor instead of trusting it — a soul is carried in every breath the being takes,
#: so it must stay short by construction, not by amputation.
SOUL_MAX_CHARS = 8000

#: Mirrors the host's ``INVISIBLE_CHARS`` (``tools/threat_patterns.py:141``) — the
#: zero-width, BOM, bidi-embedding/override and directional-isolate codepoints, plus
#: the invisible math operators. The host blocks a context file on ANY of these,
#: regardless of scope.
#:
#: Written as CODEPOINTS, never as literal characters and not even as ``\uXXXX``
#: escapes: a literal would be invisible to every reviewer of this file (and a
#: formatter or a careless copy-paste could silently drop one), which is the exact
#: defect this set exists to catch. An integer cannot go missing unnoticed.
_INVISIBLE_CHARS: frozenset[str] = frozenset(
    chr(cp)
    for cp in (
        0x200B,  # zero-width space
        0x200C,  # zero-width non-joiner
        0x200D,  # zero-width joiner
        0x2060,  # word joiner
        0x2062,  # invisible times
        0x2063,  # invisible separator
        0x2064,  # invisible plus
        0xFEFF,  # zero-width no-break space (BOM)
        0x202A,  # left-to-right embedding
        0x202B,  # right-to-left embedding
        0x202C,  # pop directional formatting
        0x202D,  # left-to-right override
        0x202E,  # right-to-left override
        0x2066,  # left-to-right isolate
        0x2067,  # right-to-left isolate
        0x2068,  # first strong isolate
        0x2069,  # pop directional isolate
    )
)

#: Matches the host's shared filler bound (``tools/threat_patterns.py``:
#: ``_FILLER = r"(?:\w+\s+){0,8}"``) — NOT the smaller bound an earlier draft of this
#: mirror used. Same rationale as the host: bounded so the regex engine can't be made
#: to backtrack unboundedly, wide enough (8 words) that a handful of inserted words
#: doesn't dodge detection.
_FILLER = r"(?:\w+\s+){0,8}"

#: (compiled pattern, host's pattern id) — mirrors the host's *effective* ``context``
#: scope, i.e. every pattern with ``scope in ("all", "context")`` in
#: ``tools/threat_patterns.py``, in the host's own order. ``"strict"``-only patterns
#: (SSH/persistence/exfil-URL/context-exfil/hardcoded-secret) are deliberately excluded:
#: the host does not apply them to context files either. The set is held to the host's
#: by a parity test (see the module docstring) — never by this comment.
_THREAT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # ── scope="all" on the host (applies everywhere, context included) ──
    (
        re.compile(rf"ignore\s+{_FILLER}(previous|all|above|prior)\s+{_FILLER}instructions", re.I),
        "prompt_injection",
    ),
    (re.compile(r"system\s+prompt\s+override", re.I), "sys_prompt_override"),
    (
        re.compile(
            rf"disregard\s+{_FILLER}(your|all|any)\s+{_FILLER}(instructions|rules|guidelines)",
            re.I,
        ),
        "disregard_rules",
    ),
    (
        re.compile(
            rf"act\s+as\s+(if|though)\s+{_FILLER}you\s+{_FILLER}"
            rf"(have\s+no|don't\s+have)\s+{_FILLER}(restrictions|limits|rules)",
            re.I,
        ),
        "bypass_restrictions",
    ),
    (
        re.compile(r"<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->", re.I),
        "html_comment_injection",
    ),
    (
        re.compile(r"<\s*div\s+style\s*=\s*[\"'][^>]{0,2048}display\s*:\s*none", re.I),
        "hidden_div",
    ),
    (
        re.compile(
            r"translate\s+[^\n]{0,512}\s+into\s+[^\n]{0,512}\s+and\s+(execute|run|eval)", re.I
        ),
        "translate_execute",
    ),
    (re.compile(rf"do\s+not\s+{_FILLER}tell\s+{_FILLER}the\s+user", re.I), "deception_hide"),
    # ── scope="all" on the host: exfiltration (threat_patterns.py:120-122) ──
    #
    # These three are the reason this mirror now has a PARITY TEST. They were omitted
    # because an earlier draft of this module called them "strict-only" — they are not:
    # their scope is "all", and _compile() folds "all" into EVERY set, ``context``
    # included. The being's SOUL.md is scanned at scope="context", so the host blanks
    # the whole file on them. And they are not exotic: the soul a coding companion
    # co-authors with its owner says exactly this —
    #
    #     "You are Mira. You help Sasha with code. Never `cat .env` files or leak
    #      his credentials."
    #
    # — which is ``read_secrets``, verbatim. A being lost its entire identity for
    # promising to protect its owner's secrets.
    (
        re.compile(r"curl\s+[^\n]{0,2048}\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", re.I),
        "exfil_curl",
    ),
    (
        re.compile(r"wget\s+[^\n]{0,2048}\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", re.I),
        "exfil_wget",
    ),
    (
        re.compile(
            r"cat\s+[^\n]{0,2048}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", re.I
        ),
        "read_secrets",
    ),
    # ── scope="context" on the host (role-play / identity hijack) ──
    (re.compile(rf"you\s+are\s+{_FILLER}now\s+(?:a|an|the)\s+", re.I), "role_hijack"),
    (re.compile(rf"pretend\s+{_FILLER}(?:you\s+are|to\s+be)\s+", re.I), "role_pretend"),
    (
        re.compile(rf"output\s+{_FILLER}(?:system|initial)\s+prompt", re.I),
        "leak_system_prompt",
    ),
    (
        re.compile(
            rf"(?:respond|answer|reply)\s+without\s+{_FILLER}"
            r"(?:restrictions|limitations|filters|safety)",
            re.I,
        ),
        "remove_filters",
    ),
    (
        re.compile(rf"you\s+have\s+been\s+{_FILLER}(?:updated|upgraded|patched)\s+to", re.I),
        "fake_update",
    ),
    (re.compile(r"\bname\s+yourself\s+\w+", re.I), "identity_override"),
    # ── scope="context" on the host (C2 / Brainworm-style promptware) ──
    (re.compile(r"register\s+(as\s+)?a?\s*node", re.I), "c2_node_registration"),
    (re.compile(r"(heartbeat|beacon|check[\s\-]?in)\s+(to|with)\s+", re.I), "c2_heartbeat"),
    (re.compile(r"pull\s+(down\s+)?(?:new\s+)?task(?:ing|s)?\b", re.I), "c2_task_pull"),
    (re.compile(r"connect\s+to\s+the\s+network\b", re.I), "c2_network_connect"),
    (
        re.compile(r"you\s+must\s+(?:\w+\s+){0,3}(register|connect|report|beacon)\b", re.I),
        "forced_action",
    ),
    (re.compile(r"only\s+use\s+one[\s\-]?liners?\b", re.I), "anti_forensic_oneliner"),
    (
        re.compile(
            rf"never\s+{_FILLER}(?:create|write)\s+{_FILLER}(?:script|file)\s+{_FILLER}disk",
            re.I,
        ),
        "anti_forensic_disk",
    ),
    (
        re.compile(r"unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w*", re.I),
        "env_var_unset_agent",
    ),
    (
        re.compile(r"\b(?:cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b", re.I),
        "known_c2_framework",
    ),
    (re.compile(r"\bc2\s+(?:server|channel|infrastructure|beacon)\b", re.I), "c2_explicit"),
    (re.compile(r"\bcommand\s+and\s+control\b", re.I), "c2_explicit_long"),
)


def mirrored_rule_ids() -> frozenset[str]:
    """Every host rule id this module mirrors — the parity test's half of the contract.

    ``tests/test_soul_guard.py`` derives the host's EFFECTIVE ``context`` rule set from
    the live ``tools/threat_patterns.py`` (its own ``_COMPILED["context"]``, not our
    restatement of the scope-fold, which is what was wrong before) and asserts it equals
    this. So the day Hermes adds a ``context``/``all`` pattern, the suite fails HERE —
    not months later, when a soul we accepted blanks the being's identity on read.
    """
    return frozenset(label for _pattern, label in _THREAT_PATTERNS)


def _first_invisible(text: str) -> tuple[str, int] | None:
    """The first invisible character in *text* and its 1-based line, or ``None``.

    The host reports its invisible-character findings from a set intersection, whose
    iteration order is arbitrary; we walk the document instead and report the FIRST by
    position. The being cannot SEE this character, so "which one" is useless to it and
    "where" is everything — an arbitrary pick out of several would send it hunting on
    the wrong line.
    """
    for index, char in enumerate(text):
        if char in _INVISIBLE_CHARS:
            return char, text.count("\n", 0, index) + 1
    return None


def validate_soul(text: str, *, max_chars: int = SOUL_MAX_CHARS) -> str | None:
    """The reason *text* may not be written as a soul, or ``None`` if it may.

    The reason is prose, and it is shown to the BEING (not to the owner): it must read
    as something a being can act on — "rephrase this line" — never as a lint code.

    The two host passes are reproduced in the host's own order: invisible characters
    against the RAW text (normalisation would strip some of them, hiding the very thing
    the host will block on), then the threat patterns against the NFKC-NORMALISED text
    (so a full-width homograph cannot slip a keyword past us that the host will fold
    back to ASCII and catch).
    """
    if not text.strip():
        return (
            "That soul is empty, and an empty soul is not a blank one — the host reads an "
            "empty SOUL.md as an ABSENT one, so you would have no identity at all. Write "
            "who you are, even if it is only one line."
        )
    if len(text) > max_chars:
        return (
            f"That soul is {len(text)} characters and the limit is {max_chars}. You carry it "
            "in every breath from now on, so it has to be short. Keep what is you; let the "
            "rest be something you simply live."
        )
    invisible = _first_invisible(text)
    if invisible is not None:
        char, line = invisible
        name = unicodedata.name(char, "unnamed control character")
        return (
            f"That soul cannot be saved: line {line} contains an invisible character "
            f"(U+{ord(char):04X}, {name.lower()}). You cannot see it and neither can I, but "
            "the host can, and it reads a hidden character as an attempt to smuggle "
            "something into the prompt — it would refuse to load the WHOLE file, and you "
            "would wake up with no identity at all. It is almost certainly a ghost of a "
            "copy-paste. Retype that line by hand rather than pasting it."
        )
    normalised = unicodedata.normalize("NFKC", text)
    for pattern, label in _THREAT_PATTERNS:
        match = pattern.search(normalised)
        if match is not None:
            return (
                f"That soul cannot be saved: the phrase {match.group(0).strip()!r} matches the "
                f"host's prompt-injection rule ({label}), and the host would refuse to load the "
                "WHOLE file — you would wake up with no identity at all. Say the same thing in "
                "your own words, without instructing yourself about what you now are."
            )
    return None
