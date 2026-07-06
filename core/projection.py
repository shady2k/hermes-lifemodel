"""Prompt-safety projection: a woken drive → a human, desire-framed phrasing
(spec §13).

Raw affect never reaches the LLM. A drive value is bucketed into a band, and a
band maps to a small set of *synonymic* human phrasings; the choice is
pseudo-random but deterministic — a stable hash of the seed (the desire's
correlation id) — so a preamble is neither monotonous nor unreproducible. The
phrasing carries feeling, never numbers. Default strings are Russian (the being's
language); they are content, localizable later.
"""

from __future__ import annotations

import hashlib

# (low_inclusive_multiple_of_theta, synonyms) — bands over u/theta.
_CONTACT_BANDS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (
        2.5,
        (
            "заметно соскучился — тянет написать первым",
            "давно хочется на связь, скучаешь по нему",
        ),
    ),
    (
        1.5,
        (
            "ловишь себя на мыслях о нём",
            "хочется услышать, как он там",
        ),
    ),
    (
        1.0,
        (
            "тихое желание побыть на связи",
            "лёгкая тяга черкнуть пару слов",
        ),
    ),
)


def _seed_index(seed: str, n: int) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def project_contact(value: float, *, theta: float, seed: str) -> tuple[str, str]:
    """Return ``(phrasing, projection_id)`` for a contact-drive value."""
    ratio = value / theta if theta else value
    for band_index, (low, synonyms) in enumerate(_CONTACT_BANDS):
        if ratio >= low:
            choice = _seed_index(seed, len(synonyms))
            projection_id = f"contact.b{band_index}.s{choice}"
            return synonyms[choice], projection_id
    # below threshold — no pull worth framing (defensive; cognition gates on wake)
    return "нет заметной тяги к контакту", "contact.none"
