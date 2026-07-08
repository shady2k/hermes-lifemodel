"""Defensive clock arithmetic shared by the pipeline (spec §17).

``minutes_between`` returns elapsed minutes from an ISO-8601 timestamp to a
``datetime``, and — like ``core/decision.py``'s private helper it generalises —
returns ``0.0`` ("no elapsed rise") for ``None``, an unparseable string, or a
tz-naive value, so a malformed ``last_tick_at`` never crashes a tick.
"""

from __future__ import annotations

from datetime import datetime


def minutes_between(a_iso: str | None, b: datetime) -> float:
    if a_iso is None:
        return 0.0
    try:
        a = datetime.fromisoformat(a_iso)
    except ValueError:
        return 0.0
    if a.tzinfo is None or a.utcoffset() is None:
        return 0.0
    return (b - a).total_seconds() / 60.0


_ELAPSED_BANDS: tuple[tuple[float, str], ...] = (
    (60.0, "совсем недавно"),
    (180.0, "пару часов назад"),
    (480.0, "несколько часов назад"),
    (1440.0, "сегодня, но уже порядочно прошло"),
    (2880.0, "со вчерашнего дня"),
    (5760.0, "уже несколько дней"),
    (11520.0, "около недели"),
    (43200.0, "не одну неделю"),
)


def humanize_elapsed(minutes: float | None) -> str:
    """Render an elapsed duration as a word-only Russian phrase (no digits).

    ``None`` means "no prior exchange to measure from". A negative value
    (clock skew) is clamped to "just now" rather than raising."""
    if minutes is None:
        return "вы ещё толком не общались"
    for upper, phrase in _ELAPSED_BANDS:
        if minutes < upper:
            return phrase
    return "очень давно"
