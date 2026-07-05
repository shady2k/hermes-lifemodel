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
