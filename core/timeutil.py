"""Defensive clock arithmetic shared by the pipeline (spec §17).

``minutes_between`` returns elapsed minutes from an ISO-8601 timestamp to a
``datetime``, and — like ``core/decision.py``'s private helper it generalises —
returns ``0.0`` ("no elapsed rise") for ``None``, an unparseable string, or a
tz-naive value, so a malformed ``last_tick_at`` never crashes a tick.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, tzinfo

_LOG = logging.getLogger("lifemodel.timeutil")


def to_iso(dt: datetime) -> str:
    """Serialize an aware ``datetime`` to canonical, fixed-width ISO-8601 UTC.

    Rejects a tz-naive *dt* (a naive value would silently misorder), converts to
    UTC, and returns ``.isoformat(timespec="microseconds")`` — always 6-digit
    microseconds, always ``+00:00``. That fixed width is load-bearing: it defeats
    Python's omission of ``.000000`` and makes the TEXT string lexically sortable
    == chronologically sortable, which is what all ordering/expiry now rests on.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"to_iso requires a timezone-aware datetime, got naive {dt!r}")
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def from_iso(s: str) -> datetime:
    """Strictly parse an ISO-8601 string to an aware UTC ``datetime``.

    The one storage parser (spec §3). Raises ``ValueError`` on a malformed
    string (``datetime.fromisoformat`` does) and, deliberately, on a string that
    parses to a tz-*naive* value — a naive stored instant would misorder against
    normalized text, so it is rejected here rather than silently assumed UTC. A
    parsed offset is normalized to UTC via ``astimezone(UTC)``.
    """
    parsed = datetime.fromisoformat(s)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"from_iso requires a timezone-aware timestamp, got naive {s!r}")
    return parsed.astimezone(UTC)


def to_epoch_seconds(dt: datetime) -> float:
    """Epoch seconds (UTC) for a legitimately epoch-VALUED metric (spec §2/§3).

    Not a storage-column path — epoch survives only as a *metric value* (e.g. a
    gauge whose value is epoch seconds). Rejects a tz-naive *dt* for the same
    reason as :func:`to_iso`, normalizes to UTC, and returns ``.timestamp()``.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"to_epoch_seconds requires a timezone-aware datetime, got naive {dt!r}")
    return dt.astimezone(UTC).timestamp()


def to_display(value: datetime | str, tz: tzinfo | None) -> str:
    """Render a stored instant in the owner's local zone for a human surface.

    The ONE fail-open path (spec §3 codex #7): a debug/status view must never be
    blanked by one bad row. Accepts an aware ``datetime`` or an ISO string; on a
    malformed or tz-naive value it logs a WARNING and returns the raw value as a
    string rather than raising (``from_iso`` stays strict — this is the display
    layer's tolerance, not the storage layer's). *tz* is the owner's zone from
    the adapter boundary; ``None`` falls back to the server's local zone, matching
    ``core/wake_packet.py``'s ``astimezone`` convention, and a bad *tz* falls back
    to UTC rather than drop the render.
    """
    if isinstance(value, str):
        try:
            dt = from_iso(value)
        except ValueError:
            _LOG.warning("to_display: unparseable stored time %r; showing raw", value)
            return value
    else:
        dt = value
        if dt.tzinfo is None or dt.utcoffset() is None:
            _LOG.warning("to_display: tz-naive datetime %r; showing raw", value)
            return str(value)
    try:
        local = dt.astimezone(tz)
    except Exception:  # noqa: BLE001 - a bad tz must never blank the view
        local = dt.astimezone(UTC)
    return local.isoformat()


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
