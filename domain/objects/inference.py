"""``InferredField`` — per-field inference metadata for the derived UserModel (§8).

A :class:`UserModel` field is not an authoritative scalar but a **derived belief
with a shelf life**: WHAT we inferred, WHEN we inferred it, and how long that
inference stays trustworthy. :class:`InferredField` bundles
``{value, inferred_at, ttl}``; once ``inferred_at + ttl`` is in the past the
field is **stale** and must read as :data:`UNKNOWN` (spec §8, "стухло →
неизвестно") — never as the stale value.

``ttl_seconds=None`` (or ``inferred_at=None``) means "never expires": an
authoritative value the owner set by hand, or a permissive default — not a
time-boxed inference — so owner-set boundaries and defaults never silently
vanish. This is the whole reason the UserModel (our cache of the Other) is split
from ``AgentState`` (the self, always fresh, no TTL).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Generic, TypeVar

T = TypeVar("T")


class _Unknown:
    """The singleton sentinel a stale field reads as (distinct from any value)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNKNOWN"


#: A field whose inference has gone stale reads as this — NOT as its old value and
#: NOT as ``None`` (a legitimate value); an explicit "we no longer know" marker.
UNKNOWN: Final = _Unknown()


@dataclass(frozen=True)
class InferredField(Generic[T]):
    """A single derived field: its ``value`` plus when it was inferred and its TTL.

    ``inferred_at`` is an ISO-8601, timezone-aware stamp; ``ttl_seconds`` is the
    inference's shelf life in seconds. Either being ``None`` means the value never
    expires (an owner-set / authoritative value or a permissive default).
    """

    value: T
    inferred_at: str | None = None
    ttl_seconds: float | None = None

    def is_stale(self, now: datetime) -> bool:
        """``True`` iff this is a time-boxed inference whose TTL has elapsed by *now*."""
        if self.inferred_at is None or self.ttl_seconds is None:
            return False
        deadline = datetime.fromisoformat(self.inferred_at) + timedelta(seconds=self.ttl_seconds)
        return now >= deadline

    def resolve(self, now: datetime) -> T | _Unknown:
        """The value if still fresh at *now*, else :data:`UNKNOWN` (stale → unknown)."""
        return UNKNOWN if self.is_stale(now) else self.value

    def resolve_or(self, now: datetime, default: T) -> T:
        """The value if still fresh at *now*, else *default* (stale reads as unknown).

        The typed convenience for consumers that have a permissive fallback (an
        empty tuple, ``""``) — a stale inference degrades to "no information",
        which for the UserModel is exactly the behavior-neutral default.
        """
        return default if self.is_stale(now) else self.value
