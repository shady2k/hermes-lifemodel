"""The ``State`` model — the being's persisted soul, as a plain dataclass.

Design constraints (HLA §4 / §13):

* **Human-readable JSON, no heavy deps.** Every field is a JSON-native type
  (number, string, or ``null``), so ``json`` alone round-trips a ``State`` with
  no custom encoder. Timestamps are ISO-8601 UTC
  *strings* produced upstream by the clock (task 0.4's ``ClockPort``), not
  ``datetime`` objects — keeping the wire format trivial and diff-friendly.
* **Extensible, not over-built (YAGNI).** Only the fields Phase 1 actually
  needs are here; the growing soul (desires, open loops, receptivity,
  temperament, neuron thresholds — HLA §4) slots in as new fields later
  *without a rewrite* because ``from_dict`` is tolerant of missing keys.

The model owns its own (de)serialization and validates types on the way in,
raising :class:`StateCorruptError` for malformed data. It imports nothing from
Hermes and stays unit-testable in isolation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .errors import StateCorruptError

#: Current on-disk state schema. Bump this when the persisted shape changes in a
#: way old readers cannot understand. Reading a *different* version is a Phase-7
#: concern (migrations / back-compat, HLA §9 / FR16); this build fails loud on
#: any mismatch (see :meth:`JsonStateStore.load`).
SCHEMA_VERSION = 1


@dataclass
class State:
    """The single source of truth for the being's state (§4).

    The first field, ``schema_version``, is the on-disk *header*: it serializes
    first (dataclass field order is preserved) so the file leads with the
    version it was written under.
    """

    # --- header ---
    schema_version: int = SCHEMA_VERSION

    # --- the being's persisted state (Phase 1 minimal; extend, don't rewrite) ---
    #: Monotonic count of heartbeat ticks (roadmap 1.1). Bumped once per tick by
    #: the tick orchestrator; the simplest proof that state persists *between*
    #: ticks (a fresh store loads it, +1, commits). Never decreases.
    tick_count: int = 0
    #: Accumulated drive to act. Persists between neuron ticks (roadmap 1.2/1.3):
    #: ticks add to it, a threshold crossing drains it.
    pressure: float = 0.0
    #: Coarse energy placeholder (HLA §4/§11). Recovered during sleep in later
    #: phases; carried now so the wake path has a slot to read.
    energy: float = 1.0
    #: ISO-8601 UTC timestamp of the last neuron tick, or ``None`` before the
    #: first tick.
    last_tick_at: str | None = None
    #: ISO-8601 UTC timestamp of the last outbound contact, for cooldown
    #: bookkeeping (roadmap 1.4). ``None`` until the being first reaches out.
    last_contact_at: str | None = None
    #: ISO-8601 UTC timestamp the current wake-cooldown expires at (roadmap 1.4).
    #: Stamped ``now + COOLDOWN`` when a wake drains the pressure; while
    #: ``now < cooldown_until`` the tick stays asleep even above threshold, so at
    #: most one message fires per threshold cycle. ``None`` until the first wake.
    #: Additive field — the schema stays v1 because :meth:`from_dict` defaults it
    #: when absent, so state files written before 1.4 load unchanged.
    cooldown_until: str | None = None
    # NB: signal dedup does *not* live here. It is owned by the SignalBus
    # consumed-ledger (``signals.consumed``), which persists "already consumed"
    # independently of this State to avoid racing the tick's own commit — see
    # :class:`~lifemodel.adapters.signal_bus.FileSignalBus`.

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-native dict, header (``schema_version``) first."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> State:
        """Build a ``State`` from a parsed mapping, validating field types.

        Tolerant of *missing* keys (documented defaults fill in — so new fields
        added in later phases load cleanly from files written by this build).
        Strict on *present* keys of the wrong type: those signal corruption and
        raise :class:`StateCorruptError`. The caller (the store) is responsible
        for gating ``schema_version`` compatibility before calling this.
        """
        return cls(
            schema_version=_as_int(data.get("schema_version", SCHEMA_VERSION), "schema_version"),
            tick_count=_as_int(data.get("tick_count", 0), "tick_count"),
            pressure=_as_float(data.get("pressure", 0.0), "pressure"),
            energy=_as_float(data.get("energy", 1.0), "energy"),
            last_tick_at=_as_opt_str(data.get("last_tick_at"), "last_tick_at"),
            # last_contact_at and cooldown_until are the timestamps the engine
            # threads into datetime comparisons (the wake cooldown, see
            # ``lifemodel.tick.run_tick``; the contact time flows into the
            # wake-packet). They are validated here as timezone-*aware* ISO-8601
            # instants: a malformed value — or a tz-*naive* one that would raise
            # ``TypeError`` when compared against the clock's aware ``now`` — is
            # corruption caught loud at load, never a mid-tick crash. (With the
            # fail-closed ``main`` such a crash would wedge the being silent rather
            # than fire; either way it must not reach the tick.) ``last_tick_at``
            # stays an opaque display string — it is never parsed or compared.
            last_contact_at=_as_opt_iso(data.get("last_contact_at"), "last_contact_at"),
            cooldown_until=_as_opt_iso(data.get("cooldown_until"), "cooldown_until"),
        )


def _as_int(value: object, field_name: str) -> int:
    # ``bool`` is a subclass of ``int`` in Python; reject it explicitly so a
    # stray ``true`` in the file is not silently read as ``1``.
    if isinstance(value, bool) or not isinstance(value, int):
        raise StateCorruptError(f"field {field_name!r} must be an int, got {_type(value)}")
    return value


def _as_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StateCorruptError(f"field {field_name!r} must be a number, got {_type(value)}")
    number = float(value)
    # json.loads accepts the non-standard NaN/Infinity/-Infinity tokens by
    # default; reject the resulting non-finite floats — they are not valid JSON
    # and would poison downstream threshold comparisons.
    if not math.isfinite(number):
        raise StateCorruptError(f"field {field_name!r} must be finite, got {number}")
    return number


def _as_opt_str(value: object, field_name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise StateCorruptError(f"field {field_name!r} must be a string or null, got {_type(value)}")


def _as_opt_iso(value: object, field_name: str) -> str | None:
    # A str-or-null *and* a timezone-AWARE ISO-8601 instant when present. The value
    # is kept as its original string (the on-disk shape stays a string, HLA §4);
    # parsing here validates it so downstream comparisons never raise. The
    # tz-aware requirement is load-bearing: the tick compares the clock's aware
    # UTC ``now`` against ``cooldown_until``, and a naive value would raise
    # ``TypeError: can't compare offset-naive and offset-aware datetimes`` — so a
    # naive value is rejected as corruption at load, not left to crash the tick.
    text = _as_opt_str(value, field_name)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise StateCorruptError(
            f"field {field_name!r} must be an ISO-8601 timestamp, got {text!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StateCorruptError(
            f"field {field_name!r} must be a timezone-aware timestamp, got naive {text!r}"
        )
    return text


def _type(value: object) -> str:
    return type(value).__name__
