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
    #: Coarse energy placeholder (HLA §4/§11). Recovered during sleep in later
    #: phases; carried now so the wake path has a slot to read.
    energy: float = 1.0
    #: The contact-desire drive's continuous urge variable (certified sim
    #: ``Drive.u``, spec §5). Rises with genuine silence, satiated by a positive
    #: exchange, drained when a wake-eligible urge is consumed — see
    #: ``lifemodel.core.decision`` (the live adapter) and ``lifemodel.sim.drive``
    #: (the certified source of truth this reconstructs each tick).
    u: float = 0.0
    #: Elapsed minutes ``u`` has continuously sat at/over the wake threshold
    #: ``θ`` (spec §5/§7). Reset to zero whenever ``u`` dips back under ``θ`` or
    #: a desire resolves; feeds the wake-decision's duration gate.
    duration_over_theta: float = 0.0
    #: ISO-8601 UTC timestamp of the last genuine (non-internal) exchange with
    #: the user (spec §4/§6). Satiates the drive and opens the active-silence
    #: window; ``None`` before the first exchange.
    last_exchange_at: str | None = None
    #: Where the current contact-desire sits in its lifecycle (spec §4/§5/§7):
    #: ``"none"`` (no live desire), ``"active"`` (woken, awaiting a verdict), or
    #: ``"deferred"`` (held for a later release condition — unreachable live in
    #: Phase 1, kept for parity with the certified ``DesireStatus`` enum).
    desire_status: str = "none"
    #: ISO-8601 UTC timestamp of the most recent REJECT verdict (spec §5/§7),
    #: feeding the growing-backoff gate so a declined desire is not re-tried too
    #: soon. ``None`` until the first reject.
    declined_at: str | None = None
    #: Consecutive REJECT count (spec §7's growing backoff — ``r0·k**n``),
    #: reset to zero by any genuine exchange or a FULFILL verdict.
    decline_count: int = 0
    #: Correlation id of the in-flight proactive turn awaiting a verdict from
    #: the final LLM output (``post_llm_call``), or ``None`` when no proactive
    #: turn is outstanding.
    pending_proactive_id: str | None = None
    #: ISO-8601 UTC timestamp the pending proactive turn above was launched at,
    #: or ``None`` when no proactive turn is outstanding.
    pending_proactive_since: str | None = None
    #: ISO-8601 UTC timestamp of the last neuron tick, or ``None`` before the
    #: first tick.
    last_tick_at: str | None = None
    #: ISO-8601 UTC timestamp of the last outbound contact, for cooldown
    #: bookkeeping (roadmap 1.4). ``None`` until the being first reaches out.
    last_contact_at: str | None = None
    #: ISO-8601 UTC liveness stamp of the in-process proactive-egress service
    #: (lm-64s). While it is fresh (within ``SERVICE_LIVENESS_MAX_AGE``) the cron
    #: heartbeat defers to the in-process brain — it owns ticking while alive — and
    #: takes over as the fallback brain when the stamp goes stale/absent (spec §6).
    #: Additive: the schema stays v1 (``from_dict`` defaults it when absent).
    egress_service_alive_at: str | None = None
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
        added in later phases load cleanly from files written by this build) and
        of *unknown* keys (a legacy field from an older build, e.g. the retired
        ``pressure``/``cooldown_until``, is silently dropped rather than raising
        or resurrecting a dead attribute — this method only ever looks up known
        field names by name, never splats ``**data``). Strict on *present*
        *known* keys of the wrong type: those signal corruption and raise
        :class:`StateCorruptError`. The caller (the store) is responsible for
        gating ``schema_version`` compatibility before calling this.
        """
        return cls(
            schema_version=_as_int(data.get("schema_version", SCHEMA_VERSION), "schema_version"),
            tick_count=_as_int(data.get("tick_count", 0), "tick_count"),
            energy=_as_float(data.get("energy", 1.0), "energy"),
            u=_as_float(data.get("u", 0.0), "u"),
            duration_over_theta=_as_float(
                data.get("duration_over_theta", 0.0), "duration_over_theta"
            ),
            # last_exchange_at, declined_at, and pending_proactive_since are
            # compared against the clock's aware ``now`` by ``core/decision.py``
            # (the live adapter), so — like last_contact_at below — they are
            # validated here as timezone-*aware* ISO-8601 instants: a malformed
            # value, or a tz-*naive* one that would raise ``TypeError`` when
            # compared, is corruption caught loud at load, never a mid-tick crash.
            last_exchange_at=_as_opt_iso(data.get("last_exchange_at"), "last_exchange_at"),
            desire_status=_as_str(data.get("desire_status", "none"), "desire_status"),
            declined_at=_as_opt_iso(data.get("declined_at"), "declined_at"),
            decline_count=_as_int(data.get("decline_count", 0), "decline_count"),
            pending_proactive_id=_as_opt_str(
                data.get("pending_proactive_id"), "pending_proactive_id"
            ),
            pending_proactive_since=_as_opt_iso(
                data.get("pending_proactive_since"), "pending_proactive_since"
            ),
            last_tick_at=_as_opt_str(data.get("last_tick_at"), "last_tick_at"),
            # last_contact_at is a timestamp the engine threads into datetime
            # comparisons (the wake-packet's cooldown context). It is validated
            # here as a timezone-*aware* ISO-8601 instant: a malformed value — or
            # a tz-*naive* one that would raise ``TypeError`` when compared
            # against the clock's aware ``now`` — is corruption caught loud at
            # load, never a mid-tick crash. (With the fail-closed ``main`` such a
            # crash would wedge the being silent rather than fire; either way it
            # must not reach the tick.) ``last_tick_at`` stays an opaque display
            # string — it is never parsed or compared.
            last_contact_at=_as_opt_iso(data.get("last_contact_at"), "last_contact_at"),
            egress_service_alive_at=_as_opt_iso(
                data.get("egress_service_alive_at"), "egress_service_alive_at"
            ),
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


def _as_str(value: object, field_name: str) -> str:
    # Unlike ``_as_opt_str``, ``None`` is not a valid value here: fields routed
    # through this validator (e.g. ``desire_status``) always default to a
    # concrete string, never ``null``.
    if isinstance(value, str):
        return value
    raise StateCorruptError(f"field {field_name!r} must be a string, got {_type(value)}")


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
