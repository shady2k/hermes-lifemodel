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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .errors import StateCorruptError

#: Current on-disk state schema. Bump this when the persisted shape changes in a
#: way old readers cannot understand. Reading a *different* version is a Phase-7
#: concern (migrations / back-compat, HLA §9 / FR16); this build fails loud on
#: any mismatch (see :meth:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore.load`).
SCHEMA_VERSION = 3


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
    #: Monotonic count of brain-loop ticks. Bumped once per tick by ``coreloop.tick()``
    #: (driven by the platform adapter's loop); the simplest proof that state persists
    #: *between* ticks (a fresh store loads it, +1, commits). Never decreases.
    tick_count: int = 0
    #: Coarse energy placeholder (HLA §4/§11). Recovered during sleep in later
    #: phases; carried now so the wake path has a slot to read.
    energy: float = 1.0
    #: Homeostatic sleep-pressure debt S (spec §8), in [0, 1]. Rises with
    #: cognition spend (Phase D); decays toward 0 during rest. Additive:
    #: ``from_dict`` defaults it when absent.
    fatigue: float = 0.0
    #: The contact-desire drive's continuous urge variable (certified sim
    #: ``Drive.u``, spec §5). Rises with genuine silence, satiated by a positive
    #: exchange, drained when a wake-eligible urge is consumed — see
    #: ``lifemodel.core.decision`` (the live adapter) and
    #: ``lifemodel.core.solitude_drive.Drive`` (the certified source of truth
    #: this reconstructs each tick).
    u: float = 0.0
    #: Elapsed minutes ``u`` has continuously sat at/over the wake threshold
    #: ``θ`` (spec §5/§7). Reset to zero whenever ``u`` dips back under ``θ`` or
    #: a desire resolves; feeds the wake-decision's duration gate.
    duration_over_theta: float = 0.0
    #: Core-affect VALENCE (lm-ukc.1) — the being's hedonic background on Russell's
    #: circumplex, in ``[-1, 1]`` (unpleasant … pleasant), 0 = neutral. A cheap 0-LLM
    #: projection of the body (loneliness/rejection pull it down; a fresh genuine
    #: exchange lifts it), eased toward its target with inertia each tick (the deriver,
    #: a later task). The range is the deriver's discipline, not enforced here — like
    #: ``energy``/``u`` the field only validates *finite*. **One-way invariant:** affect
    #: colors the voice and is NEVER read by the wake/contact decision. Additive:
    #: ``from_dict`` defaults it when absent, so an older file loads neutral (no reset).
    affect_valence: float = 0.0
    #: Core-affect AROUSAL (lm-ukc.1) — activation on Russell's circumplex, in
    #: ``[0, 1]`` (calm … keyed-up), 0 = calm cold-start. Driven by energy/fatigue,
    #: circadian alertness, and the urgency of the pull; eased with inertia. Same
    #: one-way invariant and finite-only validation as :attr:`affect_valence`.
    #: Additive default 0.0.
    affect_arousal: float = 0.0
    #: ISO-8601 UTC timestamp of the last affect update (lm-ukc.1) — the affect
    #: sibling of :attr:`last_tick_at` for the leaky-integrator's elapsed-time step.
    #: Kept as an OPAQUE string (the deriver parses it defensively, like
    #: ``last_tick_at``), so it is validated only as opt-str, not a tz-aware instant.
    #: ``None`` before the first affect update; additive (defaults absent).
    affect_updated_at: str | None = None
    #: The felt WORD (lm-ukc.3, e.g. ``"wistful"``/``"restless"``) last surfaced
    #: AMBIENTLY into an ordinary reactive turn by the ``pre_llm_call`` felt-state
    #: injector (lm-ukc.4). Persisted so the gate can suppress a repeated cue on a
    #: long non-neutral stretch — it re-injects on a felt-word CHANGE, else only
    #: after cooldown. Written ONLY by the reactive display path; it never conflicts
    #: with the affect axes the tick writes. ``None`` before the first ambient show;
    #: additive (``from_dict`` defaults it absent, so an older file loads cleanly).
    affect_display_last_word: str | None = None
    #: ISO-8601 UTC timestamp of the last ambient felt-state show (lm-ukc.4), the
    #: cooldown anchor its gate reads. Kept as an OPAQUE string (parsed defensively
    #: by :func:`~lifemodel.core.timeutil.minutes_between`, like
    #: :attr:`affect_updated_at`/:attr:`last_tick_at`), so it is validated only as
    #: opt-str. ``None`` before the first ambient show; additive (defaults absent).
    affect_display_last_at: str | None = None
    #: ISO-8601 UTC timestamp of the last genuine (non-internal) exchange with
    #: the user (spec §4/§6). This is the real EXCHANGE RECORD the wake-packet
    #: temporal fact renders ("The last time we exchanged messages was X",
    #: :func:`~lifemodel.core.wake_packet.render_temporal_facts`), so it is IMMUNE
    #: to admin/control commands (lm-md6.1): it is written ONLY by a genuine
    #: two-way exchange (``core/aggregation.py``), never forged by ``force_wake``/
    #: ``satiate``/``set`` — those move :attr:`silence_anchor_at` instead. ``None``
    #: before the first exchange.
    last_exchange_at: str | None = None
    #: ISO-8601 UTC timestamp anchoring the active-silence WINDOW gate
    #: (``core.wake`` gate 3 — suppress a wake for ``w`` minutes after contact),
    #: DECOUPLED from :attr:`last_exchange_at` (lm-md6.1). ``None`` means "no admin
    #: override" and the gate falls back to the real :attr:`last_exchange_at`; a
    #: genuine exchange clears it back to ``None``. Admin/control commands that
    #: only need to tune the silence gate (``force_wake`` backdates it past ``w``;
    #: ``satiate`` sets it to now) write HERE, so the immune exchange record the
    #: model reads is never forged. ``None`` before any override.
    silence_anchor_at: str | None = None
    # NB: the contact-desire *lifecycle* is no longer a ``State`` flag (lm-27n.3):
    # it lives in the singleton ``kind='desire'`` record ``contact:owner`` (HLA
    # §4.1), read via ``core.desire_view.live_contact_desire``. Only the residual
    # policy scalars below (backoff / pending / ActionPending / send-log) stay here.
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
    #: The load-bearing async-correlation anchor (spec §4.4): the FULL W3C
    #: ``traceparent`` of the launch span that minted ``pending_proactive_id``,
    #: written atomically beside it so the async read-back (``post_llm`` hook) and
    #: the resolving tick can re-bind the SAME origin trace across the Hermes
    #: boundary (which carries no in-band trace channel). Lives in the *precious*
    #: ``runtime_state`` (not the disposable trace DB) so losing the trace store
    #: never breaks the weave. ``None`` whenever ``pending_proactive_id`` is —
    #: cleared in lockstep at every clear-site (§4.4).
    pending_proactive_origin_traceparent: str | None = None
    #: ISO-8601 UTC timestamp of the last neuron tick, or ``None`` before the
    #: first tick.
    last_tick_at: str | None = None
    #: ISO-8601 UTC timestamp of the last outbound contact, for cooldown
    #: bookkeeping (roadmap 1.4). ``None`` until the being first reaches out.
    last_contact_at: str | None = None
    #: ISO-8601 UTC timestamp when the being's outreach was fulfilled (send
    #: happened), starting the ActionPending inhibition window (spec §9.2).
    #: ``None`` when no outreach is pending. A real exchange clears this.
    action_pending_since: str | None = None
    #: ISO-8601 UTC timestamps of recent real proactive sends, bounded by
    #: ``SEND_LOG_KEEP`` (spec §14). The global backstop reads this to enforce
    #: the hard rate limit (≤3/day, ≥60 min apart). Defaults to empty (additive).
    proactive_send_log: list[str] = field(default_factory=list)
    #: Count of consecutive proactive outreaches sent without a genuine reply
    #: since (spec §14 / Slice 3, lm-8o3.1). Feeds the unanswered-outbound gate
    #: so the being does not repeat a pure-longing bid after one unanswered
    #: outreach; reset to zero by a genuine exchange. Defaults to zero
    #: (additive).
    unanswered_outbound_count: int = 0
    #: The external-event idempotency ring (spec §8 / lm-fib.8.5): a bounded,
    #: TTL'd map of external event id → ISO-8601 UTC stamp of when it was first
    #: processed, ordered oldest-first (Python dict insertion order). This is
    #: **not** a bus cursor and **not** bus-dedup (the nervous flow is ephemeral,
    #: spec §2/§3 — a signal lives inside one ExecutionFrame and is gone). It is
    #: the BODY remembering "I already processed this external event id", so a
    #: Telegram/Hermes retry of the SAME inbound cannot satiate ``u`` twice, reset
    #: ``last_exchange_at`` again, or re-resolve the desire. The intake path
    #: (:mod:`lifemodel.core.idempotency`, called from ``core/coreloop.py``) checks
    #: it before seeding a frame, dropping a duplicate ``contact_observed`` and
    #: recording each fresh id (evicting oldest / TTL-expired to stay bounded).
    #: Durable in ``runtime_state`` (survives a restart, unlike the bus); a
    #: factory-reset simply starts it empty. Defaults to empty (additive).
    processed_external_event_ids: dict[str, str] = field(default_factory=dict)

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
            fatigue=_as_float(data.get("fatigue", 0.0), "fatigue"),
            u=_as_float(data.get("u", 0.0), "u"),
            duration_over_theta=_as_float(
                data.get("duration_over_theta", 0.0), "duration_over_theta"
            ),
            # Core affect (lm-ukc.1): two finite coordinates on Russell's circumplex.
            # Range discipline ([-1,1] valence / [0,1] arousal) is the deriver's, so —
            # like u/energy — from_dict validates only that they are finite numbers.
            affect_valence=_as_float(data.get("affect_valence", 0.0), "affect_valence"),
            affect_arousal=_as_float(data.get("affect_arousal", 0.0), "affect_arousal"),
            # The affect stamp is an opaque string (sibling of last_tick_at, parsed
            # defensively by the deriver), so it is validated only as opt-str.
            affect_updated_at=_as_opt_str(data.get("affect_updated_at"), "affect_updated_at"),
            # The reactive felt-display bookkeeping (lm-ukc.4): the last ambiently
            # shown felt word + when. Both opaque strings (the display gate parses
            # the timestamp defensively, like affect_updated_at), so opt-str only.
            affect_display_last_word=_as_opt_str(
                data.get("affect_display_last_word"), "affect_display_last_word"
            ),
            affect_display_last_at=_as_opt_str(
                data.get("affect_display_last_at"), "affect_display_last_at"
            ),
            # last_exchange_at, declined_at, and pending_proactive_since are
            # compared against the clock's aware ``now`` by ``core/decision.py``
            # (the live adapter), so — like last_contact_at below — they are
            # validated here as timezone-*aware* ISO-8601 instants: a malformed
            # value, or a tz-*naive* one that would raise ``TypeError`` when
            # compared, is corruption caught loud at load, never a mid-tick crash.
            last_exchange_at=_as_opt_iso(data.get("last_exchange_at"), "last_exchange_at"),
            # silence_anchor_at is compared against the clock's aware ``now`` by the
            # silence-window gate (aggregation / introspect), so — like the exchange
            # timestamp above — it is validated as a tz-aware ISO-8601 instant.
            silence_anchor_at=_as_opt_iso(data.get("silence_anchor_at"), "silence_anchor_at"),
            declined_at=_as_opt_iso(data.get("declined_at"), "declined_at"),
            decline_count=_as_int(data.get("decline_count", 0), "decline_count"),
            pending_proactive_id=_as_opt_str(
                data.get("pending_proactive_id"), "pending_proactive_id"
            ),
            pending_proactive_since=_as_opt_iso(
                data.get("pending_proactive_since"), "pending_proactive_since"
            ),
            # The async-correlation anchor is a W3C traceparent string, not a
            # timestamp — validated only as opt-str (an opaque header we round-trip).
            pending_proactive_origin_traceparent=_as_opt_str(
                data.get("pending_proactive_origin_traceparent"),
                "pending_proactive_origin_traceparent",
            ),
            last_tick_at=_as_opt_str(data.get("last_tick_at"), "last_tick_at"),
            # last_contact_at is a timestamp the engine threads into datetime
            # comparisons (the wake-packet's cooldown context). It is validated
            # here as a timezone-*aware* ISO-8601 instant: a malformed value — or
            # a tz-*naive* one that would raise ``TypeError`` when compared
            # against the clock's aware ``now`` — is corruption caught loud at
            # load, never a mid-tick crash. (With the fail-closed ``main`` such a
            # crash would wedge the being silent rather than fire; either way it
            # must not reach the tick.) ``last_tick_at`` is validated here only
            # as an opaque string, NOT as a parsed instant: unlike the fields
            # above, ``core/decision.py``'s ``_minutes_between`` (the sole
            # parser of this field) is deliberately defensive — a malformed or
            # tz-naive value returns ``0.0`` (no elapsed rise this tick) rather
            # than raising, so a bad ``last_tick_at`` degrades gracefully
            # instead of being corruption caught at load.
            last_contact_at=_as_opt_iso(data.get("last_contact_at"), "last_contact_at"),
            action_pending_since=_as_opt_iso(
                data.get("action_pending_since"), "action_pending_since"
            ),
            proactive_send_log=_as_str_list(data, "proactive_send_log", []),
            unanswered_outbound_count=_as_int(
                data.get("unanswered_outbound_count", 0), "unanswered_outbound_count"
            ),
            processed_external_event_ids=_as_str_str_dict(data, "processed_external_event_ids", {}),
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


def _as_str_list(data: Mapping[str, Any], key: str, default: list[str]) -> list[str]:
    if key not in data:
        return list(default)
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise StateCorruptError(f"'{key}' must be a list[str]")
    return list(value)


def _as_str_str_dict(data: Mapping[str, Any], key: str, default: dict[str, str]) -> dict[str, str]:
    # The idempotency ring is a JSON object of id → ISO stamp, both strings; a
    # missing key is the additive default (empty). Insertion order is preserved by
    # ``dict`` (and by json round-trip), so callers can treat it as oldest-first.
    if key not in data:
        return dict(default)
    value = data[key]
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise StateCorruptError(f"'{key}' must be a dict[str, str]")
    return dict(value)


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
