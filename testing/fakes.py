"""In-memory fakes for every Phase-1 port — "imitations before code" (HLA §13).

The DI contract says tests inject fakes, not real adapters. These are the
canonical ones: a fake clock, delivery, state store, memory store and pressure
sensor, each satisfying its port/ABC. They are shipped in the
package (not hidden in a test folder) so later tasks reuse the *same* fakes
their upstream defined, rather than re-rolling subtly different ones. Stdlib
only; no Hermes.

Each fake is deliberately transparent — it exposes the recorded inputs
(``FakeDelivery.sent``, ``FakeClock`` mutators) so a test can assert on them.

``FakeMemoryStore``/``FakePressureSensor`` (lm-fib.6.1, HLA §4.1/D7) back
:class:`~lifemodel.ports.memory.MemoryPort`/:class:`~lifemodel.ports.pressure.PressureSensorPort`
with a plain dict, applying the *same* semantics as
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` (upsert keeps
``created_at``, guarded ``transition``, deterministic ``find`` order, epoch
expiry) so the shared contract test suite runs identically against fake and
real store, and higher layers can unit-test against these ports without a
database. Purely additive — nothing in the live tick uses them yet.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, assert_never

from ..domain.memory import (
    MemoryDraft,
    MemoryMutation,
    MemoryPatch,
    MemoryRecord,
    PressureIndex,
    PutOp,
    StaleTransition,
    TransitionOp,
    coalesce_patch,
    describe_stale_transition,
    ensure_json_serializable,
    merge_payload,
    normalize_expires_at,
    stamp_iso_utc,
    summarize_pressure_index,
)
from ..ports.clock import ClockPort
from ..ports.memory import OrderBy
from ..ports.tracer import (
    ActiveSpan,
    SpanStatus,
    TraceContext,
)
from ..ports.tracer import (
    format_traceparent as _format_traceparent,
)
from ..ports.tracer import (
    parse_traceparent as _parse_traceparent,
)
from ..state.model import State


class FakeClock:
    """A :class:`~lifemodel.ports.clock.ClockPort` pinned to a controllable time.

    Construct with a timezone-aware UTC ``datetime``; move it with
    :meth:`advance` / :meth:`set` so tests exercise elapsed-time logic
    (cooldowns, the connection neuron) deterministically.
    """

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward by *delta*."""
        self._now += delta

    def set(self, now: datetime) -> None:
        """Pin the clock to an absolute instant."""
        self._now = now


class FakeTracer:
    """A deterministic :class:`~lifemodel.ports.tracer.TracerPort` for tests.

    Mirrors :class:`FakeClock`: instead of random ids it hands out a fixed SEQUENCE
    — trace ids ``0…01``, ``0…02``, … and span ids ``0…0001``, ``0…0002``, … — so a
    stamped provenance trace_id / a bound log field is stable to assert on.
    ``start_root`` consumes the next trace id + span id (a FRESH trace per tick,
    unless it continues an upstream traceparent, when it keeps that trace and parents
    onto the upstream span); ``child_of`` keeps the parent's trace and consumes the
    next span. All ids are valid non-zero W3C hex.
    """

    def __init__(self) -> None:
        self._next_trace = 1
        self._next_span = 1

    def _mint_trace(self) -> str:
        value = f"{self._next_trace:032x}"
        self._next_trace += 1
        return value

    def _mint_span(self) -> str:
        value = f"{self._next_span:016x}"
        self._next_span += 1
        return value

    def start_root(self, *, upstream_traceparent: str | None = None) -> TraceContext:
        if upstream_traceparent is not None:
            upstream = _parse_traceparent(upstream_traceparent)
            return TraceContext(
                trace_id=upstream.trace_id,
                span_id=self._mint_span(),
                parent_span_id=upstream.span_id,
                trace_flags=upstream.trace_flags,
            )
        return TraceContext(
            trace_id=self._mint_trace(),
            span_id=self._mint_span(),
            parent_span_id=None,
            trace_flags="01",
        )

    def child_of(self, parent: TraceContext) -> TraceContext:
        return TraceContext(
            trace_id=parent.trace_id,
            span_id=self._mint_span(),
            parent_span_id=parent.span_id,
            trace_flags=parent.trace_flags,
        )

    def format_traceparent(self, ctx: TraceContext) -> str:
        return _format_traceparent(ctx)

    def parse_traceparent(self, value: str) -> TraceContext:
        return _parse_traceparent(value)


class FakeActiveSpan:
    """A deterministic, inspectable :class:`~lifemodel.ports.tracer.ActiveSpan`.

    A plain mutable handle for tests: construct it over a
    :class:`~lifemodel.ports.tracer.TraceContext`, then assert on the ``attrs`` a
    component :meth:`set` and the ``status`` it :meth:`end`ed with. Mirrors the
    live :class:`~lifemodel.adapters.tracer.MutableActiveSpan` without importing
    the adapter layer, and satisfies the ``ActiveSpan`` protocol structurally.
    """

    def __init__(
        self,
        context: TraceContext,
        *,
        component: str | None = None,
        tick: int | None = None,
        started_at: str | None = None,
    ) -> None:
        self.context = context
        self.component = component
        self.tick = tick
        self.started_at = started_at
        self.ended_at: str | None = None
        self.status: SpanStatus = "ok"
        #: A test-only flag recording that :meth:`end` was called.
        self.ended = False
        self._attrs: dict[str, Any] = {}

    @property
    def attrs(self) -> Mapping[str, Any]:
        return self._attrs

    def set(self, **attrs: Any) -> FakeActiveSpan:
        self._attrs.update(attrs)
        return self

    def end(self, *, status: SpanStatus = "ok", ended_at: str | None = None) -> FakeActiveSpan:
        self.status = status
        self.ended = True
        if ended_at is not None:
            self.ended_at = ended_at
        return self


class FakeSpanLogger:
    """A recording :class:`~lifemodel.log.SpanLogger` for tests.

    Each level call records a ``{"level", "event", **fields}`` dict — with the
    bound span's ``trace_id``/``span_id``/``tick`` stamped in exactly as the real
    :class:`~lifemodel.log.SpanLogger` would — so a test can assert both the
    events emitted and that the span's ids were carried, without standing up a
    real trace store or ring. With no span it still records, leaving the ids off
    (a bare unit-test logger).
    """

    def __init__(self, span: ActiveSpan | None = None) -> None:
        self.span = span
        self.events: list[dict[str, Any]] = []

    def _record(self, level: str, event: str, fields: dict[str, Any]) -> None:
        record: dict[str, Any] = {"level": level, "event": event, **fields}
        if self.span is not None:
            ctx = self.span.context
            record["trace_id"] = ctx.trace_id
            record["span_id"] = ctx.span_id
            record["tick"] = self.span.tick
        self.events.append(record)

    def debug(self, event: str, **fields: Any) -> None:
        self._record("debug", event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._record("info", event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._record("warning", event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._record("error", event, fields)

    def critical(self, event: str, **fields: Any) -> None:
        self._record("critical", event, fields)


class FakeDelivery:
    """A :class:`~lifemodel.ports.delivery.DeliveryPort` that records sends.

    Nothing leaves the process; every ``send`` is appended to :attr:`sent` as a
    ``(channel, text)`` pair for the test to assert on.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, channel: str, text: str) -> None:
        self.sent.append((channel, text))


class FakeStateStore:
    """An in-memory :class:`~lifemodel.state.port.StatePort` (+ ``TickCommitPort``).

    Holds one ``State`` in memory (the documented default until first commit).
    Deep-copies on the way in and out so a caller mutating its own ``State`` can
    never reach through and change what the store holds — matching the isolation
    a real serializing store gives.

    :meth:`commit_tick` mirrors the real
    :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`'s atomic
    State+memory committer. State-only ticks need no memory backing; to also
    apply memory mutations, inject a :class:`FakeMemoryStore` (``memory=``) — the
    two then move **all-or-nothing** together (a stale transition mid-batch rolls
    back the state *and* every earlier put in the same batch), so fake and real
    agree on split-brain-freedom (HLA §4.1).
    """

    def __init__(
        self, initial: State | None = None, *, memory: FakeMemoryStore | None = None
    ) -> None:
        self._state = copy.deepcopy(initial) if initial is not None else State()
        self._memory = memory

    def load(self) -> State:
        return copy.deepcopy(self._state)

    def commit(self, state: State) -> None:
        self._state = copy.deepcopy(state)

    def reset(self) -> State:
        """Factory-wipe to a fresh ``State()`` — never requires a prior
        successful :meth:`load`, matching :class:`~lifemodel.state.port.StatePort`."""
        self._state = State()
        return copy.deepcopy(self._state)

    def commit_tick(self, state: State | None, mutations: Sequence[MemoryMutation]) -> None:
        """Atomically apply *state* (if not ``None``) then each mutation in order.

        Snapshot-then-restore gives true all-or-nothing: any exception (a stale
        transition, a serialization guard) restores both the ``State`` and the
        memory rows to their pre-batch values and re-raises — matching the real
        store's single transaction, including intra-batch ``put``-then-
        ``transition`` of the same record.
        """
        if mutations and self._memory is None:
            raise TypeError(
                "FakeStateStore.commit_tick got memory mutations but no memory store; "
                "construct it with FakeStateStore(memory=FakeMemoryStore(...))"
            )
        state_snapshot = copy.deepcopy(self._state)
        rows_snapshot = copy.deepcopy(self._memory._rows) if self._memory is not None else None
        try:
            if state is not None:
                self.commit(state)
            for mutation in mutations:
                match mutation:
                    case PutOp():
                        assert self._memory is not None  # guarded above
                        self._memory.put(mutation.draft)
                    case TransitionOp():
                        assert self._memory is not None  # guarded above
                        self._memory.transition(
                            mutation.kind,
                            mutation.id,
                            mutation.from_state,
                            mutation.to_state,
                            mutation.patch,
                        )
                    case _:  # pragma: no cover - exhaustive over the closed union
                        assert_never(mutation)
        except BaseException:
            self._state = state_snapshot
            if self._memory is not None and rows_snapshot is not None:
                self._memory._rows = rows_snapshot
            raise


class FakeMemoryStore:
    """An in-memory :class:`~lifemodel.ports.memory.MemoryPort` (+ pressure sensor).

    Backed by a plain ``dict`` keyed by ``(kind, id)`` — no SQL, no epoch
    columns — applying the same semantics as
    :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`: upsert keeps
    ``created_at`` and bumps ``revision``, ``transition`` is guarded on
    ``from_state``, ``find`` order is deterministic with an ``id`` tiebreak,
    and expiry is epoch-based. It also implements
    :class:`~lifemodel.ports.pressure.PressureSensorPort` directly — mirroring
    how the real store answers both ports from one object — via
    :func:`~lifemodel.domain.memory.summarize_pressure_index`, so the shared
    contract test suite can parametrize over "one fake object" / "one real
    store" symmetrically. Every read returns a deep copy so a caller can never
    mutate what this fake holds internally.
    """

    def __init__(self, *, clock: ClockPort) -> None:
        self._clock = clock
        self._rows: dict[tuple[str, str], MemoryRecord] = {}

    def put(self, draft: MemoryDraft) -> str:
        ensure_json_serializable(draft.payload)
        expires_at = normalize_expires_at(draft.expires_at)  # validate + normalize on write
        now = stamp_iso_utc(self._clock.now())  # canonical fixed-width UTC; rejects a naive clock
        key = (draft.kind, draft.id)
        existing = self._rows.get(key)
        created_at = existing.created_at if existing is not None else now
        revision = existing.revision + 1 if existing is not None else 0
        self._rows[key] = MemoryRecord(
            kind=draft.kind,
            id=draft.id,
            state=draft.state,
            payload=copy.deepcopy(draft.payload),
            source=draft.source,
            recipient_id=draft.recipient_id,
            salience=draft.salience,
            confidence=draft.confidence,
            expires_at=expires_at,
            created_at=created_at,
            updated_at=now,
            revision=revision,
            schema_version=draft.schema_version,
        )
        return draft.id

    def get(self, kind: str, id: str) -> MemoryRecord | None:
        record = self._rows.get((kind, id))
        return None if record is None else _copy_record(record)

    def find(
        self,
        kind: str | None = None,
        state: str | None = None,
        limit: int | None = None,
        order_by: OrderBy = "updated_desc",
    ) -> list[MemoryRecord]:
        # Match the real store: SQLite treats `LIMIT -1` as "no limit", so a
        # bare `records[:limit]` slice would silently diverge — reject instead.
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        records = [
            record
            for record in self._rows.values()
            if (kind is None or record.kind == kind) and (state is None or record.state == state)
        ]
        records = _sort_records(records, order_by)
        if limit is not None:
            records = records[:limit]
        return [_copy_record(record) for record in records]

    def transition(
        self,
        kind: str,
        id: str,
        from_state: str,
        to_state: str,
        patch: MemoryPatch | None = None,
    ) -> MemoryRecord:
        patch = patch if patch is not None else MemoryPatch()
        if patch.payload_merge is not None:
            ensure_json_serializable(patch.payload_merge)

        key = (kind, id)
        existing = self._rows.get(key)
        if existing is None or existing.state != from_state:
            actual_state = existing.state if existing is not None else None
            raise StaleTransition(describe_stale_transition(kind, id, from_state, actual_state))

        # Normalize on write (existing is already canonical; a patch value is raw).
        new_expires_at = normalize_expires_at(coalesce_patch(patch.expires_at, existing.expires_at))
        now = stamp_iso_utc(self._clock.now())  # canonical fixed-width UTC; rejects a naive clock
        updated = MemoryRecord(
            kind=existing.kind,
            id=existing.id,
            state=to_state,
            payload=merge_payload(existing.payload, patch.payload_merge),
            source=coalesce_patch(patch.source, existing.source),
            recipient_id=existing.recipient_id,
            salience=coalesce_patch(patch.salience, existing.salience),
            confidence=coalesce_patch(patch.confidence, existing.confidence),
            expires_at=new_expires_at,
            created_at=existing.created_at,
            updated_at=now,
            revision=existing.revision + 1,
            schema_version=existing.schema_version,
        )
        self._rows[key] = updated
        return _copy_record(updated)

    def read_pressure_index(self, now: datetime) -> PressureIndex:
        return summarize_pressure_index(self._rows.values(), now)


class FakePressureSensor:
    """A standalone :class:`~lifemodel.ports.pressure.PressureSensorPort` fake.

    Wraps a :class:`FakeMemoryStore`'s rows so a test that only wants to fake
    the pressure-sensor boundary — without depending on the full
    :class:`~lifemodel.ports.memory.MemoryPort` surface — can inject just this
    (interface-segregation parity with the two-port split;
    :class:`SQLiteRuntimeStore` satisfies both ports from one object, but a
    consumer of only ``PressureSensorPort`` should not have to know that).
    """

    def __init__(self, store: FakeMemoryStore) -> None:
        self._store = store

    def read_pressure_index(self, now: datetime) -> PressureIndex:
        return self._store.read_pressure_index(now)


def _copy_record(record: MemoryRecord) -> MemoryRecord:
    return copy.deepcopy(record)


def _sort_records(records: list[MemoryRecord], order_by: OrderBy) -> list[MemoryRecord]:
    # Two stable passes (minor key ascending, then major key descending) so
    # ties break by id ASC even though the major key sorts DESC — a single
    # `reverse=True` over a combined tuple key would reverse the tiebreak too.
    # Sort timestamps by the stored normalized ISO string directly: every stamp
    # is canonical fixed-width UTC TEXT (:func:`stamp_iso_utc`), so a lexical sort
    # is byte-identical to SQLite's `updated_at DESC` / `created_at DESC` under
    # any timezone offset — the same normalized-TEXT ordering the store rests on.
    by_id = sorted(records, key=lambda r: r.id)
    if order_by == "updated_desc":
        return sorted(by_id, key=lambda r: r.updated_at, reverse=True)
    if order_by == "created_desc":
        return sorted(by_id, key=lambda r: r.created_at, reverse=True)
    return sorted(by_id, key=lambda r: r.salience, reverse=True)
