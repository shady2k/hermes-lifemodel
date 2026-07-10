"""Phase-3 async bridge — one proactive attempt = one ``trace_id`` (spec §4.4/§5).

End-to-end over a REAL :class:`~lifemodel.state.trace_store.TraceWriter` writing a
real ``observability.sqlite``, driving the actual spine (``proactive_tick`` +
``post_llm`` hook + next-tick aggregation), then querying the durable store —
exactly how the debugger will. Proves the weave the whole phase exists for: launch
(tick N) → delivery → async outcome (post_llm) → resolution (tick N+k) all land
under the SAME ``trace_id`` raised from the state anchor; and a lost anchor becomes
an ``orphan_async_outcome`` on its own trace, never attached to a foreign one.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.composition import build_lifemodel
from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.core.proactive import proactive_tick
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.objects import DesireState
from lifemodel.events import EventRing
from lifemodel.hooks import make_post_llm_observer
from lifemodel.ports.memory import MemoryPort
from lifemodel.ports.tracer import parse_traceparent
from lifemodel.state.model import State
from lifemodel.state.trace_store import TraceWriter, observability_db_path
from lifemodel.testing.fakes import FakeClock, FakeTracer
from lifemodel.testing.harness import RecordingEgress

_T0 = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


def _seed_launch_ready(lm: object) -> None:
    """Seed a live ACTIVE contact desire + a rested state so tick 1 launches."""
    store = lm.state  # type: ignore[attr-defined]
    assert isinstance(store, MemoryPort)
    store.commit(State(u=2.0, energy=1.0, last_tick_at=_T0.isoformat()))
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0)))


def _event_trace_ids(conn: sqlite3.Connection, event: str) -> list[str]:
    return [r[0] for r in conn.execute("SELECT trace_id FROM trace_events WHERE event=?", (event,))]


def test_one_proactive_attempt_is_one_trace_id(tmp_path: Path) -> None:
    db_path = observability_db_path(tmp_path)
    writer = TraceWriter(db_path)
    writer.start()
    try:
        clock = FakeClock(_T0)
        lm = build_lifemodel(
            base_dir=tmp_path,
            clock=clock,
            tracer=FakeTracer(),
            trace_writer=writer,
            event_ring=EventRing(),
        )
        _seed_launch_ready(lm)
        egress = RecordingEgress(ReachOutcome.DELIVERED)

        # --- tick N: the launch (cognition mints origin trace T, delivery under T) ---
        clock.advance(timedelta(minutes=1))
        outcome = proactive_tick(lm, egress, {"chat_id": "1"})
        assert outcome is ReachOutcome.DELIVERED
        after_launch = lm.state.load()
        assert after_launch.pending_proactive_id is not None
        origin = after_launch.pending_proactive_origin_traceparent
        assert origin is not None
        expected_trace = parse_traceparent(origin).trace_id
        correlation_id = after_launch.pending_proactive_id

        # --- post_llm: the async turn finishes → its OWN frame weaves the outcome AND
        # resolves the desire under origin T, immediately (spec §3) — the outcome span
        # and the resolution span both land under T in this single frame. ---
        make_post_llm_observer(lambda: lm)(
            user_message=f"{IMPULSE_LABEL_PREFIX} a pull inside...",
            assistant_response="Hey, hi!",
        )

        assert writer.flush(timeout=5.0)
        conn = sqlite3.connect(str(db_path))

        # Every stage of the ONE attempt is under the ONE origin trace_id.
        for event in ("proactive_delivery", "proactive_outcome", "proactive_resolution"):
            ids = _event_trace_ids(conn, event)
            assert ids == [expected_trace], f"{event}: {ids} != [{expected_trace}]"

        # The correlation index resolved (so retention can reclaim it) and points at T.
        row = conn.execute(
            "SELECT origin_trace_id, resolved_at FROM trace_correlations WHERE correlation_id=?",
            (correlation_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == expected_trace
        assert row[1] is not None  # resolved_at stamped

        # The anchor is cleared in state after resolution (§4.4 clear-site).
        resolved_state = lm.state.load()
        assert resolved_state.pending_proactive_id is None
        assert resolved_state.pending_proactive_origin_traceparent is None

        # No orphan was produced on the happy path.
        assert _event_trace_ids(conn, "orphan_async_outcome") == []
        conn.close()
    finally:
        writer.stop()


def test_missing_origin_anchor_becomes_orphan_not_foreign_attachment(tmp_path: Path) -> None:
    db_path = observability_db_path(tmp_path)
    writer = TraceWriter(db_path)
    writer.start()
    try:
        lm = build_lifemodel(
            base_dir=tmp_path,
            clock=FakeClock(_T0),
            tracer=FakeTracer(),
            trace_writer=writer,
            event_ring=EventRing(),
        )
        # A pending turn whose durable origin anchor is GONE (only the id survived).
        lm.state.commit(State(pending_proactive_id="p-orphan"))
        lm.state.put(
            encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0))
        )

        make_post_llm_observer(lambda: lm)(
            user_message=f"{IMPULSE_LABEL_PREFIX} ...",
            assistant_response="[SILENT]",
        )

        assert writer.flush(timeout=5.0)
        conn = sqlite3.connect(str(db_path))

        # An explicit orphan was recorded (on its own fresh trace)...
        orphan_ids = _event_trace_ids(conn, "orphan_async_outcome")
        assert len(orphan_ids) == 1
        # ...and the outcome was NEVER attached to a fresh/foreign trace.
        assert _event_trace_ids(conn, "proactive_outcome") == []
        # The orphan carries the correlation id for the viewer.
        row = conn.execute(
            "SELECT fields_json FROM trace_events WHERE event='orphan_async_outcome'"
        ).fetchone()
        assert row is not None and "p-orphan" in row[0]
        conn.close()
    finally:
        writer.stop()
