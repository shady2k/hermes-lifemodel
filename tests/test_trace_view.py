"""Tests for the ``/lifemodel trace`` viewer (spec §4.6/§6.7).

Covers the pure seams (ring-overlay dedup, tree/render) and an end-to-end render
of a genuinely-populated ``observability.sqlite`` — one proactive attempt woven
under a single ``trace_id`` (tick → components → launch → async outcome →
resolution), plus ``last N``, an orphan, and the fail-soft edges.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lifemodel.composition import build_lifemodel
from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.core.proactive import proactive_tick
from lifemodel.core.taxonomy import verdict_signal
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.domain.egress import ReachOutcome, Verdict
from lifemodel.domain.objects import DesireState
from lifemodel.events import EventRing
from lifemodel.hooks import make_post_llm_observer
from lifemodel.ports.tracer import parse_traceparent
from lifemodel.state.model import State
from lifemodel.state.trace_store import (
    acquire_trace_writer,
    observability_db_path,
    release_trace_writer,
)
from lifemodel.testing.fakes import FakeClock
from lifemodel.testing.harness import RecordingEgress
from lifemodel.trace_view import _Event, _merge_events, trace_for_dir

T0 = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Pure seams
# --------------------------------------------------------------------------- #


def _evt(record_id: int, event: str, ts: str = "2026-07-09T12:00:00") -> _Event:
    return _Event(record_id, "T", "s1", 1, event, ts, {})


def test_merge_events_dedups_by_record_id() -> None:
    # A record present BOTH flushed and still-in-ring is not doubled (§4.2 #5).
    flushed = [_evt(1, "a"), _evt(2, "b")]
    ring = [
        {"record_id": 2, "trace_id": "T", "event": "b", "ts": "2026-07-09T12:00:00"},
        {"record_id": 3, "trace_id": "T", "event": "c", "ts": "2026-07-09T12:00:01"},
    ]
    merged = _merge_events(flushed, ring, "T")
    assert [e.record_id for e in merged] == [1, 2, 3]
    assert [e.event for e in merged] == ["a", "b", "c"]


def test_merge_events_ignores_ring_records_for_other_traces() -> None:
    ring = [{"record_id": 9, "trace_id": "OTHER", "event": "x", "ts": "t"}]
    assert _merge_events([_evt(1, "a")], ring, "T") == [_evt(1, "a")]


# --------------------------------------------------------------------------- #
# Fail-soft edges
# --------------------------------------------------------------------------- #


def test_missing_db_is_a_friendly_message_not_a_crash(tmp_path: Path) -> None:
    out = trace_for_dir(tmp_path, "anything")
    assert "no trace store yet" in out


def test_bare_and_bad_args_return_usage(tmp_path: Path) -> None:
    assert "usage:" in trace_for_dir(tmp_path, "")
    assert "usage:" in trace_for_dir(tmp_path, "last notanumber")


# --------------------------------------------------------------------------- #
# End-to-end: a real populated trace store
# --------------------------------------------------------------------------- #


@pytest.fixture
def populated(tmp_path: Path):
    """Populate observability.sqlite with one full proactive attempt + an orphan."""
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        clock = FakeClock(T0)
        lm = build_lifemodel(
            base_dir=tmp_path, clock=clock, trace_writer=writer, event_ring=EventRing()
        )
        lm.state.commit(State(u=2.0, energy=1.0, last_tick_at=T0.isoformat()))
        lm.state.put(
            encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0))
        )
        egress = RecordingEgress(ReachOutcome.DELIVERED)

        clock.advance(timedelta(minutes=1))
        proactive_tick(lm, egress, {"platform": "t", "chat_id": "1"})
        after = lm.state.load()
        corr = after.pending_proactive_id
        origin_tid = parse_traceparent(after.pending_proactive_origin_traceparent).trace_id

        make_post_llm_observer(lm)(
            user_message=f"{IMPULSE_LABEL_PREFIX} impulse",
            assistant_response="hi!",
        )
        clock.advance(timedelta(minutes=2))
        lm.bus.publish(
            verdict_signal(
                origin_id=f"verdict-{corr}",
                verdict=Verdict.FULFILL,
                timestamp=clock.now().isoformat(),
                correlation_id=corr,
            )
        )
        proactive_tick(lm, egress, {"platform": "t", "chat_id": "1"})

        # An orphan: post_llm with a pending id but NO origin anchor.
        lm.state.commit(State(pending_proactive_id="lost"))
        lm.state.put(
            encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=1.0))
        )
        make_post_llm_observer(lm)(
            user_message=f"{IMPULSE_LABEL_PREFIX} again", assistant_response="[SILENT]"
        )

        writer.flush(timeout=5.0)
        yield tmp_path, origin_tid
    finally:
        release_trace_writer(db)


def test_trace_by_id_renders_the_full_weave_under_one_trace_id(populated) -> None:
    base_dir, origin_tid = populated
    out = trace_for_dir(base_dir, origin_tid)

    # tick → components → decisions(attrs) → launch → async outcome → resolution,
    # ALL under the single origin trace_id.
    assert f"trace {origin_tid}" in out
    assert "cognition-launcher" in out  # the launch component
    assert "proactive_delivery" in out  # the delivery span
    assert "proactive_prompt" in out  # the exact prompt, durable under the trace
    assert "proactive_outcome" in out  # the async read-back
    assert "proactive_resolution" in out  # the resolving tick
    # decision attrs ride the span, self-explaining (not just reason codes).
    assert "wake_outcome=URGE" in out


def test_trace_last_n_lists_recent_root_traces_newest_first(populated) -> None:
    base_dir, origin_tid = populated
    out = trace_for_dir(base_dir, "last 5")
    # The orphan is the most recent root trace; the launch tick is older.
    assert "orphan_async_outcome" in out
    assert origin_tid in out
    orphan_pos = out.index("orphan_async_outcome")
    origin_pos = out.index(f"trace {origin_tid}")
    assert orphan_pos < origin_pos  # newest-first


def test_trace_last_default_is_one(populated) -> None:
    base_dir, _ = populated
    out = trace_for_dir(base_dir, "last")
    assert out.count("\ntrace ") == 1  # exactly one root trace rendered


def test_orphan_async_outcome_is_shown_explicitly(populated) -> None:
    base_dir, _ = populated
    out = trace_for_dir(base_dir, "last 5")
    assert "orphan_async_outcome" in out
    assert "async_correlation_missing" in out


def test_unknown_trace_id_is_a_friendly_message(populated) -> None:
    base_dir, _ = populated
    assert "no trace" in trace_for_dir(base_dir, "0" * 32)
