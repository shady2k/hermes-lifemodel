"""Unit tests for the pure memory-domain helpers (lm-fib.6.1, HLA §4.1/D7).

These pin the semantics shared by both ``MemoryPort`` implementations (the
real :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` and
:class:`~lifemodel.testing.fakes.FakeMemoryStore`) so a bug in one place is
caught here rather than only showing up as a fake/real divergence in the
contract suite.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta, timezone

import pytest

from lifemodel.domain.memory import (
    MemoryDraft,
    MemoryMutation,
    MemoryPatch,
    MemoryRecord,
    MemorySerializationError,
    PressureIndex,
    PutOp,
    TransitionOp,
    coalesce_patch,
    describe_stale_transition,
    ensure_json_serializable,
    epoch_ms,
    merge_payload,
    normalize_expires_at,
    stamp_iso_utc,
    summarize_pressure_index,
)


def _record(**overrides: object) -> MemoryRecord:
    base = dict(
        kind="desire",
        id="d1",
        state="active",
        payload={},
        source="test",
        recipient_id="owner",
        salience=0.0,
        confidence=None,
        expires_at=None,
        created_at="2026-07-06T12:00:00+00:00",
        updated_at="2026-07-06T12:00:00+00:00",
        revision=0,
        schema_version=1,
    )
    base.update(overrides)
    return MemoryRecord(**base)  # type: ignore[arg-type]


class TestEnsureJsonSerializable:
    def test_accepts_plain_json_object(self) -> None:
        ensure_json_serializable({"a": 1, "b": [1, 2, "x"], "c": None})

    def test_rejects_non_finite_float(self) -> None:
        with pytest.raises(MemorySerializationError):
            ensure_json_serializable({"a": float("nan")})

    def test_rejects_non_serializable_value(self) -> None:
        with pytest.raises(MemorySerializationError):
            ensure_json_serializable({"a": object()})  # type: ignore[dict-item]


class TestNormalizeExpiresAt:
    def test_none_passes_through(self) -> None:
        assert normalize_expires_at(None) is None

    def test_normalizes_tz_aware_iso_to_fixed_width_utc(self) -> None:
        # A +05:00 offset with whole-second precision must land as canonical,
        # fixed-width UTC TEXT — no raw caller string reaches a column.
        assert (
            normalize_expires_at("2026-07-06T17:00:00+05:00") == "2026-07-06T12:00:00.000000+00:00"
        )

    def test_rejects_naive_timestamp(self) -> None:
        with pytest.raises(MemorySerializationError):
            normalize_expires_at("2026-07-06T12:00:00")

    def test_rejects_malformed_timestamp(self) -> None:
        with pytest.raises(MemorySerializationError):
            normalize_expires_at("not-a-timestamp")


class TestEpochMs:
    def test_matches_manual_computation(self) -> None:
        # Retained only for the store's forensic filename stamps, not columns.
        dt = datetime(2026, 7, 6, 12, 0, 0, tzinfo=UTC)
        assert epoch_ms(dt) == int(dt.timestamp() * 1000)


class TestStampIsoUtc:
    def test_passes_through_utc_instant_fixed_width(self) -> None:
        dt = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
        assert stamp_iso_utc(dt) == "2026-07-06T12:00:00.000000+00:00"

    def test_normalizes_non_utc_offset_to_utc(self) -> None:
        tz = timezone(timedelta(hours=5))
        dt = datetime(2026, 7, 6, 17, 0, tzinfo=tz)  # == 12:00 UTC
        assert stamp_iso_utc(dt) == "2026-07-06T12:00:00.000000+00:00"

    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(MemorySerializationError):
            stamp_iso_utc(datetime(2026, 7, 6, 12, 0))  # no tzinfo


class TestMergePayload:
    def test_none_merge_returns_copy_of_existing(self) -> None:
        existing = {"a": 1}
        result = merge_payload(existing, None)
        assert result == existing
        assert result is not existing  # a copy, not the same dict

    def test_shallow_merges_top_level_keys(self) -> None:
        existing = {"a": 1, "b": 2}
        result = merge_payload(existing, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_dicts_replaced_wholesale_not_deep_merged(self) -> None:
        existing = {"nested": {"x": 1, "y": 2}}
        result = merge_payload(existing, {"nested": {"y": 3}})
        assert result == {"nested": {"y": 3}}


class TestCoalescePatch:
    def test_none_patch_value_keeps_existing(self) -> None:
        assert coalesce_patch(None, "old") == "old"

    def test_non_none_patch_value_replaces_existing(self) -> None:
        assert coalesce_patch("new", "old") == "new"


class TestDescribeStaleTransition:
    def test_missing_record_message(self) -> None:
        msg = describe_stale_transition("desire", "d1", "active", None)
        assert "d1" in msg
        assert "desire" in msg

    def test_wrong_state_message_mentions_actual_and_expected(self) -> None:
        msg = describe_stale_transition("desire", "d1", "active", "archived")
        assert "archived" in msg
        assert "active" in msg


class TestSummarizePressureIndex:
    def test_empty_returns_default(self) -> None:
        now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
        assert summarize_pressure_index([], now) == PressureIndex()

    def test_counts_only_active_desires_and_tracks_max_salience(self) -> None:
        now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
        records = [
            _record(id="d1", salience=0.5),
            _record(id="d2", salience=0.9),
            _record(id="d3", kind="fact"),  # wrong kind, excluded
            _record(id="d4", state="archived"),  # wrong state, excluded
        ]
        idx = summarize_pressure_index(records, now)
        assert idx.active_desire_count == 2
        assert idx.max_desire_salience == 0.9
        assert idx.contact_frame_available is True

    def test_excludes_expired_records(self) -> None:
        now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
        expired = _record(id="d1", expires_at="2026-07-06T11:59:59+00:00")
        alive = _record(id="d2", expires_at="2026-07-06T12:00:01+00:00")
        idx = summarize_pressure_index([expired, alive], now)
        assert idx.active_desire_count == 1
        assert idx.max_desire_salience == 0.0


# --- lm-27n.2: schema_version on the draft + the mutation value types ---


class TestMemoryDraftSchemaVersion:
    def test_defaults_to_one(self) -> None:
        draft = MemoryDraft(kind="desire", id="d", state="active", payload={}, source="t")
        assert draft.schema_version == 1

    def test_carries_an_explicit_version(self) -> None:
        draft = MemoryDraft(
            kind="desire", id="d", state="active", payload={}, source="t", schema_version=2
        )
        assert draft.schema_version == 2


class TestMutationValueTypes:
    def test_put_op_wraps_a_draft(self) -> None:
        draft = MemoryDraft(kind="desire", id="d", state="active", payload={}, source="t")
        op = PutOp(draft)
        assert op.draft is draft
        assert isinstance(op, MemoryMutation)  # runtime union membership

    def test_transition_op_defaults_patch_to_none(self) -> None:
        op = TransitionOp(kind="desire", id="d", from_state="active", to_state="archived")
        assert op.patch is None
        assert isinstance(op, MemoryMutation)

    def test_transition_op_carries_a_patch(self) -> None:
        patch = MemoryPatch(salience=0.5)
        op = TransitionOp(
            kind="desire", id="d", from_state="active", to_state="archived", patch=patch
        )
        assert op.patch is patch

    def test_ops_are_frozen_and_equal_by_value(self) -> None:
        a = TransitionOp(kind="desire", id="d", from_state="active", to_state="archived")
        b = TransitionOp(kind="desire", id="d", from_state="active", to_state="archived")
        assert a == b
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.kind = "fact"  # type: ignore[misc]
