"""Unit tests for the BDI object-core substrate (lm-27n.1, HLA §4.1).

These pin the *single-door* contract of :mod:`lifemodel.domain.objects`: the
:class:`~lifemodel.domain.objects.KindRegistry` is the only path that
encodes/decodes/validates a typed kind over the generic ``memory_records``
envelope. The guardrail contract (invalid payload -> ``InvalidPayload``,
invalid transition -> ``InvalidTransition``, unknown kind -> ``UnknownKind``)
and the W3C-traceparent-compatible provenance validation live here; the
per-kind round-trips and state machines live in ``test_object_kinds.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import ClassVar

import pytest

from lifemodel.domain.memory import JsonObject, MemoryDraft, MemoryRecord
from lifemodel.domain.objects import (
    CONTACT_DESIRE_ID,
    BaseObject,
    Desire,
    DesireSpring,
    DesireState,
    Intention,
    InvalidPayload,
    InvalidTransition,
    KindRegistry,
    ObjectCoreError,
    Provenance,
    Sensitivity,
    Thought,
    UnknownKind,
    UserModel,
    default_registry,
    derive_id,
    format_traceparent,
    parse_traceparent,
    qualified_id,
)

# KindSpec is the extension/test seam — imported from the submodule, not the
# package surface (default_registry() is the blessed public factory).
from lifemodel.domain.objects.registry import KindSpec

#: W3C traceparent example values (32/16 lowercase hex, not all-zero).
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"
PARENT_SPAN_ID = "00f067aa0ba902b8"

EXPECTED_KINDS = frozenset({"desire", "intention", "user_model", "thought"})


def _provenance(**overrides: object) -> Provenance:
    base: dict[str, object] = dict(
        created_by="cognition",
        component="cognition.appraise",
        reason="test",
        turn_id="turn-1",
        source_object_ids=("thought:t1",),
        source_signal_ids=("sig-1",),
        trace_id=TRACE_ID,
        creation_span_id=SPAN_ID,
        parent_span_id=PARENT_SPAN_ID,
        trace_flags="01",
    )
    base.update(overrides)
    return Provenance(**base)  # type: ignore[arg-type]


def _desire(**overrides: object) -> Desire:
    base: dict[str, object] = dict(
        id="contact:owner",
        state=DesireState.ACTIVE,
        source="aggregation",
        recipient_id="owner",
        salience=0.7,
        confidence=0.5,
        expires_at=None,
        sensitivity=Sensitivity.SENSITIVE,
        supersedes="desire:old",
        superseded_by=None,
        tags=("contact", "warm"),
        provenance=_provenance(),
        object="reach out to Alex",
        spring=DesireSpring.MIXED,
        source_drive=0.4,
        source_thought_ids=("thought:t1", "thought:t2"),
        intensity=0.8,
        valence="positive",
        urgency=0.6,
        satiation_condition="sent a warm message",
        risk_if_acted=0.2,
        risk_if_ignored=0.5,
    )
    base.update(overrides)
    return Desire(**base)  # type: ignore[arg-type]


def _record_from_draft(draft: MemoryDraft, *, schema_version: int = 1) -> MemoryRecord:
    """Wrap an encoded draft as the store would (store-stamped fields supplied)."""
    return MemoryRecord(
        kind=draft.kind,
        id=draft.id,
        state=draft.state,
        payload=draft.payload,
        source=draft.source,
        recipient_id=draft.recipient_id,
        salience=draft.salience,
        confidence=draft.confidence,
        expires_at=draft.expires_at,
        created_at="2026-07-06T12:00:00+00:00",
        updated_at="2026-07-06T12:00:00+00:00",
        revision=0,
        schema_version=schema_version,
    )


def _desire_record(**payload_overrides: object) -> MemoryRecord:
    """A decodable desire record whose payload can be corrupted per-test."""
    reg = default_registry()
    record = _record_from_draft(reg.encode(_desire()))
    payload = copy.deepcopy(record.payload)
    payload.update(payload_overrides)
    return replace(record, payload=payload)


def _with_provenance(record: MemoryRecord, **prov_overrides: object) -> MemoryRecord:
    payload = copy.deepcopy(record.payload)
    prov = payload["_provenance"]
    assert isinstance(prov, dict)
    prov.update(prov_overrides)
    return replace(record, payload=payload)


# --- An out-of-catalog kind, used to prove the registry stays closed. -------
@dataclass(frozen=True, kw_only=True)
class _Unregistered(BaseObject):
    KIND: ClassVar[str] = "mystery"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {}

    @classmethod
    def _rebuild(cls, base: object, payload: object) -> _Unregistered:  # pragma: no cover
        raise NotImplementedError


# --- A malformed kind (reserved-prefixed semantic field), used for the guard.
@dataclass(frozen=True, kw_only=True)
class _BadKind(BaseObject):
    _secret: str = "x"
    KIND: ClassVar[str] = "bad"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:  # pragma: no cover
        return {}

    @classmethod
    def _rebuild(cls, base: object, payload: object) -> _BadKind:  # pragma: no cover
        raise NotImplementedError


class TestErrorTaxonomy:
    def test_all_errors_subclass_object_core_error(self) -> None:
        assert issubclass(UnknownKind, ObjectCoreError)
        assert issubclass(InvalidPayload, ObjectCoreError)
        assert issubclass(InvalidTransition, ObjectCoreError)


class TestClosedCatalog:
    def test_kinds_are_exactly_the_four(self) -> None:
        assert default_registry().kinds() == EXPECTED_KINDS

    def test_is_known(self) -> None:
        reg = default_registry()
        assert reg.is_known("desire")
        assert not reg.is_known("mystery")

    def test_states_of_unknown_kind_raises(self) -> None:
        with pytest.raises(UnknownKind):
            default_registry().states_of("mystery")

    def test_validate_transition_unknown_kind_raises(self) -> None:
        with pytest.raises(UnknownKind):
            default_registry().validate_transition("mystery", "active", "archived")

    def test_encode_unregistered_kind_raises(self) -> None:
        obj = _Unregistered(id="x", state="active", source="test")
        with pytest.raises(UnknownKind):
            default_registry().encode(obj)

    def test_decode_unknown_kind_raises(self) -> None:
        record = _desire_record()
        alien = replace(record, kind="mystery")
        with pytest.raises(UnknownKind):
            default_registry().decode(alien)

    def test_no_public_register_method(self) -> None:
        assert not hasattr(KindRegistry, "register")


class TestDeterministicIds:
    def test_contact_desire_id_literal(self) -> None:
        assert CONTACT_DESIRE_ID == "contact:owner"

    def test_qualified_id(self) -> None:
        assert qualified_id("desire", "contact:owner") == "desire:contact:owner"

    def test_derive_id_is_reproducible(self) -> None:
        assert derive_id("thought", "abc", "42") == "thought:abc:42"
        assert derive_id("a", "b") == derive_id("a", "b")


class TestReservedKeyGuard:
    def test_registering_reserved_prefixed_field_fails(self) -> None:
        with pytest.raises(ValueError):
            KindRegistry([KindSpec(cls=_BadKind, transitions={"active": frozenset()})])

    def test_registering_duplicate_kind_fails(self) -> None:
        spec = KindSpec(cls=Desire, transitions={"active": frozenset()})
        with pytest.raises(ValueError):
            KindRegistry([spec, spec])

    def test_registering_dangling_target_state_fails(self) -> None:
        # "active" points at "archived" but "archived" is not itself a state key.
        spec = KindSpec(cls=Desire, transitions={"active": frozenset({"archived"})})
        with pytest.raises(ValueError):
            KindRegistry([spec])

    def test_registering_empty_transition_table_fails(self) -> None:
        with pytest.raises(ValueError):
            KindRegistry([KindSpec(cls=Desire, transitions={})])


class TestEnvelopeDefaults:
    def test_minimal_object_defaults_round_trip(self) -> None:
        reg = default_registry()
        obj = _desire(
            sensitivity=Sensitivity.NORMAL,
            supersedes=None,
            superseded_by=None,
            tags=(),
            provenance=None,
        )
        decoded = reg.decode(_record_from_draft(reg.encode(obj)))
        assert decoded == obj
        assert decoded.sensitivity is Sensitivity.NORMAL
        assert decoded.tags == ()
        assert decoded.supersedes is None
        assert decoded.provenance is None

    def test_sensitivity_defaults_to_normal_on_the_dataclass(self) -> None:
        obj = Desire(
            id="d1",
            state=DesireState.ACTIVE,
            source="test",
            object="x",
            spring=DesireSpring.DRIVE,
            source_drive=None,
            source_thought_ids=(),
            intensity=0.0,
            valence="neutral",
            urgency=0.0,
            satiation_condition="",
            risk_if_acted=0.0,
            risk_if_ignored=0.0,
        )
        assert obj.sensitivity is Sensitivity.NORMAL
        assert obj.recipient_id == "owner"


class TestSchemaVersion:
    def test_decode_rejects_mismatched_schema_version(self) -> None:
        reg = default_registry()
        record = _record_from_draft(reg.encode(_desire()), schema_version=99)
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_all_kinds_declare_schema_version_one(self) -> None:
        for cls in (Desire, Intention, UserModel, Thought):
            assert cls.SCHEMA_VERSION == 1, cls.__name__


class TestTagsAreNotLifecycle:
    def test_state_comes_from_column_not_tags(self) -> None:
        reg = default_registry()
        # Tags that *look* like lifecycle states must not steer the decoded state.
        record = _desire_record(_tags=["satisfied", "dropped", "expired"])
        decoded = reg.decode(record)
        assert decoded.state == "active"  # the column, unchanged
        assert decoded.tags == ("satisfied", "dropped", "expired")


class TestInvalidPayloadGuardrail:
    def test_missing_required_field_raises(self) -> None:
        reg = default_registry()
        payload = copy.deepcopy(reg.encode(_desire()).payload)
        del payload["object"]
        record = _record_from_draft(
            MemoryDraft(kind="desire", id="d1", state="active", payload=payload, source="test")
        )
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_wrong_type_field_raises(self) -> None:
        reg = default_registry()
        record = _desire_record(intensity="not-a-number")
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_non_finite_float_raises(self) -> None:
        reg = default_registry()
        record = _desire_record(urgency=float("nan"))
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_unknown_enum_value_raises(self) -> None:
        reg = default_registry()
        record = _desire_record(spring="telepathy")
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_unknown_state_on_decode_raises(self) -> None:
        reg = default_registry()
        record = replace(_desire_record(), state="on-fire")
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_unknown_sensitivity_raises(self) -> None:
        reg = default_registry()
        record = _desire_record(_sensitivity="top-secret")
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_tags_not_a_list_raises(self) -> None:
        reg = default_registry()
        record = _desire_record(_tags="not-a-list")
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_tag_item_not_a_str_raises(self) -> None:
        reg = default_registry()
        record = _desire_record(_tags=[1, 2, 3])
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_supersedes_wrong_type_raises(self) -> None:
        reg = default_registry()
        record = _desire_record(_supersedes=123)
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_provenance_not_an_object_raises(self) -> None:
        reg = default_registry()
        record = _desire_record(_provenance="not-an-object")
        with pytest.raises(InvalidPayload):
            reg.decode(record)


class TestEnvelopeRoundTrip:
    def test_superseded_by_round_trips(self) -> None:
        reg = default_registry()
        obj = _desire(supersedes=None, superseded_by="desire:newer")
        decoded = reg.decode(_record_from_draft(reg.encode(obj)))
        assert decoded.superseded_by == "desire:newer"
        assert decoded == obj

    def test_encode_omits_absent_optional_envelope_keys(self) -> None:
        reg = default_registry()
        obj = _desire(supersedes=None, superseded_by=None, provenance=None)
        draft = reg.encode(obj)
        assert "_supersedes" not in draft.payload
        assert "_superseded_by" not in draft.payload
        assert "_provenance" not in draft.payload


class TestProvenanceTraceValidation:
    def test_valid_trace_round_trips(self) -> None:
        reg = default_registry()
        obj = _desire()
        decoded = reg.decode(_record_from_draft(reg.encode(obj)))
        assert decoded.provenance is not None
        assert decoded.provenance.trace_id == TRACE_ID
        assert decoded.provenance.creation_span_id == SPAN_ID
        assert decoded.provenance.parent_span_id == PARENT_SPAN_ID
        assert decoded.provenance == obj.provenance

    def test_absent_trace_fields_are_legal(self) -> None:
        prov = Provenance(created_by="aggregation", component="agg", reason="r")
        assert prov.trace_id is None
        assert prov.creation_span_id is None

    def test_hex_is_normalized_to_lowercase(self) -> None:
        prov = Provenance(
            created_by="c",
            component="c",
            reason="r",
            trace_id=TRACE_ID.upper(),
            creation_span_id=SPAN_ID.upper(),
        )
        assert prov.trace_id == TRACE_ID
        assert prov.creation_span_id == SPAN_ID

    @pytest.mark.parametrize(
        "bad_trace_id",
        [
            "abc",  # too short
            "z" * 32,  # non-hex
            "0" * 32,  # all-zero
            TRACE_ID + "00",  # too long
        ],
    )
    def test_bad_trace_id_rejected_on_decode(self, bad_trace_id: str) -> None:
        reg = default_registry()
        record = _with_provenance(_desire_record(), trace_id=bad_trace_id)
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_bad_span_length_rejected_on_decode(self) -> None:
        reg = default_registry()
        record = _with_provenance(_desire_record(), creation_span_id="0" * 17)
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_all_zero_span_rejected_on_decode(self) -> None:
        reg = default_registry()
        record = _with_provenance(_desire_record(), creation_span_id="0" * 16)
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_bad_trace_flags_rejected_on_decode(self) -> None:
        reg = default_registry()
        record = _with_provenance(_desire_record(), trace_flags="zz")
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_constructing_bad_provenance_raises_invalid_payload(self) -> None:
        with pytest.raises(InvalidPayload):
            Provenance(created_by="c", component="c", reason="r", trace_id="nope")


class TestTraceparentHelpers:
    def test_format_matches_w3c_shape(self) -> None:
        prov = _provenance()
        assert format_traceparent(prov) == f"00-{TRACE_ID}-{SPAN_ID}-01"

    def test_format_defaults_flags_to_sampled(self) -> None:
        prov = Provenance(
            created_by="c",
            component="c",
            reason="r",
            trace_id=TRACE_ID,
            creation_span_id=SPAN_ID,
        )
        assert format_traceparent(prov) == f"00-{TRACE_ID}-{SPAN_ID}-01"

    def test_format_parse_round_trip(self) -> None:
        prov = _provenance()
        assert parse_traceparent(format_traceparent(prov)) == (TRACE_ID, SPAN_ID, "01")

    def test_format_without_trace_raises(self) -> None:
        prov = Provenance(created_by="c", component="c", reason="r")
        with pytest.raises(ValueError):
            format_traceparent(prov)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "00-only-three",
            f"01-{TRACE_ID}-{SPAN_ID}-01",  # unsupported version
            f"00-{'0' * 32}-{SPAN_ID}-01",  # all-zero trace
            f"00-{TRACE_ID}-{'0' * 16}-01",  # all-zero span
            f"00-{TRACE_ID}-{SPAN_ID}-0",  # 1-hex flags
            f"00-{TRACE_ID}-zzzzzzzzzzzzzzzz-01",  # non-hex span
        ],
    )
    def test_parse_rejects_malformed(self, bad: str) -> None:
        with pytest.raises(InvalidPayload):
            parse_traceparent(bad)


class TestCreationContextNaming:
    def test_field_is_creation_span_id_not_span_id(self) -> None:
        # Guards the semantic drift codex flagged: it is the trace context *at
        # creation*, never "this object's live span".
        prov = _provenance()
        assert hasattr(prov, "creation_span_id")
        assert not hasattr(prov, "span_id")


# --- An object whose KIND is registered but whose class is wrong. ------------
@dataclass(frozen=True, kw_only=True)
class _FakeDesire(BaseObject):
    KIND: ClassVar[str] = "desire"  # a real kind, but this is not a Desire
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:  # pragma: no cover
        return {}

    @classmethod
    def _rebuild(cls, base: object, payload: object) -> _FakeDesire:  # pragma: no cover
        raise NotImplementedError


class TestWriteDoorValidation:
    """The write door validates too — invalid state/class never reaches storage."""

    def test_encode_rejects_unknown_state(self) -> None:
        reg = default_registry()
        obj = _desire(state="on-fire")
        with pytest.raises(InvalidPayload):
            reg.encode(obj)

    def test_encode_rejects_wrong_class_for_registered_kind(self) -> None:
        reg = default_registry()
        obj = _FakeDesire(id="d1", state="active", source="test")
        with pytest.raises(InvalidPayload):
            reg.encode(obj)

    def test_encode_rejects_type_invalid_semantic_field(self) -> None:
        # Static typing can't guarantee field types at runtime (LLM-built object,
        # or a `# type: ignore`). The write door still rejects a non-float intensity.
        reg = default_registry()
        obj = _desire(intensity="not-a-number")
        with pytest.raises(InvalidPayload):
            reg.encode(obj)


class TestPayloadKeyStrictness:
    """Nothing beyond the kind's declared semantic + reserved keys survives decode."""

    def test_stray_schema_version_key_rejected(self) -> None:
        reg = default_registry()
        # The forbidden `_schema_version` payload key (schema_version is a
        # store-stamped COLUMN, never in payload — a split-brain guard).
        record = _desire_record(_schema_version=1)
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_stray_reserved_underscore_key_rejected(self) -> None:
        reg = default_registry()
        record = _desire_record(_smuggled="x")
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_unknown_semantic_key_rejected(self) -> None:
        reg = default_registry()
        record = _desire_record(not_a_real_desire_field="x")
        with pytest.raises(InvalidPayload):
            reg.decode(record)

    def test_non_dict_payload_rejected(self) -> None:
        reg = default_registry()
        record = replace(_desire_record(), payload="not-an-object")  # type: ignore[arg-type]
        with pytest.raises(InvalidPayload):
            reg.decode(record)


class TestGuardrailContract:
    """The three mandatory contract tests, together in one place."""

    def test_invalid_payload_raises_invalid_payload(self) -> None:
        with pytest.raises(InvalidPayload):
            default_registry().decode(_desire_record(intensity=object()))  # type: ignore[arg-type]

    def test_invalid_transition_raises_invalid_transition(self) -> None:
        with pytest.raises(InvalidTransition):
            default_registry().validate_transition("desire", "active", "expired")

    def test_unknown_kind_raises_unknown_kind(self) -> None:
        with pytest.raises(UnknownKind):
            default_registry().states_of("mystery")
