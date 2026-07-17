"""Per-kind round-trips and state machines for the BDI object core (lm-27n.1).

One representative instance of each of the four kinds is built with a fully
populated envelope (sensitivity/supersession/tags/provenance incl. W3C trace
fields), then driven through ``decode(encode(obj)) == obj``. Each kind's
explicit transition table is exercised edge-by-edge through the registry's
``validate_transition`` — the only door for lifecycle legality.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from lifemodel.domain.memory import MemoryDraft, MemoryRecord
from lifemodel.domain.objects import (
    BaseObject,
    Desire,
    DesireSpring,
    DesireState,
    InferredField,
    Intention,
    IntentionState,
    InvalidPayload,
    InvalidTransition,
    KindRegistry,
    Provenance,
    Sensitivity,
    Thought,
    ThoughtState,
    UserModel,
    UserModelState,
    default_registry,
)
from lifemodel.domain.objects.belief import BELIEF_TRANSITIONS
from lifemodel.domain.objects.commitment import COMMITMENT_TRANSITIONS
from lifemodel.domain.objects.desire import DESIRE_TRANSITIONS
from lifemodel.domain.objects.intention import INTENTION_TRANSITIONS
from lifemodel.domain.objects.thought import THOUGHT_TRANSITIONS
from lifemodel.domain.objects.user_model import USER_MODEL_TRANSITIONS

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"


def _provenance() -> Provenance:
    return Provenance(
        created_by="cognition",
        component="cognition.appraise",
        reason="test",
        turn_id="turn-1",
        source_object_ids=("thought:t1",),
        source_signal_ids=("sig-1",),
        trace_id=TRACE_ID,
        creation_span_id=SPAN_ID,
        parent_span_id="00f067aa0ba902b8",
        trace_flags="01",
    )


_ENVELOPE: dict[str, object] = dict(
    source="cognition",
    recipient_id="owner",
    salience=0.7,
    confidence=0.5,
    sensitivity=Sensitivity.SENSITIVE,
    supersedes="thought:old",
    superseded_by=None,
    tags=("a", "b"),
    provenance=_provenance(),
)


def _desire() -> Desire:
    return Desire(
        id="contact:owner",
        state=DesireState.ACTIVE,
        object="reach out to Alex",
        spring=DesireSpring.MIXED,
        source_drive=0.4,
        source_thought_ids=("thought:t1",),
        intensity=0.8,
        valence="positive",
        urgency=0.6,
        satiation_condition="sent a warm message",
        risk_if_acted=0.2,
        risk_if_ignored=0.5,
        **_ENVELOPE,  # type: ignore[arg-type]
    )


def _intention() -> Intention:
    return Intention(
        id="int-1",
        state=IntentionState.PENDING,
        goal="check in with Alex this week",
        commitment_strength=0.9,
        plan="draft, then send when they're free",
        implementation_trigger="Friday evening",
        constraints=("no work talk",),
        admissibility_filter="quiet-hours-aware",
        reconsideration_triggers=("they reach out first",),
        expiry=None,
        rationale="the friendship matters",
        **_ENVELOPE,  # type: ignore[arg-type]
    )


def _user_model() -> UserModel:
    return UserModel(
        id="rel-alex",
        state=UserModelState.ACTIVE,
        cadence=InferredField("weekly"),
        good_hours=InferredField((18, 19, 20)),
        bad_hours=InferredField((2, 3, 4)),
        response_valence_pattern=InferredField("warm-but-slow"),
        privacy_boundaries=InferredField(("no health details",)),
        topic_sensitivity=InferredField(("work",)),
        intimacy_depth=InferredField(0.6),
        reply_latency_norm=InferredField("hours"),
        known_load=InferredField("busy at work"),
        acceptable_styles=InferredField(("playful", "concise")),
        explicit_preferences=InferredField(("texts over calls",)),
        **_ENVELOPE,  # type: ignore[arg-type]
    )


def _thought() -> Thought:
    return Thought(
        id="thought:t9",
        state=ThoughtState.ACTIVE,
        content="I wonder how Alex's move went",
        trigger="idle",
        parent_id=None,
        attention_score=0.5,
        no_progress_count=0,
        loop_signature="alex-move",
        parked_until=None,
        park_count=0,
        sustained_attention_count=0,
        actionability=0.7,
        other_regarding_value=0.8,
        **_ENVELOPE,  # type: ignore[arg-type]
    )


def _record_from_draft(draft: MemoryDraft, *, schema_version: int = 1) -> MemoryRecord:
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


ALL_KINDS: list[BaseObject] = [_desire(), _intention(), _user_model(), _thought()]

TRANSITION_TABLES: dict[str, Mapping[str, frozenset[str]]] = {
    "desire": DESIRE_TRANSITIONS,
    "intention": INTENTION_TRANSITIONS,
    "user_model": USER_MODEL_TRANSITIONS,
    "thought": THOUGHT_TRANSITIONS,
    "commitment": COMMITMENT_TRANSITIONS,
    "belief": BELIEF_TRANSITIONS,
}


class TestRoundTrip:
    @pytest.mark.parametrize("obj", ALL_KINDS, ids=lambda o: o.KIND)
    def test_decode_encode_is_identity(self, obj: BaseObject) -> None:
        reg = default_registry()
        record = _record_from_draft(reg.encode(obj), schema_version=obj.SCHEMA_VERSION)
        assert reg.decode(record) == obj

    @pytest.mark.parametrize("obj", ALL_KINDS, ids=lambda o: o.KIND)
    def test_encode_draft_carries_the_kind_and_columns(self, obj: BaseObject) -> None:
        draft = default_registry().encode(obj)
        assert draft.kind == obj.KIND
        assert draft.id == obj.id
        assert draft.state == obj.state
        assert draft.source == obj.source

    @pytest.mark.parametrize("obj", ALL_KINDS, ids=lambda o: o.KIND)
    def test_reserved_keys_live_in_payload_semantic_at_top_level(self, obj: BaseObject) -> None:
        # Non-tautological: check the kind's OWN semantic output (not a set
        # already filtered of `_` keys) — this is the invariant that keeps the
        # underscore-namespaced envelope from colliding with semantic fields.
        semantic = obj._semantic_payload()
        assert semantic  # each kind carries at least one semantic field
        assert all(not k.startswith("_") for k in semantic), semantic
        draft = default_registry().encode(obj)
        assert "_sensitivity" in draft.payload
        assert "_provenance" in draft.payload


class TestStateMachines:
    @pytest.mark.parametrize("kind", sorted(TRANSITION_TABLES))
    def test_every_allowed_edge_passes(self, kind: str) -> None:
        reg = default_registry()
        table = TRANSITION_TABLES[kind]
        for from_state, to_states in table.items():
            for to_state in to_states:
                reg.validate_transition(kind, from_state, to_state)  # must not raise

    @pytest.mark.parametrize("kind", sorted(TRANSITION_TABLES))
    def test_states_of_matches_the_table_keys(self, kind: str) -> None:
        reg = default_registry()
        assert reg.states_of(kind) == frozenset(TRANSITION_TABLES[kind].keys())

    @pytest.mark.parametrize("kind", sorted(TRANSITION_TABLES))
    def test_terminal_states_have_empty_out_sets(self, kind: str) -> None:
        reg = default_registry()
        table = TRANSITION_TABLES[kind]
        terminals = [s for s, outs in table.items() if not outs]
        assert terminals  # every kind has at least one terminal state
        for terminal in terminals:
            for other in table:
                if other == terminal:
                    continue
                with pytest.raises(InvalidTransition):
                    reg.validate_transition(kind, terminal, other)

    @pytest.mark.parametrize("kind", sorted(TRANSITION_TABLES))
    def test_disallowed_edge_raises(self, kind: str) -> None:
        reg = default_registry()
        table = TRANSITION_TABLES[kind]
        states = set(table)
        for from_state, to_states in table.items():
            disallowed = states - to_states - {from_state}
            for to_state in disallowed:
                with pytest.raises(InvalidTransition):
                    reg.validate_transition(kind, from_state, to_state)

    @pytest.mark.parametrize("kind", sorted(TRANSITION_TABLES))
    def test_unknown_state_raises(self, kind: str) -> None:
        reg = default_registry()
        with pytest.raises(InvalidTransition):
            reg.validate_transition(kind, "active", "definitely-not-a-state")
        with pytest.raises(InvalidTransition):
            reg.validate_transition(kind, "definitely-not-a-state", "active")


class TestSpecificTransitions:
    def test_desire_active_to_deferred_allowed(self) -> None:
        default_registry().validate_transition("desire", "active", "deferred")

    def test_desire_active_to_expired_disallowed(self) -> None:
        with pytest.raises(InvalidTransition):
            default_registry().validate_transition("desire", "active", "expired")

    def test_intention_pending_to_active_allowed(self) -> None:
        default_registry().validate_transition("intention", "pending", "active")

    def test_user_model_only_archives(self) -> None:
        reg = default_registry()
        reg.validate_transition("user_model", "active", "archived")
        with pytest.raises(InvalidTransition):
            reg.validate_transition("user_model", "active", "dropped")

    def test_thought_active_to_merged_allowed(self) -> None:
        default_registry().validate_transition("thought", "active", "merged")


class TestIntTupleDecoding:
    def test_good_hours_round_trip(self) -> None:
        reg = default_registry()
        rel = _user_model()
        decoded = reg.decode(_record_from_draft(reg.encode(rel)))
        assert isinstance(decoded, UserModel)
        assert decoded.good_hours.value == (18, 19, 20)

    def test_non_int_hour_item_raises(self) -> None:
        reg = default_registry()
        record = _record_from_draft(reg.encode(_user_model()))
        payload = dict(record.payload)
        # The inferred-field value list holds a non-int item -> reject on decode.
        payload["good_hours"] = {"value": [18, "nineteen", 20], "inferred_at": None, "ttl": None}
        with pytest.raises(InvalidPayload):
            reg.decode(replace(record, payload=payload))

    def test_bool_is_not_a_valid_int_hour(self) -> None:
        reg = default_registry()
        record = _record_from_draft(reg.encode(_user_model()))
        payload = dict(record.payload)
        payload["bad_hours"] = {"value": [True, 3, 4], "inferred_at": None, "ttl": None}
        with pytest.raises(InvalidPayload):
            reg.decode(replace(record, payload=payload))


class TestRegistryIsClosed:
    def test_default_registry_returns_fresh_instances(self) -> None:
        assert default_registry() is not default_registry()

    def test_kind_registry_type_is_the_only_door(self) -> None:
        assert isinstance(default_registry(), KindRegistry)


class TestLiveStates:
    """``KindRegistry.live_states`` (lm-27n.6): the union of non-terminal states
    that drives the registry-aware coreloop snapshot."""

    def test_live_states_is_the_catalog_union(self) -> None:
        assert default_registry().live_states() == frozenset(
            {"active", "deferred", "pending", "parked"}
        )

    def test_live_states_equals_the_nonterminal_union_of_the_tables(self) -> None:
        expected: set[str] = set()
        for table in TRANSITION_TABLES.values():
            expected |= {state for state, outs in table.items() if outs}
        assert default_registry().live_states() == frozenset(expected)

    def test_terminal_states_of_matches_empty_out_sets(self) -> None:
        reg = default_registry()
        for kind, table in TRANSITION_TABLES.items():
            assert reg.terminal_states_of(kind) == frozenset(
                state for state, outs in table.items() if not outs
            )

    def test_terminal_nonterminal_consistency_invariant(self) -> None:
        # The load-bearing invariant behind a union-state snapshot: NO state string
        # may be terminal for one kind while non-terminal (live) for another — else
        # a live_states() sweep would fetch a row that is terminal for its own kind.
        # A future kind that violates it must fail HERE, loudly.
        reg = default_registry()
        terminal: set[str] = set()
        live: set[str] = set()
        for kind in TRANSITION_TABLES:
            terminal |= reg.terminal_states_of(kind)
            live |= reg.states_of(kind) - reg.terminal_states_of(kind)
        conflict = terminal & live
        assert not conflict, (
            f"states terminal for one kind but live for another: {sorted(conflict)}"
        )

    def test_live_states_excludes_every_kinds_own_terminals(self) -> None:
        # A corollary of the invariant: a state terminal in its own kind never
        # appears in live_states(), so the snapshot never surfaces a dead row.
        reg = default_registry()
        live = reg.live_states()
        for kind in TRANSITION_TABLES:
            assert reg.terminal_states_of(kind).isdisjoint(live)
