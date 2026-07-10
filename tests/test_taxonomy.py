from __future__ import annotations

import pytest

from lifemodel.core.taxonomy import (
    KIND_CONTACT,
    KIND_CONTACT_OBSERVED,
    KIND_IN_FLIGHT,
    KIND_PROACTIVE_OUTCOME,
    contact_observed_signal,
    contact_signal,
    contact_value,
    in_flight_signal,
    is_in_flight,
    is_kind,
    proactive_outcome_signal,
    read_contact_observed,
    read_proactive_outcome,
)
from lifemodel.core.taxonomy import (
    contact_signal as _contact_signal,
)
from lifemodel.domain.egress import ProactiveOutcome


def test_contact_signal_carries_value_and_delta() -> None:
    sig = contact_signal(
        origin_id="c-1", value=1.25, delta=0.02, timestamp="2026-07-06T00:00:00+00:00"
    )
    assert sig.kind == KIND_CONTACT
    assert sig.origin_id == "c-1"
    assert sig.payload["value"] == 1.25
    assert sig.payload["delta"] == 0.02
    assert is_kind(sig, KIND_CONTACT)
    assert not is_kind(sig, KIND_CONTACT_OBSERVED)


def test_contact_observed_signal_roundtrips_actor_label() -> None:
    sig = contact_observed_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None)
    assert sig.kind == KIND_CONTACT_OBSERVED
    assert read_contact_observed(sig) == ("user", "two_way")


def test_read_contact_observed_rejects_wrong_kind() -> None:
    sig = contact_signal(origin_id="c-2", value=0.0, delta=0.0, timestamp=None)
    with pytest.raises(ValueError):
        read_contact_observed(sig)


def test_read_contact_observed_rejects_bad_payload() -> None:
    from lifemodel.domain.signal import Signal

    sig = Signal(
        origin_id="e-2", kind=KIND_CONTACT_OBSERVED, payload={"actor": "user"}
    )  # missing label
    with pytest.raises(ValueError):
        read_contact_observed(sig)


def test_proactive_outcome_signal_roundtrips() -> None:
    sig = proactive_outcome_signal(origin_id="v-1", outcome=ProactiveOutcome.SENT, timestamp=None)
    assert sig.kind == KIND_PROACTIVE_OUTCOME
    assert read_proactive_outcome(sig) is ProactiveOutcome.SENT


def test_read_proactive_outcome_rejects_bad_value() -> None:
    from lifemodel.domain.signal import Signal

    with pytest.raises(ValueError):
        read_proactive_outcome(
            Signal(origin_id="v-2", kind=KIND_PROACTIVE_OUTCOME, payload={"verdict": "nope"})
        )


def test_in_flight_signal_and_reader() -> None:
    busy = in_flight_signal(origin_id="f-1", value=True, timestamp=None)
    idle = in_flight_signal(origin_id="f-2", value=False, timestamp=None)
    assert busy.kind == KIND_IN_FLIGHT
    assert is_in_flight([idle, busy]) is True
    assert is_in_flight([idle]) is False
    assert is_in_flight([]) is False


def test_contact_value_reads_transient_signal_or_default() -> None:
    c = _contact_signal(origin_id="c-9", value=2.5, delta=0.1, timestamp=None)
    assert contact_value([c], default=0.0) == 2.5
    assert contact_value([], default=1.23) == 1.23


def test_control_kinds_are_control_lane() -> None:
    from lifemodel.core.taxonomy import (
        CONTROL_KINDS,
        KIND_CONTACT_OBSERVED,
        KIND_IN_FLIGHT,
        KIND_PROACTIVE_OUTCOME,
        lane_of,
    )

    for k in (KIND_CONTACT_OBSERVED, KIND_PROACTIVE_OUTCOME, KIND_IN_FLIGHT):
        assert k in CONTROL_KINDS
        assert lane_of(k) == "control"


def test_contact_is_sensor_lane() -> None:
    from lifemodel.core.taxonomy import lane_of

    assert lane_of(KIND_CONTACT) == "sensor"


def test_unknown_kind_defaults_to_sensor_never_control() -> None:
    from lifemodel.core.taxonomy import lane_of

    assert lane_of("something-new") == "sensor"  # unknown floods can't claim lossless control


def test_proactive_outcome_signal_carries_correlation_id() -> None:
    sig = proactive_outcome_signal(
        origin_id="v9", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id="proactive-X"
    )
    assert read_proactive_outcome(sig) is ProactiveOutcome.SENT
    from lifemodel.core.taxonomy import read_proactive_outcome_correlation

    assert read_proactive_outcome_correlation(sig) == "proactive-X"


def test_verdict_correlation_defaults_empty() -> None:
    sig = proactive_outcome_signal(origin_id="v10", outcome=ProactiveOutcome.SILENT, timestamp=None)
    from lifemodel.core.taxonomy import read_proactive_outcome_correlation

    assert read_proactive_outcome_correlation(sig) == ""


# --- lm-27n.9: the thought_contact_proposal transient signal -----------------


def test_thought_contact_proposal_round_trips() -> None:
    from lifemodel.core.taxonomy import (
        KIND_THOUGHT_CONTACT_PROPOSAL,
        read_thought_contact_proposal,
        thought_contact_proposal_signal,
    )

    sig = thought_contact_proposal_signal(
        origin_id="thought-crystallization",
        thought_id="t-serve",
        score=0.72,
        reason="other-serving",
        other_regarding=0.6,
        actionability=0.3,
        salience=0.8,
        timestamp=None,
    )
    assert sig.kind == KIND_THOUGHT_CONTACT_PROPOSAL
    proposal = read_thought_contact_proposal([sig])
    assert proposal is not None
    assert proposal.thought_id == "t-serve"
    assert proposal.score == 0.72
    assert proposal.reason == "other-serving"
    assert proposal.other_regarding == 0.6


def test_read_proposal_none_when_absent_or_malformed() -> None:
    from lifemodel.core.taxonomy import KIND_THOUGHT_CONTACT_PROPOSAL, read_thought_contact_proposal
    from lifemodel.domain.signal import Signal

    assert read_thought_contact_proposal([]) is None
    # a malformed payload (missing thought_id / bad score) degrades to None, not a crash.
    bad = Signal(origin_id="x", kind=KIND_THOUGHT_CONTACT_PROPOSAL, payload={"score": "high"})
    assert read_thought_contact_proposal([bad]) is None


def test_read_proposal_returns_the_freshest() -> None:
    from lifemodel.core.taxonomy import (
        read_thought_contact_proposal,
        thought_contact_proposal_signal,
    )

    def _p(tid: str):
        return thought_contact_proposal_signal(
            origin_id="c",
            thought_id=tid,
            score=0.7,
            reason="r",
            other_regarding=0.5,
            actionability=0.5,
            salience=0.7,
            timestamp=None,
        )

    latest = read_thought_contact_proposal([_p("t-a"), _p("t-b")])
    assert latest is not None and latest.thought_id == "t-b"
