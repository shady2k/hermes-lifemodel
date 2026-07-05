from __future__ import annotations

import pytest

from lifemodel.core.taxonomy import (
    KIND_CONTACT,
    KIND_EXCHANGE,
    KIND_IN_FLIGHT,
    KIND_VERDICT,
    contact_signal,
    contact_value,
    exchange_signal,
    in_flight_signal,
    is_in_flight,
    is_kind,
    read_exchange,
    read_verdict,
    verdict_signal,
)
from lifemodel.core.taxonomy import (
    contact_signal as _contact_signal,
)
from lifemodel.sim.aggregation import Verdict


def test_contact_signal_carries_value_and_delta() -> None:
    sig = contact_signal(
        origin_id="c-1", value=1.25, delta=0.02, timestamp="2026-07-06T00:00:00+00:00"
    )
    assert sig.kind == KIND_CONTACT
    assert sig.origin_id == "c-1"
    assert sig.payload["value"] == 1.25
    assert sig.payload["delta"] == 0.02
    assert is_kind(sig, KIND_CONTACT)
    assert not is_kind(sig, KIND_EXCHANGE)


def test_exchange_signal_roundtrips_actor_label() -> None:
    sig = exchange_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None)
    assert sig.kind == KIND_EXCHANGE
    assert read_exchange(sig) == ("user", "two_way")


def test_read_exchange_rejects_wrong_kind() -> None:
    sig = contact_signal(origin_id="c-2", value=0.0, delta=0.0, timestamp=None)
    with pytest.raises(ValueError):
        read_exchange(sig)


def test_read_exchange_rejects_bad_payload() -> None:
    from lifemodel.domain.signal import Signal

    sig = Signal(origin_id="e-2", kind=KIND_EXCHANGE, payload={"actor": "user"})  # missing label
    with pytest.raises(ValueError):
        read_exchange(sig)


def test_verdict_signal_roundtrips() -> None:
    sig = verdict_signal(origin_id="v-1", verdict=Verdict.FULFILL, timestamp=None)
    assert sig.kind == KIND_VERDICT
    assert read_verdict(sig) is Verdict.FULFILL


def test_read_verdict_rejects_bad_value() -> None:
    from lifemodel.domain.signal import Signal

    with pytest.raises(ValueError):
        read_verdict(Signal(origin_id="v-2", kind=KIND_VERDICT, payload={"verdict": "nope"}))


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
