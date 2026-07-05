from __future__ import annotations

import pytest

from lifemodel.core.taxonomy import (
    KIND_CONTACT,
    KIND_EXCHANGE,
    contact_signal,
    exchange_signal,
    is_kind,
    read_exchange,
)


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
