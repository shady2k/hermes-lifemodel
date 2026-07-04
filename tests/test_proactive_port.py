from __future__ import annotations

from lifemodel.domain.egress import ReachOutcome
from lifemodel.ports import ProactiveEgressPort
from lifemodel.ports.proactive import ProactiveEgressPort as DirectPort


def test_reach_outcome_ok_only_for_delivered() -> None:
    assert ReachOutcome.DELIVERED.ok is True
    assert ReachOutcome.SKIPPED_BUSY.ok is False
    assert ReachOutcome.UNAVAILABLE.ok is False
    assert ReachOutcome.FAILED.ok is False


def test_port_is_runtime_checkable_and_reexported() -> None:
    assert ProactiveEgressPort is DirectPort

    class Impl:
        def reach_out(self, target: dict[str, str | None], impulse: str) -> ReachOutcome:
            return ReachOutcome.DELIVERED

    assert isinstance(Impl(), ProactiveEgressPort)

    class NotImpl:
        pass

    assert not isinstance(NotImpl(), ProactiveEgressPort)
