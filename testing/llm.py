"""``FakeLlmPort`` — the internal-cognition seam's test double (lm-705.6).

Scripted: construct with either an :class:`~lifemodel.core.llm_port.InternalCognitionResult`
(returned on every call) or an ``Exception`` (raised on every call), mirroring
:class:`~lifemodel.testing.fakes.FakeClock`'s "construct with the answer" shape.
Records every request it was called with, so a test can assert on what the
runner / completion path actually asked for.
"""

from __future__ import annotations

from ..core.llm_port import InternalCognitionRequest, InternalCognitionResult


class FakeLlmPort:
    """A scripted :class:`~lifemodel.core.llm_port.LlmPort` — returns or raises."""

    def __init__(self, result: InternalCognitionResult | Exception) -> None:
        self._result = result
        #: Every request handed to :meth:`complete_structured`, in call order.
        self.requests: list[InternalCognitionRequest] = []

    async def complete_structured(self, req: InternalCognitionRequest) -> InternalCognitionResult:
        self.requests.append(req)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result
