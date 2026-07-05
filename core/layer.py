"""``Layer`` / ``ProcessingLayer`` — a brain-stage extension point (HLA §1/§13).

A processing layer is one stage of the layered brain (autonomic → aggregation →
cognition, HLA §1). It runs on some context and returns a
:class:`~lifemodel.domain.layer.LayerResult` carrying a **confidence**. Each
layer owns a ``confidence_threshold``: a result at or above it is trusted; below
it is a candidate for escalation to a smarter layer (fast → smart, HLA §1). The
escalation *policy* is Phase 2 (2.2); this contract just names the threshold and
the comparison so later layers implement, not redesign, it.

``ProcessingLayer`` is an alias for ``Layer`` — HLA §13 uses both names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..domain.layer import LayerResult


class Layer(ABC):
    """One stage of the brain; ``process`` reports a confidence-scored result.

    ``ctx`` is intentionally typed ``Any``: each concrete layer takes its own
    context shape (raw signals, an aggregation summary, a wake-packet …), and a
    permissive base keeps subclasses from fighting the signature (HLA §13,
    pragmatism over ceremony).
    """

    #: Confidence at or above which this layer's result is trusted without a
    #: smarter retry. Subclasses override; ``0.0`` means "always trusted".
    confidence_threshold: float = 0.0

    @abstractmethod
    def process(self, ctx: Any) -> LayerResult:
        """Run this stage over *ctx* and return its confidence-scored result."""
        raise NotImplementedError

    def meets_confidence(self, result: LayerResult) -> bool:
        """Whether *result* clears this layer's threshold (no escalation needed).

        An explicit ``result.escalate`` request always forces escalation,
        regardless of the confidence score.
        """
        return not result.escalate and result.confidence >= self.confidence_threshold


#: HLA §13 names this extension point both ``Layer`` and ``ProcessingLayer``.
ProcessingLayer = Layer
