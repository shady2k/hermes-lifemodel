"""``LayerResult`` — what a processing layer returns (HLA §1/§13).

A layer (``ProcessingLayer``) runs a stage of the brain and reports back with a
**confidence**. Confidence is the currency of the fast→smart escalation (HLA §1,
System 1 → System 2): a result below the layer's confidence threshold is a
candidate for retry on a smarter model. The escalation *policy* lands in Phase 2
(2.2); this is only the value it reasons over. Imports nothing from Hermes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerResult:
    """The outcome of one :class:`~lifemodel.core.layer.Layer` pass.

    * ``confidence`` — how sure the layer is, in ``[0.0, 1.0]``. Compared against
      the layer's threshold to decide whether to escalate (HLA §1).
    * ``output`` — the layer-specific product (text, an intent, a signal set …);
      typed ``Any`` because each layer produces its own shape.
    * ``escalate`` — an explicit request to hand off to a smarter layer, set when
      confidence alone does not capture the need to retry.
    """

    confidence: float
    output: Any = None
    escalate: bool = False
