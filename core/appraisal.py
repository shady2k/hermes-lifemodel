"""The appraisal seam — the being's judgment of whether a completed exchange left
something worth returning to.

The post-hoc ``Appraiser`` protocol was retired in lm-705.11 (the tool is now the
sole producer). ``ThoughtSeed`` remains as the payload shape the tool→signal path
carries (content + salience + producer).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThoughtSeed:
    """An appraisal result: the content + salience of a thought worth capturing."""

    content: str
    salience: float
    actionability: float = 0.0
    other_regarding_value: float = 0.0
