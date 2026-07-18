"""The post_llm observer after the Appraiser seam retirement (lm-705.11 Task 5)."""

from __future__ import annotations

import inspect

from lifemodel.hooks import make_post_llm_observer


def test_post_llm_observer_has_no_appraiser_param() -> None:
    assert "appraiser" not in inspect.signature(make_post_llm_observer).parameters
