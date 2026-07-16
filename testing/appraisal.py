"""Test double for the appraisal seam (slice 1, lm-705.1)."""

from __future__ import annotations

from ..core.appraisal import ThoughtSeed


class FakeAppraiser:
    """Returns a fixed *seed* (or ``None`` to decline) regardless of input, and
    records the last call so a seam test can assert it was invoked."""

    def __init__(self, seed: ThoughtSeed | None) -> None:
        self._seed = seed
        self.calls: list[tuple[str, str]] = []

    def appraise(self, *, user_message: str, assistant_response: str) -> ThoughtSeed | None:
        self.calls.append((user_message, assistant_response))
        return self._seed
