"""Pure, Hermes-free simulation of the proactive-contact desire model.

This package is the certifiable core of bead ``lm-x43``: the drive-component,
the hard policy gates, and the wake-decision — plus the simulation harness that
proves them against numeric invariants before any plugin code consumes them.
Nothing here imports Hermes; it runs and is tested in plain Python.

Spec: ``docs/superpowers/specs/2026-07-04-proactive-contact-desire-model-design.md``
"""

from __future__ import annotations
