"""The certified, Hermes-free math the live components import (spec §6).

This package is the certified core of bead ``lm-x43`` — the drive-component
(:mod:`lifemodel.sim.drive`), the hard wake-decision gates
(:mod:`lifemodel.sim.wake`), and the exchange-quality classifier
(:mod:`lifemodel.sim.quality`). The LIVE components import these directly
(``SolitudeDrive`` uses ``Drive``; ``ContactAggregation`` uses ``evaluate_wake``;
``ContactSensor`` uses ``quality_of``), so the math the being runs IS the
certified math. Nothing here imports Hermes; it runs and is tested in plain Python.

T8 (spec §6) removed the OLD parallel tick model that lived here
(``sim/harness.py`` + ``sim/aggregation.py``) — the simulation now drives the REAL
``CoreLoop`` + real components through fake ports via the integration harness in
:mod:`lifemodel.testing.harness`. The unit regressions for the certified math
(``tests/sim/test_sim_drive.py``, ``test_sim_wake.py``, ``test_sim_quality.py``)
stay.
"""

from __future__ import annotations
