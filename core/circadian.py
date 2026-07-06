"""Circadian rhythm C(t) — the 24-hour alertness wave (spec §8).

A pure function of absolute wall-clock time (not ticks): ``0.5 + 0.5·cos(2π(h −
peak)/24)`` over the UTC hour-of-day ``h``. Peak alertness (C=1) at
``peak_hour_utc``; trough (C=0) twelve hours later. Part of the two-process sleep
model (Borbély): C is the circadian process; S (fatigue) is the homeostatic one.
"""

from __future__ import annotations

import math
from datetime import datetime


def circadian(now: datetime, *, peak_hour_utc: float) -> float:
    h = now.hour + now.minute / 60.0 + now.second / 3600.0
    return 0.5 + 0.5 * math.cos(2.0 * math.pi * (h - peak_hour_utc) / 24.0)
