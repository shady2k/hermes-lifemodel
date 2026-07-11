"""External-event idempotency ring — the body's "I already processed this" (spec §8).

**Idempotency ≠ bus-dedup.** The old durable signal bus (with its cursor/dedup) is
gone; the nervous flow is ephemeral (spec §2/§3). But **external** events are not
ours to make exactly-once — Telegram/Hermes retry the SAME inbound. Without a
guard, a retry would satiate ``u`` a second time, re-stamp ``last_exchange_at``,
and re-resolve the desire. So :class:`~lifemodel.state.model.State` keeps a bounded,
TTL'd ring of external event ids it has already processed
(``processed_external_event_ids``), and the intake path (``core/coreloop.py``)
runs each frame's seed signals through :func:`dedupe_external_events` before
seeding the :class:`~lifemodel.core.frame.SignalFrame`.

The rule (spec §8 / §12 scenario 6):

* An **external event** is a ``contact_observed`` seed signal (the inbound hook's
  transduction of a real Hermes event, keyed by its ``origin_id`` — the Hermes
  event id). Nothing else in the pipeline carries one.
* If its id is already in the ring **and not TTL-expired** → it is a DUPLICATE:
  the signal is dropped (it never reaches the sensor/drive/aggregation), so ``u``
  is not satiated a second time.
* If its id is new (or the prior record TTL-expired) → it is processed normally
  and recorded with a fresh ``now`` stamp.

The ring stays bounded: TTL-expired entries are swept every call, and the oldest
entries are evicted once the cap is exceeded — so an id that aged out can fire
again. Pure and stdlib-only: no I/O, no state mutation. The coreloop persists the
returned ring through the state-actor's atomic end-of-frame commit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from ..domain.signal import Signal
from .taxonomy import KIND_CONTACT_OBSERVED

#: The ring's cap: at most this many recently-processed external ids are kept.
#: The spec names no specific number ("with one sensor, MAX_INTAKE is almost
#: unnecessary", §7); this is a generous bound so a burst of distinct inbounds
#: cannot grow the durable ring without limit while still comfortably covering any
#: realistic retry window.
DEFAULT_RING_CAP: int = 256

#: How long a processed id is remembered. Telegram/Hermes retries land within
#: seconds-to-minutes, so a day is far longer than any real retry window while
#: keeping the ring from pinning ids forever — an id older than this may recur and
#: is correctly treated as a fresh event.
DEFAULT_RING_TTL: timedelta = timedelta(hours=24)


def _is_live(recorded_at: str, now: datetime, ttl: timedelta) -> bool:
    """True if *recorded_at* is a still-remembered (within-TTL) stamp.

    Defensive: an unparseable or timezone-naive stamp (only possible from a
    corrupt ring, since we only ever write ``now.isoformat()``) is treated as
    expired so it is swept out rather than pinned forever or crashing the frame.
    """
    try:
        ts = datetime.fromisoformat(recorded_at)
    except ValueError:
        return False
    if ts.tzinfo is None or ts.utcoffset() is None:
        return False
    return (now - ts) <= ttl


def dedupe_external_events(
    ring: Mapping[str, str],
    signals: Sequence[Signal],
    now: datetime,
    *,
    cap: int = DEFAULT_RING_CAP,
    ttl: timedelta = DEFAULT_RING_TTL,
) -> tuple[tuple[Signal, ...], dict[str, str]]:
    """Drop duplicate external events and return (signals to process, updated ring).

    See the module docstring for the rule. The returned ring is a fresh dict
    (never the input, which is a state snapshot): TTL-expired entries are swept,
    each fresh external id is recorded at *now*, and the oldest entries are evicted
    once ``cap`` is exceeded. Non-external signals pass through untouched and are
    never recorded. Original frame order is preserved among the kept signals.
    """
    # Start from the live (within-TTL) subset, preserving oldest-first order.
    live: dict[str, str] = {eid: at for eid, at in ring.items() if _is_live(at, now, ttl)}

    kept: list[Signal] = []
    for signal in signals:
        if signal.kind != KIND_CONTACT_OBSERVED:
            kept.append(signal)  # not an external event — never deduped or recorded
            continue
        eid = signal.origin_id
        if eid in live:
            continue  # duplicate external event — drop it (no second satiation)
        live[eid] = now.isoformat()  # fresh id: remember it, process it
        kept.append(signal)

    # Bound the ring: evict oldest (front of insertion order) beyond the cap.
    while len(live) > cap:
        oldest = next(iter(live))
        del live[oldest]

    return tuple(kept), live
