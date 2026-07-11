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


def filter_external_events(
    ring: Mapping[str, str],
    signals: Sequence[Signal],
    now: datetime,
    *,
    ttl: timedelta = DEFAULT_RING_TTL,
) -> tuple[tuple[Signal, ...], tuple[str, ...]]:
    """Drop duplicate external events; report the fresh ids to RECORD later.

    A PURE filter — it never mutates the ring. Returns ``(signals to process,
    fresh external ids)``: a ``contact_observed`` whose id is already live in the
    ring (within-TTL), or already seen earlier in THIS batch, is dropped; every
    other signal passes through in original frame order. The fresh ids (first-seen
    order, de-duplicated within the batch) are the ones the caller records via
    :func:`record_external_events` — but only AFTER the load-bearing contact
    consumer has actually processed them (spec §8), so a consumer fault leaves the
    ring untouched and the retry re-fires instead of being deduped into oblivion.
    """
    live = {eid for eid, at in ring.items() if _is_live(at, now, ttl)}

    kept: list[Signal] = []
    fresh_ids: list[str] = []
    seen_fresh: set[str] = set()  # fresh ids accepted THIS batch (batch-internal dedup)
    for signal in signals:
        if signal.kind != KIND_CONTACT_OBSERVED:
            kept.append(signal)  # not an external event — never deduped or recorded
            continue
        eid = signal.origin_id
        if eid in live or eid in seen_fresh:
            continue  # duplicate external event — drop it (no second satiation)
        seen_fresh.add(eid)
        fresh_ids.append(eid)
        kept.append(signal)

    return tuple(kept), tuple(fresh_ids)


def record_external_events(
    ring: Mapping[str, str],
    fresh_ids: Sequence[str],
    now: datetime,
    *,
    cap: int = DEFAULT_RING_CAP,
    ttl: timedelta = DEFAULT_RING_TTL,
) -> dict[str, str]:
    """Return a fresh ring with *fresh_ids* recorded, TTL-swept, and bounded.

    Applied AFTER the frame's components ran, and ONLY when the load-bearing contact
    consumer did not fault this frame — so a fault leaves the ring byte-identical and
    the retry re-fires. The returned ring is a fresh dict (never the input, a state
    snapshot): TTL-expired entries are swept (even when *fresh_ids* is empty, so the
    ring stays bounded on quiet frames), each fresh id is recorded at *now*, and the
    oldest entries are evicted once ``cap`` is exceeded. A pure-duplicate frame (no
    fresh ids, nothing swept) returns a ring equal to the input, so the coreloop
    writes no churn.
    """
    # Start from the live (within-TTL) subset, preserving oldest-first order.
    live: dict[str, str] = {eid: at for eid, at in ring.items() if _is_live(at, now, ttl)}
    for eid in fresh_ids:
        live[eid] = now.isoformat()  # remember each fresh id at the moment processed

    # Bound the ring: evict oldest (front of insertion order) beyond the cap.
    while len(live) > cap:
        oldest = next(iter(live))
        del live[oldest]

    return live


def dedupe_external_events(
    ring: Mapping[str, str],
    signals: Sequence[Signal],
    now: datetime,
    *,
    cap: int = DEFAULT_RING_CAP,
    ttl: timedelta = DEFAULT_RING_TTL,
) -> tuple[tuple[Signal, ...], dict[str, str]]:
    """Filter duplicates AND record fresh ids in one call (filter → record).

    The combined form: it records unconditionally, so it is only sound when the
    caller has no consumer whose failure could strand a recorded id. The coreloop
    instead splits the two (:func:`filter_external_events` before the component loop,
    :func:`record_external_events` after it, gated on the contact consumer's success)
    so a ContactSensor fault cannot durably lose an inbound. Behaviour is identical
    to filter-then-record-all.
    """
    kept, fresh_ids = filter_external_events(ring, signals, now, ttl=ttl)
    new_ring = record_external_events(ring, fresh_ids, now, cap=cap, ttl=ttl)
    return kept, new_ring
