"""``BrainHealth`` — the process-local single source of truth for brain liveness.

The backbone of the fail-loud invariant (spec §4.2): one small, thread-safe,
per-``base_dir`` object whose :attr:`~BrainHealth.state` answers "did the being's
brain boot, connect, keep ticking, or die?". It is WRITTEN by the wiring path
(:func:`lifemodel.register`, :meth:`BeingAdapter.connect`, ``_on_loop_death``, the
observer hooks) and READ by the platform ``check_fn`` and (Slice 3)
``/lifemodel status``. Because the same gateway process hosts BOTH the being
adapter AND the ``/lifemodel`` command handler, a per-base_dir singleton
(:func:`get_brain_health`, mirroring
:func:`~lifemodel.core.metrics.get_metric_registry` /
:func:`~lifemodel.state.trace_store.acquire_trace_writer`) is shared between them.

**Durability (spec §4.2/§4.3).** A ``boot_failed`` state persists a SMALL durable
record (``brain_boot.json`` under *base_dir*) so that after the gateway is revived
into a *fresh process*, ``/lifemodel status`` can still report *why* the boot
failed — the in-memory singleton is gone, but the file is not. Live liveness
(``last_tick_at`` / ``ticks_total``) is NOT duplicated here: it reuses
:class:`~lifemodel.state.model.State`'s already-durable ``last_tick_at`` /
``tick_count`` (advanced every tick by the CoreLoop), which callers pass into
:meth:`~BrainHealth.check`. This object never invents a parallel tick counter.

All stdlib (``json`` / ``threading`` / ``datetime``) — the plugin runs inside
Hermes' own interpreter, no third-party deps.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

#: The closed set of brain states (spec §4.2). ``never_started`` and ``connecting``
#: are the pre-connect *transients*; the rest are terminal-ish readouts.
BrainState = Literal["never_started", "connecting", "connected", "loop_dead", "boot_failed"]

#: The brain's nominal tick cadence (mirrors :class:`BeingAdapter`'s loop interval),
#: the baseline the staleness threshold derives from. Kept HERE — the leaf health
#: module, Hermes-free and with no heavy deps — so BOTH the adapter's ``check_fn``
#: (:func:`~lifemodel.adapters.being_platform.make_check_fn`) AND ``/lifemodel status``
#: read ONE threshold and can never drift.
DEFAULT_TICK_INTERVAL_SECONDS = 60.0
#: How stale the durable ``last_tick_at`` may get before the brain reads unhealthy
#: (spec §4.2, "a few intervals"). Generous so a slow tick — or a just-connected loop
#: still inside its grace — is never a false alarm.
STALE_AFTER_SECONDS = DEFAULT_TICK_INTERVAL_SECONDS * 5

#: The durable boot-health record filename, a sibling of ``lifemodel.sqlite`` /
#: ``observability.sqlite`` under *base_dir*.
_BOOT_RECORD_FILENAME = "brain_boot.json"

_module_logger = logging.getLogger("lifemodel.brain_health")


def brain_boot_path(base_dir: Path) -> Path:
    """Return the durable boot-health record path under *base_dir*."""
    return Path(base_dir) / _BOOT_RECORD_FILENAME


def read_boot_record(base_dir: Path) -> dict[str, str] | None:
    """Return the persisted boot-health record for *base_dir*, or ``None``.

    Defensive by construction (this is a *disposable* diagnostic file, spec
    §4.2): an absent file, an unreadable one, or malformed JSON all degrade to
    ``None`` rather than raising — a corrupt diagnostic must never crash the
    status command that reads it.
    """
    path = brain_boot_path(base_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    # Coerce to a flat str→str view; a hand-mangled file with odd value types is
    # simply stringified rather than rejected (still readable by the owner).
    return {str(k): str(v) for k, v in data.items()}


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 stamp defensively — ``None`` on absent/unparseable."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _latest(*stamps: str | None) -> datetime | None:
    """Return the most recent parseable instant among *stamps* (``None`` if none)."""
    parsed = [dt for dt in (_parse_iso(s) for s in stamps) if dt is not None]
    return max(parsed) if parsed else None


def tick_staleness(
    connected_at: str | None,
    last_tick_at: str | None,
    *,
    now: datetime,
    stale_after_seconds: float,
) -> tuple[float | None, bool]:
    """Return ``(age_seconds, is_stale)`` for the freshest liveness anchor.

    The single source of the staleness rule, shared by :meth:`BrainHealth.check`
    (the enablement gate) and ``/lifemodel status`` (the display) so the two can
    never disagree. The anchor is the most recent of *connected_at* and
    *last_tick_at*: ``connected_at`` anchors the grace so a just-connected loop that
    has not ticked yet is not flagged false-stale in its first interval, while a live
    ``last_tick_at`` takes over once ticks flow. ``age`` is ``None`` when NO anchor is
    parseable — an unknown age is treated as not-stale (the loud channels carry a real
    outage), never as a false alarm.
    """
    anchor = _latest(connected_at, last_tick_at)
    if anchor is None:
        return None, False
    age = (now - anchor).total_seconds()
    return age, age > stale_after_seconds


@dataclass(frozen=True)
class BrainHealthSnapshot:
    """An immutable read of a :class:`BrainHealth`'s display fields (Slice 3).

    Taken under the record's lock (:meth:`BrainHealth.snapshot`) so a status render
    sees one consistent view even while the tick thread / reconnect watcher writes
    concurrently. The ``last_observer_error`` map is copied, not aliased.
    """

    state: BrainState
    boot_error: str | None
    last_loop_death: str | None
    death_count: int
    last_observer_error: dict[str, str]
    connected_at: str | None


class BrainHealth:
    """The mutable, thread-safe liveness record for one being (spec §4.2).

    Prefer :func:`get_brain_health` over constructing directly — it enforces the
    singleton-per-base_dir lifecycle the register/connect/command sites rely on to
    share one record. Every mutator holds a lock so the gateway's tick thread, the
    reconnect watcher, and a ``/lifemodel`` command handler can write concurrently.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.state: BrainState = "never_started"
        self.boot_error: str | None = None
        self.last_loop_death: str | None = None
        self.death_count: int = 0
        self.last_observer_error: dict[str, str] = {}
        #: ISO-8601 stamp of the last :meth:`mark_connected`, the grace anchor the
        #: staleness check uses so a just-connected loop that hasn't ticked yet is
        #: not flagged false-stale in its first interval. ``None`` until connected.
        self.connected_at: str | None = None
        self._lock = threading.Lock()

    # ---- state transitions (written by connect/register/tick/observers) ---- #

    def mark_connecting(self) -> None:
        """Entering :meth:`BeingAdapter.connect` — the loop is coming up (spec §4.3)."""
        with self._lock:
            self.state = "connecting"

    def mark_connected(self, *, at: str | None = None) -> None:
        """The brain loop is up. Clears a prior ``loop_dead`` / ``boot_failed`` and
        the durable boot record — a clean (re)connect means we are healthy now
        (spec §4.3). *at* anchors the staleness grace; defaults to real now."""
        stamp = at if at is not None else datetime.now().astimezone().isoformat()
        with self._lock:
            self.state = "connected"
            self.connected_at = stamp
            self.boot_error = None
        # File I/O outside the lock (never hold a lock across a syscall).
        self._clear_boot_record()

    def mark_boot_failed(self, error: str) -> None:
        """A REQUIRED wiring step failed (spec §4.3). Records the reason in memory
        AND persists the durable record so a revived process can still explain it.
        The caller (``wire``) re-raises afterwards — this only records."""
        with self._lock:
            self.state = "boot_failed"
            self.boot_error = error
        self._write_boot_record(error)

    def mark_boot_ok(self) -> None:
        """All REQUIRED wiring for this process succeeded — wipe any stale durable
        failure record (a previously-broken deploy that is now fixed). Does not
        change :attr:`state` (the loop still has to ``connect()``)."""
        with self._lock:
            self.boot_error = None
        self._clear_boot_record()

    def record_loop_death(self, message: str, traceback_text: str | None = None) -> None:
        """The supervised brain loop died (spec §4.3). Sets ``loop_dead``, stores the
        message (+ short traceback), and bumps the cumulative :attr:`death_count`."""
        detail = message if traceback_text is None else f"{message}\n{traceback_text}"
        with self._lock:
            self.state = "loop_dead"
            self.last_loop_death = detail
            self.death_count += 1

    def record_observer_error(self, observer: str, error: str) -> None:
        """An afferent observer body raised (spec §4.3/MAJOR-4) — record the LAST
        error per observer name so ``/lifemodel status`` can show it."""
        with self._lock:
            self.last_observer_error[observer] = error

    # ---- read: a consistent snapshot of the display fields (feeds /status) -- #

    def snapshot(self) -> BrainHealthSnapshot:
        """Return an immutable, lock-consistent copy of the display fields (Slice 3).

        ``/lifemodel status`` reads this instead of the live attributes so a render
        sees ONE coherent view — never a torn read while the tick thread / reconnect
        watcher mutates. The observer map is copied, not aliased.
        """
        with self._lock:
            return BrainHealthSnapshot(
                state=self.state,
                boot_error=self.boot_error,
                last_loop_death=self.last_loop_death,
                death_count=self.death_count,
                last_observer_error=dict(self.last_observer_error),
                connected_at=self.connected_at,
            )

    # ---- read: the rich liveness verdict (feeds /status + logs, NOT the gate) - #

    def check(
        self, *, last_tick_at: str | None, now: datetime, stale_after_seconds: float
    ) -> tuple[bool, str]:
        """Return ``(healthy, reason)`` — the liveness verdict for the DISPLAY (spec §5).

        This is NOT the Hermes ``check_fn`` (codex MAJOR): ``check_fn`` is an
        *enablement* gate — a False there would brick the being at boot (the state is
        ``never_started`` at the registry pass) and block the gateway's reconnect after
        a loop death — so it is permissive/always-True (see
        :func:`~lifemodel.adapters.being_platform.make_check_fn`). This verdict instead
        drives where a False cannot brick anything: ``/lifemodel status`` and the
        poll-cadence DEBUG log. So it reports the TRUTH — genuine post-start unhealth
        (``boot_failed`` / ``loop_dead`` / a ``connected`` brain whose ticks went stale)
        returns ``(False, reason)`` — while the *pre-connect transients*
        (``never_started`` / ``connecting``) are "not a failure, just not up yet" and
        return ``(True, reason)`` with the reason still naming the state.
        """
        with self._lock:
            state = self.state
            boot_error = self.boot_error
            last_loop_death = self.last_loop_death
            connected_at = self.connected_at

        if state == "boot_failed":
            return False, f"boot_failed: {boot_error or 'unknown'}"
        if state == "loop_dead":
            first_line = (last_loop_death or "unknown").splitlines()[0]
            return False, f"loop_dead: {first_line}"
        if state == "never_started":
            return True, "never_started: awaiting first connect (enablement-safe)"
        if state == "connecting":
            return True, "connecting: loop coming up (enablement-safe)"

        # state == "connected": stale only if the freshest of (connected_at,
        # last_tick_at) is older than the window (shared :func:`tick_staleness`, so the
        # enablement gate and ``/lifemodel status`` never disagree). connected_at
        # anchors the grace so a just-connected loop that hasn't ticked yet is not
        # flagged false-stale.
        age, stale = tick_staleness(
            connected_at, last_tick_at, now=now, stale_after_seconds=stale_after_seconds
        )
        if stale and age is not None:
            return False, f"stale: no tick for {age:.0f}s (> {stale_after_seconds:.0f}s)"
        return True, "connected"

    # ---- durable record I/O (best-effort; never raises to the caller) ------ #

    def _write_boot_record(self, error: str) -> None:
        record = {
            "state": "boot_failed",
            "boot_error": error,
            "written_at": datetime.now().astimezone().isoformat(),
        }
        path = brain_boot_path(self.base_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        except OSError:
            # Losing the durable diagnostic is itself a (lesser) observability
            # failure — log it loudly, but never mask the boot failure we are in
            # the middle of reporting (the caller re-raises regardless).
            _module_logger.warning("brain_boot_record_write_failed path=%s", path, exc_info=True)

    def _clear_boot_record(self) -> None:
        try:
            brain_boot_path(self.base_dir).unlink(missing_ok=True)
        except OSError:
            _module_logger.warning(
                "brain_boot_record_clear_failed path=%s",
                brain_boot_path(self.base_dir),
                exc_info=True,
            )


# --------------------------------------------------------------------------- #
# Singleton per base_dir (mirrors get_metric_registry — idempotent, no teardown)
# --------------------------------------------------------------------------- #

_registry_lock = threading.Lock()
_instances: dict[str, BrainHealth] = {}


def _registry_key(base_dir: Path) -> str:
    return str(Path(base_dir).resolve())


def get_brain_health(base_dir: Path) -> BrainHealth:
    """Return the process-local :class:`BrainHealth` for *base_dir* (spec §4.2).

    Singleton per resolved *base_dir* and idempotent: the FIRST call constructs it,
    later calls return the SAME instance — so the being adapter, ``register()``, and
    the ``/lifemodel`` command handler (all in the one gateway process) share one
    record. Like :func:`~lifemodel.core.metrics.get_metric_registry` there is no
    refcount / teardown (a plugin ``register()`` has none).
    """
    key = _registry_key(base_dir)
    with _registry_lock:
        instance = _instances.get(key)
        if instance is None:
            instance = BrainHealth(Path(base_dir))
            _instances[key] = instance
        return instance
