"""Integration harness — drive the REAL CoreLoop + real components through fakes (spec §6).

This is the heart of the rebuild: the simulation no longer has a third tick model.
The harness runs the actual spine — ``ContactSensor → SolitudeDrive →
ContactAggregation → CognitionLauncher`` (the same code the live being runs) — over
the real SQLite store, through fake ports, so a green scenario HONESTLY predicts
live behaviour. The one thing that is not real here is the async act-gate (the
being's Hermes turn): the harness scripts its outcome (``sent`` on a message,
``silent`` on ``[SILENT]``) by seeding a ``proactive_outcome`` signal into the SAME
read-back path the ``post_llm`` hook uses live.

A scenario is a list of :class:`Step` (advance the fake clock to accumulate
silence, optionally seed a ``contact_observed`` reading, optionally script the
act-gate outcome). Each step runs ONE ExecutionFrame via ``proactive_tick`` (real
pipeline → real backstop → recording egress) and records what happened: the live
desire's state, whether a launch reached the egress (and the impulse text), the
delivery outcome, and the suppression-span reasons emitted that frame — the span
tree that makes a quiet frame as debuggable as a loud one (spec §5).
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..composition import LifeModel, build_lifemodel
from ..core.component import ComponentLayer
from ..core.desire_view import read_live_contact_desire
from ..core.frame import FrameTrigger
from ..core.noticing_buffer import NoticingBuffer
from ..core.proactive import proactive_tick
from ..core.quality import Actor, Label
from ..core.registry import ComponentManifest, UnknownComponent
from ..core.taxonomy import contact_observed_signal, proactive_outcome_signal
from ..core.thought_capture import THOUGHT_CAPTURE_ID, ThoughtCapture
from ..core.thought_processing import THOUGHT_PROCESSING_SELECTOR_ID, ThoughtProcessingSelector
from ..core.timeutil import to_iso
from ..domain.egress import ProactiveOutcome, ReachOutcome
from ..domain.memory import MemoryDraft, MemoryRecord
from ..domain.signal import Signal
from ..events import EventRing
from ..ports.clock import ClockPort
from ..ports.memory import MemoryPort
from ..state.model import State
from .fakes import FakeClock

#: The default harness being was born long ago — see :meth:`IntegrationHarness.__post_init__`
#: for why a drive scenario must not open on an unborn being.
BORN_AT = "2025-12-01T10:00:00+00:00"


def draft_to_record(draft: MemoryDraft, *, now: datetime) -> MemoryRecord:
    """Encode a :class:`~lifemodel.domain.memory.MemoryDraft` into the
    :class:`~lifemodel.domain.memory.MemoryRecord` a store's first ``put`` would
    hand back — the start-of-tick snapshot a component test's ``TickContext.objects``
    carries. Stamps ``created_at``/``updated_at`` from *now* and ``revision=0`` (what
    a fresh row gets on insert); every other field is copied straight from the draft.
    A test fixture only — a real tick reads this back through the store, never this
    helper."""
    stamped = to_iso(now)
    return MemoryRecord(
        kind=draft.kind,
        id=draft.id,
        state=draft.state,
        payload=draft.payload,
        source=draft.source,
        recipient_id=draft.recipient_id,
        salience=draft.salience,
        confidence=draft.confidence,
        expires_at=draft.expires_at,
        created_at=stamped,
        updated_at=stamped,
        revision=0,
        schema_version=draft.schema_version,
    )


@dataclass(frozen=True)
class Step:
    """One harness frame: advance the clock, seed optional signals, run the frame.

    ``advance`` accumulates silence (the drive rises by ``Δt``). ``exchange`` seeds a
    real inbound ``contact_observed`` reading (``(actor, label)`` — never
    ``proactive_internal``). ``act_gate`` scripts the async Hermes turn's outcome for
    the turn currently in flight (``pending_proactive_id``) — the read-back path. Both
    are seeded into the single ExecutionFrame this step runs (spec §3).
    """

    advance: timedelta = timedelta(0)
    exchange: tuple[Actor, Label] | None = None
    act_gate: ProactiveOutcome | None = None


@dataclass(frozen=True)
class TickRecord:
    """What happened on one harness tick — the outcome AND the span tree."""

    tick: int
    outcome: ReachOutcome | None  # delivery outcome; None = the core stayed quiet
    desire_state: str | None  # live contact-desire state (active/deferred), else None
    launched: bool  # a LaunchProactive reached the egress this tick
    delivered_impulse: str | None  # the impulse text handed to the egress (if launched)
    suppressions: tuple[str, ...]  # reasons of the suppression spans emitted this tick
    u: float  # the drive vital after the tick


class RecordingEgress:
    """A :class:`~lifemodel.ports.proactive.ProactiveEgressPort` that records
    ``reach_out`` calls without sending anything, returning a fixed outcome."""

    def __init__(self, outcome: ReachOutcome = ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple[object, str]] = []

    def reach_out(self, target: object, impulse: str) -> ReachOutcome:
        self.calls.append((target, impulse))
        return self.outcome


@dataclass
class IntegrationHarness:
    """Drive the real being spine through fake ports over a tmp dir.

    Builds the real graph once (real components + the SQLite store) and reuses it
    across ticks — the CoreLoop/StateActor are reusable, state persists in the
    store, so this runs the identical tick code the live being runs. The fake clock
    is advanced per step; the recording egress catches launches without sending; the
    recording logger captures every suppression span (the span tree)."""

    base_dir: Path
    clock: FakeClock = field(default_factory=lambda: FakeClock(datetime(2026, 1, 1, tzinfo=UTC)))
    event_ring: EventRing = field(default_factory=EventRing)
    egress: RecordingEgress = field(default_factory=RecordingEgress)
    target: dict[str, str | None] = field(
        default_factory=lambda: {"platform": "test", "chat_id": "1", "thread_id": None}
    )
    initial_state: State | None = None
    records: list[TickRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lm = build_lifemodel(
            base_dir=self.base_dir,
            clock=self.clock,
            event_ring=self.event_ring,
        )
        # Seed the start state (loaded lazily by the StateActor on the first tick).
        # The default is a rested start with last_tick_at set, so the FIRST clock
        # advance yields a real Δt (else minutes_between(None, now) = 0 and the drive
        # never rises on step 0) and cognition can afford a launch. It is also a being
        # that has ALREADY BEEN BORN (``genesis_completed_at``): the default scenario is
        # the DRIVE's, and an unborn being wakes for a different reason entirely — it
        # wakes to be born, on tick 0, without ``u`` ever crossing ``θ`` (spec §6.2), so
        # every drive scenario would otherwise open with a birth. Pass an unstamped
        # ``initial_state=State(…)`` to drive the genesis path on purpose. A caller may
        # pass ``initial_state`` to land a tick inside a specific gate (a recent exchange,
        # an active decline backoff, a rate-limiting send log) without driving the
        # whole flow there — the components then run for real on that state.
        initial = self.initial_state
        if initial is None:
            initial = State(
                u=0.0,
                energy=1.0,
                fatigue=0.0,
                last_tick_at=self.clock.now().isoformat(),
                genesis_completed_at=BORN_AT,
            )
        self._lm.state.commit(initial)

    def run(self, steps: Sequence[Step]) -> list[TickRecord]:
        """Run each step in order, appending a :class:`TickRecord` per tick."""
        for step in steps:
            self.records.append(self._step(step))
        return self.records

    def _memory(self) -> MemoryPort:
        # The live SQLite store is both StatePort and MemoryPort; narrow for the
        # typed readers (desire/intention views) that take a MemoryPort.
        memory = self._lm.state
        assert isinstance(memory, MemoryPort), "harness store must be a MemoryPort"
        return memory

    def _step(self, step: Step) -> TickRecord:
        self.clock.advance(step.advance)
        now = self.clock.now()
        # Seed this step's readings into the single ExecutionFrame it runs (spec §3):
        # ephemeral signals, not a durable bus.
        signals: list[Signal] = []
        trigger = FrameTrigger.HEARTBEAT
        if step.exchange is not None:
            actor, label = step.exchange
            signals.append(
                contact_observed_signal(
                    origin_id=f"contact-{len(self.records)}",
                    actor=actor,
                    label=label,
                    timestamp=now.isoformat(),
                )
            )
            trigger = FrameTrigger.EVENT
        if step.act_gate is not None:
            # Script the async act-gate: feed its outcome into the read-back path
            # (the proactive_outcome signal), correlated to the turn in flight.
            pending = self._lm.state.load().pending_proactive_id
            if pending:
                signals.append(
                    proactive_outcome_signal(
                        origin_id=f"outcome-{pending}",
                        outcome=step.act_gate,
                        timestamp=now.isoformat(),
                        correlation_id=pending,
                    )
                )
                trigger = FrameTrigger.ASYNC_COMPLETION
        ring_before = len(self.event_ring.read())
        egress_before = len(self.egress.calls)
        # The real delivery path: pipeline (frame) → backstop → recording egress.
        outcome = proactive_tick(
            self._lm, self.egress, self.target, initial_signals=signals, trigger=trigger
        )
        # Suppression spans route through the SpanLogger onto the freshness ring
        # (spec §4.2/§5), not the ad-hoc logger — read this step's slice back.
        new_ring = self.event_ring.read()[ring_before:]
        new_egress = self.egress.calls[egress_before:]
        suppressions = tuple(
            rec["reason"]
            for rec in new_ring
            if rec.get("event") == "suppression" and "reason" in rec
        )
        desire = read_live_contact_desire(self._memory())
        final = self._lm.state.load()
        return TickRecord(
            tick=final.tick_count,
            outcome=outcome,
            desire_state=desire.state if desire is not None else None,
            launched=bool(new_egress),
            delivered_impulse=new_egress[-1][1] if new_egress else None,
            suppressions=suppressions,
            u=final.u,
        )


def build_capture_lifemodel(
    *, base_dir: Path | None = None, clock: ClockPort | None = None
) -> LifeModel:
    """A real-code ``LifeModel`` with :class:`~lifemodel.core.thought_capture.ThoughtCapture`
    registered — the slice-1 (lm-705.1) thought-capture sim seam.

    Deliberately NOT a fake-ports harness: :class:`~lifemodel.testing.fakes.FakeStateStore`
    is a ``StatePort``/``TickCommitPort`` but NOT a ``MemoryPort`` (it exposes no
    ``get``/``find``/``put``/``transition`` of its own — only an INJECTED
    ``FakeMemoryStore`` does), so ``read_live_thoughts(lm.state)`` and the ``post_llm``
    seam's own ``isinstance(lm.state, MemoryPort)`` narrowing would both fail against
    one. The REAL :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` — exactly
    what :class:`IntegrationHarness` and ``tests/test_frame_acceptance.py`` already use
    for a "real code, honest prediction" sim — satisfies every port at once, so this
    builds the ordinary real graph (:func:`build_lifemodel`) over a fresh on-disk store.
    ``build_lifemodel`` registers ``ThoughtCapture`` itself (lm-705.1 Task 5); the
    ``try``/``UnknownComponent`` guard below only backfills it for a registry that
    somehow lacks it (mirroring every idempotent registration in ``composition.py``),
    so this stays correct even if that default ever changes.

    ``base_dir`` defaults to a fresh temp directory (stdlib ``tempfile``) so a caller
    can write ``build_capture_lifemodel()`` bare, with no ``tmp_path`` fixture to
    thread through — every call gets its own store, so concurrent tests never collide.
    The being is committed BORN (:data:`BORN_AT`) before return: an unborn being's very
    first frame is the genesis wake (spec §6.2), an entirely different path this
    capture-pipeline seam has nothing to do with — every thought-capture scenario is an
    ordinary owner↔being exchange, inside an existing relationship.
    """
    resolved_base_dir = (
        base_dir
        if base_dir is not None
        else Path(tempfile.mkdtemp(prefix="lifemodel-thought-capture-"))
    )
    resolved_clock: ClockPort = (
        clock if clock is not None else FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    lm = build_lifemodel(base_dir=resolved_base_dir, clock=resolved_clock)
    lm.state.commit(
        State(
            u=0.0,
            energy=1.0,
            fatigue=0.0,
            last_tick_at=resolved_clock.now().isoformat(),
            genesis_completed_at=BORN_AT,
        )
    )
    try:
        lm.registry.manifest(THOUGHT_CAPTURE_ID)
    except UnknownComponent:
        lm.registry.register(
            ThoughtCapture(),
            ComponentManifest(
                id=THOUGHT_CAPTURE_ID,
                type="thought-capture",
                layer=ComponentLayer.AGGREGATION,
                metric_surface=(),
                accepts_signals=True,
            ),
        )
    return lm


def build_processing_lifemodel(
    *, base_dir: Path | None = None, clock: ClockPort | None = None
) -> LifeModel:
    """A real-code ``LifeModel`` with
    :class:`~lifemodel.core.thought_processing.ThoughtProcessingSelector` registered —
    the slice-2 (lm-705.2) thought-processing sim seam (spec §6: backlog health,
    bounds terminate, idle 0-LLM, cost <= FR20 ceiling).

    Mirrors :func:`build_capture_lifemodel` exactly (see its docstring for why this
    builds the ordinary real graph over a fresh on-disk
    :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` rather than a fake-ports
    harness): ``build_lifemodel`` already registers ``ThoughtProcessingSelector``
    itself (lm-705.2 Task 7); the ``try``/``UnknownComponent`` guard below only
    backfills it for a registry that somehow lacks it, mirroring every idempotent
    registration in ``composition.py``.

    ``base_dir`` defaults to a fresh temp directory so a caller can write
    ``build_processing_lifemodel()`` bare. The being is committed BORN
    (:data:`BORN_AT`) before return, same reasoning as ``build_capture_lifemodel``:
    a thought-processing sim scenario is an ordinary tick inside an existing
    relationship, never the genesis wake.
    """
    resolved_base_dir = (
        base_dir
        if base_dir is not None
        else Path(tempfile.mkdtemp(prefix="lifemodel-thought-processing-"))
    )
    resolved_clock: ClockPort = (
        clock if clock is not None else FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    lm = build_lifemodel(base_dir=resolved_base_dir, clock=resolved_clock)
    lm.state.commit(
        State(
            u=0.0,
            energy=1.0,
            fatigue=0.0,
            last_tick_at=resolved_clock.now().isoformat(),
            genesis_completed_at=BORN_AT,
        )
    )
    try:
        lm.registry.manifest(THOUGHT_PROCESSING_SELECTOR_ID)
    except UnknownComponent:
        lm.registry.register(
            ThoughtProcessingSelector(),
            ComponentManifest(
                id=THOUGHT_PROCESSING_SELECTOR_ID,
                type="thought-processing-selector",
                layer=ComponentLayer.COGNITION,
                metric_surface=(),
                accepts_signals=True,
            ),
        )
    return lm


def build_noticing_lifemodel(
    *, buffer: NoticingBuffer, base_dir: Path | None = None, clock: ClockPort | None = None
) -> LifeModel:
    """A real-code ``LifeModel`` with the noticing pair —
    :class:`~lifemodel.core.noticing.NoticingTrigger` +
    :class:`~lifemodel.core.noticing.NoticingApply` — registered over *buffer*: the
    slice-5 (lm-705.5) sim seam (spec §6: a buffered sitting becomes real thoughts,
    source ids, closed-prefix, dedup, 0-LLM idle).

    Mirrors :func:`build_processing_lifemodel` exactly (see its docstring / the
    module docstring above for why this builds the ordinary real graph over a fresh
    on-disk :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`), with one
    difference: *buffer* is a REQUIRED, caller-supplied
    :class:`~lifemodel.core.noticing_buffer.NoticingBuffer` rather than something
    this function constructs. Unlike thought-processing's backlog (which lives
    entirely in the store), noticing's input is the PROCESS-OWNED buffer — a fresh
    one built here would never see the turns the caller seeds through its own public
    API (``open_pending``/``stamp_source``/``complete``) before/after driving a frame
    (see :mod:`lifemodel.core.noticing_buffer`'s own docstring on why it is
    process-owned, not per-graph). ``build_lifemodel``'s own ``noticing_buffer=``
    branch registers both components idempotently over *buffer*, so a caller that
    already holds a registry with them registered (an unlikely but harmless case)
    is left unchanged.

    ``base_dir`` defaults to a fresh temp directory so a caller can write
    ``build_noticing_lifemodel(buffer=NoticingBuffer())`` with only the mandatory
    keyword. The being is committed BORN (:data:`BORN_AT`) before return, same
    reasoning as :func:`build_processing_lifemodel`: a noticing sim scenario is an
    ordinary tick inside an existing relationship, never the genesis wake.
    """
    resolved_base_dir = (
        base_dir if base_dir is not None else Path(tempfile.mkdtemp(prefix="lifemodel-noticing-"))
    )
    resolved_clock: ClockPort = (
        clock if clock is not None else FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    )
    lm = build_lifemodel(base_dir=resolved_base_dir, clock=resolved_clock, noticing_buffer=buffer)
    lm.state.commit(
        State(
            u=0.0,
            energy=1.0,
            fatigue=0.0,
            last_tick_at=resolved_clock.now().isoformat(),
            genesis_completed_at=BORN_AT,
        )
    )
    return lm


@dataclass
class _RecordingEgress:
    """Records launches without sending anything — for the capture-path safety test."""

    sent: list[object] = field(default_factory=list)

    def reach_out(self, target: object, impulse: str) -> object:
        from ..domain.egress import ReachOutcome

        self.sent.append((target, impulse))
        return ReachOutcome.DELIVERED


def build_capture_harness() -> object:
    """A lightweight ``CoreLoop`` over fakes for testing the restricted capture path.

    Returns a plain object with ``.coreloop``, ``.state_store``, ``.memory``, and
    ``.egress`` — enough for the ``capture_thoughts`` safety test (lm-705.11 Task 3).
    The store is seeded with one active ``desire`` row so the \"no LaunchProactive
    stranded\" assertion is meaningful.
    """
    from ..core.component import ComponentLayer
    from ..core.coreloop import CoreLoop
    from ..core.registry import ComponentManifest, ComponentRegistry
    from ..core.state_actor import StateActor
    from ..core.thought_capture import THOUGHT_CAPTURE_ID, ThoughtCapture
    from ..domain.memory import MemoryDraft
    from ..ports.tracer import TracerPort
    from .fakes import FakeClock, FakeMemoryStore, FakeStateStore, FakeTracer

    clock = FakeClock(datetime(2026, 7, 18, tzinfo=UTC))
    memory = FakeMemoryStore(clock=clock)
    state_store = FakeStateStore(initial=State(), memory=memory)
    # Seed one active desire so the "no LaunchProactive stranded" check is real.
    memory.put(MemoryDraft(kind="desire", id="d1", state="active", payload={}, source="test"))

    registry = ComponentRegistry()
    registry.register(
        ThoughtCapture(),
        ComponentManifest(
            id=THOUGHT_CAPTURE_ID,
            type="thought-capture",
            layer=ComponentLayer.AGGREGATION,
            metric_surface=(),
            accepts_signals=True,
        ),
    )

    actor = StateActor(state_store)
    tracer: TracerPort = FakeTracer()
    egress = _RecordingEgress()

    coreloop = CoreLoop(
        registry=registry,
        state_actor=actor,
        clock=clock,
        tracer=tracer,
        memory=memory,
    )
    # The load-bearing field: the fail-closed test swaps a double onto this.
    coreloop._capture_component = ThoughtCapture()

    result = type("CaptureHarness", (), {})()
    result.coreloop = coreloop
    result.state_store = state_store
    result.memory = memory
    result.egress = egress
    # A LightModel stand-in for the tool handler: exposes coreloop + clock.
    result.lm = type("LightModel", (), {})()
    result.lm.coreloop = coreloop
    result.lm.clock = clock
    result.metrics = coreloop._metrics
    return result
