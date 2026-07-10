"""SignalFrame + ExecutionFrame — the ephemeral in-memory nerve bus (spec §2/§3)."""

from __future__ import annotations

import threading

from lifemodel.core.frame import FrameTrigger, SignalFrame, run_frame
from lifemodel.domain.signal import Signal


def _sig(origin_id: str, kind: str = "contact_observed") -> Signal:
    return Signal(origin_id=origin_id, kind=kind)


# --- SignalFrame: the in-memory bus that lives <= one frame -------------------


def test_signal_frame_seeds_with_initial_signals() -> None:
    frame = SignalFrame([_sig("a"), _sig("b")])
    assert [s.origin_id for s in frame.snapshot()] == ["a", "b"]


def test_signal_frame_defaults_empty() -> None:
    assert SignalFrame().snapshot() == ()


def test_signal_frame_emit_appends_in_order() -> None:
    frame = SignalFrame([_sig("seed")])
    frame.emit(_sig("later-1"))
    frame.emit(_sig("later-2"))
    assert [s.origin_id for s in frame.snapshot()] == ["seed", "later-1", "later-2"]


def test_signal_frame_snapshot_is_a_stable_copy() -> None:
    # A snapshot handed to a component must not change when the frame keeps growing.
    frame = SignalFrame()
    frame.emit(_sig("first"))
    snap = frame.snapshot()
    frame.emit(_sig("second"))
    assert [s.origin_id for s in snap] == ["first"]  # the earlier snapshot is frozen


# --- FrameTrigger: the four occasions a frame runs (spec §3) ------------------


def test_frame_triggers_cover_every_occasion() -> None:
    # A frame runs on a heartbeat, an incoming event, an async-cognition completion,
    # or an admin mutation — the closed set of occasions (spec §3).
    assert {t.value for t in FrameTrigger} == {
        "heartbeat",
        "event",
        "async_completion",
        "admin",
    }


# --- run_frame: serialized through the one process-wide state-actor -----------


class _NullTracer:
    pass


class _RecordingLoop:
    """A stand-in CoreLoop that records the initial_signals + trigger it was run with."""

    def __init__(self) -> None:
        self.seen_signals: list[str] = []
        self.seen_trigger: FrameTrigger | None = None
        self.report = object()

    def tick(self, initial_signals=(), *, trigger=FrameTrigger.HEARTBEAT):  # type: ignore[no-untyped-def]
        self.seen_signals = [s.origin_id for s in initial_signals]
        self.seen_trigger = trigger
        return self.report


def test_run_frame_threads_initial_signals_and_trigger_into_the_loop() -> None:
    loop = _RecordingLoop()
    report = run_frame(loop, [_sig("ext-1")], trigger=FrameTrigger.EVENT)
    assert report is loop.report
    assert loop.seen_signals == ["ext-1"]
    assert loop.seen_trigger is FrameTrigger.EVENT


def test_run_frame_defaults_to_a_heartbeat_with_no_signals() -> None:
    loop = _RecordingLoop()
    run_frame(loop)
    assert loop.seen_signals == []
    assert loop.seen_trigger is FrameTrigger.HEARTBEAT


def test_run_frame_holds_the_state_actor_lock_for_the_whole_frame() -> None:
    # Every frame is serialized through ONE process-wide state-actor lock (spec §3):
    # two frames can never interleave their snapshot→commit. Prove the lock is held
    # for the duration of the frame by trying to acquire it (non-blocking) from
    # inside the tick — it must fail while the frame runs.
    from lifemodel.core.frame import _STATE_ACTOR_LOCK

    held_during_frame = {"value": None}

    class _LockProbingLoop:
        def tick(self, initial_signals=(), *, trigger=FrameTrigger.HEARTBEAT):  # type: ignore[no-untyped-def]
            # A DIFFERENT thread must not be able to acquire the lock now.
            acquired: list[bool] = []

            def _try() -> None:
                got = _STATE_ACTOR_LOCK.acquire(blocking=False)
                acquired.append(got)
                if got:
                    _STATE_ACTOR_LOCK.release()

            t = threading.Thread(target=_try)
            t.start()
            t.join()
            held_during_frame["value"] = not acquired[0]  # not-acquired => we hold it
            return None

    run_frame(_LockProbingLoop())
    assert held_during_frame["value"] is True
