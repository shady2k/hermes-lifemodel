"""Owner-facing MUTATING ``/lifemodel`` subcommands (bead lm-2vx).

Before this existed, forcing the being to wake for testing meant hand-editing
the persisted state directly — fragile, and it races the live 60s
``BeingAdapter`` tick (the loop reads state, mutates, and commits over a
hand-edit that lands mid-cycle). These subcommands go through the SAME
:class:`~lifemodel.state.port.StatePort` the adapter loop uses (via the
composition root, exactly as :mod:`lifemodel.debug` does for the read-only
dump — backed by :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`
since lm-fib.6.2) — no parallel writer, no hand-editing.

Each mutation is a small pure function: ``(before: State, now: datetime, ...)
-> (candidate: State | None, message: str)``. ``None`` means "reject, nothing
to commit" (bad input); the message is always owner-facing text, echoing the
changed fields before -> after. The ``*_for_dir`` wrappers do the
load/validate/commit against a real profile directory, re-validating every
candidate through :meth:`State.from_dict` (the model's own type/shape/tz-aware
timestamp checks) before it is ever persisted — defense in depth, reusing the
model's validator rather than inventing a second one. ``reset`` is the one
exception: it routes through :meth:`~lifemodel.state.port.StatePort.reset`
directly (see :func:`reset_for_dir`) so a factory wipe still works even when
the previously-persisted state is unreadable.

A residual logical race with an in-progress tick (loop reads -> command writes
-> loop commits over it) is accepted here, as directed: this is a debug tool, a
mutation lands cleanly on the *next* tick, and no coordination is built for it.

``force_wake`` derives its gate values from the SAME constants the live
pipeline reads (``composition.CONTACT_PARAMS``, ``core.backstop.allow_send``'s
defaults), so it can never drift from the real wake decision
(:mod:`lifemodel.sim.wake`, :mod:`lifemodel.core.aggregation`). It only makes
the state wake-*eligible* for cognition (spec §7: an urge merely *wakes*
cognition, it never sends) — whether a turn is actually launched and delivered
is a separate, energy-gated cognition-layer decision on a later tick that this
command deliberately does not touch, bypass, or run synchronously.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import composition
from .core.backstop import allow_send
from .core.desire_view import DESIRE_KIND
from .domain.memory import MemoryMutation, StaleTransition, TransitionOp
from .domain.objects import CONTACT_DESIRE_ID, DesireState
from .log import EventLogger
from .ports.memory import MemoryPort
from .ports.tick_commit import TickCommitPort
from .state.errors import StateCorruptError, StateError
from .state.model import State

#: The terminal desire states — a live desire in any other state can still be
#: terminalized; one already here is left alone.
_TERMINAL_DESIRE_STATES: frozenset[str] = frozenset(
    {DesireState.SATISFIED.value, DesireState.DROPPED.value, DesireState.EXPIRED.value}
)

#: Margin above theta_u so the effective-pressure gate is cleared, not grazed.
_FORCE_WAKE_U_MARGIN = 1.0
#: Extra minutes past the silence window W so the silence gate is cleared, not
#: grazed by a clock-resolution wobble.
_FORCE_WAKE_SILENCE_MARGIN_MIN = 5.0

# --- the `set` whitelist -----------------------------------------------------
# Every field `set` may write, with its coercion kind. This IS the whole safety
# boundary for a generic setter over the being's persisted soul — anything not
# listed here is rejected with a clear message, never silently splatted.
_KIND_FLOAT = "float"
_KIND_INT = "int"
_KIND_TIMESTAMP = "timestamp"
_SET_WHITELIST: dict[str, str] = {
    "u": _KIND_FLOAT,
    "energy": _KIND_FLOAT,
    "fatigue": _KIND_FLOAT,
    "duration_over_theta": _KIND_FLOAT,
    "decline_count": _KIND_INT,
    "last_exchange_at": _KIND_TIMESTAMP,
    "last_contact_at": _KIND_TIMESTAMP,
}


def _field_lines(before: State, after: State, field_names: Sequence[str]) -> list[str]:
    return [
        f"  {name}: {getattr(before, name)!r} -> {getattr(after, name)!r}" for name in field_names
    ]


def _echo(label: str, before: State, after: State, field_names: Sequence[str]) -> str:
    lines = [
        f"lifemodel {label}  (mutating)",
        "=" * 30,
        "",
        *_field_lines(before, after, field_names),
    ]
    return "\n".join(lines) + "\n"


# --- the mutations themselves (pure: State in, candidate State out) --------


def nudge(before: State, now: datetime, raw_amount: str) -> tuple[State | None, str]:
    """``u += N`` (default ``+1.0``) — a quick bump of the contact drive."""
    raw = raw_amount.strip()
    if raw:
        try:
            amount = float(raw)
        except ValueError:
            return None, f"error: 'nudge' amount must be a number, got {raw!r}\n"
    else:
        amount = 1.0
    after = dataclasses.replace(before, u=before.u + amount)
    return after, _echo("nudge", before, after, ["u"])


def force_wake(before: State, now: datetime) -> tuple[State | None, str]:
    """Satisfy every wake gate (``sim.wake.evaluate_wake``) so the NEXT real
    adapter tick's aggregation pass wakes cognition — never runs a tick itself."""
    theta = composition.CONTACT_PARAMS.theta_u
    w = composition.CONTACT_PARAMS.w
    u = theta + _FORCE_WAKE_U_MARGIN
    backdate_min = w + _FORCE_WAKE_SILENCE_MARGIN_MIN
    last_exchange_at = (now - timedelta(minutes=backdate_min)).isoformat()

    send_log = before.proactive_send_log
    backstop_was_blocking = not allow_send(send_log, now)
    if backstop_was_blocking:
        send_log = []  # trim so the global backstop (spec §14) doesn't hold the send

    after = dataclasses.replace(
        before,
        u=u,
        last_exchange_at=last_exchange_at,
        pending_proactive_id=None,
        pending_proactive_since=None,
        decline_count=0,
        declined_at=None,
        action_pending_since=None,  # clears ActionPending inhibition too
        proactive_send_log=send_log,
    )

    fields = [
        "u",
        "last_exchange_at",
        "pending_proactive_id",
        "pending_proactive_since",
        "decline_count",
        "declined_at",
        "action_pending_since",
        "proactive_send_log",
    ]
    gates = [
        f"effective pressure: u={u:.2f} >= theta={theta:.2f} "
        "(action_pending cleared -> inhibition=0)",
        f"active-silence window: last_exchange_at backdated {backdate_min:.0f}m "
        f"(window w={w:.0f}m)",
        "no live desire: any live contact-desire row terminalized + pending_proactive_id/since "
        "cleared, so the next tick births a fresh one "
        "(in_flight is a per-tick signal, not persisted state -- unaffected by this command)",
        "reject-backoff clear: decline_count=0, declined_at=None",
        "backstop send-allowed: proactive_send_log "
        + (
            "cleared (was over the daily cap / min interval)"
            if backstop_was_blocking
            else "already within the daily limit"
        ),
    ]
    lines = [
        "lifemodel force-wake  (mutating)",
        "=" * 30,
        "",
        *_field_lines(before, after, fields),
        "",
        "gates satisfied:",
        *(f"  - {g}" for g in gates),
    ]
    return after, "\n".join(lines) + "\n"


def satiate(before: State, now: datetime) -> tuple[State | None, str]:
    """Simulate a fulfilled contact — as if the user just genuinely reached out."""
    now_iso = now.isoformat()
    after = dataclasses.replace(
        before,
        u=0.0,
        last_contact_at=now_iso,
        last_exchange_at=now_iso,
        pending_proactive_id=None,
        pending_proactive_since=None,
        action_pending_since=None,
    )
    fields = [
        "u",
        "last_contact_at",
        "last_exchange_at",
        "pending_proactive_id",
        "pending_proactive_since",
        "action_pending_since",
    ]
    return after, _echo("satiate", before, after, fields)


def reset(before: State, now: datetime) -> tuple[State | None, str]:
    """Factory wipe: as if newly born — write a fresh ``State()``.

    Intentionally total (the owner's explicit call, not a soft reset): this
    also clears ``tick_count``, the backstop send-count, and every "last
    talked" timestamp.
    """
    after = State()
    changed = [
        f.name
        for f in dataclasses.fields(State)
        if getattr(before, f.name) != getattr(after, f.name)
    ]
    if not changed:
        return after, (
            "lifemodel reset  (mutating)\n" + "=" * 30 + "\n\n  (state was already fresh)\n"
        )
    return after, _echo("reset", before, after, changed)


def set_field(before: State, now: datetime, raw_args: str) -> tuple[State | None, str]:
    """``set <field> <value>`` over the safe-field whitelist (see ``_SET_WHITELIST``)."""
    parts = raw_args.strip().split(None, 1)
    whitelist = ", ".join(_SET_WHITELIST)
    if len(parts) < 2:
        return None, f"usage: /lifemodel set <field> <value>\nwhitelisted fields: {whitelist}\n"
    field_name, raw_value = parts[0], parts[1].strip()

    kind = _SET_WHITELIST.get(field_name)
    if kind is None:
        return None, (
            f"error: 'set' field {field_name!r} is not writable. Whitelisted fields: {whitelist}\n"
        )

    value: object
    if kind == _KIND_FLOAT:
        try:
            value = float(raw_value)
        except ValueError:
            return None, f"error: field {field_name!r} expects a number, got {raw_value!r}\n"
    elif kind == _KIND_INT:
        try:
            value = int(raw_value)
        except ValueError:
            return None, f"error: field {field_name!r} expects an integer, got {raw_value!r}\n"
    else:  # _KIND_TIMESTAMP
        value = now.isoformat() if raw_value == "now" else raw_value

    changes: dict[str, Any] = {field_name: value}
    after = dataclasses.replace(before, **changes)
    return after, _echo(f"set {field_name}", before, after, [field_name])


# --- directory-level wrappers (the seam `__init__.py` calls) ----------------


def _terminalize_live_desire(lm: composition.LifeModel, to_state: str) -> list[MemoryMutation]:
    """A one-mutation batch that terminalizes the live contact-desire row to
    *to_state*, or ``[]`` when there is nothing live to terminalize.

    The desire lifecycle is a typed row now (lm-27n.3), so a state command that
    used to just null a ``State`` flag must move the row through the registry-
    guarded transition. Reads the singleton row; skips when absent or already
    terminal (never an illegal transition out of a terminal state)."""
    if not isinstance(lm.state, MemoryPort):
        return []
    record = lm.state.get(DESIRE_KIND, CONTACT_DESIRE_ID)
    if record is None or record.state in _TERMINAL_DESIRE_STATES:
        return []
    return [
        TransitionOp(
            kind=DESIRE_KIND,
            id=CONTACT_DESIRE_ID,
            from_state=record.state,
            to_state=to_state,
        )
    ]


def _apply(
    base_dir: Path,
    compute: Callable[[State, datetime], tuple[State | None, str]],
    *,
    logger: EventLogger | None = None,
    desire_mutations: Callable[[composition.LifeModel], list[MemoryMutation]] | None = None,
) -> str:
    """Load -> compute a candidate -> re-validate -> commit (or reject).

    ``desire_mutations`` optionally computes desire-row mutations to commit
    atomically alongside the ``State`` candidate (one ``commit_tick``), so the
    row and the vitals never split."""
    lm = composition.build_lifemodel(base_dir=base_dir, logger=logger)
    before = lm.state.load()
    now = lm.clock.now()
    candidate, message = compute(before, now)
    if candidate is None:
        return message
    try:
        State.from_dict(candidate.to_dict())  # reuse the model's own validation
    except StateCorruptError as exc:
        return f"error: refusing to persist an invalid state: {exc}\n"
    mutations = desire_mutations(lm) if desire_mutations is not None else []
    if mutations and isinstance(lm.state, TickCommitPort):
        lm.state.commit_tick(candidate, mutations)
    else:
        lm.state.commit(candidate)
    return message


def nudge_for_dir(base_dir: Path, raw_amount: str, *, logger: EventLogger | None = None) -> str:
    return _apply(base_dir, lambda before, now: nudge(before, now, raw_amount), logger=logger)


def force_wake_for_dir(base_dir: Path, *, logger: EventLogger | None = None) -> str:
    # Terminalize any stuck desire so the NEXT real tick births a fresh one via
    # the (now-satisfied) gates — the gate-proving path force-wake exists for.
    return _apply(
        base_dir,
        force_wake,
        logger=logger,
        desire_mutations=lambda lm: _terminalize_live_desire(lm, str(DesireState.DROPPED)),
    )


def satiate_for_dir(base_dir: Path, *, logger: EventLogger | None = None) -> str:
    # A simulated fulfilled contact terminalizes the live desire (satisfied).
    return _apply(
        base_dir,
        satiate,
        logger=logger,
        desire_mutations=lambda lm: _terminalize_live_desire(lm, str(DesireState.SATISFIED)),
    )


def reset_for_dir(base_dir: Path, *, logger: EventLogger | None = None) -> str:
    """Factory wipe via :meth:`~lifemodel.state.port.StatePort.reset` directly —
    NOT through :func:`_apply`'s load-mutate-commit flow, because a reset must
    still work when the previously-persisted state is unreadable (corrupt, or
    an unsupported schema version). ``before`` is loaded best-effort purely to
    render the changed-fields echo; failing that read never blocks the reset
    itself, it only degrades the message to a generic banner.
    """
    lm = composition.build_lifemodel(base_dir=base_dir, logger=logger)
    now = lm.clock.now()
    try:
        before: State | None = lm.state.load()
    except StateError:
        before = None
    lm.state.reset()
    _clear_live_desire_row(lm)  # a factory wipe also drops any live desire row
    if before is None:
        return "lifemodel reset  (mutating)\n" + "=" * 30 + "\n\n  (previous state unreadable)\n"
    _, message = reset(before, now)
    return message


def _clear_live_desire_row(lm: composition.LifeModel) -> None:
    """Best-effort: terminalize any live contact-desire row on a factory wipe.

    The state ``reset`` wipes the vitals row; this drops the separate desire
    record so a reset is truly "as if newly born". Best-effort — a missing memory
    port or a lost race never blocks the reset, which has already landed."""
    if not isinstance(lm.state, MemoryPort):
        return
    record = lm.state.get(DESIRE_KIND, CONTACT_DESIRE_ID)
    if record is None or record.state in _TERMINAL_DESIRE_STATES:
        return
    with contextlib.suppress(StaleTransition):  # a lost race is fine; the reset already landed
        lm.state.transition(DESIRE_KIND, CONTACT_DESIRE_ID, record.state, str(DesireState.DROPPED))


def set_field_for_dir(base_dir: Path, raw_args: str, *, logger: EventLogger | None = None) -> str:
    return _apply(base_dir, lambda before, now: set_field(before, now, raw_args), logger=logger)
