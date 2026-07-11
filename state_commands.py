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
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import composition
from .core.backstop import allow_send
from .core.desire_view import DESIRE_KIND
from .core.thought_view import (
    LIVE_THOUGHT_STATES,
    THOUGHT_KIND,
    build_thought,
    encode_thought,
    seed_thought_id,
)
from .core.user_model_view import (
    EXPLICIT_CONFIDENCE,
    build_owner_user_model,
    encode_owner_user_model,
    read_owner_user_model,
)
from .core.why_graph import (
    WhyNode,
    build_why_graph,
    display_id,
    why_contact_desire,
    why_contact_intention,
)
from .domain.memory import MemoryMutation, PutOp, StaleTransition, TransitionOp
from .domain.objects import (
    CONTACT_DESIRE_ID,
    DesireState,
    InferredField,
    InvalidTransition,
    Provenance,
    ThoughtState,
    UserModel,
    default_registry,
)
from .ports.memory import MemoryPort
from .ports.tick_commit import TickCommitPort
from .ports.tracer import parse_traceparent
from .state.errors import StateCorruptError, StateError
from .state.model import State
from .state.trace_store import observability_db_path, peek_trace_writer

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
    # last_exchange_at is deliberately ABSENT (lm-md6.1): the real exchange record the
    # wake packet renders is immune to admin commands. Tune the silence-window gate via
    # silence_anchor_at instead.
    "silence_anchor_at": _KIND_TIMESTAMP,
    "last_contact_at": _KIND_TIMESTAMP,
}


def _fmt_value(value: object) -> str:
    """Render a State field value for a human echo (lm-25t).

    DISPLAY ONLY: floats are rounded to 2 decimals so echoes read cleanly
    (``u: 1.42 -> 2.00``, not ``u: 1.419954456041666 -> 2.0``). The persisted
    value is untouched — this formats the ``before``/``after`` snapshot, it never
    rewrites state. Everything else (ints, strings, timestamps, None, lists)
    renders via ``repr`` so it reads exactly as stored."""
    if isinstance(value, float):
        return f"{value:.2f}"
    return repr(value)


def _field_lines(before: State, after: State, field_names: Sequence[str]) -> list[str]:
    return [
        f"  {name}: {_fmt_value(getattr(before, name))} -> {_fmt_value(getattr(after, name))}"
        for name in field_names
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
    # Satisfy the silence-window gate by backdating the DECOUPLED silence anchor —
    # NOT the real last_exchange_at, which the wake packet renders and which is immune
    # to all admin commands (lm-md6.1). The gate reads this anchor; the model still
    # sees the genuine last exchange.
    silence_anchor_at = (now - timedelta(minutes=backdate_min)).isoformat()

    send_log = before.proactive_send_log
    backstop_was_blocking = not allow_send(send_log, now)
    if backstop_was_blocking:
        send_log = []  # trim so the global backstop (spec §14) doesn't hold the send

    after = dataclasses.replace(
        before,
        u=u,
        silence_anchor_at=silence_anchor_at,
        pending_proactive_id=None,
        pending_proactive_since=None,
        pending_proactive_origin_traceparent=None,  # clear the async anchor in lockstep (§4.4)
        decline_count=0,
        declined_at=None,
        action_pending_since=None,  # clears ActionPending inhibition too
        proactive_send_log=send_log,
    )

    fields = [
        "u",
        "silence_anchor_at",
        "pending_proactive_id",
        "pending_proactive_since",
        "pending_proactive_origin_traceparent",
        "decline_count",
        "declined_at",
        "action_pending_since",
        "proactive_send_log",
    ]
    gates = [
        f"effective pressure: u={u:.2f} >= theta={theta:.2f} "
        "(action_pending cleared -> inhibition=0)",
        f"active-silence window: silence_anchor_at backdated {backdate_min:.0f}m "
        f"(window w={w:.0f}m; real last_exchange_at left untouched)",
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
    """Simulate a fulfilled contact — reset the drive as if contact just happened.

    Resets the drive (``u=0``) and opens the active-silence window (``silence_anchor_at
    =now``) exactly as a genuine exchange would for the gate — but does NOT forge the
    real ``last_exchange_at`` the wake packet renders (lm-md6.1): that record is written
    only by an actual two-way exchange. "Reset the drive" is decoupled from "record of
    a real conversation", so a satiated being still tells the model the true last-exchange
    time, not a fabricated "just now"."""
    now_iso = now.isoformat()
    after = dataclasses.replace(
        before,
        u=0.0,
        last_contact_at=now_iso,
        silence_anchor_at=now_iso,  # open the silence window WITHOUT forging last_exchange_at
        pending_proactive_id=None,
        pending_proactive_since=None,
        pending_proactive_origin_traceparent=None,  # clear the async anchor in lockstep (§4.4)
        action_pending_since=None,
    )
    fields = [
        "u",
        "last_contact_at",
        "silence_anchor_at",
        "pending_proactive_id",
        "pending_proactive_since",
        "pending_proactive_origin_traceparent",
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


# --- user-model prefs (spec §8) ---------------------------------------------
# The owner-facing path to SET the being's derived norms about its owner (good/bad
# hours, cadence min, quiet topics, allowed styles, explicit prefs). It builds a
# typed ``UserModel`` and commits it through the intent bus (a ``PutRecord``
# upsert via ``commit_tick`` — NOT direct SQL, NOT config), which
# ``appraise_receptivity`` then reads. Setting prefs marks the row EXPLICIT
# (``confidence=EXPLICIT_CONFIDENCE``) so its boundaries hard-veto; an unset being
# keeps the permissive default and behaves exactly as before.

#: Each ``key=value`` token the owner may set, mapped to its ``UserModel`` field
#: + coercion kind. This IS the whole whitelist — an unknown key is rejected with a
#: clear message, never silently splatted.
_UM_LIST_KEYS: dict[str, str] = {
    "privacy": "privacy_boundaries",
    "topics": "topic_sensitivity",
    "styles": "acceptable_styles",
    "prefs": "explicit_preferences",
}
_UM_HOUR_KEYS: dict[str, str] = {"bad-hours": "bad_hours", "good-hours": "good_hours"}
_UM_STR_KEYS: dict[str, str] = {
    "cadence": "cadence",
    "valence": "response_valence_pattern",
    "load": "known_load",
    "latency": "reply_latency_norm",
}
_UM_FLOAT_KEYS: dict[str, str] = {"intimacy": "intimacy_depth"}

#: One ``key=value`` token; the value runs up to the next ``key=`` or end-of-input,
#: so multi-word values ("load=busy at work") parse without quoting.
_UM_TOKEN = re.compile(r"(?P<key>[a-z][a-z-]*)=(?P<val>.*?)(?=\s+[a-z][a-z-]*=|$)", re.IGNORECASE)


def _um_usage() -> str:
    keys = ", ".join([*_UM_HOUR_KEYS, *_UM_STR_KEYS, *_UM_LIST_KEYS, *_UM_FLOAT_KEYS])
    return (
        "usage: /lifemodel user-model <key>=<value> ...\n"
        f"keys: {keys}\n"
        "note: bad-hours/good-hours are UTC hours 0-23 (comma-separated); "
        "keys not named are left unchanged.\n"
    )


def _parse_hours(raw: str) -> tuple[int, ...] | None:
    hours: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            hour = int(token)
        except ValueError:
            return None
        if not 0 <= hour <= 23:
            return None
        hours.append(hour)
    return tuple(hours)


def _parse_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def set_user_model_prefs(
    raw_args: str, existing: UserModel | None = None
) -> tuple[UserModel | None, str]:
    """Parse owner ``key=value`` prefs into the typed owner :class:`UserModel`.

    Returns ``(user_model, message)`` on success (``confidence=EXPLICIT_CONFIDENCE``
    so its boundaries hard-veto) or ``(None, error_message)`` on bad input — the
    same reject-nothing-to-commit contract the other mutations use. Pure: builds
    the row, persists nothing (the ``*_for_dir`` wrapper commits it through the bus).

    When *existing* is given, the parsed keys are **patched onto it** — only the
    fields named in this command change, so setting ``cadence`` later does NOT
    clear a previously-set ``bad_hours`` boundary (an owner boundary must not be
    silently dropped by an unrelated update).
    """
    if not raw_args.strip():
        return None, _um_usage()
    kwargs: dict[str, Any] = {}
    set_labels: list[str] = []
    consumed = 0
    for match in _UM_TOKEN.finditer(raw_args):
        consumed += 1
        key = match.group("key").lower()
        raw_value = match.group("val").strip()
        if key in _UM_HOUR_KEYS:
            hours = _parse_hours(raw_value)
            if hours is None:
                return (
                    None,
                    f"error: {key!r} expects comma-separated hours 0-23, got {raw_value!r}\n",
                )
            kwargs[_UM_HOUR_KEYS[key]] = hours
            set_labels.append(f"{_UM_HOUR_KEYS[key]}={hours}")
        elif key in _UM_LIST_KEYS:
            items = _parse_list(raw_value)
            kwargs[_UM_LIST_KEYS[key]] = items
            set_labels.append(f"{_UM_LIST_KEYS[key]}={items}")
        elif key in _UM_STR_KEYS:
            kwargs[_UM_STR_KEYS[key]] = raw_value
            set_labels.append(f"{_UM_STR_KEYS[key]}={raw_value!r}")
        elif key in _UM_FLOAT_KEYS:
            try:
                number = float(raw_value)
            except ValueError:
                return None, f"error: {key!r} expects a number, got {raw_value!r}\n"
            kwargs[_UM_FLOAT_KEYS[key]] = number
            set_labels.append(f"{_UM_FLOAT_KEYS[key]}={number}")
        else:
            return None, f"error: unknown user-model key {key!r}.\n{_um_usage()}"
    if consumed == 0:
        return None, _um_usage()
    if existing is not None:
        # Patch the provided keys onto the existing row (preserve other boundaries).
        # Owner-set values are authoritative → wrapped with no ttl (never go stale).
        patched: dict[str, Any] = {name: InferredField(value) for name, value in kwargs.items()}
        user_model = dataclasses.replace(existing, confidence=EXPLICIT_CONFIDENCE, **patched)
    else:
        user_model = build_owner_user_model(confidence=EXPLICIT_CONFIDENCE, **kwargs)
    lines = [
        "lifemodel user-model  (mutating)",
        "=" * 30,
        "",
        f"  owner user-model set (confidence={EXPLICIT_CONFIDENCE:.2f} -> boundaries hard-veto)",
        *(f"  {label}" for label in set_labels),
    ]
    return user_model, "\n".join(lines) + "\n"


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
    desire_mutations: Callable[[composition.LifeModel], list[MemoryMutation]] | None = None,
) -> str:
    """Load -> compute a candidate -> re-validate -> commit (or reject).

    ``desire_mutations`` optionally computes desire-row mutations to commit
    atomically alongside the ``State`` candidate (one ``commit_tick``), so the
    row and the vitals never split."""
    lm = composition.build_lifemodel(base_dir=base_dir)
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
    # A mutation that clears an in-flight proactive attempt (force-wake / satiate) is
    # a §4.4 clear-site: retire its disposable index correlation so retention can
    # reclaim the origin trace (the precious state anchor is already cleared above).
    if before.pending_proactive_id and candidate.pending_proactive_id is None:
        _mark_pending_correlation_resolved(base_dir, before, now)
    return message


def _mark_pending_correlation_resolved(base_dir: Path, before: State, now: datetime) -> None:
    """Best-effort: stamp ``resolved_at`` on the disposable correlation index for a
    pending proactive attempt an admin command just cleared (§4.4), so retention can
    reclaim its origin trace instead of protecting it forever on an unresolved row.

    Reaches the LIVE in-process writer via :func:`peek_trace_writer` (no refcount): a
    ``/lifemodel`` command runs inside the gateway process that holds the singleton,
    so this shares it. A bare CLI process with no live being simply no-ops — the
    *precious* state anchor was already cleared, and the trace DB is disposable."""
    correlation_id = before.pending_proactive_id
    origin = before.pending_proactive_origin_traceparent
    if not correlation_id or not origin:
        return
    writer = peek_trace_writer(observability_db_path(base_dir))
    if writer is None:
        return
    with contextlib.suppress(ValueError):
        stamp = now.isoformat()
        writer.submit_correlation(
            correlation_id=correlation_id,
            origin_trace_id=parse_traceparent(origin).trace_id,
            created_at=stamp,
            resolved_at=stamp,
        )


def nudge_for_dir(base_dir: Path, raw_amount: str) -> str:
    return _apply(base_dir, lambda before, now: nudge(before, now, raw_amount))


def force_wake_for_dir(base_dir: Path) -> str:
    # Terminalize any stuck desire so the NEXT real tick births a fresh one via
    # the (now-satisfied) gates — the gate-proving path force-wake exists for.
    return _apply(
        base_dir,
        force_wake,
        desire_mutations=lambda lm: _terminalize_live_desire(lm, str(DesireState.DROPPED)),
    )


def satiate_for_dir(base_dir: Path) -> str:
    # A simulated fulfilled contact terminalizes the live desire (satisfied).
    return _apply(
        base_dir,
        satiate,
        desire_mutations=lambda lm: _terminalize_live_desire(lm, str(DesireState.SATISFIED)),
    )


def reset_for_dir(base_dir: Path) -> str:
    """Factory wipe via :meth:`~lifemodel.state.port.StatePort.reset` directly —
    NOT through :func:`_apply`'s load-mutate-commit flow, because a reset must
    still work when the previously-persisted state is unreadable (corrupt, or
    an unsupported schema version). ``before`` is loaded best-effort purely to
    render the changed-fields echo; failing that read never blocks the reset
    itself, it only degrades the message to a generic banner.

    A TRUE factory wipe (lm-7lx) also deletes every ``memory_records`` row —
    every thought/desire/intention/user_model, not just the vitals row — so
    a reset being genuinely starts "as if newly born" with no rumination spiral
    left behind. See :func:`_purge_all_memory` for the best-effort seam.
    """
    lm = composition.build_lifemodel(base_dir=base_dir)
    now = lm.clock.now()
    try:
        before: State | None = lm.state.load()
    except StateError:
        before = None
    lm.state.reset()
    # Factory-wipe is a §4.4 clear-site too: the fresh ``State()`` has no anchor, so
    # retire any in-flight index correlation the wiped state carried.
    if before is not None:
        _mark_pending_correlation_resolved(base_dir, before, now)
    cleared = _purge_all_memory(lm)  # a factory wipe also drops EVERY memory row
    footer = f"  cleared {cleared} memory records\n"
    if before is None:
        return (
            "lifemodel reset  (mutating)\n"
            + "=" * 30
            + "\n\n  (previous state unreadable)\n"
            + footer
        )
    _, message = reset(before, now)
    return message + footer


def _purge_all_memory(lm: composition.LifeModel) -> int:
    """Best-effort: delete every ``memory_records`` row on a factory wipe.

    Deliberately duck-typed rather than an ``isinstance(lm.state, MemoryPort)``
    check: a hard delete-everything is out of scope for ``MemoryPort`` itself
    (that Protocol's own contract is soft-delete only, via guarded
    ``transition``), so this reaches the concrete
    :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`'s
    ``purge_memory_records`` the same permissive way ``_terminalize_live_desire``
    reaches memory, without importing the concrete adapter into this
    Hermes-free module. A store without the method (a minimal ``StatePort``
    fake, or any failure mid-purge) degrades to "0 cleared" — it never blocks
    the reset, which has already landed by the time this runs."""
    purge = getattr(lm.state, "purge_memory_records", None)
    if not callable(purge):
        return 0
    try:
        result = purge()
    except Exception:  # noqa: BLE001 - best-effort; the reset must never fail here
        return 0
    return result if isinstance(result, int) else 0


def set_field_for_dir(base_dir: Path, raw_args: str) -> str:
    return _apply(base_dir, lambda before, now: set_field(before, now, raw_args))


def set_user_model_prefs_for_dir(base_dir: Path, raw_args: str) -> str:
    """Parse owner prefs and commit the owner user-model row through the bus.

    The user-model is a typed ``kind='user_model'`` record, not a ``State``
    field, so this does NOT go through :func:`_apply` (whose ``compute`` returns a
    ``State`` candidate). It builds the typed row, wraps it in a ``PutRecord``
    upsert, and commits it via the SAME atomic ``commit_tick`` the tick pipeline
    uses (the single door) — the unchanged vitals ride along so the two never
    split. A ``StatePort`` fake without ``TickCommitPort`` cannot persist the row;
    that degrades to a no-op commit (the live ``SQLiteRuntimeStore`` always
    implements it)."""
    lm = composition.build_lifemodel(base_dir=base_dir)
    # Read the existing owner user-model so the parsed keys PATCH it (an unrelated
    # update must not clear a previously-set boundary). Absent → build fresh.
    existing = read_owner_user_model(lm.state) if isinstance(lm.state, MemoryPort) else None
    user_model, message = set_user_model_prefs(raw_args, existing)
    if user_model is None:
        return message
    state = lm.state.load()
    put = PutOp(draft=encode_owner_user_model(user_model))
    if isinstance(lm.state, TickCommitPort):
        lm.state.commit_tick(state, [put])
    else:  # pragma: no cover - the live store is always a TickCommitPort
        return "error: this store cannot persist a user-model record\n"
    return message


# --- thoughts (lm-27n.6) ----------------------------------------------------
# The being's thought engine has no generation yet (idle/event/chaining land
# later), so the ONLY way to create a thought in .6 is this owner/debug seed
# path — it lets persist → render → snapshot → transition be tested end-to-end
# and lets the owner inspect the mechanism. It builds a typed ``Thought`` (active,
# a deterministic content-digest id, a seed provenance) and commits it through the
# SAME atomic bus (a ``PutRecord``/``TransitionRecord`` via ``commit_tick``) the
# tick pipeline uses — NOT direct SQL. Transitions (park/resolve/drop) go through
# the registry's guarded edge; nothing DRIVES them yet (that engine is .7).

#: The seed provenance stamped on an owner/debug-created thought — records who/why
#: for the audit trail (no trace ids: this is a manual seed, not a traced turn).
_SEED_PROVENANCE = Provenance(
    created_by="owner",
    component="state_commands.think",
    reason="owner-seeded thought (lm-27n.6 debug/seed path — no generation yet)",
)


def think_for_dir(base_dir: Path, content: str) -> str:
    """Seed one typed ``kind='thought'`` row (active) through the intent bus.

    The debug/owner path that creates a thought so persist+render+snapshot are
    testable end-to-end (generation is deferred). Builds a typed ``Thought`` with
    a deterministic content-digest id — so re-seeding identical content upserts
    ONE row, not a pile — and commits it via the SAME atomic ``commit_tick`` the
    tick uses (the unchanged vitals ride along so the two never split)."""
    content = content.strip()
    if not content:
        return "usage: /lifemodel think <content>\n"
    lm = composition.build_lifemodel(base_dir=base_dir)
    thought = build_thought(
        id=seed_thought_id(content),
        content=content,
        trigger="seed",
        source="owner-seed",
        provenance=_SEED_PROVENANCE,
    )
    state = lm.state.load()
    put = PutOp(draft=encode_thought(thought))
    if not isinstance(lm.state, TickCommitPort):  # pragma: no cover - live store always commits
        return "error: this store cannot persist a thought record\n"
    lm.state.commit_tick(state, [put])
    lines = [
        "lifemodel think  (mutating)",
        "=" * 30,
        "",
        f"  thought seeded (active) [{thought.id}]",
        f"  {content}",
    ]
    return "\n".join(lines) + "\n"


def transition_thought_for_dir(
    base_dir: Path,
    thought_id: str,
    to_state: ThoughtState | str,
) -> str:
    """Move a live thought through a guarded, registry-legal edge (park/resolve/
    drop/...), committed atomically through the bus.

    The debug/helper transition path: reads the row, validates the edge is legal
    for the thought state machine (the registry is the single door), then commits
    a ``TransitionRecord`` via ``commit_tick``. Rejects an absent/terminal row or
    an illegal edge with a clear message — never a raw write. Nothing DRIVES these
    transitions yet (the engine is .7)."""
    to = str(to_state)
    lm = composition.build_lifemodel(base_dir=base_dir)
    memory = lm.state if isinstance(lm.state, MemoryPort) else None
    if memory is None:  # pragma: no cover - the live store is always a MemoryPort
        return "error: this store cannot read thought records\n"
    record = memory.get(THOUGHT_KIND, thought_id)
    if record is None:
        return f"error: no thought {thought_id!r}\n"
    if record.state not in LIVE_THOUGHT_STATES:
        return f"error: thought {thought_id!r} is already terminal ({record.state})\n"
    try:
        default_registry().validate_transition(THOUGHT_KIND, record.state, to)
    except InvalidTransition as exc:
        return f"error: {exc}\n"
    state = lm.state.load()
    op = TransitionOp(kind=THOUGHT_KIND, id=thought_id, from_state=record.state, to_state=to)
    if not isinstance(lm.state, TickCommitPort):  # pragma: no cover - live store always commits
        return "error: this store cannot persist a thought transition\n"
    try:
        lm.state.commit_tick(state, [op])
    except StaleTransition as exc:  # a lost race with a concurrent tick
        return f"error: {exc}\n"
    lines = [
        "lifemodel think transition  (mutating)",
        "=" * 30,
        "",
        f"  thought [{thought_id}] {record.state} -> {to}",
    ]
    return "\n".join(lines) + "\n"


# --- why (lm-27n.10): the READ-ONLY causal-chain reader ---------------------
# "Why does this desire/intention exist?" / "Why did I write?" — renders the durable
# why-graph (core/why_graph.py) as an indented text tree. Pure read: it walks the
# rows through the MemoryPort, never writes, and the walk is HARD-bounded (cycle /
# missing / depth / node caps live in the reader).


def _render_why_node(node: WhyNode, depth: int, label: str | None, lines: list[str]) -> None:
    indent = "  " * depth
    head = f"{label} -> " if label is not None else ""
    reason = f" - {node.reason}" if node.reason else ""
    trace = f" (trace {node.trace_id[:8]})" if node.trace_id else ""
    lines.append(f"{indent}{head}{display_id(node.kind, node.id)} [{node.state}]{reason}{trace}")
    for edge in node.edges:
        if edge.node is not None:
            _render_why_node(edge.node, depth + 1, edge.label, lines)
        elif edge.cycle:
            lines.append(f"{'  ' * (depth + 1)}{edge.label} -> [cycle]")
        elif edge.missing_ref is not None:
            lines.append(f"{'  ' * (depth + 1)}{edge.label} -> {edge.missing_ref} [missing]")


def render_why(node: WhyNode) -> str:
    """Render a :class:`WhyNode` tree as deterministic, indented text (read-only)."""
    lines: list[str] = ["lifemodel why  (read-only)", "=" * 30, ""]
    _render_why_node(node, 0, None, lines)
    return "\n".join(lines) + "\n"


def why_for_dir(base_dir: Path, raw_args: str) -> str:
    """Answer ``/lifemodel why [desire|intention|write|<kind>:<id>]`` — read-only.

    ``why`` / ``why write`` / ``why intention`` walk the contact intention chain ("why
    did I decide to write?"); ``why desire`` walks the contact desire chain; a bare
    ``<kind>:<id>`` walks that precise object. A target with no live/recent row renders
    a clear "no current outreach" (or "no such object") message — never a crash."""
    lm = composition.build_lifemodel(base_dir=base_dir)
    memory = lm.state if isinstance(lm.state, MemoryPort) else None
    if memory is None:  # pragma: no cover - the live store is always a MemoryPort
        return "error: this store cannot read memory records\n"

    raw = raw_args.strip()
    target = raw.lower()
    if target in ("", "write", "intention"):
        node = why_contact_intention(memory)
        empty = "lifemodel why: no current outreach — no live or recent contact intention.\n"
    elif target == "desire":
        node = why_contact_desire(memory)
        empty = "lifemodel why: no current outreach — no live or recent contact desire.\n"
    elif ":" in raw:
        kind, _, obj_id = raw.partition(":")
        node = build_why_graph(memory, kind, obj_id)
        empty = f"lifemodel why: no such object {kind}:{obj_id}\n"
    else:
        return (
            "usage: /lifemodel why [desire | intention | write | <kind>:<id>]\n"
            "  (no argument = the current contact intention chain)\n"
        )
    if node is None:
        return empty
    return render_why(node)
