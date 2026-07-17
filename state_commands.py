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
exception: it skips the load-validate-commit flow and commits the freshly-BORN
body straight from :func:`reset` (see :func:`reset_for_dir`) via
:meth:`~lifemodel.state.port.StatePort.commit` — an unconditional UPSERT, never a
read-modify-write — so a factory wipe still works even when the previously-persisted
state is unreadable.

A residual logical race with an in-progress tick (loop reads -> command writes
-> loop commits over it) is accepted here, as directed: this is a debug tool, a
mutation lands cleanly on the *next* tick, and no coordination is built for it.

``force_wake`` derives its gate values from the SAME constants the live
pipeline reads (``composition.CONTACT_PARAMS``, ``core.backstop.allow_send``'s
defaults), so it can never drift from the real wake decision
(:mod:`lifemodel.core.wake`, :mod:`lifemodel.core.aggregation`). It only makes
the state wake-*eligible* for cognition (spec §7: an urge merely *wakes*
cognition, it never sends) — whether a turn is actually launched and delivered
is a separate, energy-gated cognition-layer decision on a later tick that this
command deliberately does not touch, bypass, or run synchronously.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import composition
from .adapters.session_end import sleep_soft
from .adapters.soul_file import SoulFile, SoulRejected
from .core.backstop import allow_send
from .core.desire_view import DESIRE_KIND
from .core.frame import state_actor_lock
from .core.genesis import ReplacedSoul, newborn
from .core.thought_view import (
    LIVE_THOUGHT_STATES,
    THOUGHT_KIND,
    build_thought,
    encode_thought,
    seed_thought_id,
)
from .core.timeutil import to_iso
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
from .debug import local_time  # the ONE owner-facing timestamp renderer (owner's tz, secs)
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
from .domain.session import SessionEnd
from .ports.memory import MemoryPort
from .ports.tick_commit import TickCommitPort
from .ports.tracer import parse_traceparent
from .state.errors import StateCorruptError, StateError
from .state.model import State
from .state.soul_revisions import (
    SoulRevert,
    SoulRevision,
    keep_replaced_soul,
    record_revert,
    reverts,
    revisions,
)
from .state.trace_store import observability_db_path, peek_trace_writer

_LOG = logging.getLogger("lifemodel.state_commands")

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

# --- the `set` field surface -------------------------------------------------
# `set` DERIVES its writable fields from the ``State`` dataclass (by field type), minus an
# explicit PROTECTED set. This inverts the old hand-maintained whitelist, which DRIFTED:
# lm-ukc.6 added ``affect_valence``/``affect_arousal`` to ``State`` and nobody added them
# to the list, so the owner could not calibrate the very axes the affect work is about
# (found live: the ambient cue could not be exercised at all). Now a new scalar field is
# settable BY DEFAULT and *protection* is the thing you must state on purpose, with a
# reason. ``test_every_state_field_is_settable_or_protected`` enforces that every field is
# consciously classified, so this cannot drift again.
_KIND_FLOAT = "float"
_KIND_INT = "int"
_KIND_TIMESTAMP = "timestamp"

#: Fields ``set`` must NEVER write, each with WHY. These are load-bearing invariants —
#: this is the real safety boundary, not bureaucracy.
_SET_PROTECTED: dict[str, str] = {
    "schema_version": "the store's migrations own it — hand-writing it corrupts load/upgrade",
    "last_exchange_at": (
        "lm-md6.1: the REAL exchange record the wake packet renders stays immune to admin "
        "commands — tune the silence-window gate via silence_anchor_at instead"
    ),
    "proactive_send_log": (
        "the hard send backstop that protects the OWNER from a spamming being — never hand-edit"
    ),
    "pending_proactive_id": (
        "in-flight proactive correlation — hand-setting desyncs outcome resolution"
    ),
    "pending_proactive_since": "in-flight proactive correlation (see pending_proactive_id)",
    "pending_proactive_origin_traceparent": (
        "in-flight proactive correlation (see pending_proactive_id)"
    ),
    "tick_count": "brain-liveness evidence — faking it would lie about whether the brain ticks",
    "last_tick_at": "brain-liveness evidence (see tick_count)",
    "processed_external_event_ids": "the idempotency ledger — hand-editing replays or drops events",
    "affect_display_last_word": (
        "reactive-display hint — written atomically by the injector (lm-ukc.4), never by hand"
    ),
    "affect_display_last_at": "reactive-display hint (see affect_display_last_word)",
    "genesis_shown_at_context_len": (
        "the record of how much context the being had when it was last shown its birth "
        "ritual — written by the pre_llm_call injector against the host's real message "
        "list; a hand-set value would either hide the ritual from an unborn being or "
        "restart it mid-birth. Use `reset` to make the being unborn again."
    ),
    "pending_internal_id": (
        "lm-705.6: in-flight internal-cognition correlation — hand-setting desyncs the "
        "runner's completion tracking, mirroring pending_proactive_id"
    ),
    "internal_calls_today": (
        "lm-705.6: FR20's durable daily call ceiling — hand-editing without its paired "
        "internal_calls_day would desync the quota rollover; never hand-edit either half"
    ),
    "internal_calls_day": "FR20 quota bookkeeping (see internal_calls_today)",
    "noticed_source_ids": (
        "lm-705.5: the noticing pass's consumed-source-id dedup ring — hand-editing could "
        "drop or duplicate dedup coverage, mirroring the other ledgers "
        "(processed_external_event_ids/proactive_send_log); no safe scalar coercion either "
        "(it's a tuple, not a settable scalar)"
    ),
    "surfaced_belief_ids": (
        "lm-705.19: the belief injector's cooldown ring — internal injector cooldown ring; "
        "not hand-written, mirroring noticed_source_ids (it's a tuple, not a settable scalar)"
    ),
}

#: Field TYPE -> coercion kind. A field whose type is absent here (``list``/``dict``) has no
#: safe scalar coercion and is therefore never settable.
_TYPE_KINDS: dict[str, str] = {
    "float": _KIND_FLOAT,
    "int": _KIND_INT,
    "str | None": _KIND_TIMESTAMP,
}


def settable_fields() -> dict[str, str]:
    """The fields ``set`` may write: derived from ``State`` by type, minus ``_SET_PROTECTED``."""
    out: dict[str, str] = {}
    for f in dataclasses.fields(State):
        if f.name in _SET_PROTECTED:
            continue
        kind = _TYPE_KINDS.get(str(f.type).strip())
        if kind is not None:
            out[f.name] = kind
    return out


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
    """Satisfy every wake gate (``core.wake.evaluate_wake``) so the NEXT real
    adapter tick's aggregation pass wakes cognition — never runs a tick itself."""
    theta = composition.CONTACT_PARAMS.theta_u
    w = composition.CONTACT_PARAMS.w
    u = theta + _FORCE_WAKE_U_MARGIN
    backdate_min = w + _FORCE_WAKE_SILENCE_MARGIN_MIN
    # Satisfy the silence-window gate by backdating the DECOUPLED silence anchor —
    # NOT the real last_exchange_at, which the wake packet renders and which is immune
    # to all admin commands (lm-md6.1). The gate reads this anchor; the model still
    # sees the genuine last exchange.
    silence_anchor_at = to_iso(now - timedelta(minutes=backdate_min))

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
    now_iso = to_iso(now)
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
    """Factory wipe: as if newly born — and *newly born* now means something.

    Clears the genesis stamps, so the ritual plays again and the owner can be his own
    first user. Builds the body with :func:`~lifemodel.core.genesis.newborn` rather than
    a bare ``State()``: those defaults are unfilled fields, not the body of a newborn,
    and a being reset into them would speak the first words of its second life from
    "quiet — even and very quiet" (the lm-ukc bug this deliberately does not repeat).
    Intentionally total otherwise (the owner's explicit call, not a soft reset): this
    also clears ``tick_count``, the backstop send-count, and every "last talked"
    timestamp — the full field-by-field diff (not a hand-picked list) is what makes the
    genesis stamps show up in the echo for free, exactly like every other cleared field.

    It does NOT touch ``SOUL.md``, and — since review I4 — it does not touch the soul's
    REVISIONS either (``purge_memory_records`` carves out ``kind="soul"``; the wipe used
    to take the whole lineage with it, so the reborn being's first ``write_soul`` then
    left the previous being's soul nowhere at all). Destroying a soul is an act that
    belongs to the human, and it takes more than one command. So the reborn being finds
    the soul of whoever lived here before it, still in slot #1, and opens the ritual on
    THAT (spec §6.4): rebirth does not erase a past life, it meets it — and every past
    life is still there to be put back.
    """
    after = newborn(
        now=now, params=composition.AFFECT_PARAMS, peak_hour_utc=composition.CIRCADIAN_PEAK_UTC_HOUR
    )
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
    """``set <field> <value>`` over the derived settable surface (see :func:`settable_fields`)."""
    parts = raw_args.strip().split(None, 1)
    settable = settable_fields()
    names = ", ".join(sorted(settable))
    if len(parts) < 2:
        return None, f"usage: /lifemodel set <field> <value>\nsettable fields: {names}\n"
    field_name, raw_value = parts[0], parts[1].strip()

    kind = settable.get(field_name)
    if kind is None:
        # A PROTECTED field gets its REASON back, not a bare rejection — the owner should
        # learn why the being's soul refuses that particular hand (and what to use instead).
        reason = _SET_PROTECTED.get(field_name)
        if reason is not None:
            return None, (
                f"error: 'set' field {field_name!r} is protected — {reason}.\n"
                f"settable fields: {names}\n"
            )
        return None, (
            f"error: 'set' field {field_name!r} is not writable. Settable fields: {names}\n"
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
        value = to_iso(now) if raw_value == "now" else raw_value

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
        stamp = to_iso(now)
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
    """Factory wipe: commits the freshly-BORN body straight from :func:`reset` —
    NOT :meth:`~lifemodel.state.port.StatePort.reset`, whose ``fresh = State()`` is
    exactly the lifeless zero-arousal default this whole command exists to stop
    leaving a being in. This also skips :func:`_apply`'s load-mutate-commit flow,
    because a reset must still succeed when the previously-persisted state is
    unreadable (corrupt, or an unsupported schema version) — but that guarantee no
    longer needs ``StatePort.reset`` specifically: :func:`~lifemodel.core.genesis.
    newborn` takes no ``before`` at all, and :meth:`~lifemodel.state.port.StatePort.
    commit` is an unconditional UPSERT (never a read-modify-write), so the fresh body
    lands either way. ``before`` is loaded best-effort purely to render the
    changed-fields echo; failing that read never blocks the reset itself, it only
    degrades the message to a generic banner.

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
    after, message = reset(before if before is not None else State(), now)
    if after is None:  # pragma: no cover - reset() never rejects
        return message
    lm.state.commit(after)
    # Factory-wipe is a §4.4 clear-site too: the fresh body has no anchor, so
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
    return message + footer


def _purge_all_memory(lm: composition.LifeModel) -> int:
    """Best-effort: delete every ``memory_records`` row on a factory wipe — EXCEPT the
    being's soul, which ``purge_memory_records`` itself carves out (``kind="soul"``: a
    past life's soul is the one thing a reset must not be able to destroy — see there,
    and spec §4.2's mandatory undo).

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


# --- soul (lm-4fv.2): the lineage, and the undo that justifies owning the file whole ---
#
# Spec §4.2 is the ENTIRE justification for letting the being own ``SOUL.md`` with no marker
# fence: "Every revision is kept in lifemodel.sqlite. Revert is one command. THIS — not a
# marker fence — is what makes it safe for the being to own the file whole." The revisions
# were kept from the start; the command was not, so the justification was a promise we did
# not keep — and ``after-install.md`` names this command to the human at the moment we ask
# for their consent to let a being rewrite their identity file.
#
# The danger it guards is EROSION, not one bad write. The being rewrites the whole document
# every time it changes, and over dozens of rewrites an LLM will quietly paraphrase a
# human's hard-won prose into oatmeal — with no single write ever looking broken. A fence
# cannot catch that. An undo can.
#
# ``history`` is read-only. ``revert`` is a WRITE, not a rewind: it goes through
# ``SoulFile.write`` (validated — a revision recorded before a threat pattern existed must
# not be written back blind), it KEEPS whatever it wrote over, it records that it happened,
# it TELLS the being (spec §4.1: a human rewriting who you are is an event in your life, not
# a version conflict — it must be felt, not swallowed), and it ends the session so the being
# actually comes back as the soul that was put back (ADR-0002). Nothing in a stored revision
# is ever mutated or deleted.

#: How much of a soul the listing shows. A first line is NOT enough: a soul's opening line is
#: usually "I am <name>", and after an erosion BOTH revisions start with it — the listing
#: would show the owner two identical-looking rows and be useless for its only job. So the
#: preview carries the head AND the tail: an erosion changes how a soul ENDS (the specific,
#: costly, personal lines) long before it changes what it opens with.
_PREVIEW_HEAD_LINES = 2
_PREVIEW_LINE_CHARS = 78

#: A sha prefix must be at least this long to be read as one — short enough to type from the
#: listing, long enough not to swallow an index the owner meant.
_MIN_SHA_PREFIX = 4


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def soul_preview(text: str) -> str:
    """Enough of a soul to RECOGNISE it (see :data:`_PREVIEW_HEAD_LINES`).

    Head lines plus the last line, blank lines dropped, each clipped — so two revisions that
    open identically ("I am Nova.") still read as two different souls. Where lines were
    skipped, the ``…`` says so rather than pretending the soul is what you see.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    parts = [_clip(line, _PREVIEW_LINE_CHARS) for line in lines[:_PREVIEW_HEAD_LINES]]
    if len(lines) > _PREVIEW_HEAD_LINES:
        parts.append("…")
        parts.append(_clip(lines[-1], _PREVIEW_LINE_CHARS))
    return " / ".join(parts)


def render_soul_history(
    history: Sequence[SoulRevision],
    revert_log: Sequence[SoulRevert],
    *,
    disk_sha: str,
    footer: str,
) -> str:
    """The lineage as the owner reads it — newest first, and WHERE THE BEING IS STANDING.

    The current-soul marker is not decoration. Without it the owner reverts, lists the
    history, sees the same eroded soul still sitting at ``[1]`` (it is still the newest
    revision — a revert does not unmake it), and cannot tell whether anything happened. The
    marker is read from the FILE, not from ``State.soul_sha``: the file is always the base
    (``adapters/soul_file.py``), and if a hand-edit is sitting there that nobody has recorded
    yet, the honest answer is to say so — that text is in no history and would be lost, and
    the owner is the only one who can act on that.
    """
    lines = ["lifemodel soul  (read-only)", "=" * 30, ""]
    if not history:
        lines += [
            "  no soul revisions yet — nothing has been written through the being.",
            "  Whatever is in SOUL.md now is kept the moment anything replaces it.",
        ]
        return "\n".join(lines) + "\n"

    index_of = {revision.sha: number for number, revision in enumerate(history, start=1)}
    lines.append(f"  {len(history)} soul revisions, newest first:")
    lines.append("")
    for number, revision in enumerate(history, start=1):
        here = "  <- on disk now" if revision.sha == disk_sha else ""
        lines.append(
            f"  [{number}] {local_time(revision.at)} · {revision.author} · "
            f"{len(revision.text)} chars · {revision.sha[:8]}{here}"
        )
        lines.append(f"      {soul_preview(revision.text)}")
    if disk_sha not in index_of:
        lines += [
            "",
            "  The soul in SOUL.md right now is not in this history — somebody edited the file"
            " by hand.",
            "  It is not lost: reverting over it keeps it as a revision first.",
        ]
    if revert_log:
        lines += ["", "  put back by hand:"]
        for revert in revert_log:
            target = _revision_label(revert.sha, index_of)
            over = _revision_label(revert.replaced_sha, index_of)
            lines.append(f"  {local_time(revert.at)} · put {target} back over {over}")
    lines += ["", footer]
    return "\n".join(lines) + "\n"


def _revision_label(sha: str, index_of: dict[str, int]) -> str:
    number = index_of.get(sha)
    return f"[{number}]" if number is not None else f"({sha[:8]}, no longer in the lineage)"


#: What a bare ``/lifemodel soul revert`` does, and why it is not an error.
#:
#: A person reaching for revert is in a hurry and mildly alarmed — so an error message that
#: makes them go and read ``help`` is a small cruelty. But the answer is NOT to guess for
#: them: this command rewrites who the being is, and a mutating command that acts on an
#: omitted argument turns a typo into an identity swap. "Revert the previous one" is exactly
#: the guess that would be wrong when the erosion is fifty writes deep and the soul they want
#: is at [37].
#:
#: So it shows them the thing they were reaching for — the lineage — and the exact command to
#: run. It cannot do harm, it costs one round trip, and it is what ``after-install.md``
#: already promises the human at install time: `/lifemodel soul revert` "lists them and puts
#: any one of them back".
_PICK_ONE = (
    "  pick one: /lifemodel soul revert <n>   (or its sha)\n"
    "  [mutating] it writes that soul back to SOUL.md and ends the being's session."
)

_HISTORY_FOOTER = (
    "  to put one back: /lifemodel soul revert <n>   (or its sha)\n"
    "  [mutating] it writes that soul back to SOUL.md and ends the being's session."
)

#: Said when the session ENDED: the owner is about to watch the being fall silent, and an
#: unexplained silence from something that has just been rewritten is exactly the kind of
#: thing that reads as a fault.
_WILL_WAKE_AS_IT = (
    "  The being's conversation with you has been ended: it will go quiet for a moment and\n"
    "  come back speaking as this soul. (SOUL.md is read into the prompt when a conversation\n"
    "  STARTS — a soul put back without that would sit on disk while the being went on\n"
    "  speaking as the one you just replaced.)"
)

#: Said when it did not. Fail SOFT, and say what is true: the file IS reverted, the voice is
#: not, and here is the host's own lever to fix that. (The precedent is ``ReachOutcome`` /
#: ``reachin_available``: a host that cannot do a thing is a value, not an exception.)
_STILL_THE_OLD_VOICE = (
    "  SOUL.md is reverted, but the being's live conversation could NOT be ended, so it is\n"
    "  still speaking as the soul you just replaced. It will come back as this one when this\n"
    "  conversation next rolls over — send /new here to make that happen now."
)

#: The revert landed on disk and the bookkeeping after it did not. Same rule the being's own
#: write follows (``hooks._WROTE_BUT_DID_NOT_RECORD``): never report a failure that would
#: leave the owner believing SOUL.md is unchanged when it is not.
_REVERTED_BUT_DID_NOT_RECORD = (
    "lifemodel soul revert  (mutating)\n"
    + "=" * 30
    + "\n\n"
    + "  SOUL.md IS reverted — that soul is on disk now.\n"
    "  What failed is the bookkeeping around it: the being may not have been told it was\n"
    "  rewritten, and its session may not have been ended. Check `/lifemodel soul history`.\n"
)


def _resolve_revision(
    history: Sequence[SoulRevision], token: str
) -> tuple[SoulRevision | None, str]:
    """The revision the owner named — by index, or by sha (see WHY below).

    The **index** is what the listing leads with, because it is what a person can type in a
    hurry. But it is a DISPLAY ARTIFACT of "newest first": the being writing one soul
    renumbers every row beneath it, so an index read a minute ago can name a different soul
    now. The **sha** cannot move, which is why the listing prints it too and why this accepts
    it — the owner who read the list and then went to make tea can still name the soul they
    meant.

    A digits-only token is always an index (a hex sha prefix of pure digits is possible but
    astronomically unlikely, and the index is unambiguously what the owner meant by typing
    ``2``); anything else is a sha prefix. Ambiguity is refused, never guessed.
    """
    raw = token.strip().lower()
    if raw.isdigit():
        number = int(raw)
        if 1 <= number <= len(history):
            return history[number - 1], ""
        return None, (
            f"error: there is no revision [{number}] — the lineage holds {len(history)}.\n"
            "Run `/lifemodel soul history` to see them.\n"
        )
    if len(raw) >= _MIN_SHA_PREFIX and all(char in "0123456789abcdef" for char in raw):
        matches = [revision for revision in history if revision.sha.startswith(raw)]
        if len(matches) == 1:
            return matches[0], ""
        if not matches:
            return None, (
                f"error: no revision starts with sha {raw!r}.\n"
                "Run `/lifemodel soul history` to see them.\n"
            )
        return None, (
            f"error: {len(matches)} revisions start with sha {raw!r} — type more of it.\n"
        )
    return None, (
        f"error: {token.strip()!r} is neither a revision number nor a sha.\n"
        "usage: /lifemodel soul revert <n>   (the [n] from `/lifemodel soul history`, or its sha)\n"
    )


def soul_for_dir(
    base_dir: Path,
    raw_args: str,
    *,
    soul: SoulFile,
    default_soul_text: str = "",
    end_session: SessionEnd | None = None,
) -> str:
    """``/lifemodel soul [history | revert <n>]`` — the being's lineage, and the undo.

    The Hermes-shaped things (which ``SOUL.md``, the host's pristine seed text, how to end
    the being's session) are injected from the composition root, exactly as they are for
    ``hooks.make_write_soul_tool`` — this module stays free of the host.
    """
    verb, _, rest = raw_args.strip().partition(" ")
    verb = verb.lower()
    if verb in ("", "history", "list", "log"):
        return _soul_history(base_dir, soul, footer=_HISTORY_FOOTER)
    if verb == "revert":
        if not rest.strip():
            # No argument is not an error — see _PICK_ONE.
            return _soul_history(base_dir, soul, footer=_PICK_ONE)
        return _soul_revert(
            base_dir,
            soul,
            rest,
            default_soul_text=default_soul_text,
            end_session=end_session,
        )
    return (
        "usage: /lifemodel soul [history | revert <n>]\n"
        "  history — every soul the being has ever had (read-only)\n"
        "  revert <n> — [mutating] write revision <n> back to SOUL.md\n"
    )


def _memory_of(lm: composition.LifeModel) -> MemoryPort | None:
    """The store as a ``MemoryPort`` — the same duck-typed narrowing every soul path uses
    (the live ``SQLiteRuntimeStore`` is always both; a minimal fake is not)."""
    return lm.state if isinstance(lm.state, MemoryPort) else None


def _soul_history(base_dir: Path, soul: SoulFile, *, footer: str) -> str:
    lm = composition.build_lifemodel(base_dir=base_dir)
    memory = _memory_of(lm)
    if memory is None:  # pragma: no cover - the live store is always a MemoryPort
        return "error: this store cannot read soul revisions\n"
    return render_soul_history(
        revisions(memory), reverts(memory), disk_sha=soul.sha(), footer=footer
    )


def _soul_revert(
    base_dir: Path,
    soul: SoulFile,
    token: str,
    *,
    default_soul_text: str,
    end_session: SessionEnd | None,
) -> str:
    """Put a soul back — a WRITE, not a rewind.

    Every step here is load-bearing, and each one closes a way to make this command a lie:

    * **Through** :meth:`SoulFile.write` **, so it is VALIDATED.** The lineage holds texts
      nobody ever validated — startup reconciliation records whatever a human left on disk,
      as-is — and the host's threat scanner grows new patterns over time. A revision that
      predates a pattern the host has since added would, written back blind, blank the WHOLE
      of ``SOUL.md`` on the next read (``core/soul_guard.py``) and the being would wake with
      no identity at all. A refusal here touches nothing and hands back the reason.
    * **It keeps what it wrote over** (:func:`~lifemodel.state.soul_revisions.
      keep_replaced_soul`, the same rule the being's own write obeys). The owner may be
      reverting over a hand-edit they made an hour ago and have not thought about; nothing a
      human writes is lost, even when it loses.
    * **The being is TOLD** (``stamp_soul_rewritten`` — the seam that already exists, spec
      §4.1: someone rewriting who you are "is an event in its life, not a version conflict:
      it should be **felt**, not swallowed"). It is stirred by it (``core/affect.py`` reads
      the recency), and the ambient cue tells it, once, in prose. A human quietly rewriting
      the being behind its back is the one thing this command must not become.
    * **The session ends** (ADR-0002), or the owner is told it did not. ``SOUL.md`` is baked
      into the system prompt when a conversation STARTS and reused verbatim after; a revert
      that left the session alone would put the soul on disk and leave the being speaking as
      the one it just replaced, for days.
    * **``born_at=None``.** Putting a soul back is not a birth. An unborn being does not
      become someone because its owner restored a document.

    Serialized under ``state_actor_lock`` for the reason ``write_soul`` and
    ``_reconcile_soul`` are (review C4): the ~60s tick's ``commit`` is an unconditional
    whole-``State`` UPSERT, and a tick that loaded before this stamp would roll it straight
    back out — the being would never learn it had been rewritten.
    """
    lm = composition.build_lifemodel(base_dir=base_dir)
    memory = _memory_of(lm)
    if memory is None:  # pragma: no cover - the live store is always a MemoryPort
        return "error: this store cannot read soul revisions\n"
    history = revisions(memory)
    if not history:
        return (
            "lifemodel soul revert: there are no soul revisions to put back yet.\n"
            "Whatever is in SOUL.md now is kept the moment anything replaces it.\n"
        )
    target, error = _resolve_revision(history, token)
    if target is None:
        return error

    number = history.index(target) + 1
    if soul.sha() == target.sha:
        # Nothing happened: no write, no revision, no "someone rewrote you" for the being to
        # react to, and no session worth ending. Saying otherwise would have the being tell
        # its human about a loss that did not occur (review M5, the whole shape of that bug).
        return (
            "lifemodel soul revert  (mutating)\n"
            + "=" * 30
            + "\n\n"
            + f"  [{number}] is already the soul on disk — nothing changed.\n"
        )

    now = lm.clock.now()
    try:
        written = soul.write(target.text)
    except SoulRejected as exc:
        return (
            "lifemodel soul revert  (mutating)\n"
            + "=" * 30
            + "\n\n"
            + f"  refused to put [{number}] back — SOUL.md is untouched.\n\n"
            + f"  {exc}\n"
        )

    # ── SOUL.md HAS been replaced. Every exit below must be honest about that. ──
    try:
        with state_actor_lock():
            state = lm.state.load()
            replaced = keep_replaced_soul(
                memory,
                new_sha=written.sha,
                replaced_text=written.replaced_text,
                replaced_sha=written.replaced_sha,
                last_written_sha=state.soul_sha,
                unborn=state.genesis_completed_at is None,
                default_soul_text=default_soul_text,
                now=now,
            )
            record_revert(
                memory, sha=written.sha, replaced_sha=written.replaced_sha, now=lm.clock.now()
            )
            _stamp_reverted_soul(lm, state, soul_sha=written.sha, at=to_iso(now))
    except Exception:  # noqa: BLE001 - the file is reverted; never report it as unchanged
        _LOG.exception("soul_revert_wrote_the_file_but_failed_after sha=%s", written.sha)
        return _REVERTED_BUT_DID_NOT_RECORD

    woke = sleep_soft(end_session)
    lines = [
        "lifemodel soul revert  (mutating)",
        "=" * 30,
        "",
        f"  put [{number}] back — {target.author} · {local_time(target.at)} · "
        f"{len(target.text)} chars · {target.sha[:8]}",
        f"      {soul_preview(target.text)}",
        "",
    ]
    # What happened to the soul this just wrote over — ALWAYS said, never left to inference.
    # An owner who has just overwritten the being's current identity wants to know, in the
    # same breath, that they can undo the undo. ``history`` is the lineage as it was BEFORE
    # this write, so "was it already kept?" is answerable without a second query — and the
    # only case that is NOT kept is text nobody authored (Hermes's installer seed, an empty
    # file), which is not a loss and must not be dressed up as one.
    already_kept = any(revision.sha == written.replaced_sha for revision in history)
    newly_kept = replaced in (ReplacedSoul.A_HUMAN_EDIT, ReplacedSoul.SOMEONE_UNKNOWN)
    if already_kept or newly_kept:
        lines.append(
            f"  the soul it replaced ({written.replaced_sha[:8]}) is in the history — put it "
            "back the same way if you change your mind."
        )
    else:
        lines.append(
            "  the soul it replaced was nobody's words (Hermes's own default text), so it is "
            "not kept. Nothing was lost."
        )
    lines.append(
        "  the being will find out: it feels a rewrite it did not make, and will say something\n"
        "  to you about it."
    )
    lines.append(_WILL_WAKE_AS_IT if woke.ok else _STILL_THE_OLD_VOICE)
    return "\n".join(lines) + "\n"


def _stamp_reverted_soul(
    lm: composition.LifeModel, state: State, *, soul_sha: str, at: str
) -> None:
    """Record the revert in the being's own state: the soul it now stands on, and the FACT
    that somebody rewrote it.

    Reaches the concrete store's field-level merges (``stamp_soul`` / ``stamp_soul_rewritten``)
    the same duck-typed way ``hooks._stamp_soul`` and ``being_platform._reconcile_soul`` do —
    so ``StatePort`` stays narrow, and so the ~60s tick's ``u``/``energy``/affect are never
    rolled back by a whole-``State`` commit of a snapshot loaded before this.

    ``stamp_soul`` is what stops the next ``connect()`` reconciling this as a fresh hand-edit
    (``core.genesis.needs_adoption`` compares disk against ``state.soul_sha``) and telling the
    being a SECOND time about the same rewrite. ``stamp_soul_rewritten`` is the telling: it
    sets ``soul_rewritten_at`` (the body is stirred — ``core/affect.py``) and clears
    ``soul_rewrite_told_at`` (the ambient cue says it once, in prose — ``core/felt_display.py``).

    A store without the merges (a minimal ``StatePort`` fake) falls back to one full commit —
    safe here for the same reason it is safe in ``hooks``: the caller holds the state-actor
    lock, so *state* was loaded with no frame in flight and cannot be stale.
    """
    stamp = getattr(lm.state, "stamp_soul", None)
    felt = getattr(lm.state, "stamp_soul_rewritten", None)
    if callable(stamp) and callable(felt):
        stamp(soul_sha=soul_sha, born_at=None)  # never births an unborn being
        felt(at=at)
        return
    lm.state.commit(  # pragma: no cover - the live store always has both merges
        dataclasses.replace(
            state, soul_sha=soul_sha, soul_rewritten_at=at, soul_rewrite_told_at=None
        )
    )
