"""The debug dump — the owner's read-only inspection surface (HLA §12, NFR9).

This module renders a **structured, self-explaining** personality view (spec
``2026-07-05-lifemodel-debug-personality-view-design``): every section shows raw
value(s) + derived quantities + one terse interpretation line, so an owner can
understand what the being is doing without a calculator or reading the engine.

Two invariants shape the design (spec §2):

* **Read-only (HLA §9).** The debug path never commits, never marks signals
  consumed, never writes, never logs. This is made *structurally* true: the
  renderer accepts narrow read-only protocols (:class:`UnprocessedPeek` exposes
  only the non-mutating ``peek_unprocessed``; :class:`EventReader` only ``read``)
  and a pre-computed :class:`~lifemodel.core.introspect.PersonalityReadings`
  snapshot — there is no mutating method and no live decision in reach. The
  readings themselves run the real decision on a deep copy, touching no disk.
* **Privacy (NFR9).** The dump is the owner's own introspection and is *returned*
  to the command caller — never emitted to the shared operator logs. This module
  logs nothing.

Dependencies are injected (DI): :func:`render_debug_dump` takes the readings +
the two read-only readers, and :func:`render_dump_for_dir` wires the concrete
read-only adapters for a profile state dir (through the single composition root).
Stdlib only; imports no Hermes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from .composition import build_lifemodel
from .core.introspect import (
    DebugConfig,
    Readings,
    compute_readings,
)
from .domain.signal import Signal
from .egress_service import PROACTIVE_LOOP_INTERVAL_SEC
from .events import (
    EVENT_ACT_GATE,
    EVENT_DREAM_RUN,
    EVENT_TICK,
    EVENT_WAKE_DECISION,
    EVENTS_FILENAME,
    EventSink,
)
from .sim.wake import WakeOutcome
from .tick import SERVICE_LIVENESS_MAX_AGE

_NA = "n/a"

#: The §12 event categories the AUTONOMOUS LOOP line summarizes (display order).
_LOOP_EVENT_CATEGORIES: tuple[tuple[str, str], ...] = (
    (EVENT_WAKE_DECISION, "wake_decision"),
    (EVENT_ACT_GATE, "act_gate"),
    (EVENT_DREAM_RUN, "dream_run"),
)


class UnprocessedPeek(Protocol):
    """Read-only signal-bus view — the non-mutating ``peek_unprocessed`` only."""

    def peek_unprocessed(self) -> list[Signal]: ...


class EventReader(Protocol):
    """Read-only structured-event source (the :class:`~lifemodel.events.EventSink`)."""

    def read(self, limit: int | None = None) -> list[dict[str, Any]]: ...


def render_debug_dump(
    *,
    readings: PersonalityReadings | None,
    bus: UnprocessedPeek,
    events: EventReader,
    state_error: str | None = None,
) -> str:
    """Render the structured personality dump (read-only, spec §4).

    Every section is defensively read: a broken store yields an
    ``<unreadable: ...>`` banner rather than raising — a debug tool must still
    work when the thing it inspects is broken. When *readings* is ``None`` (the
    state store could not be loaded) the live-value sections collapse to that
    banner; the constant TEMPERAMENT block and the AUTONOMOUS LOOP (bus/events)
    still render, since they do not depend on a readable state.
    """
    lines: list[str] = ["lifemodel debug dump  (read-only)", "=" * 34, ""]

    temp = readings.temperament if readings is not None else temperament()

    if readings is None:
        lines.extend(
            [
                "STATE  (live values — read-only)",
                f"  <unreadable: {state_error or 'state load failed'}>",
                "",
            ]
        )
    else:
        lines.extend(_meta_section(readings))
        lines.append("")
        lines.extend(_drive_section(readings))
        lines.append("")
        lines.extend(_desire_section(readings))
        lines.append("")
        lines.extend(_timing_section(readings))
        lines.append("")

    lines.extend(_temperament_section(temp))
    lines.append("")
    lines.extend(_wake_section(readings))
    lines.append("")
    lines.extend(_loop_section(readings, bus, events))
    return "\n".join(lines) + "\n"


def render_dump_for_dir(base_dir: Path) -> str:
    """Build the graph via the composition root and render the dump (read-only).

    *base_dir* is the profile state dir (``lifemodel.paths.state_dir``). The
    object graph is assembled through the **single** composition root
    (:func:`~lifemodel.composition.build_lifemodel`), so the debug dump always
    reflects the same wiring/defaults the engine runs with — never a divergent
    second graph. The live readings are computed from the loaded ``State`` + the
    clock's ``now`` (the real decision run on a deep copy — read-only, spec §3.1),
    and the collaborators are handed to :func:`render_debug_dump` typed only as
    the narrow read-only protocols, so the mutating methods they also own
    (``commit`` / ``consume_unprocessed``) are structurally out of reach at the
    render surface: debug reads, it never writes (HLA §9).
    """
    lm = build_lifemodel(base_dir=base_dir)
    now = lm.clock.now()

    readings: PersonalityReadings | None
    state_error: str | None = None
    try:
        state = lm.state.load()
        readings = compute_readings(state, now=now)
    except Exception as exc:  # a debug tool must survive a corrupt/unreadable store
        readings = None
        state_error = f"{type(exc).__name__}: {exc}"

    return render_debug_dump(
        readings=readings,
        state_error=state_error,
        # The default bus is the FileSignalBus, whose read-only peek the debug
        # path uses; narrowed to UnprocessedPeek, ``consume`` is out of reach.
        bus=cast(UnprocessedPeek, lm.bus),
        events=EventSink(base_dir / EVENTS_FILENAME),
    )


# --- sections ---------------------------------------------------------------------------------


def _meta_section(r: PersonalityReadings) -> list[str]:
    return [
        "META",
        f"  {'schema_version:':21} {r.schema_version}",
        f"  {'tick_count:':21} {r.tick_count}",
    ]


def _temperament_section(temp: Temperament) -> list[str]:
    theta_over_alpha = temp.theta / temp.alpha  # minutes of silence: 0 → θ
    out = [
        "TEMPERAMENT  (fixed nature — how this being is calibrated)",
        f"  {'wake threshold θ:':21} {_num(temp.theta)}",
        f"  {'loneliness rate α:':21} {_num(temp.alpha)} /min"
        f"   (0 → θ in ~{theta_over_alpha:.0f} min of silence)",
        f"  {'silence window w:':21} {_term(temp.base_params.w)} min"
        "   (won't reach out within w of a real exchange)",
        f"  {'decline backoff:':<21} {_backoff_schedule(temp)}"
        f"   (×{temp.base_params.k:g} per decline, cap {_term(temp.base_params.r_max)} min)",
        f"  {'urge ceiling U_MAX:':21} {_num(temp.u_max)}",
        f"  {'pending-verdict timeout:':21} {_term(temp.pending_timeout_min)} min"
        "   (stale proactive turn recovers as REJECT)",
    ]
    return out


def _drive_section(r: PersonalityReadings) -> list[str]:
    # DRIVE reads the SAME risen urge as WAKE READINESS (must-fix 2): the section
    # is titled "right now", so the headline is the post-decision urge (risen by
    # elapsed silence since last_tick), never the stale persisted value.
    pct = r.u_risen / temp_theta(r) * 100
    out = [
        "DRIVE  (the contact urge right now)",
        f"  {'u:':21} {_num(r.u_risen)}   ({pct:.0f}% of θ)"
        "   ← urge right now, risen as of last_tick",
        f"  {'duration_over_theta:':21} {r.duration_over_theta:.1f} min"
        "   ← time u has sat ≥ θ (tracked; not currently gating) [lm-zhz]",
        f"  {'energy:':21} {r.energy:.1f} (placeholder)"
        "   ← body charge slot; recovery is a later phase",
    ]
    out.append(f"  → {_drive_interpretation(r)}")
    return out


def _desire_section(r: PersonalityReadings) -> list[str]:
    out = [
        "DESIRE LIFECYCLE",
        f"  {'desire_status:':21} {r.desire_status}"
        f"   ← {_desire_status_note(r.desire_status)}  (none → active → deferred)",
        f"  {'pending:':21} {'yes' if r.pending else 'no'}"
        "   ← outstanding proactive turn awaiting a verdict",
        f"  {'decline_count:':21} {r.decline_count}   ← consecutive rejects (grow the backoff)",
        f"  {'declined_at:':21} {_opt(r.declined_at)}",
    ]
    out.append(f"  → {_desire_interpretation(r)}")
    return out


def _desire_status_note(status: str) -> str:
    if status == "active":
        return "woken, awaiting a verdict"
    if status == "deferred":
        return "held for a later release"
    return "no live desire"


def _timing_section(r: PersonalityReadings) -> list[str]:
    return [
        "TIMING",
        _timing_row(
            "last_exchange:",
            r.last_exchange_at,
            r.last_exchange_ago_min,
            "last real exchange (satiates u)",
        ),
        _timing_row(
            "last_contact:",
            r.last_contact_at,
            r.last_contact_ago_min,
            "last time IT reached out (outbound bookkeeping)",
        ),
        _timing_row("last_tick:", r.last_tick_at, r.last_tick_ago_min, "heartbeat alive"),
    ]


def _wake_section(r: PersonalityReadings | None) -> list[str]:
    if r is None:
        return ["WAKE READINESS", f"  {_NA}  (state unreadable)"]
    theta = temp_theta(r)
    pct = r.u_risen / theta * 100
    out = [
        "WAKE READINESS  (what the next heartbeat would decide — run on a copy, read-only)",
    ]
    # urge now (risen) + distance to θ
    if r.risen_over_theta:
        urge_tail = f"   ({pct:.0f}% of θ={theta:g})   (over θ)"
    else:
        urge_tail = f"   ({pct:.0f}% of θ={theta:g})   [+{_fmt_until(r.time_to_theta_min)} → θ]"
    out.append(f"  {'urge now (risen):':21} {_num(r.u_risen)}{urge_tail}")
    out.append(f"  {'gate verdict:':21} {r.gate_verdict}   (assuming in_flight=false)")
    if r.risen_over_theta:
        out.append(
            "  ⚠ in_flight is runtime-only: if a turn is executing now,"
            " actual verdict = no_wake_in_flight"
        )
    out.append(f"  {'would launch outreach:':21} {_launch_label(r)}")
    out.append(f"  {'stale-pending recovery:':21} {_stale_label(r)}")
    out.append("  gate ladder (precedence):")
    for rung in r.gate_ladder:
        out.append(f"    {rung.name:<18} {rung.status:<13} {rung.detail}")
    return out


def _loop_section(
    r: PersonalityReadings | None, bus: UnprocessedPeek, events: EventReader
) -> list[str]:
    out = ["AUTONOMOUS LOOP"]
    # signal bus (read-only peek)
    unprocessed: str
    try:
        pending = bus.peek_unprocessed()
        unprocessed = str(len(pending))
        recent = ", ".join(f"{s.kind}({s.origin_id})" for s in pending[-3:]) if pending else ""
    except Exception as exc:  # a debug tool must survive a broken bus
        unprocessed = f"<unreadable: {type(exc).__name__}>"
        recent = ""
    out.append(f"  {'signal bus unprocessed:':27} {unprocessed}")
    if recent:
        out.append(f"  {'recent:':27} {recent}")

    # in-process egress service liveness (alive-bool computed against the imported
    # SERVICE_LIVENESS_MAX_AGE — spec §3.3; the raw stamp + ago come from readings).
    out.append(f"  {'in-proc egress service:':27} {_egress_label(r)}")
    # the loop cadence — imported from its owner so it can never drift (must-fix 4).
    out.append(f"  {'proactive loop interval:':27} {PROACTIVE_LOOP_INTERVAL_SEC:g} s")

    # structured events (read-only)
    try:
        records = events.read()
        latest = _latest_by_event(records)
        tick_record_error: str | None = None
    except Exception as exc:  # a debug tool must survive a torn events file
        latest = {}
        tick_record_error = f"<unreadable: {type(exc).__name__}>"

    if tick_record_error is not None:
        out.append(f"  {'last tick outcome:':27} {tick_record_error}")
    else:
        out.append(f"  {'last tick outcome:':27} {_fmt_event(latest.get(EVENT_TICK)) or _NA}")
    summary = " · ".join(
        f"{label} {_compact_event(latest.get(name))}" for name, label in _LOOP_EVENT_CATEGORIES
    )
    out.append(f"  {'events:':27} {summary}")
    out.append(f"  {'lock status:':27} {_NA}  (no lock held in Phase 1)")
    return out


# --- per-section helpers ----------------------------------------------------------------------


def _drive_interpretation(r: PersonalityReadings) -> str:
    if r.risen_over_theta:
        return "urge above threshold (u ≥ θ); a pull to reach out is present"
    return (
        f"calm & satiated; needs {_fmt_until(r.time_to_theta_min)}"
        " of continued silence to feel a pull"
    )


def _desire_interpretation(r: PersonalityReadings) -> str:
    # The lifecycle reflects the post-decision snapshot, so a recovery that fired
    # this eval already shows up as (none + a fresh decline): explain both halves.
    if r.stale_pending_recovery:
        return (
            f"recovered a stale pending as REJECT ({_fmt_ago(r.stale_pending_age_min)}); "
            f"in decline backoff ({_fmt_until(r.backoff_remaining_min)} left)"
        )
    if r.desire_status == "active":
        return "woken this eval; awaiting a verdict"
    if r.pending:
        return "a proactive turn is pending; awaiting its post-llm_call verdict"
    if r.backoff_remaining_min is not None:
        return f"in decline backoff ({_fmt_until(r.backoff_remaining_min)} left)"
    return "nothing pending; not in backoff"


def _timing_row(label: str, iso: str | None, ago: float | None, note: str) -> str:
    if iso is None:
        return f"  {label:<21} {_NA}   ← {note}"
    return f"  {label:<21} {iso}   ({_fmt_ago(ago)})   ← {note}"


def _launch_label(r: PersonalityReadings) -> str:
    if r.would_launch:
        return "yes   (gate=URGE AND new-desire allowed)"
    # Not launching: either the gate itself did not reach URGE, or it did but a
    # desire was already live (Aggregator dedup — the anti-drum guarantee).
    if r.gate_verdict == WakeOutcome.URGE.value:
        return "no    (desire already active → Aggregator dedup)"
    return f"no    (gate did not reach URGE: {r.gate_verdict})"


def _stale_label(r: PersonalityReadings) -> str:
    if r.stale_pending_recovery:
        age = r.stale_pending_age_min
        return f"fires: pending {_fmt_ago(age)} → recover as REJECT"
    return "none this eval"


def _egress_label(r: PersonalityReadings | None) -> str:
    if r is None or r.egress_service_alive_at is None or r.egress_service_ago_min is None:
        return f"{_NA}  (no liveness stamp)"
    max_age_min = SERVICE_LIVENESS_MAX_AGE.total_seconds() / 60.0
    alive = r.egress_service_ago_min <= max_age_min
    state_word = "alive" if alive else "stale"
    return f"{state_word}  (stamp {_fmt_ago(r.egress_service_ago_min)})"


def _latest_by_event(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        name = record.get("event")
        if isinstance(name, str):
            latest[name] = record  # later record wins → most recent per category
    return latest


def _fmt_event(record: dict[str, Any] | None) -> str:
    """One-line summary of an event's fields (``event`` key elided), or ``""``."""
    if record is None:
        return ""
    parts = [f"{key}={value}" for key, value in record.items() if key != "event"]
    if not parts:
        return ""
    text = " ".join(parts)
    return text if len(text) <= 200 else f"{text[:197]}..."


def _compact_event(record: dict[str, Any] | None) -> str:
    """A one-word status for the AUTONOMOUS LOOP events summary, or ``n/a``."""
    if record is None:
        return _NA
    for key in ("outcome", "reason", "wake", "deferred"):
        if key in record:
            return str(record[key])
    return "recorded"


def _backoff_schedule(temp: Temperament) -> str:
    terms = temp.backoff_schedule
    shown = " → ".join(_term(t) for t in terms[:3])
    if len(terms) > 3:
        shown += " …"
    return shown


def _opt(value: str | None) -> str:
    return _NA if value is None else value


def temp_theta(r: PersonalityReadings) -> float:
    """θ from the readings' temperament (single source, no restated constant)."""
    return r.temperament.theta


def _num(value: float) -> str:
    """Compact general-purpose number formatting (≤4 sig figs)."""
    return f"{value:.4g}"


def _term(value: float) -> str:
    """A schedule term: whole numbers render as ints (``30`` not ``30.0``)."""
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


def _fmt_ago(minutes: float | None) -> str:
    if minutes is None:
        return _NA
    if minutes < 60.0:
        return f"{minutes:.1f} min ago"
    return f"{minutes / 60.0:.1f} h ago"


def _fmt_until(minutes: float | None) -> str:
    if minutes is None:
        return _NA
    if minutes < 60.0:
        return f"~{minutes:.0f} min"
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))
    return f"~{hours}h {mins:02d}m"
