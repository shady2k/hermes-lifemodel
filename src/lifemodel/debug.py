"""The debug dump — the owner's read-only inspection surface (HLA §12, NFR9).

HLA §12 fixes the *minimum inspection set* a developer needs to see the engine
working: current state (pressures/energy/timestamps), the signal bus backlog,
and the last of each structured decision event (tick, wake, act-gate, dream)
plus lock status. This module renders that as a human-readable dump.

Two invariants shape the design:

* **Read-only (HLA §9).** The debug path never commits, never marks signals
  consumed, never writes. This is made *structurally* true: the renderer accepts
  narrow read-only protocols (:class:`StateReader` exposes only ``load``;
  :class:`UnprocessedPeek` only the non-mutating ``peek_unprocessed``;
  :class:`EventReader` only ``read``), so there is no mutating method in reach.
* **Privacy (NFR9).** The dump is the owner's own introspection and is *returned*
  to the command caller — it is never emitted to the shared operator logs, so
  private soul content cannot leak there. This module logs nothing.

Dependencies are injected (DI): :func:`render_debug_dump` takes the three
readers, and :func:`render_dump_for_dir` wires the concrete read-only adapters
for a profile state dir. Stdlib only; imports no Hermes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from .composition import build_lifemodel
from .domain.signal import Signal
from .events import (
    EVENT_ACT_GATE,
    EVENT_DREAM_RUN,
    EVENT_TICK,
    EVENT_WAKE_DECISION,
    EVENTS_FILENAME,
    EventSink,
)
from .state.model import State

_NA = "n/a"

#: The §12 event categories, in display order, paired with their label.
_EVENT_CATEGORIES: tuple[tuple[str, str], ...] = (
    (EVENT_TICK, "last tick"),
    (EVENT_WAKE_DECISION, "last wake_decision"),
    (EVENT_ACT_GATE, "last act_gate"),
    (EVENT_DREAM_RUN, "last dream_run"),
)


class StateReader(Protocol):
    """Read-only slice of :class:`~lifemodel.state.port.StatePort` — only ``load``."""

    def load(self) -> State: ...


class UnprocessedPeek(Protocol):
    """Read-only signal-bus view — the non-mutating ``peek_unprocessed`` only."""

    def peek_unprocessed(self) -> list[Signal]: ...


class EventReader(Protocol):
    """Read-only structured-event source (the :class:`~lifemodel.events.EventSink`)."""

    def read(self, limit: int | None = None) -> list[dict[str, Any]]: ...


def render_debug_dump(
    *,
    state: StateReader,
    bus: UnprocessedPeek,
    events: EventReader,
) -> str:
    """Render the HLA §12 inspection set as a human-readable dump (read-only).

    Every section is defensively read: a broken store or a torn event line
    yields an ``<unreadable: ...>`` note rather than raising — a debug tool must
    still work when the thing it inspects is broken. Categories with nothing yet
    produced in Phase 1 render as ``n/a``.
    """
    lines: list[str] = ["lifemodel debug dump  (read-only)", "=" * 34, ""]
    lines.extend(_state_section(state))
    lines.append("")
    lines.extend(_bus_section(bus))
    lines.append("")
    lines.extend(_events_section(events))
    lines.append("")
    lines.append(f"  {'lock status:':21} {_NA}  (no lock held in Phase 1; HLA §9)")
    return "\n".join(lines) + "\n"


def render_dump_for_dir(base_dir: Path) -> str:
    """Build the graph via the composition root and render the dump (read-only).

    *base_dir* is the profile state dir (``lifemodel.paths.state_dir``). The
    object graph is assembled through the **single** composition root
    (:func:`~lifemodel.composition.build_lifemodel`), so the debug dump always
    reflects the same wiring/defaults the engine runs with — never a divergent
    second graph. The collaborators are then handed to :func:`render_debug_dump`
    typed only as the narrow read-only protocols, so the mutating methods they
    also own (``commit`` / ``consume_unprocessed``) are structurally out of reach
    at the render surface: debug reads, it never writes (HLA §9).
    """
    lm = build_lifemodel(base_dir=base_dir)
    return render_debug_dump(
        # ``lm.state`` is a StatePort; narrowed to StateReader only ``load`` shows.
        state=lm.state,
        # The default bus is the FileSignalBus, whose read-only peek the debug
        # path uses; narrowed to UnprocessedPeek, ``consume`` is out of reach.
        bus=cast(UnprocessedPeek, lm.bus),
        events=EventSink(base_dir / EVENTS_FILENAME),
    )


def _state_section(state: StateReader) -> list[str]:
    out = ["state (StatePort.load, read-only):"]
    try:
        current = state.load()
    except Exception as exc:  # a debug tool must survive a corrupt/unreadable store
        out.append(f"  <unreadable: {type(exc).__name__}: {exc}>")
        return out
    out.append(f"  {'schema_version:':21} {current.schema_version}")
    out.append(f"  {'u:':21} {current.u}")
    out.append(f"  {'duration_over_theta:':21} {current.duration_over_theta}")
    out.append(f"  {'desire_status:':21} {current.desire_status}")
    out.append(f"  {'energy:':21} {current.energy}")
    out.append(f"  {'last_tick_at:':21} {_opt(current.last_tick_at)}")
    out.append(f"  {'last_contact_at:':21} {_opt(current.last_contact_at)}")
    out.append(f"  {'last_exchange_at:':21} {_opt(current.last_exchange_at)}")
    out.append(f"  {'decline_count:':21} {current.decline_count}")
    return out


def _bus_section(bus: UnprocessedPeek) -> list[str]:
    out = ["signal bus (peek, read-only):"]
    try:
        pending = bus.peek_unprocessed()
    except Exception as exc:
        out.append(f"  <unreadable: {type(exc).__name__}: {exc}>")
        return out
    out.append(f"  {'unprocessed:':21} {len(pending)}")
    recent = ", ".join(f"{s.kind}({s.origin_id})" for s in pending[-3:]) if pending else _NA
    out.append(f"  {'recent:':21} {recent}")
    return out


def _events_section(events: EventReader) -> list[str]:
    out = ["events (last of each; source: events.jsonl):"]
    try:
        records = events.read()
    except Exception as exc:
        out.append(f"  <unreadable: {type(exc).__name__}: {exc}>")
        return out
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        name = record.get("event")
        if isinstance(name, str):
            latest[name] = record  # later record wins → most recent per category
    for name, label in _EVENT_CATEGORIES:
        out.append(f"  {label + ':':21} {_fmt_event(latest.get(name))}")
    return out


def _fmt_event(record: dict[str, Any] | None) -> str:
    """One-line summary of an event's fields (``event`` key elided), or ``n/a``."""
    if record is None:
        return _NA
    parts = [f"{key}={value}" for key, value in record.items() if key != "event"]
    if not parts:
        return "(no fields)"
    text = " ".join(parts)
    return text if len(text) <= 200 else f"{text[:197]}..."


def _opt(value: str | None) -> str:
    return _NA if value is None else value
