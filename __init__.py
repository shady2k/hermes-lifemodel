"""hermes-lifemodel — a personality-and-decision layer plugin for Hermes.

This module is the **adapter** between Hermes and the plugin: it is the only
place that touches the host ``ctx`` and the host ``get_hermes_home()`` API. All
reusable logic lives in Hermes-free submodules (:mod:`lifemodel.paths`,
:mod:`lifemodel.log`), which stay importable and unit-testable in isolation.

MVP skeleton (task 0.1): prove the plugin loads and is per-profile aware. It
registers one introspection command and emits a ``plugin_registered`` event.
No engine/neurons yet — those land in later tasks (see docs/roadmap.md §0).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from .adapters.origin import resolve_home_origin
from .composition import build_lifemodel
from .config import read_log_level, set_log_level_for_dir
from .debug import render_dump_for_dir
from .events import EventRing
from .hooks import make_inbound_observer, make_post_llm_observer
from .log import apply_log_level, parse_log_level
from .paths import state_dir
from .state.trace_store import acquire_trace_writer, observability_db_path
from .state_commands import (
    force_wake_for_dir,
    nudge_for_dir,
    reset_for_dir,
    satiate_for_dir,
    set_field_for_dir,
    set_user_model_prefs_for_dir,
    think_for_dir,
    why_for_dir,
)
from .stats_view import stats_for_dir
from .trace_view import trace_for_dir

__version__ = "0.0.0"

#: Registration/lifecycle/boundary events with no ambient span go through the
#: native stdlib logger (spec §4.5 allowlist); Hermes routes ``lifemodel.*`` into
#: ``agent.log``. Tick-path events go through SpanLogger, never this.
_LOG = logging.getLogger("lifemodel")


class _Subcommand(NamedTuple):
    """One ``_SUBCOMMANDS`` entry: its help text, plus whether it WRITES to
    the being's persisted state (owner-facing mutation, via the ``StatePort``
    store) or is read-only. ``help``/the bare view render ``[mutating]`` for
    the former so the owner never confuses a status peek with a state
    change."""

    description: str
    mutating: bool = False


# Single source of truth for `/lifemodel` subcommands: dispatch, the `help`
# text, and args_hint are all derived from this dict, so adding a subcommand
# means editing here — nowhere else.
_SUBCOMMANDS: dict[str, _Subcommand] = {
    "status": _Subcommand("Show the one-line plugin status (default)."),
    "debug": _Subcommand("Read-only state/event dump for owner introspection."),
    "why": _Subcommand(
        "why [desire|intention|write|<kind>:<id>] — trace the causal chain behind a "
        "desire/intention (read-only)."
    ),
    "trace": _Subcommand(
        "trace [<trace_id> | last [N]] — render a durable execution trace "
        "(tick → components → decisions → launch → async outcome), read-only."
    ),
    "stats": _Subcommand(
        "stats [last N] — operational telemetry: live tick/writer/counters (NOW) "
        "+ rates & approx p95 from metrics.sqlite over the last N samples (WINDOW), "
        "read-only."
    ),
    "help": _Subcommand("List these subcommands."),
    "nudge": _Subcommand(
        "nudge [N] — bump the contact drive: u += N (default +1.0).", mutating=True
    ),
    "force-wake": _Subcommand(
        "Set state so the NEXT real adapter tick wakes (satisfies every wake gate).",
        mutating=True,
    ),
    "satiate": _Subcommand(
        "Simulate a fulfilled contact: u->0, clocks reset, desire cleared.", mutating=True
    ),
    "reset": _Subcommand(
        "Factory wipe: write a fresh State() AND delete every memory record "
        "(thoughts/desires/intentions/user-models) — truly as if newly born.",
        mutating=True,
    ),
    "set": _Subcommand("set <field> <value> — write one whitelisted state field.", mutating=True),
    "user-model": _Subcommand(
        "user-model <key>=<value> ... — set owner norms (bad-hours, cadence, "
        "privacy, topics, styles, ...) that gate proactive contact.",
        mutating=True,
    ),
    "think": _Subcommand(
        "think <content> — seed a thought (active) the being turns over and "
        "surfaces in its proactive prompt.",
        mutating=True,
    ),
    "loglevel": _Subcommand(
        "loglevel [level] — show the current log level, or set it "
        "(debug|info|warning|error|critical); persists across restarts.",
        mutating=True,
    ),
}


def _hermes_home() -> Path:
    """Resolve the active Hermes profile home via the host API.

    This is the plugin's only Hermes touchpoint besides ``ctx``. The import is
    lazy so :mod:`lifemodel` stays importable — and :func:`register` stays
    unit-testable — without Hermes on ``sys.path``; tests override this seam to
    inject a profile home. See HLA §3/§4: our state anchors on
    ``get_hermes_home()`` (= the active profile home).
    """
    from hermes_constants import get_hermes_home

    # get_hermes_home() already returns a Path; re-wrap so the host module's
    # untyped (Any) return narrows to Path for the strict type checker.
    return Path(get_hermes_home())


def _status_line(profile: str, sdir: Path) -> str:
    """One-line 'alive' summary printed by the introspection command."""
    return f"lifemodel {__version__} alive · profile={profile} · state_dir={sdir}"


def _command_list() -> str:
    """Render every registered subcommand with its one-line description.

    Shared by the bare `/lifemodel` view and `/lifemodel help` so the two can
    never drift — both read straight from :data:`_SUBCOMMANDS`. One command
    per line, ``**name** — description`` (bold name via standard markdown
    ``**...**``, an em-dash separator, plain description) — no
    column-alignment padding, matching the plain-line/bold-label house style
    ``debug.py``'s ``_metrics`` uses for ``/lifemodel debug`` (Telegram's
    proportional font makes space-padded columns go ragged). Mutating
    subcommands keep the ``[mutating]`` marker so the owner can tell a status
    peek from a state change at a glance.
    """
    lines = [
        f"**{name}** — {'[mutating] ' if info.mutating else ''}{info.description}"
        for name, info in _SUBCOMMANDS.items()
    ]
    return "\n".join(["**commands:**", *lines])


def register(ctx: Any) -> None:
    """Register the plugin's surface with Hermes (the adapter boundary).

    ``ctx`` is duck-typed: only ``profile_name`` and ``register_command`` are
    used, so a fake ctx exercises this without importing Hermes. Kept thin —
    all logic lives in the Hermes-free submodules.
    """
    profile = str(getattr(ctx, "profile_name", "default") or "default")
    home = _hermes_home()
    sdir = state_dir(home)

    # Boot at the persisted log level (lm-j2w B2) — defaults to 'info' when no
    # config.json exists yet (a fresh being). read_log_level() is itself
    # safe-by-construction (falls back to the default on a missing/malformed
    # config or an invalid persisted name), so parse_log_level() should never
    # raise here; this try/except is pure defense in depth — NO failure while
    # resolving the boot level may ever abort register() and take the plugin
    # down (degrade to logging.INFO and keep booting). apply_log_level() only
    # setLevel()s the ``lifemodel`` logger (Hermes owns handler setup), so it is
    # idempotent and safe to call unconditionally on every register().
    try:
        desired_level = parse_log_level(read_log_level(sdir))
    except Exception:
        desired_level = logging.INFO
    apply_log_level(desired_level)

    def lifemodel_command(raw_args: str = "") -> str:
        """`/lifemodel` — 'status' (default), 'debug', 'help', or a mutating
        subcommand (nudge/force-wake/satiate/reset/set — see _SUBCOMMANDS)."""
        parts = raw_args.strip().split(None, 1)
        sub = parts[0] if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        # Every subcommand handler funnels through this ONE dispatch table so
        # a bug in ANY of them (read-only or mutating) is caught at the SAME
        # call site below — never left to propagate out of this command
        # callback. Left uncaught, Hermes' gateway degrades a handler
        # exception into a misleading generic "Unknown command /lifemodel"
        # notice instead of the real reason (lm-zhh; confirmed incident: a
        # StateSchemaError from `set` surfaced to the owner as "unknown
        # command"). Mutating subcommands all go through the SAME StatePort
        # store (SQLiteRuntimeStore, lm-fib.6.2) the adapter loop uses (via the
        # composition root), never a hand-edited file and never a synchronous
        # tick — see lifemodel.state_commands for the gate-satisfaction
        # rationale; "debug"/"why" are read-only introspection (NFR9, HLA §9).
        dispatch: dict[str, Callable[[], str]] = {
            "debug": lambda: render_dump_for_dir(sdir),
            "why": lambda: why_for_dir(sdir, rest),
            "trace": lambda: trace_for_dir(sdir, rest),
            "stats": lambda: stats_for_dir(sdir, rest),
            "help": lambda: _command_list() + "\n",
            "nudge": lambda: nudge_for_dir(sdir, rest),
            "force-wake": lambda: force_wake_for_dir(sdir),
            "satiate": lambda: satiate_for_dir(sdir),
            "reset": lambda: reset_for_dir(sdir),
            "set": lambda: set_field_for_dir(sdir, rest),
            "user-model": lambda: set_user_model_prefs_for_dir(sdir, rest),
            "think": lambda: think_for_dir(sdir, rest),
            "loglevel": lambda: set_log_level_for_dir(sdir, rest),
        }
        handler = dispatch.get(sub)
        if handler is not None:
            try:
                return handler()
            except Exception as exc:  # noqa: BLE001 - command boundary: an owner
                # typing a command must NEVER see "unknown command" for a
                # handler bug (lm-zhh) — log it (with traceback) and surface
                # the real reason, one readable owner-facing block.
                _LOG.info(
                    "lifemodel_command_failed subcommand=%s error=%s",
                    sub,
                    f"{type(exc).__name__}: {exc}",
                    exc_info=True,
                )
                return f"lifemodel: command failed — {exc}\n"

        status = _status_line(profile, sdir)
        if sub == "":
            # Bare invocation: keep the status line, then surface the full
            # subcommand list (discoverability — no separate `help` round trip
            # needed just to learn what's available).
            return f"{status}\n\n{_command_list()}\n"
        return status

    ctx.register_command(
        "lifemodel",
        lifemodel_command,
        description="Show plugin status; 'help' lists read-only and mutating subcommands.",
        args_hint=" | ".join(_SUBCOMMANDS),
    )

    _LOG.info(
        "plugin_registered plugin=lifemodel version=%s profile=%s state_dir=%s",
        __version__,
        profile,
        str(sdir),
    )

    # --- Verdict feedback wiring (Task 5, spec §5/§7) -------------------------
    # Resolves the pending proactive desire from the FINAL LLM output
    # (NO_REPLY -> reject + growing backoff, real text -> fulfill) via the
    # post_llm_call lifecycle hook — this is the anti-drum guarantee: a wake
    # that produces nothing genuine to say never queues a duplicate reach-out.
    # See lifemodel.hooks for the SPIKE findings (real payload shape) and the
    # correlation-needs-field-verification caveat. Best-effort: a host without
    # post_llm_call in VALID_HOOKS, or any wiring hiccup, must not break load.
    try:
        # The async read-back MUST reach the LIVE durable trace writer (spec §4.4):
        # ``acquire_trace_writer`` returns the SAME singleton-per-db-path instance the
        # ``BeingAdapter.connect()`` tick loop acquires (both resolve
        # ``observability_db_path(sdir)`` to one registry key), so the outcome span the
        # hook writes lands in the SAME ``observability.sqlite`` as the launch — one
        # attempt, one ``trace_id``. This refcount is held for the plugin's lifetime
        # (register has no teardown), independent of the platform connect/disconnect
        # cycle, so the hook can always write regardless of loop state.
        # Acquire the durable trace writer ONCE (refcount held for the plugin's life)
        # and share one freshness ring; the observer builds a FRESH LifeModel per call
        # (spec §3: each frame loads state fresh under the one state-actor lock) but
        # keeps writing to the SAME observability.sqlite as the launch — one attempt,
        # one trace_id.
        _outcome_writer = acquire_trace_writer(observability_db_path(sdir))
        _outcome_ring = EventRing()
        ctx.register_hook(
            "post_llm_call",
            make_post_llm_observer(
                lambda: build_lifemodel(
                    base_dir=sdir, trace_writer=_outcome_writer, event_ring=_outcome_ring
                )
            ),
        )
        _LOG.info("post_llm_observer_registered")
    except Exception as exc:  # noqa: BLE001 - best-effort; never break load
        _LOG.info("post_llm_observer_registration_skipped error=%s", f"{type(exc).__name__}: {exc}")

    # --- Inbound observation wiring (Task 6, spec §4/§6) ----------------------
    # RC1: the being is currently deaf to inbound user messages. On a genuine
    # user message, satiate the drive + stamp last_exchange_at + clear the
    # reject record + resolve any live desire, so silence resets on real
    # contact. Wired on pre_gateway_dispatch (SPIKE, see lifemodel.hooks module
    # docstring): it fires once per incoming MessageEvent, and the host itself
    # never invokes it for our own injected proactive impulse (internal=True
    # skips the hook entirely at the call site) — disjoint from the
    # post_llm_call verdict path by host guarantee, reinforced defensively in
    # the observer via the impulse-label prefix. Best-effort: a host without
    # pre_gateway_dispatch in VALID_HOOKS, or any wiring hiccup, must not break
    # load.
    try:
        ctx.register_hook(
            "pre_gateway_dispatch",
            make_inbound_observer(lambda: build_lifemodel(base_dir=sdir)),
        )
        _LOG.info("inbound_observer_registered")
    except Exception as exc:  # noqa: BLE001 - best-effort; never break load
        _LOG.info("inbound_observer_registration_skipped error=%s", f"{type(exc).__name__}: {exc}")

    # --- Proactive brain wiring (the being as a gateway platform) -------------
    # The autonomic brain is hosted as a gateway-supervised platform adapter: its
    # connect() runs the tick loop, and the gateway's reconnect watcher restarts
    # it on failure (no self-spawned task, no cron fallback, no loop-timing luck).
    # The import is lazy + best-effort: the adapter subclasses ``BasePlatformAdapter``
    # (a top-level ``gateway`` import), so importing it off-host would fail — a
    # failure here must only skip the registration, never break plugin load.
    try:
        from .adapters.being_platform import register_being_platform

        register_being_platform(ctx, base_dir=sdir, target=resolve_home_origin())
        _LOG.info("being_platform_registered")
    except Exception as exc:  # noqa: BLE001 - best-effort; never break load
        _LOG.info("being_platform_registration_skipped error=%s", f"{type(exc).__name__}: {exc}")
