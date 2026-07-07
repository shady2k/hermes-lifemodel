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

from pathlib import Path
from typing import Any

from .adapters.origin import resolve_home_origin
from .composition import build_lifemodel
from .debug import render_dump_for_dir
from .events import EVENTS_FILENAME, EventSink
from .hooks import make_inbound_observer, make_post_llm_observer
from .log import EventTee, get_logger
from .paths import state_dir

__version__ = "0.0.0"

# Single source of truth for `/lifemodel` subcommands: dispatch, the `help`
# text, and args_hint are all derived from this dict, so adding a subcommand
# means editing here — nowhere else.
_SUBCOMMANDS: dict[str, str] = {
    "status": "Show the one-line plugin status (default).",
    "debug": "Read-only state/event dump for owner introspection.",
    "help": "List these subcommands.",
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
    never drift — both read straight from :data:`_SUBCOMMANDS`.
    """
    width = max(len(name) for name in _SUBCOMMANDS)
    lines = [f"  {name:<{width}}  {description}" for name, description in _SUBCOMMANDS.items()]
    return "\n".join(["commands:", *lines])


def register(ctx: Any) -> None:
    """Register the plugin's surface with Hermes (the adapter boundary).

    ``ctx`` is duck-typed: only ``profile_name`` and ``register_command`` are
    used, so a fake ctx exercises this without importing Hermes. Kept thin —
    all logic lives in the Hermes-free submodules.
    """
    profile = str(getattr(ctx, "profile_name", "default") or "default")
    home = _hermes_home()
    sdir = state_dir(home)

    # Tee structured events into a bounded on-disk sink so the debug command can
    # query them (HLA §12/§13) instead of scraping operator logs.
    sink = EventSink(sdir / EVENTS_FILENAME)
    logger = EventTee(get_logger("lifemodel"), sink)

    def lifemodel_command(raw_args: str = "") -> str:
        """`/lifemodel` — 'status' (default), 'debug', or 'help' (see _SUBCOMMANDS)."""
        sub = raw_args.strip()
        if sub == "debug":
            # Owner introspection (NFR9): returned to the caller, never logged,
            # and read-only (HLA §9) — no commit, no bus mutation.
            return render_dump_for_dir(sdir)
        if sub == "help":
            return _command_list() + "\n"
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
        description="Show plugin status, or 'debug'/'help' for a dump or subcommand list.",
        args_hint=" | ".join(_SUBCOMMANDS),
    )

    logger.info(
        "plugin_registered",
        plugin="lifemodel",
        version=__version__,
        profile=profile,
        state_dir=str(sdir),
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
        verdict_lm = build_lifemodel(base_dir=sdir, logger=logger)
        ctx.register_hook("post_llm_call", make_post_llm_observer(verdict_lm))
        logger.info("post_llm_observer_registered")
    except Exception as exc:  # noqa: BLE001 - best-effort; never break load
        logger.info("post_llm_observer_registration_skipped", error=f"{type(exc).__name__}: {exc}")

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
        inbound_lm = build_lifemodel(base_dir=sdir, logger=logger)
        ctx.register_hook("pre_gateway_dispatch", make_inbound_observer(inbound_lm))
        logger.info("inbound_observer_registered")
    except Exception as exc:  # noqa: BLE001 - best-effort; never break load
        logger.info("inbound_observer_registration_skipped", error=f"{type(exc).__name__}: {exc}")

    # --- Proactive brain wiring (the being as a gateway platform) -------------
    # The autonomic brain is hosted as a gateway-supervised platform adapter: its
    # connect() runs the tick loop, and the gateway's reconnect watcher restarts
    # it on failure (no self-spawned task, no cron fallback, no loop-timing luck).
    # The import is lazy + best-effort: the adapter subclasses ``BasePlatformAdapter``
    # (a top-level ``gateway`` import), so importing it off-host would fail — a
    # failure here must only skip the registration, never break plugin load.
    try:
        from .adapters.being_platform import register_being_platform

        register_being_platform(ctx, base_dir=sdir, target=resolve_home_origin(), logger=logger)
        logger.info("being_platform_registered")
    except Exception as exc:  # noqa: BLE001 - best-effort; never break load
        logger.info("being_platform_registration_skipped", error=f"{type(exc).__name__}: {exc}")
