"""hermes-lifemodel — a personality-and-decision layer plugin for Hermes.

This module is the **adapter** between Hermes and the plugin: it is the only
place that touches the host ``ctx`` and the host ``get_hermes_home()`` API. All
reusable logic lives in Hermes-free submodules (:mod:`lifemodel.paths`,
:mod:`lifemodel.logging`), which stay importable and unit-testable in isolation.

MVP skeleton (task 0.1): prove the plugin loads and is per-profile aware. It
registers one introspection command and emits a ``plugin_registered`` event.
No engine/neurons yet — those land in later tasks (see docs/roadmap.md §0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .debug import render_dump_for_dir
from .events import EVENTS_FILENAME, EventSink
from .heartbeat import register_heartbeat
from .logging import EventTee, get_logger
from .paths import state_dir

__version__ = "0.0.0"


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
        """`/lifemodel` — 'status' (default) or 'debug' (read-only inspection)."""
        if raw_args.strip() == "debug":
            # Owner introspection (NFR9): returned to the caller, never logged,
            # and read-only (HLA §9) — no commit, no bus mutation.
            return render_dump_for_dir(sdir)
        return _status_line(profile, sdir)

    ctx.register_command(
        "lifemodel",
        lifemodel_command,
        description="Show plugin status, or 'debug' for a read-only state/event dump.",
        args_hint="status | debug",
    )

    logger.info(
        "plugin_registered",
        plugin="lifemodel",
        version=__version__,
        profile=profile,
        state_dir=str(sdir),
    )

    # Register the ~1-minute heartbeat cron (roadmap 1.1, HLA D1). Best-effort:
    # a host without the cron API (or any registration hiccup) must not break
    # plugin load, and the registration is idempotent so repeated loads never
    # duplicate the job. ``src`` = the plugin package's parent, added to the
    # launcher shim's ``sys.path`` so Hermes' interpreter can import us.
    src_dir = Path(__file__).resolve().parent.parent
    try:
        register_heartbeat(home, src_dir, logger=logger)
    except Exception as exc:  # noqa: BLE001 - registration is best-effort (see above)
        logger.info("heartbeat_registration_skipped", error=f"{type(exc).__name__}: {exc}")
