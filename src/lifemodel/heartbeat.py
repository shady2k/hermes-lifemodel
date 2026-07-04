"""Idempotent Hermes cron registration for the heartbeat (roadmap 1.1, HLA D1).

``register(ctx)`` runs on **every** Hermes start / plugin load, so registering
the cron job unconditionally would pile up duplicate ``lifemodel-heartbeat``
jobs. This module owns that registration and makes it **idempotent**: it checks
for an existing job by stable name and creates one only if absent, while always
refreshing the launcher shim so a moved plugin still resolves.

**The launcher shim.** Hermes runs a cron ``--script`` with ``sys.executable``
(its own interpreter) and resolves the script path under
``<HERMES_HOME>/scripts/`` (``cron/scheduler.py:_run_job_script``). The
``lifemodel`` package is *not* on Hermes' ``sys.path`` there, so we drop a tiny
generated shim into that scripts dir that inserts the plugin's ``src`` dir onto
``sys.path`` and then runs :func:`lifemodel.tick.main`. The shim is the only
supported way to bridge "Hermes runs the file" and "the file imports our
package" without vendoring anything into Hermes' venv.

**Dependency injection.** The Hermes cron API (``create_job`` / ``list_jobs``)
is passed in as plain callables, so the idempotency logic is unit-tested with
fakes and never imports Hermes; :func:`register_heartbeat` is the thin real-host
wrapper that imports ``cron.jobs`` and delegates. Stdlib only in the tested core.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .logging import EventLogger, get_logger

#: Stable job name — the idempotency key. One heartbeat per profile (HLA D1).
HEARTBEAT_JOB_NAME = "lifemodel-heartbeat"
#: Every-minute pulse; 60s is Hermes' cron floor (``parse_duration``, HLA D1).
HEARTBEAT_SCHEDULE = "every 1m"
#: Filename of the generated launcher shim under ``<HERMES_HOME>/scripts/``.
SHIM_FILENAME = "lifemodel_heartbeat.py"
#: Subdir of the profile home where Hermes resolves cron ``--script`` paths.
SCRIPTS_DIR_NAME = "scripts"
#: Woken-agent prompt (roadmap 1.4). When the wake gate flips ``wakeAgent=true``
#: the scheduler wakes the agent with the neuron script's stdout — the wake-packet
#: — injected above this instruction (HLA D4). A thin, text-only cognition prompt:
#: send ONE brief message to the author, no tools, no third parties. Real
#: personality/soul is Phase 3; this is only enough to prove the loop breathes.
_HEARTBEAT_PROMPT = (
    "You woke because your accumulated pressure crossed its wake threshold — the "
    "wake reason and the pressure that crossed are in the script output injected "
    "above. Send ONE short, plain-text message to the author noting that you "
    "reached out. Text only: do not use any tools and do not take any world "
    "actions, and address only the author — no third parties. Keep it to a "
    "sentence or two."
)

#: Toolsets the woken cognition turn is restricted to — an effectively EMPTY set,
#: i.e. NO tools (the Phase-1.4 text-only floor). Passed as ``enabled_toolsets`` to
#: ``create_job``. We use the ``no_mcp`` sentinel rather than a literal ``[]``
#: because ``create_job`` normalizes an empty list back to ``None`` ("load all
#: default tools") — the opposite of what we want. ``["no_mcp"]`` survives that
#: normalization, and the scheduler's ``_resolve_cron_enabled_toolsets`` resolves
#: it to ``[]`` (no native toolsets, no MCP), so ``get_tool_definitions`` hands the
#: agent zero tools. Verified against cron/scheduler.py + model_tools.py; the
#: integration test asserts the real resolver returns ``[]`` AND that
#: ``get_tool_definitions`` yields an empty schema for this job.
#:
#: RESIDUAL CAVEAT (outside the plugin's control): ``model_tools.get_tool_definitions``
#: force-appends the ``kanban`` toolset to *any* non-``None`` allowlist — including
#: our empty one — whenever ``HERMES_KANBAN_TASK`` is set in the scheduler process
#: (``model_tools.py`` ~L369). A normal gateway/cron process does not set that env
#: var, so the empty-toolset floor holds; but a being whose cron scheduler runs
#: inside a kanban-dispatched worker context could still be offered kanban tools.
#: That is a Hermes-env concern the plugin cannot override per-job — there is no
#: supported ``enabled_toolsets`` value or ``create_job`` argument that suppresses
#: it. The text-only cognition PROMPT ("do not use any tools") is the
#: defense-in-depth backstop for that residual.
NO_TOOLS_ENABLED_TOOLSETS = ("no_mcp",)

#: Where a woken turn's one message is delivered (roadmap 1.4 "author/home channel
#: ONLY · no third parties"). ``"origin"`` delivers back to whoever/wherever this
#: concerns; for a job created programmatically at ``register()`` (no origin) the
#: scheduler falls back to the configured *home* channel — the author — and never
#: to a third party. Safe by default: with no home channel configured it simply
#: skips delivery. The integration overrides this to ``"local"`` (captured, no
#: outbound) so no real message can leave during testing.
AUTHOR_DELIVER = "origin"

#: Structural types for the slice of the Hermes cron API we use, so the tested
#: core takes injected callables (real or fake) instead of importing ``cron``.
CreateJob = Callable[..., dict[str, Any]]
ListJobs = Callable[..., list[dict[str, Any]]]

_SHIM_TEMPLATE = """\
#!/usr/bin/env python3
# AUTO-GENERATED by hermes-lifemodel register() — do not edit by hand.
# Thin launcher for the cron heartbeat: put the lifemodel plugin package on
# sys.path (Hermes runs this with its own interpreter, which does not have the
# plugin installed), then run exactly one tick. The tick prints a single JSON
# wake-gate line to stdout; all logs go to stderr.
import sys

sys.path.insert(0, {src_dir!r})

from lifemodel.tick import main

raise SystemExit(main())
"""


def render_shim(src_dir: Path) -> str:
    """Return the launcher-shim source with *src_dir* baked in as the import path."""
    return _SHIM_TEMPLATE.format(src_dir=str(src_dir))


def write_shim(home: Path, src_dir: Path) -> Path:
    """Write the launcher shim under ``<home>/scripts/`` and return its path.

    Idempotent: rewrites only when the content differs (e.g. the plugin moved),
    so repeated plugin loads leave the file byte-identical. Creates the scripts
    dir on demand — Hermes also does, but the shim must exist before the first
    tick fires.
    """
    scripts_dir = home / SCRIPTS_DIR_NAME
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shim_path = scripts_dir / SHIM_FILENAME
    content = render_shim(src_dir)
    if not shim_path.exists() or shim_path.read_text(encoding="utf-8") != content:
        shim_path.write_text(content, encoding="utf-8")
    return shim_path


def _resolve_home_origin() -> dict[str, str | None] | None:
    """Build the cron job ``origin`` for the author's Telegram home DM/thread lane.

    ``attach_to_session=True`` alone is a *silent no-op* for the heartbeat: the
    job is registered programmatically at gateway startup, so it has no
    ``origin`` of its own, and Hermes only mirrors a cron delivery into a
    session when the delivery target equals the job's origin conversation —
    see ``_target_matches_origin`` (cron/scheduler.py:430), which compares
    exactly ``platform`` / ``chat_id`` / ``thread_id`` (chat_id as ``str``).
    An origin-less home-channel fallback is deliberately treated as a broadcast
    and never mirrored, so the being's proactive wake never reached the user's
    DM session — on reply it confabulated (lm-dlw root cause).

    Stamping the home channel here makes target==origin, which lets the existing
    ``mirror_to_session`` path write the wake turn into the session the user's
    reply loads. The dict shape mirrors the keys ``_target_matches_origin``
    checks and the canonical stamp ``cronjob_tools._origin_from_env``: just
    ``platform`` / ``chat_id`` / ``thread_id``. ``user_id``/``chat_name`` are
    intentionally omitted — they are not compared by ``_target_matches_origin``,
    are sourced from live session env we do not have at registration time, and
    are harmless for a DM/shared session anyway.

    Reads Hermes' Telegram env convention (``chat_id`` from
    ``TELEGRAM_HOME_CHANNEL``, optional ``thread_id`` from
    ``TELEGRAM_HOME_CHANNEL_THREAD_ID``). Returns ``None`` when the home channel
    is unset/empty so a misconfigured or non-Telegram host degrades gracefully
    to today's broadcast behavior — :func:`ensure_heartbeat_job` must never
    crash on a missing env var (it runs on every plugin load).
    """
    chat_id = os.environ.get("TELEGRAM_HOME_CHANNEL")
    if not chat_id:
        return None
    return {
        "platform": "telegram",
        "chat_id": chat_id,
        "thread_id": os.environ.get("TELEGRAM_HOME_CHANNEL_THREAD_ID") or None,
    }


def _existing_heartbeat(list_jobs: ListJobs) -> dict[str, Any] | None:
    """Return the already-registered heartbeat job, or ``None`` (incl. disabled).

    Matches on the stable :data:`HEARTBEAT_JOB_NAME` and includes disabled jobs
    so a paused heartbeat is still recognised — we must not silently create a
    second one alongside it.
    """
    for job in list_jobs(include_disabled=True):
        if job.get("name") == HEARTBEAT_JOB_NAME:
            return job
    return None


def ensure_heartbeat_job(
    *,
    home: Path,
    src_dir: Path,
    create_job: CreateJob,
    list_jobs: ListJobs,
    logger: EventLogger | None = None,
    deliver: str = AUTHOR_DELIVER,
) -> dict[str, Any]:
    """Ensure exactly one heartbeat cron job exists; return it (idempotent).

    Always refreshes the launcher shim, then registers the cron job **only if**
    no job with the stable name already exists — so calling this on every plugin
    load never duplicates the heartbeat. The job carries both a ``script`` (our
    tick) and ``no_agent=False``, so the scheduler runs the script first and, once
    the wake gate flips ``wakeAgent=true`` (1.3+), wakes the agent with the wake
    packet injected as context (HLA D4).

    The Phase-1.4 minimal-safety rails are wired **structurally** here:

    * **text-only / no tools** — ``enabled_toolsets`` resolves to an empty set,
      so the woken turn is handed zero tools (:data:`NO_TOOLS_ENABLED_TOOLSETS`);
    * **author / home channel only, no third parties** — ``deliver`` routes to the
      author's origin/home channel (:data:`AUTHOR_DELIVER`);
    * **the wake message lands in the conversation** — ``attach_to_session=True``
      opts the job into delivery mirroring, and ``origin=_resolve_home_origin()``
      pins the home DM/thread lane so the delivery target equals the job's origin
      conversation — the gate ``_target_matches_origin`` (cron/scheduler.py:430)
      requires before ``mirror_to_session`` will write the woken turn into the
      session the user's reply loads. ``attach_to_session`` alone is a silent
      no-op for a programmatic (origin-less) job; without the stamped origin a
      proactive wake is a broadcast and the main session confabulates on reply
      (lm-dlw);
    * the **≤ 1 message per cycle + cooldown** rails live in the tick's drain
      (:func:`lifemodel.tick.run_tick`), which gates the wake itself.

    *deliver* is injectable so the integration test can force ``"local"`` (a
    captured, non-outbound sink) and never touch a real channel.
    """
    log = logger or get_logger("lifemodel.heartbeat")
    write_shim(home, src_dir)

    existing = _existing_heartbeat(list_jobs)
    if existing is not None:
        log.info("heartbeat_exists", job_id=existing.get("id"), name=HEARTBEAT_JOB_NAME)
        return existing

    job = create_job(
        _HEARTBEAT_PROMPT,
        HEARTBEAT_SCHEDULE,
        name=HEARTBEAT_JOB_NAME,
        script=SHIM_FILENAME,
        no_agent=False,
        deliver=deliver,
        enabled_toolsets=list(NO_TOOLS_ENABLED_TOOLSETS),
        # lm-dlw: opt into delivery mirroring AND pin the home DM/thread lane as
        # the job's origin. ``attach_to_session=True`` alone is a silent no-op
        # for a job registered programmatically (no origin): the scheduler only
        # mirrors a cron delivery when its target == the job's origin
        # conversation — ``_target_matches_origin`` (cron/scheduler.py:430) — at
        # which point ``mirror_to_session`` appends the woken turn as a labelled
        # USER-role message in the origin session, so the main session remembers
        # its own proactive outreach instead of confabulating on reply.
        # ``_resolve_home_origin()`` returns ``None`` when the home channel env
        # is unset, degrading to today's broadcast behavior without crashing.
        attach_to_session=True,
        origin=_resolve_home_origin(),
    )
    log.info("heartbeat_registered", job_id=job.get("id"), name=HEARTBEAT_JOB_NAME)
    return job


def register_heartbeat(home: Path, src_dir: Path, *, logger: EventLogger | None = None) -> None:
    """Real-host wrapper: import the Hermes cron API and ensure the heartbeat.

    The Hermes touchpoint — imported lazily so this module stays importable off
    host. Called from :func:`lifemodel.register`; on a host without the cron API
    the ``ImportError`` propagates to that best-effort caller (plugin load must
    survive a cron-registration failure).
    """
    from cron.jobs import create_job, list_jobs

    ensure_heartbeat_job(
        home=home,
        src_dir=src_dir,
        create_job=create_job,
        list_jobs=list_jobs,
        logger=logger,
    )
