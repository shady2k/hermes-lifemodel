"""``wire`` — the fail-loud context manager for every wiring boundary (spec §4.3).

This inverts the default at the failure boundary that let the 2026-07-11 incident
happen. The old pattern — ``except Exception → _LOG.info("…_skipped"); continue``
— logged a *fatal* error as *benign*, at a *lost* level, *without a traceback*, so
a brain-dead plugin looked "enabled". :func:`wire` replaces it everywhere in
``register()`` and :meth:`BeingAdapter.connect`:

* on success → DEBUG (silence means healthy);
* on ANY exception → **ERROR with a full traceback** (``exc_info=True``), ALWAYS,
  independent of ``HERMES_PLUGINS_DEBUG`` — the failure is never invisible;
* ``required=True`` → record ``boot_failed`` in :class:`BrainHealth` (which
  persists the durable boot record) and **re-raise**, so Hermes' ``_load_plugin``
  marks the plugin not-enabled + logs it (the loud channel), or the gateway sees
  ``connect()`` fail;
* ``required=False`` → downgrade to **WARNING with a traceback** and swallow, so a
  capability the host genuinely lacks degrades the being instead of killing it —
  never a bare INFO-without-traceback.

All stdlib (``contextlib`` / ``logging``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from .brain_health import BrainHealth


@contextmanager
def wire(
    step_name: str,
    *,
    required: bool,
    health: BrainHealth,
    logger: logging.Logger,
) -> Iterator[None]:
    """Guard one wiring step *step_name* (spec §4.3).

    The signature carries *health* + *logger* explicitly so the SAME helper serves
    both wiring boundaries (``register()`` in the plugin root and ``connect()`` in
    the adapter), each supplying its own ``BrainHealth`` (the shared per-base_dir
    singleton) and its own logger. See the module docstring for the level policy.
    """
    logger.debug("wire_start step=%s required=%s", step_name, required)
    try:
        yield
    except Exception as exc:
        detail = f"{step_name}: {type(exc).__name__}: {exc}"
        if required:
            # LOUD + observable + fatal: ERROR+traceback, flip health to boot_failed
            # (persists the durable record), then re-raise so the host marks us
            # not-enabled / the gateway sees the connect fail.
            logger.error("wire_failed step=%s error=%s", step_name, detail, exc_info=True)
            health.mark_boot_failed(detail)
            raise
        # OPTIONAL: a capability the host genuinely lacks — degrade, don't die.
        # WARNING still carries the full traceback (never a silent INFO).
        logger.warning(
            "wire_skipped_optional step=%s error=%s", step_name, detail, exc_info=True
        )
        return
    else:
        logger.debug("wire_ok step=%s", step_name)
