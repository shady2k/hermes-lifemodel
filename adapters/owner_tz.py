"""``resolve_owner_tz`` — the owner's display timezone, read from Hermes.

The Hermes boundary for the wake-packet's temporal facts (HLA §11): the core
renders ``now`` / ``last_exchange_at`` in the owner's local zone so the being
judges "morning / evening / is he asleep" against the owner's wall clock, not UTC
(UTC 22:00 = 01:00 MSK reads as a false "evening" while he sleeps). The core stays
Hermes-free — it takes a plain stdlib ``tzinfo``; THIS adapter is the only place
that reaches into Hermes to obtain it.

Source of truth is Hermes' own ``hermes_time`` module (``get_timezone()``), which
resolves — in order — the ``HERMES_TIMEZONE`` env var, then the ``timezone`` key in
``~/.hermes/config.yaml``, then ``None`` (meaning server-local). We import it
LAZILY inside the function so this module stays importable in a dev checkout /
tests where Hermes is absent: any failure (Hermes missing, bad config) is
fail-open and returns ``None`` — the renderer then falls back to server-local, and
finally UTC. A missing timezone must never drop the impulse.
"""

from __future__ import annotations

from datetime import tzinfo


def resolve_owner_tz() -> tzinfo | None:
    """The owner's configured display zone as a stdlib ``tzinfo``, or ``None``.

    ``None`` means "no IANA timezone configured (or Hermes unavailable)" — the
    wake-packet renderer then uses the server's local zone, then UTC. Never raises:
    the boundary is fail-open so a timezone problem cannot take down a brain tick."""
    try:
        import hermes_time  # Hermes-provided top-level module (present in its venv)

        # get_timezone() already returns a validated ZoneInfo | None (it logs and
        # falls back to None on a bad value), and caches the result internally. The
        # module is untyped, so narrow defensively: anything that is not a real
        # ``tzinfo`` (incl. None) degrades to None → server-local.
        tz = hermes_time.get_timezone()
        if isinstance(tz, tzinfo):
            return tz
        return None
    except Exception:  # noqa: BLE001 - Hermes absent / any resolution error → server-local
        return None
