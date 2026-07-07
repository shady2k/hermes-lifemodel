"""The /lifemodel debug dump — the owner's read-only window (spec §16).

Renders the being's new-model state in labelled sections (physiology, drive,
desire lifecycle, gates, backstop, timing), each showing raw + derived values.
Read-only: builds the graph via the single composition root, computes pure
:class:`Readings`, never writes. Stdlib only; imports Hermes-free adapters.
"""

from __future__ import annotations

from datetime import datetime, tzinfo
from pathlib import Path
from typing import cast

from . import composition
from .core.introspect import DebugConfig, Readings, compute_readings
from .state.errors import StateError


def _cfg() -> DebugConfig:
    # max_per_day / min_interval mirror core.backstop.allow_send's defaults —
    # the live egress calls allow_send() without overrides, so those defaults ARE
    # the live limits (spec §14: <=3/day, 60 min).
    return DebugConfig(
        params=composition.CONTACT_PARAMS,
        theta=composition.CONTACT_PARAMS.theta_u,
        i0=composition.CONTACT_I0,
        grace_min=composition.CONTACT_GRACE_MIN,
        halflife_min=composition.CONTACT_INHIBITION_HALFLIFE_MIN,
        peak_hour_utc=composition.CIRCADIAN_PEAK_UTC_HOUR,
        max_per_day=3,
        min_interval_min=60.0,
        alpha=composition.CONTACT_ALPHA,
        u_max=composition.CONTACT_U_MAX,
    )


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _n(x: float) -> str:
    return f"{x:.4g}"


def _opt(x: float | None, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:.1f}{unit}"


def _resolve_tz() -> tzinfo | None:
    """Best-effort owner/Hermes-configured timezone for rendering timestamps.

    ``hermes_time.get_timezone()`` is Hermes' own resolver (``HERMES_TIMEZONE``
    env var, then ``timezone:`` in ``config.yaml``, already fails open to
    ``None`` on a bad value) — the cleanest available source of the *owner's*
    zone, so it is tried first. It is an optional runtime dependency (this
    plugin is loaded inside Hermes' interpreter, not pip-installed — see
    ``log.py``/``__init__.py`` for the same optional-Hermes-import discipline):
    absent, or any other resolution hiccup, this returns ``None`` instead of
    raising. Callers pass that straight into ``datetime.astimezone(None)``,
    which is the stdlib's own "assume system-local" behaviour — so a bad or
    missing Hermes TZ degrades to system-local, never UTC-confusion, and never
    crashes the read-only dump.
    """
    try:
        from hermes_time import get_timezone  # optional Hermes dependency

        # hermes_time is untyped (host-provided, ignore_missing_imports=true),
        # so its return narrows to Any; cast back to the real declared type.
        return cast("tzinfo | None", get_timezone())
    except Exception:  # noqa: BLE001 - best-effort; a debug dump must never crash on TZ
        return None


def _local(iso: str | None) -> str:
    """Render a stored UTC ISO timestamp in the owner's local timezone.

    The single conversion point for every timestamp field the dump shows
    (last_contact, last_exchange, pending_since, ...): Hermes-configured zone
    when :func:`_resolve_tz` cleanly provides one, else system-local — and
    trimmed to whole seconds so the dump stays scannable.
    """
    if iso is None:
        return "n/a"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso  # malformed timestamp: show it raw rather than crash
    try:
        local = dt.astimezone(_resolve_tz())
    except Exception:  # noqa: BLE001 - degrade to the original instant, never raise
        local = dt
    offset = local.strftime("%z") or "+00:00"
    if len(offset) == 5:  # "+0300" -> "+03:00"
        offset = f"{offset[:3]}:{offset[3:]}"
    return local.strftime("%Y-%m-%d %H:%M:%S") + " " + offset


def render_dump_for_dir(base_dir: Path) -> str:
    lm = composition.build_lifemodel(base_dir=base_dir)
    now = lm.clock.now()
    try:
        state = lm.state.load()
    except StateError as exc:
        return f"lifemodel debug  (read-only)\n{'=' * 30}\n\n<unreadable: {exc}>\n"
    return render_debug_dump(readings=compute_readings(state, now=now, cfg=_cfg()))


def _section(pairs: list[tuple[str, str]]) -> list[str]:
    """Render one section's ``(label, value)`` pairs as an aligned table.

    One datum per line, colons lined up on the widest label in the section, so
    the block reads top-to-bottom like a clean table instead of several
    metrics crammed onto one line.
    """
    width = max(len(label) for label, _ in pairs) + 1  # +1 for the colon
    return [f"  {label + ':':<{width}} {value}" for label, value in pairs]


def render_debug_dump(*, readings: Readings) -> str:
    r = readings
    lines: list[str] = ["lifemodel debug  (read-only)", "=" * 30, ""]

    phase = r.action_pending_phase
    if r.action_pending_remaining_min:
        phase += f"   ({_opt(r.action_pending_remaining_min, 'm grace left')})"

    pending = str(r.pending)
    if r.pending and r.pending_since:
        pending += f"   (since {_local(r.pending_since)})"

    lines.append("PHYSIOLOGY")
    lines += _section(
        [
            ("energy(E)", _pct(r.energy)),
            ("fatigue(S)", _n(r.fatigue)),
            ("circadian(C)", _n(r.circadian)),
            ("alertness", f"~{_n(r.alertness)}   (higher C, lower S = sharper)"),
        ]
    )
    lines.append("")

    lines.append("DRIVE (contact)")
    lines += _section(
        [
            ("latent u", _n(r.u)),
            ("inhibition", _n(r.inhibition)),
            ("phase", phase),
            ("effective", f"{_n(r.effective)}   (= u * (1 - inhibition))"),
            ("theta", _n(r.theta)),
            ("pct_to_wake", _pct(r.pct_to_wake)),
            (
                "interpretation",
                "over threshold" if r.effective >= r.theta else "below threshold",
            ),
        ]
    )
    lines.append("")

    lines.append("DESIRE")
    lines += _section(
        [
            ("status", r.desire_status),
            ("pending_turn", pending),
            ("last_contact", _local(r.last_contact_at)),
            ("last_exchange", _local(r.last_exchange_at)),
        ]
    )
    lines.append("")

    lines.append("GATES (why wake / no wake)")
    lines += _section(
        [
            ("would_wake", str(r.would_wake)),
            ("reason", r.wake_reason),
            ("silence_window", _opt(r.silence_window_remaining_min, " min")),
            ("backoff", _opt(r.backoff_remaining_min, " min")),
            ("declines", str(r.decline_count)),
        ]
    )
    lines.append("")

    lines.append("BACKSTOP (hard send limit)")
    lines += _section(
        [
            ("sends_today", f"{r.sends_today}/{r.sends_cap}"),
            ("send_allowed", str(r.send_allowed)),
        ]
    )
    lines.append("")

    lines.append("HEALTH (is the brain ticking)")
    lines += _section(
        [
            ("brain", "alive" if r.brain_alive else "STALE — loop may be down"),
            ("last_tick", _opt(r.last_tick_ago_min, " min ago")),
            ("tick", str(r.tick_count)),
        ]
    )
    return "\n".join(lines) + "\n"
