"""The /lifemodel debug dump — the owner's read-only window (spec §16).

Renders the being's new-model state in labelled sections (physiology, drive,
desire lifecycle, gates, backstop, timing), each showing raw + derived values.
Read-only: builds the graph via the single composition root, computes pure
:class:`Readings`, never writes. Stdlib only; imports Hermes-free adapters.
"""

from __future__ import annotations

from pathlib import Path

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


def render_dump_for_dir(base_dir: Path) -> str:
    lm = composition.build_lifemodel(base_dir=base_dir)
    now = lm.clock.now()
    try:
        state = lm.state.load()
    except StateError as exc:
        return f"lifemodel debug  (read-only)\n{'=' * 30}\n\n<unreadable: {exc}>\n"
    return render_debug_dump(readings=compute_readings(state, now=now, cfg=_cfg()))


def render_debug_dump(*, readings: Readings) -> str:
    r = readings
    lines: list[str] = ["lifemodel debug  (read-only)", "=" * 30, ""]

    lines += [
        "PHYSIOLOGY",
        f"  energy(E) {_pct(r.energy)}   fatigue(S) {_n(r.fatigue)}"
        f"   circadian(C) {_n(r.circadian)}",
        f"  alertness ~{_n(r.alertness)}   (higher C, lower S = sharper)",
        "",
        "DRIVE (contact)",
        f"  latent u {_n(r.u)}   inhibition {_n(r.inhibition)}"
        f"   [{r.action_pending_phase}"
        + (
            f", {_opt(r.action_pending_remaining_min, 'm grace left')}"
            if r.action_pending_remaining_min
            else ""
        )
        + "]",
        f"  effective = u*(1-inhibition) = {_n(r.effective)}   theta {_n(r.theta)}"
        f"   -> {_pct(r.pct_to_wake)} to wake",
        f"  interpretation: {'over threshold' if r.effective >= r.theta else 'below threshold'}",
        "",
        "DESIRE",
        f"  status {r.desire_status}   pending_turn {r.pending}"
        + (f" (since {r.pending_since})" if r.pending else ""),
        f"  last_contact {r.last_contact_at or 'n/a'}"
        f"   last_exchange {r.last_exchange_at or 'n/a'}",
        "",
        "GATES (why wake / no wake)",
        f"  would_wake {r.would_wake}   reason {r.wake_reason}",
        f"  silence_window {_opt(r.silence_window_remaining_min, ' min')}"
        f"   backoff {_opt(r.backoff_remaining_min, ' min')}"
        f"   (declines {r.decline_count})",
        "",
        "BACKSTOP (hard send limit)",
        f"  sends_today {r.sends_today}/{r.sends_cap}   send_allowed_now {r.send_allowed}",
        "",
        "HEALTH (is the brain ticking)",
        f"  brain {'alive' if r.brain_alive else 'STALE — loop may be down'}"
        f"   last_tick {_opt(r.last_tick_ago_min, ' min ago')}   tick {r.tick_count}",
    ]
    return "\n".join(lines) + "\n"
