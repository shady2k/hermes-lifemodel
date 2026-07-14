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
from .core.affect import CONTRIBUTION_DISPLAY_DEADBAND
from .core.desire_view import read_live_contact_desire
from .core.intention_view import read_live_contact_intention
from .core.introspect import DebugConfig, Readings, compute_readings, contact_chain_summary
from .core.thought_view import read_live_thoughts
from .core.user_model_view import read_owner_user_model
from .core.why_graph import why_contact_intention
from .ports.clock import ClockPort
from .ports.memory import MemoryPort
from .state.errors import StateError
from .trace_view import LastWakeOutcome, read_last_wake_outcome

#: How many live thoughts the debug audit lists (most-salient first). Bounded so
#: a large open-loop set never floods the read-only dump.
DEBUG_THOUGHTS_LIMIT = 10


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
        affect_params=composition.AFFECT_PARAMS,
    )


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _n(x: float) -> str:
    # Display precision (lm-25t): 2 decimals so the read-only dump reads cleanly
    # (0.22, not 0.2167). The stored/computed value keeps full precision — this
    # only formats the echo.
    return f"{x:.2f}"


def _opt(x: float | None, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:.1f}{unit}"


def _display_line(last_word: str | None, local_at: str) -> str:
    """The AFFECT ``display:`` value (lm-ukc.4): the felt word last surfaced
    ambiently + when, or ``"never shown"`` before the mood first proves itself.
    *local_at* is already the owner-local render (``"n/a"`` when unset)."""
    if last_word is None:
        return "never shown"
    return f"{last_word} @ {local_at}"


def _contribs(pairs: tuple[tuple[str, float], ...], top: int = 3) -> str:
    """Render the strongest signed pushes on an affect axis as one scannable line.

    Magnitude-ranked pairs in (from :class:`Readings`), the top few with a
    non-trivial push out — e.g. ``"u -0.33; exchange +0.05"``. All sub-deadband →
    ``"—"`` (nothing is moving the axis). 2-decimal like the rest of the dump."""
    ranked = [(name, v) for name, v in pairs if abs(v) >= CONTRIBUTION_DISPLAY_DEADBAND][:top]
    if not ranked:
        return "—"
    return "; ".join(f"{name} {v:+.2f}" for name, v in ranked)


def _reasons(reasons: tuple[str, ...]) -> str:
    """Render a receptivity reason/constraint tuple as one scannable line.

    Empty → ``"none"``; otherwise the reasons joined by ``"; "`` (single space,
    so the no-column-padding house style holds — see ``tests/test_debug.py``)."""
    return "none" if not reasons else "; ".join(reasons)


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


def local_time(iso: str | None) -> str:
    """Render a stored UTC ISO timestamp in the owner's local timezone.

    The single conversion point for every timestamp field an OWNER-FACING view shows
    (this dump's last_contact/last_exchange/pending_since, and ``/lifemodel soul
    history``'s lineage): Hermes-configured zone when :func:`_resolve_tz` cleanly
    provides one, else system-local — and trimmed to whole seconds so the view stays
    scannable. A raw stored instant (``2026-07-14T13:29:00.551071+00:00``) is precise and
    unreadable, and a listing whose entire job is to be compared AT A GLANCE cannot afford
    six digits of microseconds in every row.
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


def render_dump_for_dir(base_dir: Path, *, clock: ClockPort | None = None) -> str:
    # ``clock`` is injectable for deterministic tests (lm-5ac): the drive `u` is
    # reconstructed from elapsed time since ``last_tick_at``, so a pinned clock
    # keeps the rendered `u` stable. Production passes nothing → live SystemClock.
    lm = composition.build_lifemodel(base_dir=base_dir, clock=clock)
    now = lm.clock.now()
    try:
        state = lm.state.load()
    except StateError as exc:
        return f"🫀 **lifemodel debug** (read-only)\n\n<unreadable: {exc}>\n"
    memory = lm.state if isinstance(lm.state, MemoryPort) else None
    desire = read_live_contact_desire(memory) if memory is not None else None
    desire_state = desire.state if desire is not None else "none"
    desire_spring = str(desire.spring) if desire is not None else "drive"
    desire_source_thought_ids = desire.source_thought_ids if desire is not None else ()
    intention = read_live_contact_intention(memory) if memory is not None else None
    intention_state = intention.state if intention is not None else "none"
    user_model = read_owner_user_model(memory) if memory is not None else None
    thoughts = read_live_thoughts(memory, limit=DEBUG_THOUGHTS_LIMIT) if memory is not None else ()
    # The COMPACT "why did I write" chain (lm-27n.10): a single summary line, NOT the
    # full graph each dump. The full walk lives behind `/lifemodel why`.
    contact_chain = contact_chain_summary(
        why_contact_intention(memory) if memory is not None else None
    )
    last_wake = read_last_wake_outcome(base_dir)  # fail-soft: None when no/unreadable trace store
    return render_debug_dump(
        readings=compute_readings(
            state,
            now=now,
            cfg=_cfg(),
            desire_state=desire_state,
            desire_spring=desire_spring,
            desire_source_thought_ids=desire_source_thought_ids,
            intention_state=intention_state,
            user_model=user_model,
            thoughts=thoughts,
            contact_chain=contact_chain,
        ),
        last_wake=last_wake,
    )


def _metrics(pairs: list[tuple[str, str]]) -> list[str]:
    """Render one section's ``(label, value)`` pairs as plain lines.

    One datum per line, ``**label:** value`` with a single space after the
    bold colon — no column-alignment padding. Telegram renders in a
    proportional font, where space-padded columns go ragged; this matches
    Hermes' own ``/status`` command, which reads cleanly precisely because it
    never tries to line up a colon column. The bold wrapping (standard
    markdown ``**...**``, colon included inside the markers) matches
    ``/status`` byte-for-byte — see e.g. ``locales/en.yaml``'s
    ``"**Session ID:** \\`{session_id}\\`"`` and ``"**Agent Running:** {state}"``
    — which the Telegram adapter converts to MarkdownV2 and auto-escapes
    around.
    """
    return [f"**{label}:** {value}" for label, value in pairs]


def render_debug_dump(*, readings: Readings, last_wake: LastWakeOutcome | None = None) -> str:
    r = readings
    lines: list[str] = ["🫀 **lifemodel debug** (read-only)", ""]

    phase = r.action_pending_phase
    if r.action_pending_remaining_min:
        phase += f" ({_opt(r.action_pending_remaining_min, 'm grace left')})"

    pending = str(r.pending)
    if r.pending and r.pending_since:
        pending += f" (since {local_time(r.pending_since)})"

    lines.append("**PHYSIOLOGY**")
    lines += _metrics(
        [
            ("energy(E)", _pct(r.energy)),
            ("fatigue(S)", _n(r.fatigue)),
            ("circadian(C)", _n(r.circadian)),
            ("alertness", f"~{_n(r.alertness)} (higher C, lower S = sharper)"),
        ]
    )
    lines.append("")

    # AFFECT (lm-ukc.6): the being's felt state — current eased axes, this-tick target,
    # and what tugs each axis most. The felt WORD is lm-ukc.3's slot (a line here).
    lines.append("**AFFECT (self-model)**")
    lines += _metrics(
        [
            ("felt", r.affect_word),
            ("valence(v)", _n(r.affect_valence)),
            ("arousal(a)", _n(r.affect_arousal)),
            ("target", f"v {_n(r.affect_target_valence)} / a {_n(r.affect_target_arousal)}"),
            ("tugging v", _contribs(r.affect_valence_contributions)),
            ("tugging a", _contribs(r.affect_arousal_contributions)),
            ("updated", local_time(r.affect_updated_at)),
            # The reactive show (lm-ukc.4): the felt word last surfaced AMBIENTLY
            # into ordinary talk + when, so the owner sees whether/when the mood
            # proves itself (calibration, spec §9). "never shown" until it first does.
            (
                "display",
                _display_line(r.affect_display_last_word, local_time(r.affect_display_last_at)),
            ),
        ]
    )
    lines.append("")

    lines.append("**DRIVE (contact)**")
    lines += _metrics(
        [
            ("latent u", _n(r.u)),
            ("inhibition", _n(r.inhibition)),
            ("phase", phase),
            ("effective", f"{_n(r.effective)} (= u × (1 - inhibition))"),
            ("theta", _n(r.theta)),
            ("pct_to_wake", _pct(r.pct_to_wake)),
            (
                "interpretation",
                "over threshold" if r.effective >= r.theta else "below threshold",
            ),
        ]
    )
    lines.append("")

    spring = r.desire_spring
    if r.desire_source_thought_ids:
        spring += f" (from {', '.join(r.desire_source_thought_ids)})"

    lines.append("**DESIRE**")
    lines += _metrics(
        [
            ("status", r.desire_status),
            ("spring", spring),
            ("intention", r.intention_status),
            ("why", r.contact_chain),
            ("pending_turn", pending),
            ("last_contact", local_time(r.last_contact_at)),
            ("last_exchange", local_time(r.last_exchange_at)),
        ]
    )
    lines.append("")

    lines.append("**GATES (why wake / no wake)**")
    lines += _metrics(
        [
            ("would_wake", str(r.would_wake)),
            ("reason", r.wake_reason),
            ("silence_window", _opt(r.silence_window_remaining_min, " min")),
            ("backoff", _opt(r.backoff_remaining_min, " min")),
            ("declines", str(r.decline_count)),
        ]
    )
    lines.append("")

    lines.append("**BACKSTOP (hard send limit)**")
    lines += _metrics(
        [
            ("sends_today", f"{r.sends_today}/{r.sends_cap}"),
            ("send_allowed", str(r.send_allowed)),
        ]
    )
    lines.append("")

    lines.append("**RECEPTIVITY (owner appropriateness)**")
    lines += _metrics(
        [
            ("allowed", str(r.receptivity_allowed)),
            ("multiplier", _n(r.receptivity_multiplier)),
            ("confidence", _n(r.receptivity_confidence)),
            ("hard_veto", _reasons(r.receptivity_hard_reasons)),
            ("soft_downweight", _reasons(r.receptivity_soft_reasons)),
            ("constraints", _reasons(r.receptivity_constraints)),
        ]
    )
    lines.append("")

    lines.append("**THOUGHTS (what I'm turning over)**")
    lines += _metrics([("live", "none" if not r.thoughts else str(len(r.thoughts)))])
    lines += [f"— {thought}" for thought in r.thoughts]
    lines.append("")

    lines.append("**HEALTH (is the brain ticking)**")
    lines += _metrics(
        [
            ("brain", "alive" if r.brain_alive else "STALE — loop may be down"),
            ("last_tick", _opt(r.last_tick_ago_min, " min ago")),
            ("tick", str(r.tick_count)),
        ]
    )
    lines.append("")

    lines.append("**LAST WAKE OUTCOME**")
    if last_wake is None:
        lines.append("  (no wake outcome recorded yet)")
    else:
        lines += _metrics(
            [
                ("outcome", last_wake.outcome),
                ("when", local_time(last_wake.ts)),
                ("trace", f"`{last_wake.trace_id}`  (→ /lifemodel trace {last_wake.trace_id})"),
            ]
        )
    lines.append("")

    return "\n".join(lines) + "\n"
