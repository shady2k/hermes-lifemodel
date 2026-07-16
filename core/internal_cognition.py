"""The internal-cognition completion frame (lm-705.6, design §3.2/§3.3).

:func:`run_internal_completion` is the seam's completion-frame body: it seeds an
``ASYNC_COMPLETION`` frame with the aux call's typed re-entry (an
``internal_result`` signal), lets an injected result-applying
:class:`~lifemodel.core.component.Component` turn it into intents (committed
atomically by the state-actor), then — the strand fix, codex #2 —
:func:`~lifemodel.core.proactive.dispatch_launches` ANY launch that frame
returned (an unrelated already-registered ``CognitionLauncher`` does not know or
care that this frame's purpose was internal; it wakes on its own gate regardless
of trigger, spec §3.2), and finally clears :attr:`~lifemodel.state.model.State.pending_internal_id`
— on success, failure, or an empty result alike, so a strand here can never block
every future internal launch (mirroring the proactive in-flight gate).

Non-delivery is structural here too: this module never calls
``egress.reach_out``/``inject_proactive_turn`` directly — the *only* reach any
call in this module makes to the egress is via :func:`dispatch_launches`, and
only for a *proactive* launch some OTHER component incidentally returned.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..composition import LifeModel
from ..domain.egress import ReachOutcome
from ..domain.session import VoiceCheck
from ..ports.proactive import ProactiveEgressPort
from .component import Component, ComponentLayer, TickContext
from .frame import FrameTrigger, run_frame
from .intents import Intent, UpdateState
from .llm_port import InternalCognitionResult
from .proactive import dispatch_launches
from .registry import ComponentManifest, UnknownComponent
from .taxonomy import internal_result_signal
from .timeutil import to_iso

#: The seam's own placeholder result-applying component (lm-705.6's "trivial
#: internal pass" — the Global Constraints of the plan this module realizes).
#: Ignores the result entirely; noticing (lm-705.5) and processing (lm-705.2)
#: supply the real ``apply`` components that turn a result into thoughts/state.
NULL_INTERNAL_APPLY_ID = "internal-cognition-null-apply"


class NullInternalApply:
    """The trivial internal pass: reads nothing, applies nothing.

    Exercises the seam end-to-end (aux call → typed re-entry → completion frame →
    pending clear) without claiming any real cognition — the default ``apply``
    until a real consumer (noticing/processing) is wired in.
    """

    id = NULL_INTERNAL_APPLY_ID

    def step(self, ctx: TickContext) -> list[Intent]:
        return []


def run_internal_completion(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    correlation_id: str,
    result: InternalCognitionResult,
    apply: Component,
    voice: VoiceCheck | None = None,
) -> ReachOutcome | None:
    """Apply *result* + dispatch any launches + clear ``pending_internal_id``.

    Registers *apply* into *lm*'s registry (idempotent by id) so the ONE
    ``ASYNC_COMPLETION`` frame this function runs actually invokes it alongside
    every other already-enabled component — the same "a frame runs everything,
    regardless of trigger" contract every other frame honours (``core/coreloop.py``).
    *result* seeds the frame as an ``internal_result`` signal keyed by
    *correlation_id*; *apply* turns it into intents (or nothing, on a failed/empty
    *result* — see :class:`NullInternalApply`).

    *voice* is forwarded verbatim to :func:`~lifemodel.core.proactive.dispatch_launches`
    (the birth pre-flight, spec §6.2/lm-4fv.4): an unborn being's INCIDENTAL
    proactive launch on THIS completion frame — surfaced by some other
    already-registered component, never this seam itself (the strand fix,
    codex #2) — must not be able to skip the same voice gate a native
    ``proactive_tick`` launch goes through. ``None`` (the default) reproduces
    the prior no-gate behaviour for a caller that has no voice check to offer.

    Returns whatever :func:`~lifemodel.core.proactive.dispatch_launches` returns —
    ``None`` unless some OTHER component (never this seam) also surfaced a
    ``LaunchProactive`` this frame, in which case it is genuinely dispatched (the
    strand fix). This function itself never delivers anything.
    """
    assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
    assert lm.state_actor is not None, "state_actor must be wired by build_lifemodel"
    try:
        lm.registry.manifest(apply.id)
    except UnknownComponent:
        lm.registry.register(
            apply,
            ComponentManifest(
                id=apply.id,
                type="internal-cognition-apply",
                layer=ComponentLayer.COGNITION,
                metric_surface=(),
                accepts_signals=True,
            ),
        )

    signal = internal_result_signal(
        origin_id=f"internal-result-{correlation_id}",
        correlation_id=correlation_id,
        raw=result.raw,
        parsed=result.parsed,
        timestamp=to_iso(lm.clock.now()),
    )
    report = run_frame(lm.coreloop, [signal], trigger=FrameTrigger.ASYNC_COMPLETION)
    outcome = dispatch_launches(lm, report, egress, target, voice=voice)
    # A second, small commit — clearing the correlation is NOT part of the frame's
    # own intent set (no component "owns" pending_internal_id), mirroring how
    # core/proactive.py's _rollback commits its own follow-up patch after a frame.
    # The subject anchor clears in lockstep — it is set alongside
    # pending_internal_id by the runner's reserve frame, so it must clear alongside
    # it too, on every path (success, failure, or an empty result alike).
    lm.state_actor.apply(
        [UpdateState({"pending_internal_id": None, "pending_internal_subject_id": None})]
    )
    return outcome
