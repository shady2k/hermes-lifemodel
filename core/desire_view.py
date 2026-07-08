"""The contact-desire view — the single door onto the live ``kind='desire'`` row.

lm-27n.3 kills ``State.desire_status``: the contact desire's lifecycle now lives
in the singleton ``kind='desire'`` record ``contact:owner`` (HLA §4.1), mutated
only through the intent bus (:class:`~lifemodel.core.intents.PutRecord` /
:class:`~lifemodel.core.intents.TransitionRecord`) and committed atomically with
``State`` by the tick committer. This module is the ONE place that reads that row
back into a typed :class:`~lifemodel.domain.objects.Desire`, so every "is there a
desire?" site asks the SAME question — a **live non-terminal** desire
(``active``/``deferred``), never "any desire row" (a ``satisfied``/``dropped``/
``expired`` row is absence, the old ``none``).

Two readers, one predicate:

* :func:`live_contact_desire` reads the start-of-tick records snapshot
  (:attr:`~lifemodel.core.component.TickContext.objects`) — what aggregation and
  cognition consume in-tick;
* :func:`read_live_contact_desire` reads a :class:`~lifemodel.ports.memory.MemoryPort`
  point-in-time (``get`` by id) — what the out-of-band hooks / debug view use.

:func:`build_contact_desire` is the sole constructor of the contact desire: it
never hand-builds a draft, it hands a typed :class:`Desire` to the registry (the
single encode door). The semantic payload fields are constants — aggregation
never reads them back (it decides purely on the ``state`` + residual ``State``
scalars), so they carry description, not behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import MemoryDraft, MemoryRecord
from ..domain.objects import (
    CONTACT_DESIRE_ID,
    Desire,
    DesireSpring,
    DesireState,
    Provenance,
    default_registry,
)
from ..ports.memory import MemoryPort

#: The kind of the contact desire record (``kind`` column, HLA §4.1).
DESIRE_KIND = "desire"

#: The non-terminal states — a desire in one of these is *live*; anything else
#: (terminal, or absent) reads as the old ``none``.
LIVE_DESIRE_STATES: frozenset[str] = frozenset(
    {DesireState.ACTIVE.value, DesireState.DEFERRED.value}
)

#: Built once; :func:`default_registry` validates its four-kind catalog on every
#: call, so the per-tick readers reuse one instance rather than rebuild it.
_REGISTRY = default_registry()


def _decode_live(record: MemoryRecord | None) -> Desire | None:
    """Decode *record* into a live contact :class:`Desire`, or ``None``.

    ``None`` when the record is absent, is not the contact-desire singleton, or
    is terminal (``satisfied``/``dropped``/``expired``). Decoding goes through
    the registry (the single read door), so a malformed row surfaces as its
    :class:`~lifemodel.domain.objects.InvalidPayload`, never a silent miss.
    """
    if record is None or record.kind != DESIRE_KIND or record.id != CONTACT_DESIRE_ID:
        return None
    if record.state not in LIVE_DESIRE_STATES:
        return None
    desire = _REGISTRY.decode(record)
    return desire if isinstance(desire, Desire) else None


def live_contact_desire(objects: Sequence[MemoryRecord]) -> Desire | None:
    """The live (``active``/``deferred``) contact desire in a records snapshot.

    Scans the start-of-tick :attr:`~lifemodel.core.component.TickContext.objects`
    snapshot for the ``contact:owner`` desire and returns it typed, or ``None``
    if there is no live one. The snapshot the CoreLoop builds is ``state="active"``
    only, so in the live path this returns the active desire; a ``deferred`` row
    is decoded here for completeness but is not reachable through Model A's
    fulfil/reject-only cognition.
    """
    for record in objects:
        desire = _decode_live(record)
        if desire is not None:
            return desire
    return None


def read_live_contact_desire(memory: MemoryPort) -> Desire | None:
    """The live contact desire read point-in-time from a :class:`MemoryPort`.

    For out-of-band readers (the verdict-publish hook, the debug view) that hold
    a store rather than a tick snapshot. ``get`` by the singleton id, then the
    same live/terminal predicate as :func:`live_contact_desire`.
    """
    return _decode_live(memory.get(DESIRE_KIND, CONTACT_DESIRE_ID))


def build_contact_desire(
    *,
    state: DesireState,
    salience: float = 0.0,
    source_drive: float | None = None,
    spring: DesireSpring = DesireSpring.DRIVE,
    source_thought_ids: tuple[str, ...] = (),
    risk_if_ignored: float = 0.0,
    source: str = "contact-aggregation",
    provenance: Provenance | None = None,
) -> Desire:
    """Construct the singleton contact :class:`Desire` in *state*.

    The one constructor for the ``contact:owner`` desire. ``salience`` is stamped
    from the effective pressure (or, top-down, the proposal score) at creation so
    cognition / the pressure index can read intensity; ``source_drive`` records the
    latent drive ``u``.

    Two springs (lm-27n.9): the bottom-up default is ``spring=DRIVE`` with no
    ``source_thought_ids`` — behaviour-identical to .3. A top-down crystallization
    passes ``spring=THOUGHT`` (or ``MIXED`` when a drive urge co-fires) and the
    source thought id(s), so the desire carries the CONCRETE reason it sprang from
    (the ``[SILENT]`` fix + the lm-8o3 wake-framing seam) rather than a timer-derived
    urge. ``risk_if_ignored`` lets the top-down path carry the thought's stakes;
    every other semantic field is a fixed description (aggregation never reads them
    back).

    ``provenance`` (lm-27n.11) records the creation lineage + the tick's execution
    trace; ``None`` (the default) leaves an un-stamped desire — behaviour-identical to
    before .11 for the test builders that don't pass one."""
    return Desire(
        id=CONTACT_DESIRE_ID,
        state=str(state),
        source=source,
        salience=salience,
        provenance=provenance,
        object="owner",
        spring=spring,
        source_drive=source_drive,
        source_thought_ids=source_thought_ids,
        intensity=salience,
        valence="connection",
        urgency=salience,
        satiation_condition="a genuine two-way exchange with the owner",
        risk_if_acted=0.0,
        risk_if_ignored=risk_if_ignored,
    )


def encode_contact_desire(desire: Desire) -> MemoryDraft:
    """Encode *desire* through the registry (the single write door)."""
    return _REGISTRY.encode(desire)
