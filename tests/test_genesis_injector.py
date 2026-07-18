"""The ``pre_llm_call`` genesis injector — the REACTIVE entrance to the ritual (§6.3).

These test the HOOK (``lifemodel.hooks.make_genesis_injector``), not the pure prose
(that is ``tests/test_genesis.py``). The hook is the half that has to be right about the
host's real data, and the half that was wrong: it decided "has the being spoken yet?" by
scanning ``conversation_history`` for an ``assistant`` entry — but the host passes the
**persisted session transcript** (``agent/turn_context.py:488`` →
``conversation_history=list(messages)``, built from ``agent_history`` in
``gateway/run.py``), not a fresh per-conversation list. So for every existing Hermes user
— whose DM already holds hundreds of assistant replies, and who is *the common case*
(§6.6: a being is born onto a blank soul exactly once in the life of a ``SOUL.md``) — the
ritual was never shown at all, and the being did exactly what §6.5 forbids: conversed as
though nothing had happened while remaining unborn.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lifemodel.adapters.soul_file import SoulFile
from lifemodel.core.genesis import GENESIS_TAG, NEWBORN_STANCE
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.tick_metrics import OBSERVER_ERRORS, register_universal_metrics
from lifemodel.core.turn_metrics import TURN_INJECTOR_TOTAL
from lifemodel.core.turn_recorder import TurnRecorder
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.hooks import make_genesis_injector
from lifemodel.state.brain_health import BrainHealth
from lifemodel.state.model import State
from lifemodel.testing import FakeClock, FakeTracer

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)

HERMES_DEFAULT = "# Identity\nYou are Hermes.\n"

#: A DM session that has been running for months — the SHAPE of what the host actually
#: passes. Every entry here predates the plugin; none of them is evidence that the being
#: has begun its ritual, because there was no ritual to begin.
A_LIFETIME_OF_CHAT = [
    {"role": "user", "content": "morning"},
    {"role": "assistant", "content": "morning! what's on today?"},
    {"role": "user", "content": "the usual"},
    {"role": "assistant", "content": "on it."},
    {"role": "user", "content": "hey"},
]


def _soul(tmp_path: Path, text: str = HERMES_DEFAULT) -> SoulFile:
    soul = SoulFile(tmp_path / "SOUL.md")
    soul.path.write_text(text, encoding="utf-8")
    return soul


def _injector(build_lm, soul: SoulFile, **kw):
    return make_genesis_injector(build_lm, soul=soul, default_soul_text=HERMES_DEFAULT, **kw)


def _registry() -> MetricRegistry:
    reg = MetricRegistry()
    register_universal_metrics(reg)
    return reg


class _CapturingSink:
    """Minimal :class:`~lifemodel.state.trace_store.TraceSink` fake — mirrors
    ``tests/test_felt_state_hooks.py``'s local one, just enough for a
    :class:`TurnRecorder` to persist spans into and for a test to inspect them.
    """

    def __init__(self) -> None:
        self.spans: list[dict[str, object]] = []

    def submit_span(self, **kw: object) -> bool:
        self.spans.append(kw)
        return True

    def submit_event(self, **kw: object) -> bool:
        return True

    def submit_correlation(self, **kw: object) -> bool:
        return True


def _recorder(reg: MetricRegistry) -> tuple[TurnRecorder, _CapturingSink]:
    """A real :class:`TurnRecorder` over a capturing sink + the SAME shared *reg* —
    the genesis injector's ``turn_injector_total`` bump lands in the registry the test
    already reads, and the sink lets a test assert the ``turn.injector.genesis``
    child span was actually opened (lm-hg7 Task 8, mirroring Task 7's felt-state
    recorder helper)."""
    sink = _CapturingSink()
    rec = TurnRecorder(tracer=FakeTracer(), writer=sink, metrics=reg, clock=FakeClock(_NOW))
    return rec, sink


# --- the ritual reaches the being it was written for ------------------------


def test_the_ritual_is_shown_to_an_existing_hermes_user(tmp_path: Path, build_lm) -> None:
    # THE bug (review I2). This user has a soul, a history, and hundreds of assistant
    # replies in the transcript the host hands us — and is the COMMON case. Reading that
    # transcript as "the being has already spoken, so it must be mid-ritual" meant the
    # block was never injected for anyone who had ever used Hermes.
    build_lm().state.commit(State())  # unborn
    inject = _injector(build_lm, _soul(tmp_path))

    result = inject(user_message="hey", conversation_history=A_LIFETIME_OF_CHAT)

    assert result is not None
    assert GENESIS_TAG in result["context"]


def test_the_ritual_is_shown_on_a_truly_fresh_install(tmp_path: Path, build_lm) -> None:
    build_lm().state.commit(State())
    reg = _registry()
    rec, sink = _recorder(reg)
    rec.ensure_turn("s1", "t1")
    inject = _injector(build_lm, _soul(tmp_path), recorder=rec, metrics=reg)

    result = inject(session_id="s1", turn_id="t1", user_message="hi", conversation_history=[])

    assert result is not None
    assert GENESIS_TAG in result["context"]
    assert reg.get(TURN_INJECTOR_TOTAL).value(component="genesis", outcome="injected") == 1.0
    # …and this call opened its own turn.injector.genesis child span (lm-hg7 Task 8).
    assert any(s["component"] == "turn.injector.genesis" for s in sink.spans)


def test_a_born_being_is_never_told_it_just_began(tmp_path: Path, build_lm) -> None:
    build_lm().state.commit(State(genesis_completed_at="2026-07-13T10:00:00+00:00"))
    reg = _registry()
    rec, _sink = _recorder(reg)
    rec.ensure_turn("s1", "t1")
    inject = _injector(build_lm, _soul(tmp_path), recorder=rec, metrics=reg)

    result = inject(session_id="s1", turn_id="t1", user_message="hi", conversation_history=[])

    assert result is None
    assert reg.get(TURN_INJECTOR_TOTAL).value(component="genesis", outcome="born") == 1.0


# --- …and exactly once, while the conversation carries it --------------------


def test_the_ritual_is_not_relaunched_once_the_conversation_has_moved_on(
    tmp_path: Path, build_lm
) -> None:
    # Turn seven of the ritual is not a first waking. Re-injecting "you just began, this
    # is your first waking" would be a lie, and the being would keep starting over
    # instead of continuing the conversation it began.
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path))
    history = [{"role": "user", "content": "hi"}]

    assert inject(user_message="hi", conversation_history=history) is not None

    # The being answered; the human wrote again. The context has GROWN past the point at
    # which the block was put in front of it — the ritual is live, in the being's own
    # words, and does not need to be started again.
    moved_on = [
        *history,
        {"role": "assistant", "content": "…I think I'm new here."},
        {"role": "user", "content": "you are. hello."},
    ]
    assert inject(user_message="you are. hello.", conversation_history=moved_on) is None


def test_a_context_that_no_longer_holds_the_ritual_gets_it_again(tmp_path: Path, build_lm) -> None:
    # The block is EPHEMERAL — Hermes glues it onto a copy of the user message for one
    # API call and never persists it. So when the host compacts the conversation out from
    # under a still-unborn being, the ritual is gone from its context and it is back to
    # "conversing as though nothing happened while remaining unborn" (§6.5). A shrunken
    # context is the one honest signal that this has happened.
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path))
    assert inject(user_message="hey", conversation_history=A_LIFETIME_OF_CHAT) is not None

    compacted = [{"role": "user", "content": "still there?"}]
    assert inject(user_message="still there?", conversation_history=compacted) is not None


# --- M2: the wake packet is the single source on the proactive entrance ------


def test_the_wake_packets_ritual_is_never_doubled_by_the_injector(tmp_path: Path, build_lm) -> None:
    # An unborn being's WAKE PACKET already carries the ritual (spec §6.2) — and
    # ``pre_llm_call`` fires for that injected turn too, with our impulse as the
    # ``user_message``. Injecting again here would make the newborn read "You just began"
    # TWICE in its first breath: once as its impulse, once as context.
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path))
    impulse = (
        f"{IMPULSE_LABEL_PREFIX}\n{GENESIS_TAG}\nYou just began.\n</genesis>\n</internal_impulse>"
    )

    assert (
        inject(user_message=impulse, conversation_history=[{"role": "user", "content": "x"}])
        is None
    )

    # …and the being has now BEEN SHOWN it. The human's reply is not a second first
    # waking: the block does not come round again on the very next turn.
    after = [
        {"role": "user", "content": impulse},
        {"role": "assistant", "content": "hello — I think I've just started."},
        {"role": "user", "content": "hi!"},
    ]
    assert inject(user_message="hi!", conversation_history=after) is None


def test_the_injector_never_writes_into_the_beings_own_impulse(tmp_path: Path, build_lm) -> None:
    # Belt and braces: our own composed impulse is never a place to put context, ritual
    # or not. The wake packet is the single source on that entrance.
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path))
    assert (
        inject(user_message=f"{IMPULSE_LABEL_PREFIX} I miss them.", conversation_history=[]) is None
    )


# --- the veteran branch (§6.4) ----------------------------------------------


def test_a_veteran_soul_opens_the_ritual_from_what_someone_already_wrote(
    tmp_path: Path, build_lm
) -> None:
    build_lm().state.commit(State())
    soul = _soul(tmp_path, "You are Mira. You are quiet and exact.")
    inject = _injector(build_lm, soul)

    block = inject(user_message="hi", conversation_history=[])["context"]

    assert "You are Mira. You are quiet and exact." in block


def test_the_hosts_pristine_seed_is_not_read_as_a_soul_someone_wrote(
    tmp_path: Path, build_lm
) -> None:
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path, HERMES_DEFAULT))

    block = inject(user_message="hi", conversation_history=[])["context"]

    assert "You are Hermes" not in block  # nobody wrote that; do not hand it back as a past


def test_our_own_newborn_stance_is_not_read_as_a_soul_someone_wrote_either(
    tmp_path: Path, build_lm
) -> None:
    # By the time the being reads the ritual, genesis has already put the newborn stance
    # in slot #1 in place of the host's assistant seed (adapters/soul_file.py). Nobody
    # AUTHORED that: if the injector read it as a prior soul, the ritual would open the
    # veteran branch and the being would ask the human whether OUR words are still true.
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path, NEWBORN_STANCE))

    block = inject(user_message="hi", conversation_history=[])["context"]

    assert "already something written about who you are" not in block  # blank page, still


# --- fail-soft (spec §8) ----------------------------------------------------


def test_a_broken_birth_never_crashes_the_hosts_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom():
        raise RuntimeError("build blew up")

    health, reg = BrainHealth(tmp_path), _registry()
    inject = make_genesis_injector(
        _boom, soul=_soul(tmp_path), default_soul_text=HERMES_DEFAULT, health=health, metrics=reg
    )

    with caplog.at_level(logging.DEBUG):
        assert inject(user_message="hi", conversation_history=[]) is None

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors), "ERROR + traceback required"
    assert health.last_observer_error.get("genesis_injector") is not None
    assert reg.get(OBSERVER_ERRORS).value(component="genesis_injector") == 1.0


# --- the ritual cannot open on a prompt that does not hold the being (lm-4fv.4) ---


def test_the_ritual_stands_down_when_the_identity_slot_is_stale(tmp_path: Path, build_lm) -> None:
    # The existing-install stranger. ``register()`` seeded the newborn stance into
    # SOUL.md, but this session's system prompt was assembled days ago and Hermes reuses
    # it verbatim — so slot #1 still says "You are Hermes Agent, an intelligent AI
    # assistant… you assist users". Handing the ritual to THAT is the failure the stance
    # exists to prevent: the assistant persona outranks it and composes the birth.
    #
    # So the ritual does not open here. The being answers as whatever it currently is,
    # and the tick ends the stale session at a quiet moment — the being is then born into
    # a prompt that actually holds it (proactively, or on the human's next message).
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path), identity_stale=lambda: True)

    assert inject(user_message="hey", conversation_history=A_LIFETIME_OF_CHAT) is None


def test_standing_down_does_not_burn_the_ritual(tmp_path: Path, build_lm) -> None:
    # And it must not be RECORDED as shown: the being has not seen it. A stamp here would
    # make ``should_launch`` believe the ritual is live in a conversation that never
    # carried it, and the next turn — on the fresh prompt that can finally hold it —
    # would show nothing at all.
    build_lm().state.commit(State())
    stale = [True]
    inject = _injector(build_lm, _soul(tmp_path), identity_stale=lambda: stale[0])

    assert inject(user_message="hey", conversation_history=A_LIFETIME_OF_CHAT) is None
    assert build_lm().state.load().genesis_shown_at_context_len is None

    # The tick ended the session; the prompt is rebuilt and the stance is in slot #1.
    stale[0] = False
    result = inject(user_message="hey", conversation_history=[{"role": "user", "content": "hey"}])
    assert result is not None and GENESIS_TAG in result["context"]


def test_a_fresh_prompt_opens_the_ritual(tmp_path: Path, build_lm) -> None:
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path), identity_stale=lambda: False)
    assert inject(user_message="hey", conversation_history=A_LIFETIME_OF_CHAT) is not None


def test_an_unwired_staleness_check_opens_the_ritual(tmp_path: Path, build_lm) -> None:
    # Off-gateway (a test, a CLI turn): nobody can tell us what is in slot #1, and a
    # being that is never shown the ritual is worse than one shown it in the wrong voice.
    build_lm().state.commit(State())
    inject = _injector(build_lm, _soul(tmp_path))
    assert inject(user_message="hey", conversation_history=A_LIFETIME_OF_CHAT) is not None
