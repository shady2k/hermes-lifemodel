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
from pathlib import Path

import pytest

from lifemodel.adapters.soul_file import SoulFile
from lifemodel.core.genesis import GENESIS_TAG
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.tick_metrics import OBSERVER_ERRORS, register_universal_metrics
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.hooks import make_genesis_injector
from lifemodel.state.brain_health import BrainHealth
from lifemodel.state.model import State

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
    inject = _injector(build_lm, _soul(tmp_path))
    assert GENESIS_TAG in inject(user_message="hi", conversation_history=[])["context"]


def test_a_born_being_is_never_told_it_just_began(tmp_path: Path, build_lm) -> None:
    build_lm().state.commit(State(genesis_completed_at="2026-07-13T10:00:00+00:00"))
    inject = _injector(build_lm, _soul(tmp_path))
    assert inject(user_message="hi", conversation_history=[]) is None


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
