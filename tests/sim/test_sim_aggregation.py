"""The aggregation layer — the contact-desire lifecycle (spec §4, §5, §7).

The drive produces pressure; the *aggregation* layer owns the desire it becomes:
it creates one desire on the first wake-eligible urge, **dedups** every further
urge against that live desire (no duplicate wakes — the anti-drum guarantee),
holds a *deferred* desire until a release condition re-presents it, and clears
the desire on a real user exchange or a resolving cognition verdict. Logic lives
in this layer, never smeared into the drive (project convention).
"""

from __future__ import annotations

from lifemodel.sim.aggregation import Aggregator, DesireStatus, Verdict


def test_starts_with_no_desire() -> None:
    assert Aggregator().status is DesireStatus.NONE


def test_first_urge_creates_a_desire_and_wakes() -> None:
    agg = Aggregator()

    woke = agg.on_urge()

    assert woke is True
    assert agg.status is DesireStatus.ACTIVE


def test_repeat_urge_while_active_is_deduped_no_second_wake() -> None:
    # acked_urge_does_not_refire: a live desire absorbs further crossings.
    agg = Aggregator()
    agg.on_urge()

    woke_again = agg.on_urge()

    assert woke_again is False
    assert agg.status is DesireStatus.ACTIVE


def test_fulfill_resolves_the_desire() -> None:
    agg = Aggregator()
    agg.on_urge()

    agg.apply_verdict(Verdict.FULFILL)

    assert agg.status is DesireStatus.NONE


def test_reject_clears_the_desire() -> None:
    # reject → desire cleared (the growing backoff is the wake-decision's job,
    # not the aggregator's; here the desire simply resolves to NONE).
    agg = Aggregator()
    agg.on_urge()

    agg.apply_verdict(Verdict.REJECT)

    assert agg.status is DesireStatus.NONE


def test_defer_holds_the_desire() -> None:
    # defer → the intention is *held*, not dropped (never forgotten, never drums).
    agg = Aggregator()
    agg.on_urge()

    agg.apply_verdict(Verdict.DEFER)

    assert agg.status is DesireStatus.DEFERRED


def test_urge_while_deferred_is_deduped_no_new_wake() -> None:
    # A held desire absorbs new urges too — no duplicate wake while deferred.
    agg = Aggregator()
    agg.on_urge()
    agg.apply_verdict(Verdict.DEFER)

    woke = agg.on_urge()

    assert woke is False
    assert agg.status is DesireStatus.DEFERRED


def test_release_re_presents_a_deferred_desire() -> None:
    # deferred_intention_releases: a release condition re-wakes cognition.
    agg = Aggregator()
    agg.on_urge()
    agg.apply_verdict(Verdict.DEFER)

    re_woke = agg.on_release()

    assert re_woke is True
    assert agg.status is DesireStatus.ACTIVE


def test_release_is_a_noop_when_nothing_is_deferred() -> None:
    agg = Aggregator()
    assert agg.on_release() is False  # NONE

    agg.on_urge()
    assert agg.on_release() is False  # ACTIVE, not DEFERRED
    assert agg.status is DesireStatus.ACTIVE


def test_user_exchange_clears_a_live_desire() -> None:
    # user_message_satiates_and_resets: an inbound exchange clears the desire,
    # whether it was active or held.
    agg = Aggregator()
    agg.on_urge()
    agg.on_exchange()
    assert agg.status is DesireStatus.NONE

    agg.on_urge()
    agg.apply_verdict(Verdict.DEFER)
    agg.on_exchange()
    assert agg.status is DesireStatus.NONE
