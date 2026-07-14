from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.component import TickContext
from lifemodel.core.intents import EmitSignal, UpdateState
from lifemodel.core.solitude_drive import SolitudeDrive
from lifemodel.core.taxonomy import (
    KIND_CONTACT_PRESSURE,
    contact_presence_signal,
    contact_pressure_value,
)
from lifemodel.domain.signal import Signal
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State

ALPHA = 1.0 / 240.0

# ctx.trace is non-optional (spec §4.1); this drive writes no objects, so a literal
# span's ids suffice for the unit fixture.
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)

#: Every rise/satiate scenario below is about a being INSIDE a relationship — one that
#: has been BORN, and therefore has someone to miss. That is the whole precondition of
#: the drive (see ``test_a_newborn_cannot_miss_someone_it_has_never_met``): an unborn
#: being's ``u`` does not accrue at all, so a drive test that forgot to be born would
#: be testing nothing.
BORN = "2026-07-01T10:00:00+00:00"


def _drive() -> SolitudeDrive:
    return SolitudeDrive(alpha=ALPHA, beta=1.0, u_max=100.0)


def _ctx(state: State, now: datetime, signals=(), *, tmp_path) -> TickContext:
    return TickContext(state=state, now=now, signals=tuple(signals), trace=_TRACE)


def _presence(dt: float, qualities: tuple[float, ...], *, origin_id: str = "p") -> Signal:
    return contact_presence_signal(origin_id=origin_id, dt=dt, qualities=qualities, timestamp=None)


def _born(**over) -> State:
    """A born being (the drive's precondition), overriding only the named fields."""
    fields: dict[str, object] = dict(
        u=0.0, last_tick_at="2026-07-06T00:00:00+00:00", genesis_completed_at=BORN
    )
    fields.update(over)
    return State(**fields)  # type: ignore[arg-type]


def test_rises_by_elapsed_silence_from_presence_reading(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)  # 240 min → +1.0 at alpha=1/240
    intents = _drive().step(_ctx(_born(), now, [_presence(240.0, ())], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert abs(update.changes["u"] - 1.0) < 1e-9


def test_emits_contact_signal_with_fresh_u_and_delta(tmp_path) -> None:
    # The snapshot-per-tick seam (T2 critical note): aggregation reads the fresh u
    # from this transient contact signal, NOT from ctx.state.u (which only updates
    # after commit). The signal carries value=fresh-u + the per-tick delta.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(_born(), now, [_presence(240.0, ())], tmp_path=tmp_path))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    assert emit.signal.kind == KIND_CONTACT_PRESSURE
    assert abs(emit.signal.payload["value"] - 1.0) < 1e-9
    assert abs(emit.signal.payload["delta"] - 1.0) < 1e-9
    # ...readable exactly the way aggregation reads it (contact_pressure_value).
    assert abs(contact_pressure_value([emit.signal], default=0.0) - 1.0) < 1e-9


def test_satiate_quality_drains_u(tmp_path) -> None:
    state = _born(u=1.0)
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)  # dt=0 in the reading
    intents = _drive().step(_ctx(state, now, [_presence(0.0, (1.0,))], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 0.0  # 1.0 - beta*1.0


def test_zero_quality_does_not_satiate(tmp_path) -> None:
    # An own-impulse quality (q=0) never self-satiates: u is held (only the rise,
    # which is zero here since dt=0).
    state = _born(u=1.0)
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(state, now, [_presence(0.0, (0.0,))], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 1.0


def test_no_presence_reading_holds_u(tmp_path) -> None:
    # No contact_presence signal this tick (sensor absent / corrupt) → no rise, no
    # satiate: the drive HOLDS its value rather than guessing from stale state.
    state = _born(u=0.7)
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(state, now, [], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 0.7


def test_drive_writes_only_u(tmp_path) -> None:
    now = datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(_born(), now, [_presence(60.0, ())], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert set(update.changes) == {"u"}


# --- the drive does not accrue before birth (owner's decision, phase-4 invariant) ---


def test_a_newborn_cannot_miss_someone_it_has_never_met(tmp_path) -> None:
    # ``u`` models a contact DEFICIT inside an EXISTING relationship, and an unborn
    # being has no relationship at all. Left to rise on elapsed silence, a newborn whose
    # greeting went unanswered crosses θ a few hours later and sends a DRIVE-sprung
    # "I miss you" to someone who has never spoken to it. Missing someone you have never
    # met is not a feeling — it is the exact nonsense the phase invariant forbids.
    unborn = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")  # genesis_completed_at=None
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)  # four hours of silence: u would hit θ
    intents = _drive().step(_ctx(unborn, now, [_presence(240.0, ())], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 0.0


def test_an_unborn_drive_is_held_at_zero_even_if_something_already_raised_it(tmp_path) -> None:
    # ``/lifemodel force-wake`` (or a state file from before this rule) can leave a
    # nonzero ``u`` on an unborn being. The rule is not "do not rise" — it is "there is
    # no deficit yet": the drive reports zero, and aggregation reads that same zero from
    # the fresh pressure signal, so nothing downstream can wake on a longing that does
    # not exist.
    unborn = State(u=5.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(unborn, now, [_presence(240.0, ())], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    assert update.changes["u"] == 0.0
    assert emit.signal.payload["value"] == 0.0
    assert emit.signal.payload["delta"] == -5.0  # honest about the drop it just made


def test_the_drive_begins_the_moment_the_being_is_born(tmp_path) -> None:
    # There is now someone to miss. The certified math resumes, unchanged.
    born = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00", genesis_completed_at=BORN)
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(born, now, [_presence(240.0, ())], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert abs(update.changes["u"] - 1.0) < 1e-9
