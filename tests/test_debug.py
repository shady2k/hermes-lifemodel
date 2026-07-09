# tests/test_debug.py
from __future__ import annotations

import sqlite3
import sys
import types
from contextlib import closing
from datetime import timedelta, timezone

from lifemodel import debug as debug_mod
from lifemodel.adapters.clock import SystemClock
from lifemodel.debug import render_dump_for_dir
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore


def test_dump_renders_the_sections(tmp_path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(u=2.0, energy=0.6, fatigue=0.2, last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    for section in ("PHYSIOLOGY", "DRIVE", "DESIRE", "GATES", "BACKSTOP", "HEALTH"):
        assert section in out
    assert "effective" in out.lower()


def test_dump_rounds_floats_and_leaves_no_long_tails(tmp_path) -> None:
    # lm-25t: the read-only dump shows numbers at a readable precision — no raw
    # tails like "0.4199544..." or "u=5.250034172361112". DISPLAY ONLY: the
    # persisted state keeps full precision (asserted below).
    import re

    dirty = State(
        u=5.250034172361112,
        energy=0.6333333333,
        fatigue=0.2166666667,
        last_tick_at="2026-07-09T11:59:00+00:00",
    )
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(dirty)
    out = render_dump_for_dir(tmp_path)
    assert re.search(r"\d\.\d{3,}", out) is None, out  # no long float tail
    assert "**latent u:** 5.25" in out  # the rounded value is what shows
    assert SQLiteRuntimeStore(tmp_path, clock=SystemClock()).load().u == 5.250034172361112


def test_dump_puts_one_metric_per_line(tmp_path) -> None:
    # Owner complaint #1: several metrics used to be crammed onto one line.
    # Every metric now gets its own "label: value" line.
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(u=2.0, energy=0.6, fatigue=0.2, last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    lines = out.splitlines()

    energy_line = next(line for line in lines if "energy(E)" in line)
    assert "fatigue(S)" not in energy_line
    assert "circadian(C)" not in energy_line

    fatigue_line = next(line for line in lines if line.strip().startswith("**fatigue(S):**"))
    assert "circadian" not in fatigue_line
    assert "energy" not in fatigue_line

    drive_u_line = next(line for line in lines if line.strip().startswith("**latent u:**"))
    assert "inhibition" not in drive_u_line

    gates_wake_line = next(line for line in lines if line.strip().startswith("**would_wake:**"))
    assert "reason" not in gates_wake_line

    backstop_line = next(line for line in lines if line.strip().startswith("**sends_today:**"))
    assert "send_allowed" not in backstop_line


def test_dump_has_no_column_alignment_padding(tmp_path) -> None:
    # Owner complaint #3: right-padding labels to line up a colon column goes
    # ragged in Telegram's proportional font. Match Hermes' native /status
    # style instead: plain "**label:** value", a single space after the bold
    # colon, no leading indent, no padding run lining anything up.
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(u=2.0, energy=0.6, fatigue=0.2, last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    lines = out.splitlines()

    energy_line = next(line for line in lines if "energy(E)" in line)
    fatigue_line = next(line for line in lines if line.strip().startswith("**fatigue(S):**"))
    assert energy_line == "**energy(E):** 60%"
    assert fatigue_line == "**fatigue(S):** 0.20"

    for line in lines:
        if ":" not in line:
            continue
        assert not line.startswith(" ")  # no leading indent
        assert ":  " not in line  # no run of spaces after the colon


def test_dump_timestamps_render_in_the_hermes_local_timezone(monkeypatch, tmp_path) -> None:
    # Owner complaint #2: UTC confuses the owner (+03:00). Force a fixed
    # non-UTC offset via the single conversion helper and confirm every
    # rendered timestamp uses it — trimmed to whole seconds, not raw UTC.
    monkeypatch.setattr(debug_mod, "_resolve_tz", lambda: timezone(timedelta(hours=3)))
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(
            last_contact_at="2026-07-07T07:40:42.297762+00:00",
            last_tick_at="2026-07-06T00:00:00+00:00",
        )
    )
    out = render_dump_for_dir(tmp_path)
    assert "2026-07-07 10:40:42 +03:00" in out
    assert "+00:00" not in out  # no raw-UTC timestamp leaks through
    assert "297762" not in out  # microseconds trimmed
    assert "T10:40:42" not in out  # not the raw ISO 'T' separator


def test_dump_title_matches_hermes_status_style(tmp_path) -> None:
    # Match Hermes' own /status command: an emoji title line, no fenced code
    # block, no "====" divider — just plain lines in the native proportional
    # font (see the owner-cited /status sample in the bead).
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    assert out.startswith("🫀 **lifemodel debug** (read-only)")
    assert "```" not in out
    assert "=" * 10 not in out  # no leftover "====" divider rule


def test_dump_labels_and_title_are_bold_like_hermes_status(tmp_path) -> None:
    # Owner ask (lm-fib.4 follow-up): match /status's bold-label convention
    # byte-for-byte — standard-markdown **bold**, which the Telegram adapter
    # converts to MarkdownV2 (see locales/en.yaml's "**Session ID:** `{...}`"
    # and "**Agent Running:** {state}"). Title, section headers, and every
    # metric label (colon included) are bold; values stay plain.
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(u=2.0, energy=0.6, fatigue=0.2, last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    assert "**lifemodel debug**" in out
    assert "**energy(E):**" in out
    assert "**PHYSIOLOGY**" in out
    assert "**DRIVE (contact)**" in out


def test_dump_effective_hint_has_no_lone_asterisk(tmp_path) -> None:
    # A bare "*" in a value (the old "(= u * (1 - inhibition))" hint) can be
    # mistaken for a markdown bold/italic marker by the markdown->MarkdownV2
    # conversion. It must be replaced with an unambiguous multiplication
    # glyph so no stray "*" survives outside the bold-label markers.
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(u=2.0, energy=0.6, fatigue=0.2, last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    assert "u * (" not in out
    stripped = out.replace("**", "")
    assert "*" not in stripped


def test_resolve_tz_returns_none_offhost_when_hermes_time_is_unavailable() -> None:
    # No hermes_time on sys.path in the dev/test venv -> degrade to None, which
    # callers feed straight into datetime.astimezone(None) (system-local),
    # never raising over an absent Hermes dependency.
    assert "hermes_time" not in sys.modules
    assert debug_mod._resolve_tz() is None


def test_resolve_tz_prefers_the_hermes_configured_zone_when_available(monkeypatch) -> None:
    tz = timezone(timedelta(hours=3))
    fake = types.ModuleType("hermes_time")
    fake.get_timezone = lambda: tz  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_time", fake)
    assert debug_mod._resolve_tz() is tz


def test_resolve_tz_degrades_to_none_on_any_hermes_failure(monkeypatch) -> None:
    fake = types.ModuleType("hermes_time")

    def _boom() -> object:
        raise RuntimeError("config.yaml unreadable")

    fake.get_timezone = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_time", fake)
    assert debug_mod._resolve_tz() is None


def test_health_shows_brain_liveness_and_drops_service_alive(tmp_path) -> None:
    # last_tick far in the past (relative to real now) -> brain reads STALE
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    assert "HEALTH" in out
    assert "brain" in out.lower()
    assert "STALE" in out  # a stale last_tick surfaces as an explicit warning
    assert "service_alive" not in out  # the old two-brain liveness line is gone


def test_dump_survives_a_corrupt_store(tmp_path) -> None:
    # Construct the store first (creates + migrates lifemodel.sqlite), then
    # plant an unparseable state_json directly -- the DB *file* stays healthy
    # (so construction's own recovery/quarantine never kicks in); only the
    # runtime_state row's payload is garbage, which is load()'s job to catch.
    SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    with closing(sqlite3.connect(str(tmp_path / "lifemodel.sqlite"))) as conn, conn:
        conn.execute(
            "INSERT INTO runtime_state (id, state_json, updated_at, updated_at_epoch, revision) "
            "VALUES (1, ?, ?, 0, 0)",
            ("{ not json", "2026-07-06T00:00:00+00:00"),
        )
    out = render_dump_for_dir(tmp_path)
    assert "unreadable" in out.lower()  # graceful banner, no crash
    # Same bold title treatment as the healthy-store path, for consistency.
    assert out.startswith("🫀 **lifemodel debug** (read-only)")


def test_dump_renders_the_receptivity_section(tmp_path) -> None:
    from lifemodel.core.relationship_view import (
        EXPLICIT_CONFIDENCE,
        build_owner_relationship,
        encode_owner_relationship,
    )

    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.commit(State(u=2.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    store.put(
        encode_owner_relationship(
            build_owner_relationship(bad_hours=(1,), confidence=EXPLICIT_CONFIDENCE)
        )
    )
    out = render_dump_for_dir(tmp_path)
    assert "RECEPTIVITY" in out
    assert "allowed" in out.lower()


def test_dump_renders_the_thoughts_section_empty_by_default(tmp_path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(u=2.0, last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    assert "THOUGHTS" in out
    assert "**live:** none" in out  # behavior-neutral: no thoughts -> "none"


def test_dump_lists_live_thoughts(tmp_path) -> None:
    from lifemodel.core.thought_view import build_thought, encode_thought
    from lifemodel.domain.objects import ThoughtState

    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.commit(State(u=2.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    store.put(encode_thought(build_thought(id="t-hi", content="the flat question", salience=0.9)))
    store.put(encode_thought(build_thought(id="t-lo", content="the book they liked", salience=0.2)))
    store.put(
        encode_thought(
            build_thought(id="t-dead", content="handled", state=ThoughtState.DROPPED, salience=1.0)
        )
    )
    out = render_dump_for_dir(tmp_path)
    assert "**live:** 2" in out  # the two live ones, not the dropped one
    assert "the flat question [t-hi]" in out
    assert "the book they liked [t-lo]" in out
    assert "handled" not in out  # terminal thought excluded from the audit
    assert out.index("[t-hi]") < out.index("[t-lo]")  # salience order


def test_dump_shows_the_compact_contact_chain_when_present(tmp_path) -> None:
    # lm-27n.10: the debug dump carries ONE compact "why did I write" line — the
    # current contact intention chain — not the full graph.
    from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
    from lifemodel.core.intention_view import build_contact_intention, encode_contact_intention
    from lifemodel.core.trace import creation_provenance
    from lifemodel.domain.objects import (
        CONTACT_DESIRE_ID,
        DesireState,
        IntentionState,
        qualified_id,
    )
    from lifemodel.testing import FakeTracer

    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.commit(State(u=2.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0)))
    store.put(
        encode_contact_intention(
            build_contact_intention(
                state=IntentionState.ACTIVE,
                commitment_strength=2.0,
                provenance=creation_provenance(
                    FakeTracer().start_root(),
                    created_by="cognition",
                    component="cognition",
                    reason="crystallized contact intention",
                    source_object_ids=(qualified_id("desire", CONTACT_DESIRE_ID),),
                ),
            )
        )
    )
    out = render_dump_for_dir(tmp_path)
    assert "**why:** intention:contact:owner <- desire:contact:owner (source)" in out


def test_dump_contact_chain_is_no_outreach_when_absent(tmp_path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(u=2.0, last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    assert "**why:** no current outreach" in out
