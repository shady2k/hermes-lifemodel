from __future__ import annotations

import sys
import types
from datetime import timedelta, timezone, tzinfo

from lifemodel.adapters.owner_tz import resolve_owner_tz


def _fake_hermes_time(get_timezone) -> types.ModuleType:
    mod = types.ModuleType("hermes_time")
    mod.get_timezone = get_timezone  # type: ignore[attr-defined]
    return mod


def test_returns_the_hermes_configured_zone(monkeypatch) -> None:
    zone = timezone(timedelta(hours=3), "MSK")
    monkeypatch.setitem(sys.modules, "hermes_time", _fake_hermes_time(lambda: zone))
    assert resolve_owner_tz() is zone


def test_none_when_hermes_reports_no_zone(monkeypatch) -> None:
    # hermes_time.get_timezone() returns None when no IANA tz is configured → we
    # pass that through as "server-local" (the renderer's next fallback).
    monkeypatch.setitem(sys.modules, "hermes_time", _fake_hermes_time(lambda: None))
    assert resolve_owner_tz() is None


def test_fail_open_to_none_when_resolution_raises(monkeypatch) -> None:
    def boom() -> tzinfo:
        raise RuntimeError("bad config")

    monkeypatch.setitem(sys.modules, "hermes_time", _fake_hermes_time(boom))
    assert resolve_owner_tz() is None  # never propagates — a tz quirk can't kill a tick


def test_none_when_hermes_is_absent(monkeypatch) -> None:
    # A dev checkout / non-Hermes venv has no ``hermes_time`` — importing it raises,
    # and the boundary degrades to None rather than exploding.
    monkeypatch.setitem(sys.modules, "hermes_time", None)  # None → ImportError on import
    assert resolve_owner_tz() is None


def test_never_raises_in_the_ambient_environment() -> None:
    # Whatever the real environment (Hermes present or not), the call is total.
    result = resolve_owner_tz()
    assert result is None or isinstance(result, tzinfo)
