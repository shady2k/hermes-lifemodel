"""Tests for :func:`lifemodel.adapters.origin.resolve_home_origin`."""

from __future__ import annotations

from lifemodel.adapters.origin import resolve_home_origin


def test_returns_none_when_home_channel_unset(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    assert resolve_home_origin() is None


def test_builds_telegram_origin_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "12345")
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", raising=False)
    assert resolve_home_origin() == {
        "platform": "telegram",
        "chat_id": "12345",
        "thread_id": None,
    }


def test_includes_thread_id_when_set(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "12345")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "7")
    assert resolve_home_origin()["thread_id"] == "7"
