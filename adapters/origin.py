"""Resolve the author's Telegram home DM/thread lane — the proactive target.

The being's proactive reach-out must land in the user's *existing* Telegram DM
(so a reply loads the same session), which means the delivery target must equal
the home-origin conversation. This helper builds that ``{platform, chat_id,
thread_id}`` dict from Hermes' Telegram env convention.

It lives in the Hermes-boundary layer (``adapters/``) because it encodes Hermes'
env convention, but it imports no Hermes package — only ``os`` — so it stays
importable off-host. Returns ``None`` when the home channel is unset/empty, so a
misconfigured or non-Telegram host degrades gracefully (the caller must never
crash on a missing env var: it runs on every plugin load).
"""

from __future__ import annotations

import os


def resolve_home_origin() -> dict[str, str | None] | None:
    """Return the home-DM origin ``{platform, chat_id, thread_id}`` or ``None``.

    Reads ``TELEGRAM_HOME_CHANNEL`` (chat_id) and optional
    ``TELEGRAM_HOME_CHANNEL_THREAD_ID``. The dict shape mirrors the keys Hermes'
    cron mirror path compares (``_target_matches_origin``): ``user_id`` /
    ``chat_name`` are intentionally omitted — not compared, not available at
    registration time, harmless for a DM/shared session.
    """
    chat_id = os.environ.get("TELEGRAM_HOME_CHANNEL")
    if not chat_id:
        return None
    return {
        "platform": "telegram",
        "chat_id": chat_id,
        "thread_id": os.environ.get("TELEGRAM_HOME_CHANNEL_THREAD_ID") or None,
    }
