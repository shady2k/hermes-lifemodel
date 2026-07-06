from __future__ import annotations

from lifemodel.core.output_lint import LintResult, lint_proactive


def test_passes_a_warm_natural_message() -> None:
    r = lint_proactive("Саш, привет! Давно тебя не слышал, как ты?")
    assert isinstance(r, LintResult)
    assert r.ok is True


def test_flags_mechanical_timer_justification() -> None:
    assert lint_proactive("Прошло шесть часов тишины, решил проверить.").ok is False
    assert lint_proactive("6 hours of silence detected — checking in.").ok is False


def test_flags_contentless_filler() -> None:
    assert lint_proactive("Молчу, мне нечего сказать, но решил написать.").ok is False
    assert lint_proactive("Nothing to add, just checking in.").ok is False


def test_flag_gives_a_reason() -> None:
    r = lint_proactive("инициирую проверку статуса")
    assert r.ok is False and r.reason


def test_natural_time_mention_is_not_flagged() -> None:
    # a human "it's been a while" must pass — barrier is on mechanism, not time
    assert lint_proactive("Сто лет не общались, скучаю по нашим разговорам.").ok is True
