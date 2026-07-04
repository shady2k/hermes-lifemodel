"""Unit tests for idempotent cron heartbeat registration (roadmap 1.1).

The Hermes cron API is injected as plain fakes, so these prove the idempotency
and shim-generation logic without importing Hermes. The critical footgun —
``register(ctx)`` runs on every plugin load, so registering unconditionally
duplicates the job — is pinned by :func:`test_ensure_is_idempotent`.
"""

from __future__ import annotations

import compileall
from pathlib import Path
from typing import Any

from lifemodel.heartbeat import (
    AUTHOR_DELIVER,
    HEARTBEAT_JOB_NAME,
    HEARTBEAT_SCHEDULE,
    NO_TOOLS_ENABLED_TOOLSETS,
    SHIM_FILENAME,
    ensure_heartbeat_job,
    render_shim,
    write_shim,
)


class FakeCron:
    """In-memory stand-in for the slice of ``cron.jobs`` the heartbeat uses."""

    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    def create_job(
        self,
        prompt: str,
        schedule: str,
        *,
        name: str | None = None,
        script: str | None = None,
        no_agent: bool = False,
        deliver: str | None = None,
        enabled_toolsets: list[str] | None = None,
        attach_to_session: bool | None = None,
    ) -> dict[str, Any]:
        job = {
            "id": f"job{len(self.jobs)}",
            "name": name,
            "prompt": prompt,
            "schedule": schedule,
            "script": script,
            "no_agent": no_agent,
            "deliver": deliver,
            "enabled_toolsets": enabled_toolsets,
            "attach_to_session": attach_to_session,
            "enabled": True,
        }
        self.jobs.append(job)
        self.create_calls.append(job)
        return job

    def list_jobs(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        if include_disabled:
            return list(self.jobs)
        return [j for j in self.jobs if j.get("enabled", True)]


def _ensure(home: Path, cron: FakeCron) -> dict[str, Any]:
    src_dir = home / "plugin" / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    return ensure_heartbeat_job(
        home=home,
        src_dir=src_dir,
        create_job=cron.create_job,
        list_jobs=cron.list_jobs,
    )


def test_ensure_creates_heartbeat_when_absent(tmp_path: Path) -> None:
    cron = FakeCron()

    job = _ensure(tmp_path, cron)

    assert len(cron.create_calls) == 1
    assert job["name"] == HEARTBEAT_JOB_NAME
    assert job["schedule"] == HEARTBEAT_SCHEDULE
    assert job["script"] == SHIM_FILENAME
    # Must be able to wake later (1.3+): a script tick, not a no_agent watchdog.
    assert job["no_agent"] is False
    # The launcher shim exists under the profile's scripts dir.
    assert (tmp_path / "scripts" / SHIM_FILENAME).is_file()


def test_heartbeat_job_wires_phase_1_4_minimal_safety(tmp_path: Path) -> None:
    # The woken turn must be text-only (no tools), author/home-channel only, and
    # carry the cognition prompt — the Phase-1.4 rails wired structurally at
    # registration (the ≤1/cycle + cooldown rails live in the tick's drain).
    cron = FakeCron()

    job = _ensure(tmp_path, cron)

    # No tools: the empty-set sentinel (a literal [] would normalize to "all
    # tools"); the real scheduler resolves ["no_mcp"] to an empty toolset.
    assert job["enabled_toolsets"] == list(NO_TOOLS_ENABLED_TOOLSETS)
    assert job["enabled_toolsets"] == ["no_mcp"]
    # Author / home channel only, no third parties.
    assert job["deliver"] == AUTHOR_DELIVER == "origin"
    # A real, text-only cognition instruction that addresses only the author and
    # forbids tools.
    prompt = job["prompt"].lower()
    assert "author" in prompt
    assert "one" in prompt
    assert "tool" in prompt  # "do not use any tools"


def test_heartbeat_job_attaches_wake_message_to_session(tmp_path: Path) -> None:
    # lm-dlw: a proactive wake fires from cron, whose reply lives only in the
    # cron job's own session unless attach_to_session=True — then the scheduler
    # mirrors the woken turn into the origin chat (cron/scheduler.py ~L407-419,
    # persisted only when explicitly set per cron/jobs.py ~L1011-1015). Without
    # it the main session has amnesia about its own outreach and confabulates.
    # Assert the flag is wired structurally at registration, like the other
    # Phase-1.4 rails.
    cron = FakeCron()

    _ensure(tmp_path, cron)

    assert len(cron.create_calls) == 1
    assert cron.create_calls[0]["attach_to_session"] is True


def test_production_registration_delivers_to_author_origin(tmp_path: Path) -> None:
    # FINDING 4: the integration forces deliver="local" for no-outbound safety, so
    # the PRODUCTION author route is asserted here at the config layer instead. The
    # real registration call shape (no deliver override — exactly what
    # register_heartbeat passes) must create the job with deliver="origin", i.e.
    # the author's origin/home channel, never a third party.
    cron = FakeCron()
    src_dir = tmp_path / "plugin" / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    job = ensure_heartbeat_job(
        home=tmp_path,
        src_dir=src_dir,
        create_job=cron.create_job,
        list_jobs=cron.list_jobs,
        # NB: no `deliver=` — mirrors register_heartbeat's production call, which
        # relies on the AUTHOR_DELIVER default.
    )

    assert len(cron.create_calls) == 1
    assert cron.create_calls[0]["deliver"] == "origin"
    assert job["deliver"] == AUTHOR_DELIVER == "origin"


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    # The footgun: register() runs on every load. Calling ensure repeatedly must
    # never create a second heartbeat.
    cron = FakeCron()

    first = _ensure(tmp_path, cron)
    second = _ensure(tmp_path, cron)
    third = _ensure(tmp_path, cron)

    assert len(cron.create_calls) == 1
    assert first["id"] == second["id"] == third["id"]


def test_ensure_recognizes_a_disabled_heartbeat(tmp_path: Path) -> None:
    # A paused/disabled heartbeat is still "the heartbeat" — don't create a
    # duplicate beside it (we match include_disabled=True).
    cron = FakeCron()
    cron.jobs.append({"id": "old", "name": HEARTBEAT_JOB_NAME, "enabled": False})

    job = _ensure(tmp_path, cron)

    assert cron.create_calls == []
    assert job["id"] == "old"


def test_write_shim_bakes_src_dir_and_targets_scripts(tmp_path: Path) -> None:
    src_dir = tmp_path / "some" / "src"
    src_dir.mkdir(parents=True)

    shim_path = write_shim(tmp_path, src_dir)

    assert shim_path == tmp_path / "scripts" / SHIM_FILENAME
    content = shim_path.read_text(encoding="utf-8")
    assert str(src_dir) in content
    assert "from lifemodel.tick import main" in content


def test_write_shim_is_idempotent_but_rewrites_on_move(tmp_path: Path) -> None:
    src_a = tmp_path / "a"
    src_a.mkdir()
    shim = write_shim(tmp_path, src_a)
    content_a = shim.read_text(encoding="utf-8")

    # Same src → byte-identical (no needless rewrite).
    write_shim(tmp_path, src_a)
    assert shim.read_text(encoding="utf-8") == content_a

    # Moved plugin → shim updates to the new import path.
    src_b = tmp_path / "b"
    src_b.mkdir()
    write_shim(tmp_path, src_b)
    updated = shim.read_text(encoding="utf-8")
    assert str(src_b) in updated
    assert str(src_a) not in updated


def test_generated_shim_is_valid_python(tmp_path: Path) -> None:
    # The shim runs under Hermes' interpreter — a syntax error would silently
    # break every tick. Compile it to prove it parses.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    write_shim(tmp_path, src_dir)

    assert compileall.compile_dir(str(tmp_path / "scripts"), quiet=1)


def test_render_shim_inserts_path_before_import() -> None:
    shim = render_shim(Path("/opt/plugin/src"))
    insert_at = shim.index("sys.path.insert(0, '/opt/plugin/src')")
    import_at = shim.index("from lifemodel.tick import main")
    assert insert_at < import_at  # path must be on sys.path before we import
