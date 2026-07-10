# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Build & Test

The plugin runs **inside Hermes's own venv** (`~/.hermes/hermes-agent/venv`), which lacks our dev deps — runtime code must use **only Python stdlib + what Hermes provides** (treat any extra dep as optional, with a stdlib fallback). Dev tooling runs under `uv` in this repo.

```bash
make check   # full gate: ruff format --check, ruff check, mypy -p lifemodel, pytest
make test    # pytest only
make fmt     # auto-format
```

## Deploy (to the live being)

The plugin is installed from git at `~/.hermes/plugins/lifemodel` (a clone of `origin/main`). `hermes plugins update` pulls from **git**, not the working tree — so you must **commit + push first**. Command/adapter changes only take effect after a gateway restart (they register at plugin `register()`).

```bash
make deploy   # refuses on a dirty tree, then: git push → hermes plugins update lifemodel → hermes gateway restart → status
```

Equivalent by hand:

```bash
git push origin main
hermes plugins update lifemodel
hermes gateway restart
hermes gateway status
```

⚠️ `make deploy` targets the **owner's live being** (`~/.hermes`). For integration tests use an **isolated `HERMES_HOME`**, never the live being.

## Operating the Live Being (`/lifemodel` admin commands)

Proactive contact fires only when the drive `u` organically crosses `θ` — that can take **hours**. To make the being reach out **now** (e.g. to verify a wake-packet / prompt change live without waiting), the owner types this **in their DM chat with the being** — NOT from a shell:

```
/lifemodel force-wake
```

`force-wake` (see `state_commands.py:force_wake`) sets `u = θ + margin`, backdates `last_exchange_at` past the silence window, and clears decline-backoff / action-pending / the send backstop. It does **not** run a tick itself — the **next** brain tick's aggregation pass wakes cognition and the being reaches out. It echoes the satisfied gates so you can confirm.

- **Must be sent from the chat**, not a shell/python one-liner: the command runs inside the gateway process that holds the singleton state. An out-of-band mutation races the live tick and desyncs the in-memory singleton.
- **It won't defeat itself:** `hooks._is_control_command` excludes any `/…` message from the exchange signal, so sending the command does not count as contact and does not satiate the `u` it just raised.
- Sibling subcommands: `nudge [N]` (`u += N`, default +1.0), `satiate` (simulate a fulfilled contact — resets the drive), `reset` (factory wipe), `set <field> <value>`. `/lifemodel help` lists them all (mutating ones marked `[mutating]`).

## Architecture Overview

Docs live under `docs/` — product [`business-requirements.md`](docs/business-requirements.md), architecture [`hla.md`](docs/hla.md), delivery [`roadmap.md`](docs/roadmap.md) (phases = bd epics). Hexagonal layout: `core/` (Hermes-free layered engine: AUTONOMIC → AGGREGATION → COGNITION), `domain/`, `ports/`, `adapters/` (the only Hermes boundary; `being_platform.py` hosts the being as a supervised platform adapter), `state/`.

## Conventions & Patterns

_Add your project-specific conventions here_
