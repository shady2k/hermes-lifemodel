# hermes-lifemodel — dev commands. Everything runs under uv.
.DEFAULT_GOAL := help

# The Hermes runtime interpreter — the ONLY venv with the `gateway` package.
# Override if your being lives elsewhere: `make smoke HERMES_VENV_PY=/path/to/python`.
HERMES_VENV_PY ?= $(HOME)/.hermes/hermes-agent/venv/bin/python

.PHONY: help check fmt test mypy smoke deploy

help:  ## List the available commands
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk -F':.*## ' '{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

check:  ## Run the full quality gate: format check, lint, types, tests
	uv run ruff format --check .
	uv run ruff check .
	$(MAKE) mypy
	uv run pytest

mypy:  ## Type-check the package (dir-name independent — works from any worktree name)
	@tmp="$$(mktemp -d)"; \
		ln -s "$(CURDIR)" "$$tmp/lifemodel"; \
		MYPYPATH="$$tmp" uv run mypy -p lifemodel; status=$$?; \
		rm -rf "$$tmp"; \
		exit $$status

fmt:  ## Auto-format the code with ruff
	uv run ruff format .

test:  ## Run the test suite
	uv run pytest

smoke:  ## Adapter-shell smoke check against the Hermes venv (pre-deploy, needs gateway)
	@test -x "$(HERMES_VENV_PY)" || { echo "!! Hermes venv python not found at $(HERMES_VENV_PY) — set HERMES_VENV_PY=/path/to/python"; exit 1; }
	PYTHONPATH=.. "$(HERMES_VENV_PY)" -m lifemodel.smoke

deploy: smoke  ## Deploy to the live being: smoke, push, pull into ~/.hermes, restart gateway
	@git diff --quiet && git diff --cached --quiet || { echo "!! commit your changes first — 'hermes plugins update' pulls from git, not the working tree"; exit 1; }
	git push origin main
	hermes plugins update lifemodel
	hermes gateway restart
	hermes gateway status
