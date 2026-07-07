# hermes-lifemodel — dev commands. Everything runs under uv.
.DEFAULT_GOAL := help

.PHONY: help check fmt test deploy

help:  ## List the available commands
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk -F':.*## ' '{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

check:  ## Run the full quality gate: format check, lint, types, tests
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy -p lifemodel
	uv run pytest

fmt:  ## Auto-format the code with ruff
	uv run ruff format .

test:  ## Run the test suite
	uv run pytest

deploy:  ## Deploy to the live being: push, pull into ~/.hermes, restart gateway
	@git diff --quiet && git diff --cached --quiet || { echo "!! commit your changes first — 'hermes plugins update' pulls from git, not the working tree"; exit 1; }
	git push origin main
	hermes plugins update lifemodel
	hermes gateway restart
	hermes gateway status
