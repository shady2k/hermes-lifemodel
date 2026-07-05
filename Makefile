# hermes-lifemodel — dev commands. Everything runs under uv.
.DEFAULT_GOAL := help

.PHONY: help check fmt test

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
