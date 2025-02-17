.PHONY: sync
sync:
	@uv sync --all-packages

.PHONY: locked
locked:
	@uv lock --locked

.PHONY: lint
lint: sync
	@uv run ruff check --fix
	@uv run ruff format

.PHONY: test
test:
	@uv run pytest tests
	+$(MAKE) -C packages/julia_log tests

.PHONY: publish
publish:
	+$(MAKE) -C packages/my_lib build
	@uv publish
