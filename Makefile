export SRC_DIR=src

format:
	uv run ruff format $(SRC_DIR)

lint:
	uv run ruff check --fix $(SRC_DIR)
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package $(SRC_DIR)