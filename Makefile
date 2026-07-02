SHELL := /bin/bash

UV := $(shell if [ -x "$(HOME)/.local/bin/uv" ]; then echo "$(HOME)/.local/bin/uv"; else echo uv; fi)

.PHONY: install run debug clean lint lint-strict test grade

install:
	$(UV) sync

run:
	$(UV) run python -m src

debug:
	$(UV) run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --exclude '^llm_sdk/' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --exclude '^llm_sdk/' --strict

test:
	$(UV) run pytest

grade:
	cd moulinette && $(UV) run python -m moulinette grade_student_answers ../data/output/function_calling_results.json