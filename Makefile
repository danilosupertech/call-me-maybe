UV := $(if $(wildcard .venv/Scripts/uv.exe),.venv/Scripts/uv.exe,uv)

.PHONY: install run run-verbose demo-flow demo-flow-fast debug clean lint lint-strict test doctor

install:
	python -m ensurepip --upgrade
	python -m pip install uv
	$(UV) sync

run:
	$(UV) run python -u -m src --verbose

run-verbose:
	$(UV) run python -u -m src --verbose

demo-flow:
	python scripts/demo_execution_flow.py --delay 2

demo-flow-fast:
	python scripts/demo_execution_flow.py --delay 0 --all

debug:
	$(UV) run python -m pdb -m src

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.mypy_cache'), pathlib.Path('.pytest_cache')]]"

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

test:
	$(UV) run pytest

doctor:
	$(UV) run python scripts/check_environment.py
