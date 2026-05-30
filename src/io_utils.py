"""Input and output helpers."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import UserFacingError
from .models import FunctionCallResult, FunctionDefinition, PromptCase

ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_json(path: Path) -> Any:
    """Load JSON from disk with clear errors."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise UserFacingError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        message = (
            f"Invalid JSON in {path}: line {exc.lineno}, "
            f"column {exc.colno}"
        )
        raise UserFacingError(message) from exc
    except OSError as exc:
        raise UserFacingError(f"Cannot read {path}: {exc}") from exc


def _parse_list(path: Path, model: type[ModelT]) -> list[ModelT]:
    """Parse a JSON array into Pydantic models."""
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise UserFacingError(f"{path} must contain a JSON array")
    try:
        return [model.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise UserFacingError(f"Invalid structure in {path}: {exc}") from exc


def read_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Read function definitions."""
    definitions = _parse_list(path, FunctionDefinition)
    if not definitions:
        raise UserFacingError("No functions were provided")
    return definitions


def read_prompt_cases(path: Path) -> list[PromptCase]:
    """Read prompt cases."""
    return _parse_list(path, PromptCase)


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write output results as strict JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [result.model_dump(mode="json") for result in results]
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        raise UserFacingError(f"Cannot write {path}: {exc}") from exc
