"""Tests for constrained function decoding."""

import json
import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decoder import ConstrainedFunctionDecoder
from src.errors import UserFacingError
from src.io_utils import read_function_definitions, read_prompt_cases
from src.llm_client import TextScorer


class FakeScorer(TextScorer):
    """Test scorer that mimics constrained LLM selection."""

    preferred_name: str = ""

    def choose_constrained(
        self,
        prompt: str,
        candidates: list[str],
    ) -> str:
        """Select a candidate by function name for deterministic tests."""
        for candidate in candidates:
            if self.preferred_name in candidate:
                return candidate
        return candidates[0]


def test_decodes_addition(tmp_path: Path) -> None:
    """The decoder should choose addition and keep numeric types."""
    definitions_path = tmp_path / "functions.json"
    definitions_path.write_text(
        json.dumps(
            [
                {
                    "name": "fn_add_numbers",
                    "description": "Add two numbers together.",
                    "parameters": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "returns": {"type": "number"},
                }
            ]
        ),
        encoding="utf-8",
    )
    decoder = ConstrainedFunctionDecoder(
        functions=read_function_definitions(definitions_path),
        scorer=FakeScorer(preferred_name="fn_add_numbers"),
    )

    result = decoder.decode("What is the sum of 40 and 2?")

    assert result.name == "fn_add_numbers"
    assert result.parameters == {"a": 40.0, "b": 2.0}


def test_decodes_quoted_string(tmp_path: Path) -> None:
    """The decoder should preserve quoted string arguments."""
    definitions_path = tmp_path / "functions.json"
    definitions_path.write_text(
        json.dumps(
            [
                {
                    "name": "fn_reverse_string",
                    "description": "Reverse a string.",
                    "parameters": {"s": {"type": "string"}},
                    "returns": {"type": "string"},
                }
            ]
        ),
        encoding="utf-8",
    )
    decoder = ConstrainedFunctionDecoder(
        functions=read_function_definitions(definitions_path),
        scorer=FakeScorer(preferred_name="fn_reverse_string"),
    )

    result = decoder.decode("Reverse the string 'hello'")

    assert result.name == "fn_reverse_string"
    assert result.parameters == {"s": "hello"}


def test_trace_lists_candidates(tmp_path: Path) -> None:
    """Verbose trace should expose candidates."""
    definitions_path = tmp_path / "functions.json"
    definitions_path.write_text(
        json.dumps(
            [
                {
                    "name": "fn_add_numbers",
                    "description": "Add two numbers together.",
                    "parameters": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "returns": {"type": "number"},
                },
                {
                    "name": "fn_greet",
                    "description": "Generate a greeting message.",
                    "parameters": {"name": {"type": "string"}},
                    "returns": {"type": "string"},
                },
            ]
        ),
        encoding="utf-8",
    )
    decoder = ConstrainedFunctionDecoder(
        functions=read_function_definitions(definitions_path),
        scorer=FakeScorer(preferred_name="fn_greet"),
    )

    trace = decoder.decode_with_trace("Greet Ada")

    assert trace.selected.name == "fn_greet"
    assert len(trace.candidates) == 2


def test_decodes_boolean_array_and_nested_object(tmp_path: Path) -> None:
    """Nested schema support should keep values schema-compliant."""
    definitions_path = tmp_path / "functions.json"
    definitions_path.write_text(
        json.dumps(
            [
                {
                    "name": "fn_configure_job",
                    "description": "Configure a job with enabled flag.",
                    "parameters": {
                        "enabled": {"type": "boolean"},
                        "values": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "count": {"type": "integer"},
                            },
                        },
                    },
                    "returns": {"type": "boolean"},
                }
            ]
        ),
        encoding="utf-8",
    )
    decoder = ConstrainedFunctionDecoder(
        functions=read_function_definitions(definitions_path),
        scorer=FakeScorer(preferred_name="fn_configure_job"),
    )

    result = decoder.decode("Enable job name alpha, count 3 values 1 2 3")

    assert result.parameters["enabled"] is True
    assert result.parameters["values"] == [3, 1, 2, 3]
    assert result.parameters["metadata"]["name"] == "alpha"
    assert result.parameters["metadata"]["count"] == 3


def test_missing_input_file_is_user_facing(tmp_path: Path) -> None:
    """Missing input files should not leak tracebacks."""
    with pytest.raises(UserFacingError):
        read_prompt_cases(tmp_path / "missing.json")


def test_invalid_input_json_is_user_facing(tmp_path: Path) -> None:
    """Malformed JSON should be reported as a user-facing error."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ invalid json", encoding="utf-8")

    with pytest.raises(UserFacingError):
        read_prompt_cases(bad_json)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
