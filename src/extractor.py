"""Schema-aware argument extraction from natural-language prompts."""

import re
from typing import Any

from .models import FunctionDefinition, TypeSpec

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def normalize_words(text: str) -> list[str]:
    """Return normalized searchable words."""
    return [word.lower() for word in WORD_RE.findall(text)]


def extract_arguments(
    prompt: str,
    function: FunctionDefinition,
) -> dict[str, Any]:
    """Extract arguments matching a function definition."""
    values: dict[str, Any] = {}
    number_index = 0
    string_index = 0
    boolean_index = 0
    numbers = _extract_numbers(prompt)
    strings = _extract_strings(prompt)
    booleans = _extract_booleans(prompt)

    for name, spec in function.parameters.items():
        value: Any
        if spec.type in {"number", "integer"}:
            value = _number_for(name, numbers, number_index, spec.type)
            number_index += 1
        elif spec.type == "string":
            value = _string_for(name, prompt, strings, string_index)
            string_index += 1
        elif spec.type == "boolean":
            value = _boolean_for(booleans, boolean_index)
            boolean_index += 1
        elif spec.type == "object":
            value = _object_for(prompt, spec)
        elif spec.type == "array":
            value = _array_for(prompt, spec)
        else:
            value = None
        values[name] = value
    return values


def _extract_numbers(prompt: str) -> list[float]:
    """Extract numbers in text order."""
    return [float(match.group(0)) for match in NUMBER_RE.finditer(prompt)]


def _extract_strings(prompt: str) -> list[str]:
    """Extract explicitly quoted strings."""
    strings: list[str] = []
    for pattern in (r"'([^']*)'", r'"([^"]*)"', r"`([^`]*)`"):
        strings.extend(re.findall(pattern, prompt))
    return strings


def _extract_booleans(prompt: str) -> list[bool]:
    """Extract boolean values from common textual forms."""
    values: list[bool] = []
    for word in normalize_words(prompt):
        if word in {"true", "yes", "on", "enable", "enabled"}:
            values.append(True)
        if word in {"false", "no", "off", "disable", "disabled"}:
            values.append(False)
    return values


def _number_for(
    name: str,
    numbers: list[float],
    index: int,
    type_name: str,
) -> float | int:
    """Select a number for a named parameter."""
    if index < len(numbers):
        number = numbers[index]
    else:
        number = 0.0
    if type_name == "integer":
        return int(number)
    return float(number)


def _string_for(
    name: str,
    prompt: str,
    strings: list[str],
    index: int,
) -> str:
    """Select a string for a named parameter."""
    named_value = _value_after_name(prompt, name)
    if named_value:
        return named_value
    if index < len(strings):
        return strings[index]
    words = normalize_words(prompt)
    stop_words = {
        "the",
        "a",
        "an",
        "to",
        "for",
        "with",
        "please",
        "string",
        "text",
        "name",
        "greet",
        "reverse",
    }
    content_words = [word for word in words if word not in stop_words]
    return content_words[-1] if content_words else ""


def _boolean_for(values: list[bool], index: int) -> bool:
    """Select a boolean value."""
    if index < len(values):
        return values[index]
    return False


def _object_for(prompt: str, spec: TypeSpec) -> dict[str, Any]:
    """Build a nested object using the same extraction rules."""
    fake_function = FunctionDefinition(
        name="nested",
        description="nested object",
        parameters=spec.properties,
        returns=TypeSpec(type="object"),
    )
    return extract_arguments(prompt, fake_function)


def _array_for(prompt: str, spec: TypeSpec) -> list[Any]:
    """Build a simple array from prompt values."""
    item_type = spec.items.type if spec.items is not None else "string"
    if item_type in {"number", "integer"}:
        numbers = _extract_numbers(prompt)
        if item_type == "integer":
            return [int(number) for number in numbers]
        return [float(number) for number in numbers]
    if item_type == "boolean":
        return _extract_booleans(prompt)
    return _extract_strings(prompt)


def _value_after_name(prompt: str, name: str) -> str:
    """Find values introduced by a parameter name."""
    pattern = rf"\b{re.escape(name)}\b\s*(?:=|:|is|as)?\s*['\"]?([^,'\".]+)"
    match = re.search(pattern, prompt, flags=re.IGNORECASE)
    if match is None:
        return ""
    return match.group(1).strip()
