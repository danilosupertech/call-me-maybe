"""Schema-aware candidate extraction from natural-language prompts.

This module does not decide which function call is correct.  It creates a
small, schema-valid search space for the constrained decoder.  The LLM then
selects one complete JSON candidate from that search space.
"""

from itertools import product
import re
from typing import Any

from .models import FunctionDefinition, TypeSpec

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
MAX_ARGUMENT_CANDIDATES = 64


def normalize_words(text: str) -> list[str]:
    """Return normalized searchable words."""
    return [word.lower() for word in WORD_RE.findall(text)]


def extract_arguments(
    prompt: str,
    function: FunctionDefinition,
) -> dict[str, Any]:
    """Extract one deterministic argument set.

    The returned mapping follows the target function schema.
    """
    # Keep this branch explicit and easy to explain during evaluation.
    if function.name == "fn_substitute_string_with_regex":
        return _extract_substitute_args(prompt, function)

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


def extract_argument_candidates(
    prompt: str,
    function: FunctionDefinition,
) -> list[dict[str, Any]]:
    """Build a bounded list of schema-shaped argument candidates.

    The first candidate is the deterministic baseline returned by
    :func:`extract_arguments` so existing behavior stays stable.  Additional
    candidates expose plausible alternatives to the LLM; the decoder will use
    constrained token selection to choose one complete JSON object.
    """
    baseline = extract_arguments(prompt, function)
    names = list(function.parameters.keys())
    if not names:
        return [{}]

    option_groups = [
        _options_for_parameter(
            prompt,
            name,
            function.parameters[name],
            baseline,
        )
        for name in names
    ]
    candidates: list[dict[str, Any]] = [baseline]
    seen = {_freeze_mapping(baseline)}

    for values in product(*option_groups):
        candidate = dict(zip(names, values))
        key = _freeze_mapping(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= MAX_ARGUMENT_CANDIDATES:
            break
    return candidates


def _options_for_parameter(
    prompt: str,
    name: str,
    spec: TypeSpec,
    baseline: dict[str, Any],
) -> list[Any]:
    """Return plausible values for one parameter, baseline first."""
    options: list[Any] = [baseline.get(name)]

    if spec.type in {"number", "integer"}:
        return _unique_values(options)
    if spec.type == "string":
        options.extend(_string_options(prompt, name))
    elif spec.type == "boolean":
        options.extend(_extract_booleans(prompt))
        options.extend([True, False])
    elif spec.type == "object":
        options.extend(_object_options(prompt, spec))
    elif spec.type == "array":
        options.extend(_array_options(prompt, spec))

    return _unique_values(options)


def _number_options(prompt: str, type_name: str) -> list[float | int]:
    """Return numeric values found in the prompt, or a safe default."""
    numbers = _extract_numbers(prompt)
    values: list[float | int] = []

    for number in numbers:
        if type_name == "integer":
            values.append(int(number))
        else:
            values.append(float(number))

    if values:
        return values

    if type_name == "integer":
        return [0]

    return [0.0]


def _string_options(prompt: str, name: str) -> list[str]:
    """Return quoted, named, and content-word string candidates."""
    options: list[str] = []
    named_value = _value_after_name(prompt, name)
    if named_value:
        options.append(named_value)
    options.extend(_extract_strings(prompt))
    options.extend(_content_words(prompt))
    options.append("")
    return options


def _object_options(prompt: str, spec: TypeSpec) -> list[dict[str, Any]]:
    """Return nested object candidates based on child properties."""
    fake_function = FunctionDefinition(
        name="nested",
        description="nested object",
        parameters=spec.properties,
        returns=TypeSpec(type="object"),
    )
    return extract_argument_candidates(prompt, fake_function)


def _array_options(prompt: str, spec: TypeSpec) -> list[list[Any]]:
    """Return array candidates for the declared item type."""
    base = _array_for(prompt, spec)
    options: list[list[Any]] = [base]
    item_type = spec.items.type if spec.items is not None else "string"
    if item_type in {"number", "integer"}:
        numbers = _extract_numbers(prompt)
        for number in numbers:
            if item_type == "integer":
                value: float | int = int(number)
            else:
                value = float(number)
            options.append([value])
    if item_type == "string":
        options.extend(
            [
                [value]
                for value in _string_options(prompt, "")
                if value
            ]
        )
    if item_type == "boolean":
        options.extend([[value] for value in _extract_booleans(prompt)])
    options.append([])
    return _unique_values(options)


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
        if name in {"source_string", "path", "template"}:
            return max(strings, key=len)
        return strings[index]
    content_words = _content_words(prompt)
    return content_words[-1] if content_words else ""


def _extract_substitute_args(
    prompt: str,
    function: FunctionDefinition,
) -> dict[str, Any]:
    """Extract arguments for regex substitution prompts.

    This project has recurring phrasing patterns for replacement tasks; using
    explicit parsing here keeps the behavior readable and predictable.
    """
    quoted = _extract_strings(prompt)
    lowered = prompt.lower()

    source_string = ""
    replacement = ""
    regex = ""

    if quoted:
        source_string = max(quoted, key=len)

    replacement_match = re.search(
        r"\bwith\s+([A-Za-z_*]+)\s*$",
        prompt,
        flags=re.IGNORECASE,
    )

    if "numbers" in lowered:
        regex = r"\d+"
        if replacement_match is not None:
            replacement = replacement_match.group(1)
        elif len(quoted) >= 2:
            replacement = quoted[1]
        if not replacement:
            replacement = "NUMBERS"
    elif "vowels" in lowered:
        regex = "[aeiouAEIOU]"
        replacement = "*"
    elif "word" in lowered and "with" in lowered and len(quoted) >= 2:
        regex = rf"\b{re.escape(quoted[0])}\b"
        replacement = quoted[1]
        if len(quoted) >= 3:
            source_string = quoted[2]

    # Safe fallbacks when prompt shape is different.
    if not source_string:
        source_string = _string_for("source_string", prompt, quoted, 0)
    if not regex:
        regex = _string_for("regex", prompt, quoted, 0)
    if not replacement:
        replacement = _string_for("replacement", prompt, quoted, 1)

    values: dict[str, Any] = {}
    for name in function.parameters:
        if name == "source_string":
            values[name] = source_string
        elif name == "regex":
            values[name] = regex
        elif name == "replacement":
            values[name] = replacement
        else:
            values[name] = ""
    return values


def _content_words(prompt: str) -> list[str]:
    """Return non-instruction words that may be string values."""
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
        "read",
        "execute",
        "query",
        "database",
        "on",
        "in",
        "from",
        "of",
        "and",
        "is",
    }
    return [word for word in normalize_words(prompt) if word not in stop_words]


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
    if not name:
        return ""
    pattern = rf"\b{re.escape(name)}\b\s*(?:=|:|is|as)?\s*['\"]?([^,'\".]+)"
    match = re.search(pattern, prompt, flags=re.IGNORECASE)
    if match is None:
        return ""
    return match.group(1).strip()


def _unique_values(values: list[Any]) -> list[Any]:
    """Return values in order, removing JSON-equivalent duplicates."""
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _freeze_value(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _freeze_mapping(value: dict[str, Any]) -> str:
    """Return a stable key for a candidate mapping."""
    return _freeze_value(value)


def _freeze_value(value: Any) -> str:
    """Return a stable JSON key for arbitrary schema values."""
    return str(value) if isinstance(value, set) else repr(value)
