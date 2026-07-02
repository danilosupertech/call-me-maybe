"""Command line entry point for the function calling tool."""

from typing import Any

from .cli import main


def cleanup_output(data: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Clean common noisy string parameters without changing schema shape."""
    if "parameters" not in data or not isinstance(data["parameters"], dict):
        return data
    cleaned = dict(data)
    parameters = dict(data["parameters"])
    for key, value in parameters.items():
        if isinstance(value, str):
            parameters[key] = _clean_string_parameter(key, value, prompt)
    cleaned["parameters"] = parameters
    return cleaned


def _clean_string_parameter(key: str, value: str, prompt: str) -> str:
    """Normalize one extracted string parameter."""
    cleaned = value.replace("\\\\", "\\")
    if (cleaned.startswith("'") and "'" in cleaned[1:]) or (
        cleaned.startswith('"') and '"' in cleaned[1:]
    ):
        quote = cleaned[0]
        cleaned = cleaned[1:].split(quote, 1)[0]
    if key == "database" and cleaned.lower().endswith(" database"):
        cleaned = cleaned[: -len(" database")]
    if " on the database" in cleaned.lower():
        index = cleaned.lower().find(" on the database")
        cleaned = cleaned[:index]
    return cleaned.strip()


if __name__ == "__main__":
    raise SystemExit(main())
