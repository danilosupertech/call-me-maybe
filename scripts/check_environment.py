"""Check project compliance and llm_sdk runtime availability."""

import importlib
import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


FORBIDDEN = (
    "torch",
    "pytorch",
    "transformers",
    "huggingface",
    "dspy",
    "outlines",
)


def main() -> int:
    """Run environment checks and return a status code."""
    root = Path(__file__).resolve().parents[1]
    print("Project root:", root)
    _check_root_dependencies(root)
    _check_project_imports(root)
    _check_llm_sdk_metadata(root)
    return _check_llm_sdk(root)


def _check_root_dependencies(root: Path) -> None:
    """Check direct project dependencies separately from SDK dependencies."""
    path = root / "pyproject.toml"
    if not path.exists():
        print("pyproject.toml: missing")
        return
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    dependencies = data.get("project", {}).get("dependencies", [])
    direct_names = {_dependency_name(item) for item in dependencies}
    forbidden = sorted(direct_names.intersection(FORBIDDEN))
    if forbidden:
        print(f"pyproject.toml: forbidden direct dependencies: {forbidden}")
    else:
        print("pyproject.toml: direct dependencies clean")
    if "llm-sdk" in direct_names:
        print("pyproject.toml: uses local llm_sdk package")


def _dependency_name(requirement: str) -> str:
    """Return a normalized package name from a dependency requirement."""
    name = requirement.split(";", maxsplit=1)[0]
    name = name.split("[", maxsplit=1)[0]
    name = name.split("@", maxsplit=1)[0]
    for separator in ("<", ">", "=", "!", "~"):
        name = name.split(separator, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _check_project_imports(root: Path) -> None:
    """Check project source files for direct forbidden imports."""
    checked_roots = (root / "src", root / "tests", root / "scripts")
    matches: list[str] = []
    for source_root in checked_roots:
        for path in source_root.rglob("*.py"):
            content = path.read_text(encoding="utf-8").lower()
            for name in FORBIDDEN:
                if f"import {name}" in content or f"from {name}" in content:
                    matches.append(str(path.relative_to(root)))
    if matches:
        print("project imports: forbidden direct imports found")
        for match in matches:
            print(f"  - {match}")
    else:
        print("project imports: clean")


def _check_llm_sdk_metadata(root: Path) -> None:
    """Report SDK dependencies as allowed SDK-internal requirements."""
    path = root / "llm_sdk" / "pyproject.toml"
    if not path.exists():
        print("llm_sdk/pyproject.toml: missing")
        return
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    dependencies = data.get("project", {}).get("dependencies", [])
    names = sorted(_dependency_name(item) for item in dependencies)
    print(f"llm_sdk dependencies: {json.dumps(names)}")


def _check_llm_sdk(root: Path) -> int:
    """Check whether the copied SDK can be imported in this environment."""
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("llm_sdk")
    except Exception as exc:
        print("llm_sdk: cannot import")
        print(f"reason: {exc}")
        print(
            "The SDK is part of the subject and owns its internal model "
            "dependencies. Run `uv sync` from the project root so the local "
            "`llm-sdk` path dependency can install them."
        )
        return 1
    if not hasattr(module, "Small_LLM_Model"):
        print("llm_sdk: imported, but Small_LLM_Model is missing")
        return 1
    print("llm_sdk: import ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
