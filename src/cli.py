"""Command line interface."""

import argparse
import sys
from pathlib import Path

from .decoder import ConstrainedFunctionDecoder, DecodeTrace
from .errors import UserFacingError
from .io_utils import (
    read_function_definitions,
    read_prompt_cases,
    write_results,
)
from .llm_client import build_scorer

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the prompt test JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated output JSON file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name passed to llm_sdk when available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show prompt-by-prompt decoding details in the terminal.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the program and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        functions = read_function_definitions(args.functions_definition)
        prompts = read_prompt_cases(args.input)
        # print(f"Prompt Cases: {len(prompts)} -- {args.input}")
        scorer = build_scorer(args.model)
        # print(f"Scorer: {scorer}")
        if not scorer.available:
            raise UserFacingError(
                "llm_sdk could not be loaded. The subject requires the "
                "function choice to be made by the LLM: "
                f"{scorer.unavailable_reason}"
            )
        decoder = ConstrainedFunctionDecoder(
            functions=functions,
            scorer=scorer,
        )
        results = []
        for index, item in enumerate(prompts, start=1):
            print(f"Processing prompt {index}/{len(prompts)}")
            trace = decoder.decode_with_trace(item.prompt)
            print(f"Selected function: {trace.selected.name}")
            results.append(trace.selected)
            if args.verbose:
                _print_trace(index, len(prompts), trace)
        write_results(args.output, results)
        print(f"Wrote {len(results)} function calls to {args.output}")
        return 0
    except UserFacingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130


def _print_trace(index: int, total: int, trace: DecodeTrace) -> None:
    """Print one decoding trace without changing the JSON output."""
    print(f"\n[{index}/{total}] Prompt")
    print(f"  {trace.prompt}")
    print("  Candidates")
    for candidate in trace.candidates:
        print(
            "  - "
            f"{candidate.name} score={candidate.score} "
            f"params={candidate.parameters}"
        )
    print("  Selected")
    print(
        "  -> "
        f"{trace.selected.name} "
        f"params={trace.selected.parameters}"
    )
