"""Show the execution flow step by step in the terminal.

This is an educational demo. It uses the real project modules, but replaces the
LLM with a small fake scorer so it can run even when llm_sdk/torch is not
installed.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pydantic import PrivateAttr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decoder import ConstrainedFunctionDecoder, _coerce_value
from src.extractor import extract_arguments
from src.io_utils import read_function_definitions, read_prompt_cases, write_results
from src.llm_client import TextScorer
from src.models import FunctionDefinition


TOKEN_RE = re.compile(
    r'"[^"]*"|-?\d+(?:\.\d+)?|true|false|null|[A-Za-z_][A-Za-z0-9_]*|[{}[\]:,?.]'
)


class TokenDebugScorer(TextScorer):
    """Visible constrained decoder used only for this terminal demo."""

    available: bool = True
    delay: float = 2.0
    pause: bool = False
    _vocab: dict[str, int] = PrivateAttr(default_factory=dict)
    _tokens_by_id: dict[int, str] = PrivateAttr(default_factory=dict)

    def wait(self) -> None:
        """Pause between visible runtime steps."""
        wait(self.delay, self.pause)

    def choose_constrained(
        self,
        prompt: str,
        candidates: list[str],
    ) -> str:
        """Show tokenization, logits, masking, and final constrained choice."""
        print_step("LLM scorer receives the prompt")
        print(f"prompt = {prompt!r}")
        self.wait()

        print_step("LLM scorer receives only valid JSON candidates")
        for index, candidate in enumerate(candidates, start=1):
            print(f"{index}. {candidate}")
        self.wait()

        preferred = self._preferred_candidate(prompt, candidates)
        print_step("For the demo, fake logits will prefer this candidate")
        print(preferred)
        print(f"reason     = {self._selection_reason(prompt, preferred)}")
        self.wait()

        print_step("Encode prompt context")
        context = (
            "Select the correct function call for this prompt.\n"
            f"Prompt: {prompt}\n"
            "JSON:"
        )
        context_ids = self.encode(context)
        print(f"context text = {context!r}")
        print(f"context ids  = {context_ids}")
        print(f"decode(ids)  = {self.decode(context_ids)!r}")
        self.wait()

        print_step("Encode candidates")
        tokenized = {}
        for candidate in candidates:
            token_ids = self.encode(candidate)
            tokenized[candidate] = token_ids
            print(f"\ncandidate = {candidate}")
            print(f"ids       = {token_ids}")
            print(f"decode    = {self.decode(token_ids)}")
        self.wait()

        target_ids = tokenized[preferred]
        generated: list[int] = []
        max_steps = max(len(ids) for ids in tokenized.values())

        print_step("Start constrained decoding")
        print(
            "At this point every candidate is represented as a list of token "
            "ids. The decoder will not let the model choose any token freely; "
            "it only allows tokens that continue at least one valid candidate."
        )
        self.wait()
        for step in range(1, max_steps + 1):
            exact_match = self._candidate_for_tokens(tokenized, generated)
            if exact_match is not None:
                print("\nGenerated tokens already match a full candidate.")
                return exact_match

            allowed = self._allowed_next_tokens(tokenized, generated)
            if not allowed:
                break

            input_ids = context_ids + generated
            logits = self._fake_logits(target_ids, generated, allowed)
            selected_token = self._best_allowed_token(logits, allowed)

            print(f"\nstep {step}")
            print(f"generated ids    = {generated}")
            print(f"generated text   = {self.decode(generated)!r}")
            print(f"model input size = {len(input_ids)} tokens")
            print(f"allowed tokens   = {self._format_tokens(allowed)}")
            print(f"raw logits       = {self._format_logits(logits, allowed)}")
            print("numpy mask       = invalid tokens become -inf")
            print(
                "selected token   = "
                f"{selected_token} {self._token_label(selected_token)}"
            )
            generated.append(selected_token)
            self.wait()

        exact_match = self._candidate_for_tokens(tokenized, generated)
        if exact_match is not None:
            print("\nGenerated tokens match a complete candidate.")
            return exact_match

        print("\nNo exact match reached, fallback to first candidate.")
        return candidates[0]

    def encode(self, text: str) -> list[int]:
        """Tokenize text and assign a stable integer ID to each token."""
        token_strings = TOKEN_RE.findall(text)
        if not token_strings:
            token_strings = text.split()
        print("\nencode(text)")
        print(f"input text = {text}")
        print(
            "idea       = encode breaks text into tokens and converts each "
            "token to a number"
        )
        print(
            "vocabulary = internal table that remembers token -> id mappings"
        )
        print(f"tokens     = {token_strings}")
        print("\nToken by token:")

        token_ids = []
        for token in token_strings:
            token_id, is_new = self._id_for(token)
            token_ids.append(token_id)
            if is_new:
                status = (
                    "new: this token was not in the vocabulary, "
                    "so the next free id was assigned"
                )
            else:
                status = (
                    "already known: this token appeared before, "
                    "so the same id is reused"
                )
            print(f"  token {token!r:<24} -> id {token_id:<3} ({status})")
            self.wait()

        print(f"output ids = {token_ids}")
        print(
            "meaning    = these ids are what the model/scorer works with; "
            "the text itself is no longer used directly"
        )
        print("current vocabulary:")
        for known_token, known_id in sorted(
            self._vocab.items(),
            key=lambda item: item[1],
        ):
            print(f"  id {known_id:<3} -> {known_token!r}")
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to token strings."""
        return "".join(self._tokens_by_id[token_id] for token_id in token_ids)

    def _preferred_candidate(self, prompt: str, candidates: list[str]) -> str:
        """Pick a target so the fake logits can mimic model preference."""
        selected = candidates[0]
        prompt_lower = prompt.lower()
        for candidate in candidates:
            if "sum" in prompt_lower and "fn_add_numbers" in candidate:
                selected = candidate
            if "reverse" in prompt_lower and "fn_reverse_string" in candidate:
                selected = candidate
        return selected

    def _selection_reason(self, prompt: str, selected: str) -> str:
        """Explain why the demo scorer prefers one candidate."""
        prompt_lower = prompt.lower()
        if "sum" in prompt_lower and "fn_add_numbers" in selected:
            return "the prompt contains 'sum', so addition is preferred"
        if "reverse" in prompt_lower and "fn_reverse_string" in selected:
            return "the prompt contains 'reverse', so string reversal is preferred"
        return "no demo keyword matched, so the first candidate is used"

    def _id_for(self, token: str) -> tuple[int, bool]:
        """Return an integer ID for a token string."""
        if token not in self._vocab:
            token_id = len(self._vocab)
            self._vocab[token] = token_id
            self._tokens_by_id[token_id] = token
            return token_id, True
        return self._vocab[token], False

    def _fake_logits(
        self,
        target_ids: list[int],
        generated: list[int],
        allowed: set[int],
    ) -> list[float]:
        """Create fake next-token logits for a visible NumPy mask demo."""
        vocab_size = len(self._vocab)
        logits = [-4.0] * vocab_size

        for token_id in range(vocab_size):
            if token_id not in allowed:
                logits[token_id] = 9.0
                break

        next_index = len(generated)
        if next_index < len(target_ids):
            target_token = target_ids[next_index]
            logits[target_token] = 8.0

        for token_id in allowed:
            if token_id < vocab_size and logits[token_id] < 2.0:
                logits[token_id] = 2.0
        return logits

    def _best_allowed_token(self, logits: list[float], allowed: set[int]) -> int:
        """Same masking idea used by src.llm_client._best_allowed_token()."""
        logits_array = np.asarray(logits, dtype=np.float64)
        masked_logits = np.full_like(logits_array, -np.inf)
        for token_id in allowed:
            if token_id < masked_logits.size:
                masked_logits[token_id] = logits_array[token_id]
        return int(np.argmax(masked_logits))

    def _candidate_for_tokens(
        self,
        candidates: dict[str, list[int]],
        token_ids: list[int],
    ) -> str | None:
        """Return the candidate matching the generated token sequence."""
        for candidate, candidate_tokens in candidates.items():
            if candidate_tokens == token_ids:
                return candidate
        return None

    def _allowed_next_tokens(
        self,
        candidates: dict[str, list[int]],
        prefix: list[int],
    ) -> set[int]:
        """Return tokens that keep the generated prefix valid."""
        allowed: set[int] = set()
        prefix_length = len(prefix)
        for candidate_tokens in candidates.values():
            if len(candidate_tokens) <= prefix_length:
                continue
            if candidate_tokens[:prefix_length] == prefix:
                allowed.add(candidate_tokens[prefix_length])
        return allowed

    def _format_tokens(self, token_ids: set[int]) -> str:
        """Format token IDs with their decoded strings."""
        parts = [
            f"{token_id}:{self._token_label(token_id)}"
            for token_id in sorted(token_ids)
        ]
        return "{" + ", ".join(parts) + "}"

    def _format_logits(self, logits: list[float], allowed: set[int]) -> str:
        """Show the most relevant logits for this step."""
        interesting = set(allowed)
        if logits:
            interesting.add(int(np.argmax(np.asarray(logits))))
        parts = [
            f"{token_id}:{self._token_label(token_id)}={logits[token_id]:.1f}"
            for token_id in sorted(interesting)
            if token_id < len(logits)
        ]
        return "{" + ", ".join(parts) + "}"

    def _token_label(self, token_id: int) -> str:
        """Return a readable token label."""
        return repr(self._tokens_by_id.get(token_id, "?"))


def print_title(title: str) -> None:
    """Print a section title."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_step(title: str) -> None:
    """Print a smaller step title."""
    print(f"\n--- {title} ---")


def wait(delay: float, pause: bool) -> None:
    """Wait after a visible step."""
    if pause:
        input("\nPress Enter to continue...")
    elif delay > 0:
        time.sleep(delay)


def explain_candidate_generation(
    prompt: str,
    functions: list,
    delay: float,
    pause: bool,
) -> None:
    """Show how functions become candidates before the scorer chooses one."""
    print_step("6A. Where do the candidate functions come from?")
    print(
        "The available functions come from functions_definition.json. "
        "The decoder does not invent function names."
    )
    print(
        "For one prompt, the decoder tries every available function and "
        "turns each one into a valid JSON candidate."
    )
    wait(delay, pause)

    for index, function in enumerate(functions, start=1):
        print(f"\nFunction {index}: {function.name}")
        print(f"description = {function.description}")
        print("schema parameters:")
        for name, spec in function.parameters.items():
            print(f"  {name}: {spec.type}")
        wait(delay, pause)

        print("\nextract_arguments(prompt, function)")
        print(f"prompt     = {prompt!r}")
        raw_params = extract_arguments(prompt, function)
        print(f"raw params = {raw_params}")
        print(
            "meaning    = first attempt to read values from the prompt "
            "according to this function schema"
        )
        wait(delay, pause)

        print("\n_coerce_value(...) for each parameter")
        params = {}
        for name, spec in function.parameters.items():
            raw_value = raw_params.get(name)
            final_value = _coerce_value(raw_value, spec)
            params[name] = final_value
            print(
                f"  {name}: raw={raw_value!r} -> "
                f"final={final_value!r} ({spec.type})"
            )
        print(
            "meaning    = even if extraction is weak, the final candidate "
            "still matches the schema type"
        )
        wait(delay, pause)

        candidate_json = json.dumps(
            {"name": function.name, "parameters": params},
            sort_keys=True,
            separators=(",", ":"),
        )
        print("\nCandidate JSON created for this function:")
        print(candidate_json)
        print(
            "This candidate is one option the LLM/scorer may choose. "
            "It is already valid JSON."
        )
        wait(delay, pause)

    print(
        "\nImportant: at this stage nothing has been selected yet. "
        "The decoder has only prepared valid options."
    )
    wait(delay, pause)


def explain_read_function_definitions(
    functions_path: Path,
    delay: float,
    pause: bool,
) -> list[FunctionDefinition]:
    """Show what read_function_definitions() does internally."""
    print_step("2. read_function_definitions()")
    print("Goal      = load the available function schemas from JSON")
    print(f"file path = {functions_path}")
    wait(delay, pause)

    raw_text = functions_path.read_text(encoding="utf-8")
    print("\nA. The file content is plain JSON text")
    print(raw_text)
    wait(delay, pause)

    raw_data = json.loads(raw_text)
    print("\nB. json.loads(...) converts text into Python data")
    print(f"type(raw_data) = {type(raw_data).__name__}")
    print(f"items         = {len(raw_data)}")
    print(
        "meaning       = at this point it is only dictionaries/lists, "
        "not validated project objects yet"
    )
    wait(delay, pause)

    print("\nC. Each item is validated as FunctionDefinition with Pydantic")
    validated = []
    for index, item in enumerate(raw_data, start=1):
        print(f"\nRaw item {index}:")
        print(json.dumps(item, indent=2))
        function = FunctionDefinition.model_validate(item)
        validated.append(function)
        print("\nValidated object:")
        print(f"  class       = {function.__class__.__name__}")
        print(f"  name        = {function.name!r}")
        print(f"  description = {function.description!r}")
        print(f"  returns     = {function.returns.type!r}")
        print("  parameters:")
        for name, spec in function.parameters.items():
            print(f"    {name}: type={spec.type!r}")
        print(
            "  meaning     = this function is now safe for the decoder "
            "to use as a candidate source"
        )
        wait(delay, pause)

    print("\nD. The real helper is called")
    print("read_function_definitions(path) returns list[FunctionDefinition]")
    functions = read_function_definitions(functions_path)
    print(f"returned functions = {[function.name for function in functions]}")
    wait(delay, pause)
    return functions


def explain_build_scorer(delay: float, pause: bool) -> TokenDebugScorer:
    """Show the scorer creation step in a beginner-friendly way."""
    print_step("4. build scorer")
    print("Goal = create the object that will choose between valid candidates")
    print(
        "In the real app, cli.main() calls build_scorer(args.model) from "
        "src/llm_client.py."
    )
    wait(delay, pause)

    print("\nReal project flow:")
    print("  build_scorer('Qwen/Qwen3-0.6B')")
    print("    -> tries to create SmallLLMScorer")
    print("    -> SmallLLMScorer imports llm_sdk.Small_LLM_Model")
    print("    -> the SDK loads the tokenizer and the local LLM")
    print("    -> available=True if everything worked")
    print("    -> otherwise NullScorer explains why the LLM is unavailable")
    wait(delay, pause)

    print("\nWhy the demo uses TokenDebugScorer instead:")
    print("  The real SDK may require torch/model files and can be slow.")
    print("  This demo wants to show the mechanics in the terminal.")
    print("  So it uses a visible scorer that simulates encode/logits/selection.")
    print("  The decoder does not care which scorer is used, as long as it has")
    print("  choose_constrained(prompt, candidates).")
    wait(delay, pause)

    print("\nCreating the demo scorer now:")
    scorer = TokenDebugScorer(delay=delay, pause=pause)
    print(f"scorer class     = {scorer.__class__.__name__}")
    print(f"scorer available = {scorer.available}")
    print("main method      = choose_constrained(prompt, candidates)")
    print("extra demo tools = encode(), decode(), fake logits, NumPy mask")
    wait(delay, pause)

    print("\nWhat this object will receive later:")
    print("  prompt     -> the original user sentence")
    print("  candidates -> JSON strings already valid for the available functions")
    print("\nWhat it must return:")
    print("  exactly one JSON string from the candidates list")
    wait(delay, pause)
    return scorer


def build_parser() -> argparse.ArgumentParser:
    """Create the parser for the visual demo."""
    parser = argparse.ArgumentParser(
        description="Show the callmemaybe execution flow slowly in the terminal.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between steps. Use 0 for no delay.",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Wait for Enter between steps instead of sleeping.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demo prompts. By default only the first prompt is shown.",
    )
    return parser


def write_demo_inputs(folder: Path) -> tuple[Path, Path, Path]:
    """Create small JSON input files for the demo run."""
    functions_path = folder / "functions_definition.json"
    prompts_path = folder / "function_calling_tests.json"
    output_path = folder / "function_calling_results.json"

    functions = [
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
            "name": "fn_reverse_string",
            "description": "Reverse a string.",
            "parameters": {
                "s": {"type": "string"},
            },
            "returns": {"type": "string"},
        },
    ]
    prompts = [
        {"prompt": "What is the sum of 2 and 3?"},
        {"prompt": "Reverse the string 'hello'"},
    ]

    functions_path.write_text(json.dumps(functions, indent=2), encoding="utf-8")
    prompts_path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    return functions_path, prompts_path, output_path


def main(argv: list[str] | None = None) -> int:
    """Run a visible end-to-end demo of the project flow."""
    args = build_parser().parse_args(argv)
    print_title("Demo: call me maybe execution flow")

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        functions_path, prompts_path, output_path = write_demo_inputs(root)

        print_step("1. Input files created")
        print(f"functions_definition = {functions_path}")
        print(f"input prompts        = {prompts_path}")
        print(f"output file          = {output_path}")
        wait(args.delay, args.pause)

        functions = explain_read_function_definitions(
            functions_path,
            args.delay,
            args.pause,
        )

        print_step("3. read_prompt_cases()")
        prompts = read_prompt_cases(prompts_path)
        if not args.all:
            prompts = prompts[:1]
        print(f"loaded prompts: {len(prompts)}")
        for item in prompts:
            print(f"- {item.prompt}")
        wait(args.delay, args.pause)

        scorer = explain_build_scorer(args.delay, args.pause)

        print_step("5. ConstrainedFunctionDecoder(...)")
        decoder = ConstrainedFunctionDecoder(
            functions=functions,
            scorer=scorer,
        )
        print("decoder created with validated functions and scorer")
        wait(args.delay, args.pause)

        results = []
        for index, item in enumerate(prompts, start=1):
            print_title(f"Prompt {index}/{len(prompts)}")
            print(f"input prompt = {item.prompt!r}")
            wait(args.delay, args.pause)

            explain_candidate_generation(
                item.prompt,
                functions,
                args.delay,
                args.pause,
            )

            print_step("6B. decoder.decode_with_trace(prompt)")
            print(
                "Now the real decoder runs. It builds the same candidates "
                "internally and passes their JSON strings to the scorer."
            )
            wait(args.delay, args.pause)
            trace = decoder.decode_with_trace(item.prompt)

            print_step("7. candidates generated by decoder")
            for candidate in trace.candidates:
                print(
                    f"name={candidate.name} "
                    f"score={candidate.score} "
                    f"parameters={candidate.parameters}"
                )
            wait(args.delay, args.pause)

            print_step("8. selected FunctionCallResult")
            print(f"prompt     = {trace.selected.prompt!r}")
            print(f"name       = {trace.selected.name}")
            print(f"parameters = {trace.selected.parameters}")
            wait(args.delay, args.pause)

            print_step("9. append selected result")
            results.append(trace.selected)
            print(f"results size = {len(results)}")
            wait(args.delay, args.pause)

        print_title("Final output")
        print_step("10. write_results()")
        write_results(output_path, results)
        print(output_path.read_text(encoding="utf-8"))
        wait(args.delay, args.pause)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
