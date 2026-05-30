*This project has been created as part of the 42 curriculum by danicort.*

# call me maybe

## Description

`call me maybe` translates natural-language prompts into structured function calls.
Given a prompt and a JSON list of available function definitions, it writes a JSON array
containing the original prompt, the selected function name, and schema-compliant
parameters.

The output is not free-form model text. The implementation builds candidates from the
declared function schemas and serializes only validated Pydantic models, so the final
file remains parseable JSON with the expected keys.

## Instructions

Install and run with `uv`:

```sh
uv sync
uv run python -m src
```

Custom paths are supported:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

To see the prompt-by-prompt decision process in the terminal, use verbose mode:

```sh
uv run python -m src --verbose
```

Verbose mode prints every candidate function, its extracted parameters, its score, and
the selected call. It does not add anything to the final JSON file, keeping the submitted
output compliant with the subject.

The Makefile contains the required rules:

```sh
make install
make run
make debug
make clean
make lint
```

### SDK Runtime Setup

The project depends on the copied `llm_sdk` directory as a local package. The main
project code uses only `numpy`, `pydantic`, `json`, and the public SDK wrapper. The
SDK itself declares its internal runtime dependencies, including `torch`,
`transformers`, and `huggingface-hub`, in `llm_sdk/pyproject.toml`.

Run `uv sync` from the project root. This installs the root project and the local SDK
path dependency together, which is required before `uv run python -m src` can load
`llm_sdk.Small_LLM_Model`.

## Algorithm Explanation

The decoder first loads `functions_definition.json` into Pydantic models. For each
prompt, it creates one candidate call per available function. Candidate parameters are
extracted according to the declared JSON types: numbers are read in text order, quoted
strings are preferred for string parameters, booleans use common true/false words, and
nested object or array schemas are filled recursively.

Each candidate is then coerced back through the expected schema. Missing or malformed
values receive valid defaults, which guarantees that every candidate has exactly the
required keys and valid parameter types.

The application initializes `llm_sdk.Small_LLM_Model` through the public SDK wrapper.
For each prompt, it serializes the schema-valid candidates to compact JSON, tokenizes
them with the SDK's public `encode` method, and computes which next tokens can still
lead to a valid candidate. At every generation step, it calls
`get_logits_from_input_ids`, masks the choice to only tokens that continue at least one
valid JSON candidate, and appends the highest-logit allowed token. The mask is
implemented with NumPy: invalid token logits are set to negative infinity and `argmax`
selects the best remaining token. Generation stops only when the token sequence exactly
matches one complete candidate. The final output is then serialized with Python's
`json` module.

This is constrained decoding: the model chooses the function call token by token, but
invalid JSON, unknown function names, extra keys, missing parameters, and wrong schema
types are never reachable. If the SDK cannot be loaded, the program stops with a clear
error.

## Design Decisions

Pydantic is used for every project data class because the subject requires validated
classes. The SDK integration is isolated in `src/llm_client.py`; no private SDK methods
or attributes are used. File handling is centralized in `src/io_utils.py` to keep JSON
errors user-facing and avoid tracebacks for missing or malformed inputs.

## Performance Analysis

The constrained candidate generation is linear in the number of prompts times the
number of function definitions. It does not sample arbitrary text, so JSON validity is
independent of model quality. Runtime depends mainly on model initialization and
candidate scoring through `llm_sdk`.

## Challenges Faced

The main challenge was keeping the model involved while guaranteeing JSON validity.
The solution constrains the LLM at token-selection time instead of trusting a prompt to
produce JSON. Another challenge was dependency compliance: the project uses only the
provided `llm_sdk` wrapper and does not import or declare forbidden model libraries in
the root project.

## Testing Strategy

The test suite checks numeric extraction, string extraction, function selection, and JSON
schema compliance at the decoder level. It also covers nested object arguments, arrays,
booleans, missing input files, and malformed JSON. Additional manual checks should
include empty prompt arrays, ambiguous prompts, larger function catalogs, and unusual
special characters.

Run tests with:

```sh
uv run pytest
```

## Bonus Features

- Multiple model names can be passed with `--model`; the value is forwarded to
  `llm_sdk.Small_LLM_Model` when the SDK is available.
- `--verbose` / `make run` visualizes the decoding process by printing every prompt,
  candidate function, extracted parameters, selected marker, and selected call.
- Nested object, array, boolean, integer, number, and string parameters are supported
  by the schema coercion layer.
- The test suite includes unit tests for simple calls, trace output, nested arguments,
  and input error handling.

The tokenizer-reimplementation bonus is not claimed: the project uses the public
`encode` method exposed by `llm_sdk` when the SDK is installed.

## Resources

- Pydantic documentation: https://docs.pydantic.dev/
- Python `json` module documentation: https://docs.python.org/3/library/json.html
- Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
- The project subject PDF supplied in this repository.

AI was used to analyze the subject, scaffold the implementation, and prepare validation
tests and documentation. The generated code was reviewed and adjusted to keep the
schema enforcement explicit and testable.
