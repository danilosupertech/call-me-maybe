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

## Beginner Quick Start

If you are starting now, use this exact sequence:

```sh
uv sync
make run
make grade
```

What each command does:

- `uv sync`: installs project dependencies.
- `make run`: reads prompts from `data/input` and writes results to `data/output/function_calling_results.json`.
- `make grade`: checks your output against the public moulinette tests.

If `make grade` shows `SCORE: 11/11`, your output is compliant with the public subject tests.

For a full beginner-friendly walkthrough in Portuguese with visual diagrams,
see [GUIA_PASSO_A_PASSO.md](GUIA_PASSO_A_PASSO.md).

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
prompt, it creates a bounded set of candidate calls for every available function. The
search space is schema-aware: each parameter receives one deterministic baseline value
and additional plausible alternatives from the prompt, such as quoted strings, named
values, numbers, booleans, arrays, and nested object values.

Each candidate is then coerced back through the expected schema. Missing or malformed
values receive valid defaults, which guarantees that every candidate has exactly the
required keys and valid parameter types. The LLM therefore chooses a complete function
call candidate, including both the function name and parameter values, rather than only
choosing a function label.

The application initializes `llm_sdk.Small_LLM_Model` through the public SDK wrapper.
For each prompt, it serializes the schema-valid candidates to compact JSON, tokenizes
them with the SDK's public `encode` method, and computes which next tokens can still
lead to a valid candidate. At every generation step, it calls
`get_logits_from_input_ids`, masks the choice to only tokens that continue at least one
valid JSON candidate, and appends the highest-logit allowed token. The mask is
implemented with NumPy: invalid token logits are set to negative infinity and `argmax`
selects the best remaining token. Generation stops only when the token sequence exactly
matches one complete candidate or no more constrained tokens are available. If the
sequence ends without an exact match, the candidate with the longest shared token
prefix is returned as the closest approximation. The final output is then serialized with Python's
`json` module.

This is constrained decoding: the model chooses the function call token by token, but
invalid JSON, unknown function names, extra keys, missing parameters, and wrong schema
types are never reachable. If the SDK cannot be loaded, the program stops with a clear
error.

## End-to-End Visual Flow

The diagram below shows the complete execution path from the command line to the final
JSON file.

```text
+---------------------------------------------------------------+
| 1. User runs the project                                      |
|                                                               |
|    uv run python -m src                                       |
|    make run                                                   |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 2. src/__main__.py starts the application                     |
|                                                               |
|    - imports the CLI entry point                              |
|    - exits cleanly with a clear error message if needed        |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 3. src/cli.py reads command-line arguments                    |
|                                                               |
|    Default paths:                                             |
|    - data/input/functions_definition.json                     |
|    - data/input/function_calling_tests.json                   |
|    - data/output/function_calling_results.json                |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 4. src/io_utils.py loads and validates JSON inputs            |
|                                                               |
|    functions_definition.json   -> FunctionDefinition models   |
|    function_calling_tests.json -> PromptCase models           |
|                                                               |
|    Missing or malformed files are reported as user-facing      |
|    errors instead of raw tracebacks.                           |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 5. src/llm_client.py loads the LLM through the public SDK     |
|                                                               |
|    Small_LLM_Model                                            |
|    - encode(text)                                             |
|    - get_logits_from_input_ids(token_ids)                     |
|                                                               |
|    No private SDK methods or attributes are used.              |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 6. src/decoder.py receives each prompt                        |
|                                                               |
|    For every available function, it asks src/extractor.py      |
|    to build schema-shaped parameter candidates.                |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 7. src/extractor.py builds valid candidate parameters         |
|                                                               |
|    Prompt example:                                            |
|    "Please add -12.5 and 7.25"                                |
|                                                               |
|    Candidate example:                                         |
|    {"name":"fn_add_numbers","parameters":{"a":-12.5,"b":7.25}}|
|                                                               |
|    The candidate space is bounded, schema-aware, and ordered   |
|    to avoid uncontrolled free-form generation.                 |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 8. Constrained token selection chooses one complete JSON call |
|                                                               |
|    At each step:                                              |
|    - candidate JSON strings are tokenized                     |
|    - only tokens that still match at least one valid candidate |
|      remain allowed                                           |
|    - invalid token logits are masked with negative infinity    |
|    - the highest-logit allowed token is selected               |
|                                                               |
|    Result: the LLM chooses, but invalid JSON is unreachable.   |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 9. Pydantic models and schema coercion protect the output     |
|                                                               |
|    Final object shape:                                        |
|    {                                                          |
|      "prompt": "...",                                        |
|      "name": "fn_...",                                       |
|      "parameters": { ... }                                   |
|    }                                                          |
|                                                               |
|    Unknown functions, extra keys, missing arguments, and wrong |
|    schema types are prevented before writing the file.         |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
| 10. src/io_utils.py writes the final JSON output              |
|                                                               |
|     data/output/function_calling_results.json                 |
|                                                               |
|     The output directory is generated at runtime and should    |
|     not be committed to the repository.                        |
+---------------------------------------------------------------+
```

A short version of the flow is:

```text
Command
  -> CLI
  -> JSON input validation
  -> SDK-backed LLM scorer
  -> schema-aware candidate generation
  -> constrained token selection
  -> Pydantic-validated function call
  -> final JSON output
```

Concrete example:

```text
Input prompt:
  Please add -12.5 and 7.25

Available function:
  fn_add_numbers(a: number, b: number)

Generated schema-valid candidate:
  {"name":"fn_add_numbers","parameters":{"a":-12.5,"b":7.25}}

Final output item:
  {
    "prompt": "Please add -12.5 and 7.25",
    "name": "fn_add_numbers",
    "parameters": {
      "a": -12.5,
      "b": 7.25
    }
  }
```


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

The test suite checks numeric extraction, string extraction, constrained function-call
selection, output cleanup, and JSON schema compliance at the decoder level. It also
covers nested object arguments, arrays, booleans, missing input files, and malformed
JSON. Additional manual checks should include empty prompt arrays, ambiguous prompts,
larger function catalogs, and unusual special characters.

Run tests with:

```sh
uv run pytest
```

## Additional Robustness Features

These features support the mandatory part and improve reliability, but this submission
does not claim the bonus part.

- Multiple model names can be passed with `--model`; the value is forwarded to
  `llm_sdk.Small_LLM_Model` when the SDK is available.
- `--verbose` helps inspect the decoding process by printing each prompt, candidate
  function, extracted parameters, and selected call.
- Nested object, array, boolean, integer, number, and string parameters are handled by
  the schema coercion layer.
- The test suite includes unit tests for simple calls, trace output, nested arguments,
  and input error handling.

The tokenizer-reimplementation bonus is not claimed: the project uses the public
`encode` method exposed by `llm_sdk` when the SDK is installed.

## Resources

- Pydantic documentation: https://docs.pydantic.dev/
- Python `json` module documentation: https://docs.python.org/3/library/json.html
- Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
- The project subject PDF supplied in this repository.

### How AI was used

All AI-generated content was reviewed, understood, and validated by the project author.

