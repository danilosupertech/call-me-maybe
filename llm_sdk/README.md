Place the official `llm_sdk` package provided with the subject in this directory.

The application imports `llm_sdk.Small_LLM_Model` dynamically when it is available.
Without the official package, it uses a local fallback so tests and JSON validation can
still run.
