"""LLM adapter and constrained token selection."""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr


class TextScorer(BaseModel):
    """Base class for constrained model selection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    available: bool = False
    unavailable_reason: str = ""

    def choose_constrained(
        self,
        prompt: str,
        candidates: list[str],
    ) -> str:
        """Choose one candidate from an already constrained set."""
        if not candidates:
            return ""
        return candidates[0]


class NullScorer(TextScorer):
    """No-op scorer used when llm_sdk is not installed."""

    available: bool = False
    unavailable_reason: str = "llm_sdk is not available"


class SmallLLMScorer(TextScorer):
    """Best-effort public-API wrapper around llm_sdk.Small_LLM_Model."""

    available: bool = True
    model_name: str = "Qwen/Qwen3-0.6B"
    _model: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the SDK model using only public attributes."""
        from llm_sdk import Small_LLM_Model

        try:
            self._model = Small_LLM_Model(model_name=self.model_name)
        except TypeError:
            self._model = Small_LLM_Model(self.model_name)

    def choose_constrained(
        self,
        prompt: str,
        candidates: list[str],
    ) -> str:
        """Generate one valid candidate with token-level constraints.

        A trie is built from tokenized candidate JSON strings. At every step,
        only continuations of valid JSON candidates are allowed.
        """
        tokenized = {
            candidate: self._token_ids(candidate)
            for candidate in candidates
        }
        valid_tokenized = {
            candidate: token_ids
            for candidate, token_ids in tokenized.items()
            if token_ids
        }
        if not valid_tokenized:
            return candidates[0] if candidates else ""

        context = self._token_ids(_context_for_prompt(prompt))
        generated: list[int] = []
        max_steps = max(len(tokens) for tokens in valid_tokenized.values())

        for _ in range(max_steps):
            exact_match = _candidate_for_tokens(valid_tokenized, generated)
            if exact_match is not None:
                return exact_match
            allowed = _allowed_next_tokens(valid_tokenized, generated)
            if not allowed:
                break
            logits = self._logits_for(context + generated)
            next_token = _best_allowed_token(logits, allowed)
            generated.append(next_token)

        exact_match = _candidate_for_tokens(valid_tokenized, generated)
        if exact_match is not None:
            return exact_match
        return candidates[0]

    def _token_ids(self, text: str) -> list[int]:
        """Encode text and normalize SDK return types to a flat list."""
        encoded = self._model.encode(text)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

    def _logits_for(self, token_ids: list[int]) -> list[float]:
        """Read next-token logits through the SDK public method."""
        return _to_float_list(self._model.get_logits_from_input_ids(token_ids))


def _context_for_prompt(prompt: str) -> str:
    """Build the natural-language context for constrained generation."""
    return (
        "Select the correct function call for this prompt.\n"
        f"Prompt: {prompt}\n"
        "JSON:"
    )


def _candidate_for_tokens(
    candidates: dict[str, list[int]],
    token_ids: list[int],
) -> str | None:
    """Return the candidate matching a full token sequence."""
    for candidate, candidate_tokens in candidates.items():
        if candidate_tokens == token_ids:
            return candidate
    return None


def _allowed_next_tokens(
    candidates: dict[str, list[int]],
    prefix: list[int],
) -> set[int]:
    """Return tokens that keep the current prefix valid."""
    allowed: set[int] = set()
    prefix_length = len(prefix)
    for candidate_tokens in candidates.values():
        if len(candidate_tokens) <= prefix_length:
            continue
        if candidate_tokens[:prefix_length] == prefix:
            allowed.add(candidate_tokens[prefix_length])
    return allowed


def _best_allowed_token(logits: list[float], allowed: set[int]) -> int:
    """Mask invalid logits with NumPy and select the best allowed token."""
    logits_array = np.asarray(logits, dtype=np.float64)
    masked_logits = np.full_like(logits_array, -np.inf)
    for token_id in allowed:
        if token_id < masked_logits.size:
            masked_logits[token_id] = logits_array[token_id]
    return int(np.argmax(masked_logits))


def _to_float_list(values: Any) -> list[float]:
    """Normalize SDK logits to a plain list of floats."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [float(value) for value in values]


def build_scorer(model_name: str = "Qwen/Qwen3-0.6B") -> TextScorer:
    """Create an LLM scorer if llm_sdk is importable."""
    try:
        return SmallLLMScorer(model_name=model_name)
    except Exception as exc:
        return NullScorer(unavailable_reason=str(exc))
