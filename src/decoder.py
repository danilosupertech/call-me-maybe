"""Constrained generation of schema-compliant function calls."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from .extractor import extract_arguments
from .llm_client import TextScorer
from .models import FunctionCallResult, FunctionDefinition, TypeSpec


class Candidate(BaseModel):
    """One schema-valid function call candidate."""

    model_config = ConfigDict(frozen=True)

    result: FunctionCallResult
    score: float
    json_text: str


class TraceCandidate(BaseModel):
    """Human-readable candidate information for verbose output."""

    name: str
    parameters: dict[str, Any]
    score: float


class DecodeTrace(BaseModel):
    """Detailed information about one decoding decision."""

    prompt: str
    candidates: list[TraceCandidate]
    selected: FunctionCallResult


class ConstrainedFunctionDecoder(BaseModel):
    """Generate only function-call objects allowed by definitions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions: list[FunctionDefinition]
    scorer: TextScorer

    def decode(self, prompt: str) -> FunctionCallResult:
        """Return the highest-scoring schema-compliant call."""
        return self.decode_with_trace(prompt).selected

    def decode_with_trace(self, prompt: str) -> DecodeTrace:
        """Return the selected call and all constrained candidates."""
        candidates = [
            self._candidate_for(prompt, function)
            for function in self.functions
        ]
        chosen_text = self.scorer.choose_constrained(
            prompt,
            [candidate.json_text for candidate in candidates],
        )
        best = _candidate_by_text(candidates, chosen_text)
        trace_candidates = [
            TraceCandidate(
                name=candidate.result.name,
                parameters=candidate.result.parameters,
                score=1.0 if candidate.json_text == best.json_text else 0.0,
            )
            for candidate in _selected_first(candidates, best)
        ]
        return DecodeTrace(
            prompt=prompt,
            candidates=trace_candidates,
            selected=best.result,
        )

    def _candidate_for(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> Candidate:
        """Build and score one function call candidate."""
        raw_params = extract_arguments(prompt, function)
        params = {
            name: _coerce_value(raw_params.get(name), spec)
            for name, spec in function.parameters.items()
        }
        result = FunctionCallResult(
            prompt=prompt,
            name=function.name,
            parameters=params,
        )
        json_text = json.dumps(
            {"name": function.name, "parameters": params},
            sort_keys=True,
            separators=(",", ":"),
        )
        return Candidate(result=result, score=0.0, json_text=json_text)


def _candidate_by_text(
    candidates: list[Candidate],
    json_text: str,
) -> Candidate:
    """Find the candidate selected by constrained generation."""
    for candidate in candidates:
        if candidate.json_text == json_text:
            return candidate
    return candidates[0]


def _selected_first(
    candidates: list[Candidate],
    selected: Candidate,
) -> list[Candidate]:
    """Return candidates with the selected one first."""
    return [selected] + [
        candidate
        for candidate in candidates
        if candidate.json_text != selected.json_text
    ]


def _coerce_value(value: Any, spec: TypeSpec) -> Any:
    """Coerce values to the exact schema type."""
    try:
        if spec.type == "number":
            return float(value)
        if spec.type == "integer":
            return int(float(value))
        if spec.type == "string":
            return "" if value is None else str(value)
        if spec.type == "boolean":
            return bool(value)
        if spec.type == "object":
            raw = value if isinstance(value, dict) else {}
            return {
                name: _coerce_value(raw.get(name), child_spec)
                for name, child_spec in spec.properties.items()
            }
        if spec.type == "array":
            raw_items = value if isinstance(value, list) else []
            if spec.items is None:
                return raw_items
            return [_coerce_value(item, spec.items) for item in raw_items]
    except (TypeError, ValueError):
        return _default_for(spec)
    return _default_for(spec)


def _default_for(spec: TypeSpec) -> Any:
    """Return a valid default for a schema type."""
    if spec.type == "number":
        return 0.0
    if spec.type == "integer":
        return 0
    if spec.type == "string":
        return ""
    if spec.type == "boolean":
        return False
    if spec.type == "object":
        return {
            name: _default_for(child_spec)
            for name, child_spec in spec.properties.items()
        }
    if spec.type == "array":
        return []
    return None
