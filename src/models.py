"""Pydantic models for project inputs and outputs."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


JsonType = Literal["string", "number", "integer", "boolean", "object", "array"]


class TypeSpec(BaseModel):
    """Schema fragment describing one JSON value."""

    model_config = ConfigDict(extra="allow")

    type: JsonType
    description: str | None = None
    properties: dict[str, "TypeSpec"] = Field(default_factory=dict)
    items: Optional["TypeSpec"] = None


class FunctionDefinition(BaseModel):
    """Definition of one callable function."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, TypeSpec] = Field(default_factory=dict)
    returns: TypeSpec

    @field_validator("name")
    @classmethod
    def non_empty_name(cls, value: str) -> str:
        """Reject empty function names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("function name cannot be empty")
        return cleaned


class PromptCase(BaseModel):
    """One natural-language prompt to process."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCallResult(BaseModel):
    """Final schema-compliant function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
