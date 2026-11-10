"""Structured output: typed responses validated against a Pydantic model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from catena.models import Usage
from catena.tools import Tool

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar("T", bound="BaseModel")


@dataclass
class StructuredResponse(Generic[T]):
    """A validated, typed model output plus the usage it cost across all attempts."""

    output: T
    usage: Usage = field(default_factory=Usage)
    attempts: int = 1


class StructuredOutputError(Exception):
    """Raised when the model fails to produce valid output within the retry budget."""

    def __init__(
        self,
        message: str,
        raw_output: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.usage = usage or Usage()


def tool_name_for(schema: type[BaseModel]) -> str:
    """Turn a class name into a tool name the OpenAI and Anthropic APIs accept."""
    # Both APIs require ^[a-zA-Z0-9_-]{1,64}$; generics like Page[Item] would be rejected.
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", schema.__name__).strip("_")[:64]
    return name or "output"


def output_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema to ask the model for.

    Tool parameters must be a JSON object, so a RootModel (a bare list, string,
    union, ...) is asked for as `{"value": ...}`; `parse_output` unwraps it.
    """
    from pydantic import RootModel

    json_schema = schema.model_json_schema()
    if not issubclass(schema, RootModel):
        return json_schema
    wrapped: dict[str, Any] = {"type": "object", "required": ["value"]}
    if "$defs" in json_schema:
        wrapped["$defs"] = json_schema.pop("$defs")  # $ref paths point at the root
    wrapped["properties"] = {"value": json_schema}
    return wrapped


def parse_output(schema: type[T], data: Any) -> T:
    """Validate data shaped by `output_schema(schema)` into `schema`."""
    from pydantic import RootModel

    if issubclass(schema, RootModel):
        data = data.get("value") if isinstance(data, dict) else None
    return schema.model_validate(data)


def schema_tool(schema: type[BaseModel]) -> Tool:
    """Build the tool the model is forced to call to return `schema`-shaped data."""
    description = schema.__doc__ or f"Return the result as a {schema.__name__}."
    return Tool(
        name=tool_name_for(schema),
        description=description.strip(),
        parameters=output_schema(schema),
    )
