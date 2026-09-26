"""Core data models for Catena."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A chat message."""

    role: Role | str
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> Message:
        """Create a message carrying the result of a tool call."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)

    def to_dict(self) -> dict[str, Any]:
        role = self.role.value if isinstance(self.role, Role) else self.role
        d: dict[str, Any] = {"role": role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class Usage:
    """Token usage and cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass
class Response:
    """LLM response with metadata."""

    content: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def message(self) -> Message:
        return Message.assistant(self.content, tool_calls=self.tool_calls or None)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class StreamChunk:
    """A single chunk of a streamed completion."""

    delta: str = ""
    done: bool = False
    usage: Usage | None = None
    finish_reason: str | None = None


# Pricing per 1M tokens (as of 2024)
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts."""
    # Normalize model name
    model_key = model.lower()
    for key in PRICING:
        if key in model_key:
            input_price, output_price = PRICING[key]
            return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return 0.0
