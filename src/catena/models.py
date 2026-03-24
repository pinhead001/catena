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


@dataclass
class Message:
    """A chat message."""

    role: Role | str
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=Role.ASSISTANT, content=content)

    def to_dict(self) -> dict[str, str]:
        role = self.role.value if isinstance(self.role, Role) else self.role
        return {"role": role, "content": self.content}


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

    @property
    def message(self) -> Message:
        return Message.assistant(self.content)


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
