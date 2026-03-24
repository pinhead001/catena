"""Execution context for chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from catena.models import Message, Usage


@dataclass
class Context:
    """Shared state across chain steps.

    The context flows through each step, accumulating:
    - Messages: Full conversation history
    - Data: Arbitrary key-value storage for inter-step communication
    - Usage: Aggregated token/cost tracking
    """

    messages: list[Message] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)

    def add_usage(self, usage: Usage) -> None:
        """Accumulate usage statistics."""
        self.usage = self.usage + usage

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in context data."""
        self.data[key] = value

    def clone(self) -> Context:
        """Create a shallow copy of the context."""
        return Context(
            messages=self.messages.copy(),
            data=self.data.copy(),
            usage=Usage(
                input_tokens=self.usage.input_tokens,
                output_tokens=self.usage.output_tokens,
                cost_usd=self.usage.cost_usd,
            ),
        )
