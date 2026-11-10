"""Execution context for chains."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from catena.models import Message, Response, Usage


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

    def add_response(self, response: Response) -> None:
        """Record a model response: its message joins the history, its usage the total."""
        self.add_message(response.message)
        self.add_usage(response.usage)

    @property
    def text(self) -> str:
        """The text of the latest message ("" if there are none)."""
        return self.messages[-1].content if self.messages else ""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in context data."""
        self.data[key] = value

    def clone(self) -> Context:
        """Copy the context so changes to the copy leave this one alone.

        Messages are copied; values in `data` are shared, not copied.
        """
        return Context(
            messages=[replace(message) for message in self.messages],
            data=self.data.copy(),
            usage=Usage(
                input_tokens=self.usage.input_tokens,
                output_tokens=self.usage.output_tokens,
                cost_usd=self.usage.cost_usd,
            ),
        )
