"""Base provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from catena.models import Message, Response


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion from the given messages.

        Args:
            messages: Conversation history
            model: Model to use (provider-specific)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options

        Returns:
            Response with content and usage statistics
        """
        ...

    async def __aenter__(self) -> Provider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass
