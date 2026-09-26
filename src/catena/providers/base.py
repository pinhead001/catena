"""Base provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from catena.models import Message, Response, StreamChunk
from catena.tools import Tool


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
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion from the given messages.

        Args:
            messages: Conversation history
            model: Model to use (provider-specific)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            tools: Tools the model may call
            **kwargs: Provider-specific options

        Returns:
            Response with content and usage statistics
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from the given messages.

        Yields `StreamChunk`s as text arrives, with a final chunk carrying
        `done=True` and the total `usage`.
        """
        ...

    async def __aenter__(self) -> Provider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass
