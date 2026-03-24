"""Anthropic provider implementation."""

from __future__ import annotations

import os
from typing import Any

from catena.models import Message, Response, Role, Usage, calculate_cost
from catena.providers.base import Provider


class AnthropicProvider(Provider):
    """Anthropic Claude API provider.

    Example:
        ```python
        from catena import providers

        provider = providers.anthropic(model="claude-3-5-sonnet-20241022")
        response = await provider.complete([Message.user("Hello!")])
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install catena-ai[anthropic]"
            )

        self.default_model = model
        self._client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion using Anthropic Claude."""
        model_name = model or self.default_model

        # Extract system message if present
        system_msg = system
        chat_messages = []

        for msg in messages:
            role = msg.role.value if isinstance(msg.role, Role) else msg.role
            if role == "system":
                system_msg = msg.content
            else:
                chat_messages.append({"role": role, "content": msg.content})

        # Anthropic requires at least one message
        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Hello"}]

        response = await self._client.messages.create(
            model=model_name,
            messages=chat_messages,  # type: ignore
            max_tokens=max_tokens or 4096,
            temperature=temperature,
            system=system_msg or "",
            **kwargs,
        )

        content = ""
        if response.content:
            content = response.content[0].text if hasattr(response.content[0], "text") else ""

        return Response(
            content=content,
            model=model_name,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=calculate_cost(
                    model_name, response.usage.input_tokens, response.usage.output_tokens
                ),
            ),
            metadata={"stop_reason": response.stop_reason},
        )

    async def __aexit__(self, *args: Any) -> None:
        await self._client.close()
