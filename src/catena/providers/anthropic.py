"""Anthropic provider implementation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from catena.models import Message, Response, Role, StreamChunk, ToolCall, Usage, calculate_cost
from catena.providers.base import Provider
from catena.tools import Tool


def _split_system_and_messages(
    messages: list[Message], system: str | None
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract a system message and convert the rest to Anthropic's format."""
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

    return system_msg, chat_messages


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
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion using Anthropic Claude."""
        model_name = model or self.default_model
        system_msg, chat_messages = _split_system_and_messages(messages, system)

        if tools:
            kwargs["tools"] = [t.to_anthropic() for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        response = await self._client.messages.create(
            model=model_name,
            messages=chat_messages,  # type: ignore
            max_tokens=max_tokens or 4096,
            temperature=temperature,
            system=system_msg or "",
            **kwargs,
        )

        content = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                content += block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

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
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion using Anthropic Claude."""
        model_name = model or self.default_model
        system_msg, chat_messages = _split_system_and_messages(messages, system)

        async with self._client.messages.stream(
            model=model_name,
            messages=chat_messages,  # type: ignore
            max_tokens=max_tokens or 4096,
            temperature=temperature,
            system=system_msg or "",
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text)

            final = await stream.get_final_message()
            yield StreamChunk(
                done=True,
                finish_reason=final.stop_reason,
                usage=Usage(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                    cost_usd=calculate_cost(
                        model_name, final.usage.input_tokens, final.usage.output_tokens
                    ),
                ),
            )

    async def __aexit__(self, *args: Any) -> None:
        await self._client.close()
