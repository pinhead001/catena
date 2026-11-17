"""Anthropic provider implementation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from catena.models import Message, Response, Role, StreamChunk, ToolCall, Usage, calculate_cost
from catena.providers.base import Provider
from catena.tools import Tool, tool_choice_instruction


def _split_system_and_messages(
    messages: list[Message], system: str | None
) -> tuple[str | None, list[dict[str, Any]]]:
    """Collect system text and convert the rest to Anthropic's format."""
    system_parts: list[str] = []
    chat_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.role.value if isinstance(msg.role, Role) else msg.role
        if role == "system":
            system_parts.append(msg.content)
        elif role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content,
            }
            previous = chat_messages[-1] if chat_messages else None
            # The API wants all results for one assistant turn in a single user message.
            if previous and previous["role"] == "user" and isinstance(previous["content"], list):
                previous["content"].append(result)
            else:
                chat_messages.append({"role": "user", "content": [result]})
        elif role == "assistant" and msg.tool_calls:
            blocks: list[dict[str, Any]] = (
                [{"type": "text", "text": msg.content}] if msg.content else []
            )
            blocks += [
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in msg.tool_calls
            ]
            chat_messages.append({"role": "assistant", "content": blocks})
        else:
            chat_messages.append({"role": role, "content": msg.content})

    if system:
        system_parts.append(system)

    # Anthropic requires at least one message
    if not chat_messages:
        chat_messages = [{"role": "user", "content": "Hello"}]

    return "\n\n".join(system_parts) or None, chat_messages


def _request_kwargs(
    model: str,
    messages: list[Message],
    system: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build the arguments shared by messages.create() and messages.stream()."""
    system_msg, chat_messages = _split_system_and_messages(messages, system)
    request: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        # Current models always think, and thinking counts toward max_tokens.
        "max_tokens": max_tokens or 16000,
    }
    if system_msg:
        request["system"] = system_msg
    if temperature is not None:
        request["temperature"] = temperature
    return request


def _tool_kwargs(
    tools: list[Tool] | None, tool_choice: str | None, system: str | None
) -> tuple[dict[str, Any], str | None]:
    """Tool arguments for a request, and the system prompt they imply."""
    request: dict[str, Any] = {}
    if tools:
        request["tools"] = [t.to_anthropic() for t in tools]
    if tool_choice == "none":
        request["tool_choice"] = {"type": "none"}
    elif tool_choice:
        # Current Claude models reject forced tool_choice ("tool"/"any") with a 400,
        # so ask in the system prompt and leave the choice on auto.
        request["tool_choice"] = {"type": "auto"}
        instruction = tool_choice_instruction(tool_choice)
        if instruction:
            system = f"{system}\n\n{instruction}" if system else instruction
    return request, system


def _to_response(message: Any, model_name: str) -> Response:
    """Convert an Anthropic message to a Response; thinking blocks are left out."""
    content = ""
    tool_calls: list[ToolCall] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            content += block.text
        elif getattr(block, "type", None) == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

    return Response(
        content=content,
        model=model_name,
        usage=Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cost_usd=calculate_cost(
                model_name, message.usage.input_tokens, message.usage.output_tokens
            ),
        ),
        metadata={"stop_reason": message.stop_reason},
        tool_calls=tool_calls,
    )


class AnthropicProvider(Provider):
    """Anthropic Claude API provider.

    Example:
        ```python
        from catena import providers

        provider = providers.anthropic()  # claude-opus-5-5
        response = await provider.complete([Message.user("Hello!")])
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-5-5",
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
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion using Anthropic Claude.

        `temperature` is sent only when given: current Claude models reject it.
        """
        model_name = model or self.default_model
        tool_kwargs, system = _tool_kwargs(tools, tool_choice, system)

        response = await self._client.messages.create(
            **_request_kwargs(model_name, messages, system, temperature, max_tokens),
            **tool_kwargs,
            **kwargs,
        )
        return _to_response(response, model_name)

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion using Anthropic Claude."""
        model_name = model or self.default_model
        tool_kwargs, system = _tool_kwargs(tools, tool_choice, system)

        async with self._client.messages.stream(
            **_request_kwargs(model_name, messages, system, temperature, max_tokens),
            **tool_kwargs,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text)

            final = _to_response(await stream.get_final_message(), model_name)
            yield StreamChunk(
                done=True,
                finish_reason=final.metadata["stop_reason"],
                usage=final.usage,
                response=final,
            )

    async def __aexit__(self, *args: Any) -> None:
        await self._client.close()
