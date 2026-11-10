"""OpenAI provider implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from catena.models import Message, Response, Role, StreamChunk, ToolCall, Usage, calculate_cost
from catena.providers.base import Provider
from catena.tools import Tool


def _parse_tool_call(id: str, name: str, raw_arguments: str | None) -> ToolCall:
    """Parse tool-call arguments, recording bad JSON on the call instead of raising."""
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        error = f"Arguments were not valid JSON ({e}): {raw_arguments}"
        return ToolCall(id=id, name=name, error=error)
    if not isinstance(arguments, dict):
        error = f"Arguments must be a JSON object: {raw_arguments}"
        return ToolCall(id=id, name=name, error=error)
    return ToolCall(id=id, name=name, arguments=arguments)


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert to Chat Completions messages, including tool calls and tool results."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.role.value if isinstance(msg.role, Role) else msg.role
        if role == "tool":
            converted.append(
                {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
            )
        elif role == "assistant" and msg.tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in msg.tool_calls
                    ],
                }
            )
        else:
            converted.append({"role": role, "content": msg.content})
    return converted


def _request_kwargs(
    model: str,
    messages: list[Message],
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build the arguments shared by complete() and stream()."""
    request: dict[str, Any] = {"model": model, "messages": _to_openai_messages(messages)}
    # Sent only when given: reasoning models reject a non-default temperature.
    if temperature is not None:
        request["temperature"] = temperature
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    return request


class OpenAIProvider(Provider):
    """OpenAI API provider.

    Example:
        ```python
        from catena import providers

        provider = providers.openai(model="gpt-4o")
        response = await provider.complete([Message.user("Hello!")])
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install catena-ai[openai]"
            )

        self.default_model = model
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion using OpenAI.

        `temperature` and `max_tokens` are sent only when given. For models
        that take `max_completion_tokens` instead, pass it as a keyword argument.
        """
        model_name = model or self.default_model

        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
        if tool_choice in ("auto", "none", "required"):
            kwargs["tool_choice"] = tool_choice
        elif tool_choice:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        response = await self._client.chat.completions.create(
            **_request_kwargs(model_name, messages, temperature, max_tokens),
            **kwargs,
        )

        message = response.choices[0].message
        content = message.content or ""
        usage_data = response.usage

        input_tokens = usage_data.prompt_tokens if usage_data else 0
        output_tokens = usage_data.completion_tokens if usage_data else 0

        tool_calls = [
            _parse_tool_call(tc.id, tc.function.name, tc.function.arguments)
            for tc in (message.tool_calls or [])
            if tc.type == "function"
        ]

        return Response(
            content=content,
            model=model_name,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=calculate_cost(model_name, input_tokens, output_tokens),
            ),
            metadata={"finish_reason": response.choices[0].finish_reason},
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion using OpenAI."""
        model_name = model or self.default_model

        response = await self._client.chat.completions.create(
            **_request_kwargs(model_name, messages, temperature, max_tokens),
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0

        async for chunk in response:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta.content or ""
            if delta:
                yield StreamChunk(delta=delta)

        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=calculate_cost(model_name, input_tokens, output_tokens),
            ),
        )

    async def __aexit__(self, *args: Any) -> None:
        await self._client.close()
