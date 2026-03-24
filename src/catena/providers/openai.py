"""OpenAI provider implementation."""

from __future__ import annotations

import os
from typing import Any

from catena.models import Message, Response, Usage, calculate_cost
from catena.providers.base import Provider


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
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion using OpenAI."""
        model_name = model or self.default_model

        response = await self._client.chat.completions.create(
            model=model_name,
            messages=[m.to_dict() for m in messages],  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        content = response.choices[0].message.content or ""
        usage_data = response.usage

        input_tokens = usage_data.prompt_tokens if usage_data else 0
        output_tokens = usage_data.completion_tokens if usage_data else 0

        return Response(
            content=content,
            model=model_name,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=calculate_cost(model_name, input_tokens, output_tokens),
            ),
            metadata={"finish_reason": response.choices[0].finish_reason},
        )

    async def __aexit__(self, *args: Any) -> None:
        await self._client.close()
