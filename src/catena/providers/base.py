"""Base provider interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from catena.models import Message, Response, StreamChunk, Usage
from catena.structured import (
    StructuredOutputError,
    StructuredResponse,
    T,
    schema_tool,
)
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
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion from the given messages.

        Args:
            messages: Conversation history
            model: Model to use (provider-specific)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            tools: Tools the model may call
            tool_choice: Name of a tool the model must call
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

    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        retries: int = 2,
        **kwargs: Any,
    ) -> StructuredResponse[T]:
        """Generate output validated against a Pydantic model.

        The model is forced to call a tool whose parameters are `schema`'s JSON
        schema. Invalid output is sent back to the model with the validation
        error, up to `retries` more times.

        Example:
            ```python
            class Invoice(BaseModel):
                company: str
                amount: float

            result = await provider.complete_structured(messages, schema=Invoice)
            result.output.amount  # float
            ```
        """
        from pydantic import ValidationError

        if retries < 0:
            raise ValueError(f"retries must be >= 0, got {retries}")
        if "tools" in kwargs or "tool_choice" in kwargs:
            raise TypeError(
                "complete_structured() sets tools and tool_choice itself; don't pass them"
            )

        tool = schema_tool(schema)
        history = list(messages)
        usage = Usage()
        error = ""
        raw = ""

        for attempt in range(1, retries + 2):
            response = await self.complete(
                history, tools=[tool], tool_choice=tool.name, **kwargs
            )
            usage = usage + response.usage

            call = next((tc for tc in response.tool_calls if tc.name == tool.name), None)
            if call is None:
                raw = response.content
                error = f"You did not call the {tool.name} tool."
                feedback = f"{error} Your reply was:\n{raw}"
            elif call.error:
                raw = error = call.error
                feedback = error
            else:
                raw = json.dumps(call.arguments)
                try:
                    output = schema.model_validate(call.arguments)
                    return StructuredResponse(output=output, usage=usage, attempts=attempt)
                except ValidationError as e:
                    error = str(e)
                feedback = f"Your previous output was:\n{raw}\n\nIt failed validation:\n{error}"

            history.append(
                Message.user(
                    f"{feedback}\n\nCall the {tool.name} tool again with corrected arguments."
                )
            )

        raise StructuredOutputError(
            f"No valid {schema.__name__} after {retries + 1} attempts: {error}",
            raw_output=raw,
            usage=usage,
        )

    async def __aenter__(self) -> Provider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass
