"""Base provider interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypeVar

from catena.models import Message, Response, StreamChunk, Usage
from catena.structured import (
    StructuredOutputError,
    StructuredResponse,
    T,
    parse_output,
    schema_tool,
)
from catena.tools import Tool, ToolLoopError, ToolRun, run_tool_calls

if TYPE_CHECKING:
    from catena.middleware.cache import Cache

ProviderT = TypeVar("ProviderT", bound="Provider")


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
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
        """Generate a completion from the given messages.

        Args:
            messages: Conversation history
            model: Model to use (provider-specific)
            system: System prompt, added after any system messages in `messages`
            temperature: Sampling temperature; the model's default when omitted
            max_tokens: Maximum tokens to generate
            tools: Tools the model may call
            tool_choice: "auto", "none", "required", or the name of one tool to call
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
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from the given messages.

        Yields `StreamChunk`s as text arrives, then a final chunk with
        `done=True`, the total `usage`, and the complete `response`
        (including any tool calls the model made).
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

        The model is asked to call a tool whose parameters are `schema`'s JSON
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
        try:
            from pydantic import ValidationError
        except ImportError:
            raise ImportError(
                "Structured output needs Pydantic. "
                'Install with: pip install "catena-ai[structured]"'
            ) from None

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
            elif call.error:
                raw = error = call.error
            else:
                raw = json.dumps(call.arguments)
                try:
                    output = parse_output(schema, call.arguments)
                    return StructuredResponse(output=output, usage=usage, attempts=attempt)
                except ValidationError as e:
                    error = f"The arguments failed validation:\n{e}"

            # Hand the failure back the way the model expects: its own turn,
            # then a result for every tool call it made.
            retry = f"Call the {tool.name} tool again with corrected arguments."
            if response.tool_calls:
                history.append(response.message)
                for made in response.tool_calls:
                    feedback = error if made is call or call is None else "Not used."
                    history.append(Message.tool(f"{feedback}\n\n{retry}", tool_call_id=made.id))
            else:
                if raw:
                    history.append(Message.assistant(raw))
                history.append(Message.user(f"{error} {retry}"))

        raise StructuredOutputError(
            f"No valid {schema.__name__} after {retries + 1} attempts: {error}",
            raw_output=raw,
            usage=usage,
        )

    async def run_tools(
        self,
        messages: list[Message],
        tools: list[Tool],
        *,
        max_turns: int = 10,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ToolRun:
        """Let the model use tools until it answers.

        Calls the model, runs the tools it asks for (in parallel), sends the
        results back, and repeats until the model replies without a tool call.
        A tool that raises is reported to the model as an error so it can
        recover. `tool_choice` applies to the first model call only; `max_turns`
        caps the number of model calls, raising `ToolLoopError` beyond it.

        Example:
            ```python
            run = await provider.run_tools(ctx.messages, tools=[get_weather])
            ctx.messages = run.messages
            ctx.add_usage(run.usage)
            ```
        """
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        history = list(messages)
        usage = Usage()

        for turn in range(1, max_turns + 1):
            response = await self.complete(
                history, tools=tools, tool_choice=tool_choice if turn == 1 else None, **kwargs
            )
            usage = usage + response.usage
            history.append(response.message)

            if not response.tool_calls:
                return ToolRun(response=response, messages=history, usage=usage, turns=turn)

            for tool_call_id, content in await run_tool_calls(tools, response.tool_calls):
                history.append(Message.tool(content, tool_call_id=tool_call_id))

        raise ToolLoopError(
            f"The model was still calling tools after {max_turns} turns",
            messages=history,
            usage=usage,
        )

    def cached(self: ProviderT, cache: Cache, ttl: float | None = None) -> ProviderT:
        """Serve repeated `complete()` calls from `cache`; returns this provider.

        Covers everything built on `complete()`: `run_tools()` and
        `complete_structured()` too. Streaming is not cached.

        Example:
            ```python
            provider = providers.openai().cached(MemoryCache())
            ```
        """
        self.complete = cache.wrap(self.complete, ttl)  # type: ignore[method-assign]
        return self

    async def __aenter__(self) -> Provider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass
