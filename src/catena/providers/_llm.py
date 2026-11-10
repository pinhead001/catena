"""Adapter for the `llm` library: use any model an llm plugin provides."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from catena.models import Message, Response, Role, StreamChunk, ToolCall, Usage, calculate_cost
from catena.providers.base import Provider
from catena.structured import StructuredOutputError, StructuredResponse, T
from catena.tools import Tool

if TYPE_CHECKING:
    from pydantic import BaseModel


class LLMProvider(Provider):
    """Run Catena chains on any model the `llm` library can reach.

    `llm` (https://llm.datasette.io) reaches OpenAI, Anthropic, Gemini, local
    models and many more through plugins. Keys and the default model come from
    llm's own configuration (`llm keys set ...`, `llm models default ...`).

    Example:
        ```python
        from catena import providers

        provider = providers.llm("gpt-4o-mini")
        response = await provider.complete([Message.user("Hello!")])
        ```
    """

    def __init__(self, model: str | None = None, key: str | None = None) -> None:
        try:
            import llm
        except ImportError:
            raise ImportError(
                "llm package not installed. Install with: pip install catena-ai[llm]"
            )

        self._llm = llm
        self._key = key
        default: str = llm.get_default_model()  # type: ignore[no-untyped-call]
        self.default_model = model or default
        # Fail now, not on the first call, if the model or its plugin is missing.
        llm.get_async_model(self.default_model)

    def _turn_kwargs(self, messages: list[Message], extra_system: str | None) -> dict[str, Any]:
        """Split a conversation into the pieces llm's prompt() takes.

        llm wants the system text, the prior history (`messages`) and this
        turn's input (`prompt` or `tool_results`) separately, and joins them
        itself. Older plugins read only the separate fields.
        """
        llm = self._llm
        system_parts: list[str] = []
        turns: list[tuple[str, Message]] = []
        tool_names: dict[str, str] = {}
        for msg in messages:
            role = msg.role.value if isinstance(msg.role, Role) else msg.role
            if role == "system":
                system_parts.append(msg.content)
                continue
            turns.append((role, msg))
            for call in msg.tool_calls or []:
                tool_names[call.id] = call.name
        if extra_system:
            system_parts.append(extra_system)

        kwargs: dict[str, Any] = {}
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        def result_name(msg: Message) -> str:
            return msg.name or tool_names.get(msg.tool_call_id or "", "")

        # Peel this turn's input off the end; the rest is history.
        if turns and turns[-1][0] == "user":
            kwargs["prompt"] = turns.pop()[1].content
        else:
            results = []
            while turns and turns[-1][0] == "tool":
                msg = turns.pop()[1]
                results.append(
                    llm.ToolResult(
                        name=result_name(msg), output=msg.content, tool_call_id=msg.tool_call_id
                    )
                )
            if results:
                kwargs["tool_results"] = results[::-1]

        history: list[Any] = []
        pending_results: list[Any] = []
        for role, msg in turns:
            if role == "tool":
                pending_results.append(
                    llm.parts.ToolResultPart(
                        name=result_name(msg), output=msg.content, tool_call_id=msg.tool_call_id
                    )
                )
                continue
            if pending_results:
                history.append(llm.tool_message(*pending_results))
                pending_results = []
            if role == "assistant":
                parts: list[Any] = [msg.content] if msg.content else []
                parts += [
                    llm.parts.ToolCallPart(
                        name=call.name, arguments=call.arguments, tool_call_id=call.id
                    )
                    for call in msg.tool_calls or []
                ]
                history.append(llm.assistant(*parts))
            else:
                history.append(llm.user(msg.content))
        if pending_results:
            history.append(llm.tool_message(*pending_results))
        if history:
            kwargs["messages"] = history
        return kwargs

    def _prompt(
        self,
        messages: list[Message],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[Tool] | None,
        tool_choice: str | None,
        schema: type[BaseModel] | dict[str, Any] | None,
        stream: bool,
        options: dict[str, Any],
    ) -> tuple[Any, str]:
        """Start an llm prompt; returns the (lazy) llm response and the model id."""
        llm = self._llm
        llm_model = llm.get_async_model(model or self.default_model)

        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            # Plugins disagree on the name (llm-gemini uses max_output_tokens).
            fields = llm_model.Options.model_fields
            name = "max_output_tokens" if "max_tokens" not in fields else "max_tokens"
            options[name] = max_tokens

        # llm has no tool_choice setting, so ask for the tool in the system prompt.
        instruction = f"Respond by calling the {tool_choice} tool." if tool_choice else None
        kwargs = self._turn_kwargs(messages, instruction)
        kwargs["stream"] = stream

        if tools:
            kwargs["tools"] = [
                llm.Tool(name=t.name, description=t.description, input_schema=t.parameters)
                for t in tools
            ]
        if schema is not None:
            kwargs["schema"] = schema
        if options:
            kwargs["options"] = options
        if self._key is not None:
            kwargs["key"] = self._key

        return llm_model.prompt(**kwargs), llm_model.model_id

    async def _usage(self, llm_response: Any, model_id: str) -> Usage:
        usage = await llm_response.usage()
        input_tokens = usage.input or 0
        output_tokens = usage.output or 0
        priced_as = getattr(llm_response, "resolved_model", None) or model_id
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_cost(priced_as, input_tokens, output_tokens),
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
        schema: type[BaseModel] | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion through llm.

        `temperature` and `max_tokens` are sent only when given. `schema` uses
        llm's own structured output. Other keyword arguments are passed to the
        model as llm options (the names `llm models --options` lists).
        """
        llm_response, model_id = self._prompt(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            schema=schema,
            stream=False,
            options=kwargs,
        )
        content = await llm_response.text()
        tool_calls = [
            ToolCall(
                id=call.tool_call_id or f"call_{uuid.uuid4().hex[:12]}",
                name=call.name,
                arguments=call.arguments,
            )
            for call in await llm_response.tool_calls()
        ]
        return Response(
            content=content,
            model=model_id,
            usage=await self._usage(llm_response, model_id),
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
        """Stream a completion through llm."""
        llm_response, model_id = self._prompt(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None,
            tool_choice=None,
            schema=None,
            stream=True,
            options=kwargs,
        )
        async for chunk in llm_response:
            if chunk:
                yield StreamChunk(delta=chunk)
        yield StreamChunk(done=True, usage=await self._usage(llm_response, model_id))

    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        retries: int = 2,
        **kwargs: Any,
    ) -> StructuredResponse[T]:
        """Generate output validated against a Pydantic model.

        Uses the model's own schema support when its llm plugin has it, and
        falls back to the forced-tool approach otherwise.
        """
        from pydantic import ValidationError

        llm_model = self._llm.get_async_model(kwargs.get("model") or self.default_model)
        if not llm_model.supports_schema:
            return await super().complete_structured(messages, schema, retries=retries, **kwargs)

        if retries < 0:
            raise ValueError(f"retries must be >= 0, got {retries}")

        history = list(messages)
        usage = Usage()
        error = ""
        raw = ""

        for attempt in range(1, retries + 2):
            response = await self.complete(history, schema=schema, **kwargs)
            usage = usage + response.usage
            raw = response.content
            try:
                output = schema.model_validate_json(raw)
                return StructuredResponse(output=output, usage=usage, attempts=attempt)
            except ValidationError as e:
                error = str(e)

            history.append(Message.assistant(raw))
            history.append(
                Message.user(
                    f"That failed validation:\n{error}\n\nReply again with corrected JSON only."
                )
            )

        raise StructuredOutputError(
            f"No valid {schema.__name__} after {retries + 1} attempts: {error}",
            raw_output=raw,
            usage=usage,
        )
