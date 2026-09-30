"""Tests for structured output."""

import re
from datetime import date
from typing import Any, Generic, TypeVar

import pytest
from pydantic import BaseModel

from catena import Message, Provider, Response, StructuredOutputError, ToolCall, Usage
from catena.structured import schema_tool

ItemT = TypeVar("ItemT")


class Invoice(BaseModel):
    """An extracted invoice."""

    company: str
    amount: float
    due_date: date


class ScriptedProvider(Provider):
    """Returns pre-scripted responses and records every call."""

    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": list(messages), **kwargs})
        return self.responses.pop(0)

    async def stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError
        yield


def _tool_response(arguments: dict[str, Any], cost: float = 0.01) -> Response:
    return Response(
        content="",
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=cost),
        tool_calls=[ToolCall(id="call_1", name="Invoice", arguments=arguments)],
    )


def test_schema_tool_uses_model_schema():
    tool = schema_tool(Invoice)
    assert tool.name == "Invoice"
    assert tool.description == "An extracted invoice."
    assert set(tool.parameters["properties"]) == {"company", "amount", "due_date"}


@pytest.mark.asyncio
async def test_complete_structured_returns_typed_output():
    provider = ScriptedProvider(
        [_tool_response({"company": "Acme", "amount": "1200.50", "due_date": "2026-10-01"})]
    )

    result = await provider.complete_structured([Message.user("extract")], schema=Invoice)

    assert result.output == Invoice(company="Acme", amount=1200.5, due_date=date(2026, 10, 1))
    assert result.attempts == 1
    assert provider.calls[0]["tool_choice"] == "Invoice"
    assert provider.calls[0]["tools"][0].name == "Invoice"


@pytest.mark.asyncio
async def test_complete_structured_retries_with_validation_error():
    provider = ScriptedProvider(
        [
            _tool_response({"company": "Acme", "amount": "a lot"}),
            _tool_response({"company": "Acme", "amount": 1200, "due_date": "2026-10-01"}),
        ]
    )

    result = await provider.complete_structured([Message.user("extract")], schema=Invoice)

    assert result.attempts == 2
    assert result.output.amount == 1200
    assert result.usage.cost_usd == pytest.approx(0.02)
    retry_prompt = provider.calls[1]["messages"][-1].content
    assert "failed validation" in retry_prompt
    assert "due_date" in retry_prompt


@pytest.mark.asyncio
async def test_complete_structured_retries_when_tool_not_called():
    provider = ScriptedProvider(
        [
            Response(content="Sure, here is the invoice..."),
            _tool_response({"company": "Acme", "amount": 5, "due_date": "2026-10-01"}),
        ]
    )

    result = await provider.complete_structured([Message.user("extract")], schema=Invoice)

    assert result.attempts == 2
    assert "did not call" in provider.calls[1]["messages"][-1].content


@pytest.mark.asyncio
async def test_complete_structured_raises_after_retries_exhausted():
    provider = ScriptedProvider([_tool_response({"company": "Acme"}) for _ in range(2)])

    with pytest.raises(StructuredOutputError) as exc_info:
        await provider.complete_structured([Message.user("extract")], schema=Invoice, retries=1)

    assert len(provider.calls) == 2
    assert exc_info.value.raw_output == '{"company": "Acme"}'


@pytest.mark.asyncio
async def test_complete_structured_does_not_mutate_caller_messages():
    provider = ScriptedProvider(
        [
            _tool_response({"company": "Acme"}),
            _tool_response({"company": "Acme", "amount": 1, "due_date": "2026-10-01"}),
        ]
    )
    messages = [Message.user("extract")]

    await provider.complete_structured(messages, schema=Invoice)

    assert len(messages) == 1


def test_schema_tool_sanitizes_generic_model_name():
    class Page(BaseModel, Generic[ItemT]):
        items: list[ItemT]

    name = schema_tool(Page[Invoice]).name

    assert name == "Page_Invoice"
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name)


@pytest.mark.asyncio
async def test_complete_structured_retries_on_unparseable_arguments():
    bad = Response(
        content="",
        usage=Usage(cost_usd=0.01),
        tool_calls=[ToolCall(id="c1", name="Invoice", error="Arguments were not valid JSON")],
    )
    provider = ScriptedProvider(
        [bad, _tool_response({"company": "Acme", "amount": 1, "due_date": "2026-10-01"})]
    )

    result = await provider.complete_structured([Message.user("extract")], schema=Invoice)

    assert result.attempts == 2
    assert "not valid JSON" in provider.calls[1]["messages"][-1].content


@pytest.mark.asyncio
async def test_structured_output_error_reports_usage():
    provider = ScriptedProvider([_tool_response({"company": "Acme"}) for _ in range(3)])

    with pytest.raises(StructuredOutputError) as exc_info:
        await provider.complete_structured([Message.user("extract")], schema=Invoice)

    assert exc_info.value.usage.cost_usd == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_complete_structured_rejects_negative_retries():
    provider = ScriptedProvider([])

    with pytest.raises(ValueError):
        await provider.complete_structured([Message.user("x")], schema=Invoice, retries=-1)

    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kwarg", ["tools", "tool_choice"])
async def test_complete_structured_rejects_tools_kwargs(kwarg):
    provider = ScriptedProvider([])

    with pytest.raises(TypeError, match="sets tools and tool_choice"):
        await provider.complete_structured([Message.user("x")], schema=Invoice, **{kwarg: None})
