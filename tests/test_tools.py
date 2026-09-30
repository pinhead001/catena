"""Tests for tool/function-calling support."""

import pytest

from catena.models import ToolCall
from catena.tools import Tool, run_tool_calls, tool


def test_tool_decorator_builds_schema():
    @tool
    def get_weather(city: str, units: str = "celsius") -> str:
        """Get the current weather for a city."""
        return f"Sunny in {city} ({units})"

    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get the current weather for a city."
    assert get_weather.parameters["properties"]["city"] == {"type": "string"}
    assert get_weather.parameters["required"] == ["city"]


def test_tool_decorator_custom_name():
    @tool(name="weather_lookup")
    def get_weather(city: str) -> str:
        return city

    assert get_weather.name == "weather_lookup"


def test_tool_to_openai_format():
    params = {"type": "object", "properties": {}}
    t = Tool(name="add", description="Add two numbers", parameters=params)
    payload = t.to_openai()
    assert payload["type"] == "function"
    assert payload["function"]["name"] == "add"


def test_tool_to_anthropic_format():
    params = {"type": "object", "properties": {}}
    t = Tool(name="add", description="Add two numbers", parameters=params)
    payload = t.to_anthropic()
    assert payload["name"] == "add"
    assert payload["input_schema"] == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_tool_call_invokes_sync_func():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = await add.call(a=1, b=2)
    assert result == 3


@pytest.mark.asyncio
async def test_tool_call_invokes_async_func():
    @tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = await add.call(a=1, b=2)
    assert result == 3


@pytest.mark.asyncio
async def test_run_tool_calls():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    calls = [ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})]
    results = await run_tool_calls([add], calls)

    assert results == [("1", "3")]


@pytest.mark.asyncio
async def test_run_tool_calls_unknown_tool():
    calls = [ToolCall(id="1", name="missing", arguments={})]
    results = await run_tool_calls([], calls)

    assert results[0][0] == "1"
    assert "unknown tool" in results[0][1]


@pytest.mark.asyncio
async def test_run_tool_calls_reports_argument_error_without_calling():
    called = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        called.append((a, b))
        return a + b

    calls = [ToolCall(id="1", name="add", error="Arguments were not valid JSON")]
    results = await run_tool_calls([add], calls)

    assert results == [("1", "Error: Arguments were not valid JSON")]
    assert called == []
