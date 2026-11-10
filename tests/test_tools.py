"""Tests for tool/function-calling support."""

import pytest

from catena import Message, Provider, Response, ToolLoopError, Usage
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


def _parameters(func):  # type: ignore[no-untyped-def]
    return tool(func).parameters


def test_schema_for_collections_and_optionals():
    def search(
        tags: list[str],
        limit: int | None = None,
        weights: dict[str, float] | None = None,
        point: tuple[float, float] = (0, 0),
    ) -> str:
        return ""

    props = _parameters(search)["properties"]

    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["limit"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert props["weights"]["anyOf"][0] == {
        "type": "object",
        "additionalProperties": {"type": "number"},
    }
    assert props["point"] == {
        "type": "array",
        "prefixItems": [{"type": "number"}, {"type": "number"}],
    }
    assert _parameters(search)["required"] == ["tags"]


def test_schema_for_literal_and_enum():
    import enum
    from typing import Literal

    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    def paint(units: Literal["c", "f"], color: Color) -> str:
        return ""

    props = _parameters(paint)["properties"]

    assert props["units"] == {"enum": ["c", "f"]}
    assert props["color"] == {"enum": ["red", "blue"]}


def test_schema_for_pydantic_model_hoists_definitions():
    from pydantic import BaseModel

    class Address(BaseModel):
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    def save(person: Person) -> str:
        return ""

    parameters = _parameters(save)

    assert parameters["properties"]["person"]["properties"]["address"] == {
        "$ref": "#/$defs/Address"
    }
    assert "Address" in parameters["$defs"]
    assert "$defs" not in parameters["properties"]["person"]


def test_unknown_types_are_unconstrained_and_unannotated_are_strings():
    from typing import Any

    def f(anything: Any, plain, *args: int, **kwargs: str) -> str:  # type: ignore[no-untyped-def]
        return ""

    parameters = _parameters(f)

    assert parameters["properties"] == {"anything": {}, "plain": {"type": "string"}}
    assert parameters["required"] == ["anything", "plain"]


def test_argument_descriptions_come_from_the_docstring():
    @tool
    def get_weather(city: str, units: str = "celsius", days: int = 1) -> str:
        """Get the weather forecast for a city.

        Use this for any question about weather.

        Args:
            city: The city name, e.g. "Tokyo".
            units (str): "celsius" or "fahrenheit";
                defaults to celsius.

        Returns:
            A one-line forecast.
        """
        return ""

    props = get_weather.parameters["properties"]

    assert get_weather.description == (
        "Get the weather forecast for a city.\n\nUse this for any question about weather."
    )
    assert props["city"] == {"type": "string", "description": 'The city name, e.g. "Tokyo".'}
    assert props["units"]["description"] == '"celsius" or "fahrenheit"; defaults to celsius.'
    assert props["days"] == {"type": "integer"}


def test_annotated_string_describes_a_parameter():
    from typing import Annotated

    def f(city: Annotated[str, "The city name"]) -> str:
        """Look something up.

        Args:
            city: Ignored, the annotation wins.
        """
        return ""

    assert _parameters(f)["properties"]["city"] == {
        "type": "string",
        "description": "The city name",
    }


@pytest.mark.asyncio
async def test_structured_tool_results_are_sent_as_json():
    from pydantic import BaseModel

    class Forecast(BaseModel):
        high: int

    tools = [
        Tool(name="a", description="", func=lambda: {"high": 72, "ok": True}),
        Tool(name="b", description="", func=lambda: Forecast(high=72)),
        Tool(name="c", description="", func=lambda: "plain text"),
        Tool(name="d", description="", func=lambda: None),
    ]
    calls = [ToolCall(id=t.name, name=t.name, arguments={}) for t in tools]

    results = dict(await run_tool_calls(tools, calls))

    assert results == {
        "a": '{"high": 72, "ok": true}',
        "b": '{"high":72}',
        "c": "plain text",
        "d": "null",
    }


class _Scripted(Provider):
    """Returns pre-scripted responses and records every call."""

    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": list(messages), **kwargs})
        return self.responses.pop(0)

    async def stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError
        yield


def _asks(*calls: ToolCall, cost: float = 0.01) -> Response:
    return Response(content="", usage=Usage(cost_usd=cost), tool_calls=list(calls))


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@pytest.mark.asyncio
async def test_run_tools_loops_until_the_model_answers():
    provider = _Scripted(
        [
            _asks(ToolCall(id="c1", name="add", arguments={"a": 2, "b": 40})),
            _asks(ToolCall(id="c2", name="add", arguments={"a": 42, "b": 1})),
            Response(content="43", usage=Usage(cost_usd=0.01)),
        ]
    )
    question = [Message.user("2 + 40 + 1?")]

    run = await provider.run_tools(question, tools=[add])

    assert run.content == "43"
    assert run.turns == 3
    assert run.usage.cost_usd == pytest.approx(0.03)
    assert [m.role.value for m in run.messages] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]  # fmt: skip
    assert [m.content for m in run.messages if m.role.value == "tool"] == ["42", "43"]
    assert run.messages[2].tool_call_id == "c1"
    assert len(question) == 1
    # The second model call saw the first tool result.
    assert provider.calls[1]["messages"][-1].content == "42"


@pytest.mark.asyncio
async def test_run_tools_without_tool_calls_is_one_turn():
    provider = _Scripted([Response(content="hello")])

    run = await provider.run_tools([Message.user("hi")], tools=[add], temperature=0.2)

    assert (run.content, run.turns) == ("hello", 1)
    assert provider.calls[0]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_run_tools_answers_parallel_calls_and_reports_tool_errors():
    provider = _Scripted(
        [
            _asks(
                ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="c2", name="add", arguments={"a": 1}),
                ToolCall(id="c3", name="missing", arguments={}),
            ),
            Response(content="done"),
        ]
    )

    run = await provider.run_tools([Message.user("go")], tools=[add])

    results = [m for m in run.messages if m.role.value == "tool"]
    assert [m.tool_call_id for m in results] == ["c1", "c2", "c3"]
    assert results[0].content == "3"
    assert results[1].content.startswith("Error:")
    assert results[2].content == "Error: unknown tool 'missing'"


@pytest.mark.asyncio
async def test_run_tools_applies_tool_choice_to_the_first_turn_only():
    provider = _Scripted(
        [
            _asks(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})),
            Response(content="3"),
        ]
    )

    await provider.run_tools([Message.user("go")], tools=[add], tool_choice="required")

    assert [c["tool_choice"] for c in provider.calls] == ["required", None]


@pytest.mark.asyncio
async def test_run_tools_stops_at_max_turns_and_reports_what_was_spent():
    provider = _Scripted(
        [_asks(ToolCall(id=f"c{i}", name="add", arguments={"a": 1, "b": 1})) for i in range(5)]
    )

    with pytest.raises(ToolLoopError, match="2 turns") as exc_info:
        await provider.run_tools([Message.user("go")], tools=[add], max_turns=2)

    assert len(provider.calls) == 2
    assert exc_info.value.usage.cost_usd == pytest.approx(0.02)
    assert len(exc_info.value.messages) == 5


@pytest.mark.asyncio
async def test_run_tools_rejects_a_zero_turn_budget():
    with pytest.raises(ValueError, match="max_turns"):
        await _Scripted([]).run_tools([Message.user("go")], tools=[add], max_turns=0)


def test_description_starting_with_example_is_kept():
    @tool
    def demo(city: str) -> str:
        """Example lookup tool: returns a canned forecast.

        Note that the data is fake.
        """
        return ""

    assert demo.description.startswith("Example lookup tool")
    assert "data is fake" in demo.description


def test_annotated_with_unhashable_metadata_still_builds_a_schema():
    from typing import Annotated

    def f(count: Annotated[int, {"unit": "items"}], nothing: None = None) -> str:
        return ""

    props = _parameters(f)["properties"]

    assert props == {"count": {"type": "integer"}, "nothing": {"type": "null"}}


@pytest.mark.asyncio
async def test_arguments_are_converted_to_the_annotated_types():
    import enum

    from pydantic import BaseModel

    class Color(enum.Enum):
        RED = "red"

    class Address(BaseModel):
        city: str

    received = {}

    @tool
    def save(address: Address, color: Color, count: int, tags: list[str], note=None) -> str:
        received.update(address=address, color=color, count=count, tags=tags, note=note)
        return "saved"

    arguments = {
        "address": {"city": "Tokyo"},
        "color": "red",
        "count": "3",
        "tags": ["a"],
        "note": 5,
    }
    results = await run_tool_calls([save], [ToolCall(id="c1", name="save", arguments=arguments)])

    assert results == [("c1", "saved")]
    assert received == {
        "address": Address(city="Tokyo"),
        "color": Color.RED,
        "count": 3,
        "tags": ["a"],
        "note": 5,  # unannotated: passed through untouched
    }


@pytest.mark.asyncio
async def test_unconvertible_arguments_are_reported_to_the_model_without_calling_the_tool():
    calls = []

    @tool
    def repeat(times: int) -> str:
        calls.append(times)
        return "x" * times

    results = await run_tool_calls(
        [repeat], [ToolCall(id="c1", name="repeat", arguments={"times": "many"})]
    )

    assert calls == []
    assert results[0][1].startswith("Error:")
    assert "times" in results[0][1] or "integer" in results[0][1]
