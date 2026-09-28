"""Tests for provider implementations, using mocked SDK clients."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic as anthropic_module
import openai as openai_module
import pytest

from catena.models import Message
from catena.providers.anthropic import AnthropicProvider
from catena.providers.openai import OpenAIProvider

# --- OpenAI fakes ---------------------------------------------------------


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage, finish_reason: str = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletion:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.choices = [_FakeChoice(_FakeMessage(content, tool_calls))]
        self.usage = _FakeUsage(10, 5)


class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        self.delta = _FakeDelta(content)
        self.finish_reason = finish_reason


class _FakeStreamChunk:
    def __init__(self, choices: list, usage: _FakeUsage | None = None) -> None:
        self.choices = choices
        self.usage = usage


class _FakeAsyncStream:
    def __init__(self, chunks: list) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_openai_complete_parses_tool_calls(monkeypatch):
    fake_response = _FakeCompletion(
        content="",
        tool_calls=[_FakeToolCall("call_1", "get_weather", '{"city": "NYC"}')],
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=fake_response)))
    )
    monkeypatch.setattr(openai_module, "AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIProvider(api_key="test")
    response = await provider.complete([Message.user("weather?")])

    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "NYC"}
    assert response.usage.input_tokens == 10


@pytest.mark.asyncio
async def test_openai_complete_forces_tool_choice(monkeypatch):
    create = AsyncMock(return_value=_FakeCompletion(content="ok"))
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai_module, "AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIProvider(api_key="test")
    await provider.complete([Message.user("hi")], tool_choice="get_weather")

    assert create.call_args.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


@pytest.mark.asyncio
async def test_openai_stream_yields_deltas_and_usage(monkeypatch):
    chunks = [
        _FakeStreamChunk([_FakeStreamChoice("Hello")]),
        _FakeStreamChunk([_FakeStreamChoice(" world", finish_reason="stop")]),
        _FakeStreamChunk([], usage=_FakeUsage(5, 2)),
    ]
    fake_stream = _FakeAsyncStream(chunks)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=fake_stream)))
    )
    monkeypatch.setattr(openai_module, "AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIProvider(api_key="test")
    deltas = []
    final = None
    async for chunk in provider.stream([Message.user("hi")]):
        if chunk.done:
            final = chunk
        else:
            deltas.append(chunk.delta)

    assert deltas == ["Hello", " world"]
    assert final is not None
    assert final.finish_reason == "stop"
    assert final.usage.input_tokens == 5


# --- Anthropic fakes -------------------------------------------------------


class _FakeBlock:
    def __init__(
        self,
        type: str,
        text: str | None = None,
        id: str | None = None,
        name: str | None = None,
        input: dict | None = None,
    ) -> None:
        self.type = type
        self.text = text
        self.id = id
        self.name = name
        self.input = input


class _FakeAnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeAnthropicResponse:
    def __init__(
        self,
        content: list,
        usage: _FakeAnthropicUsage,
        stop_reason: str = "end_turn",
    ) -> None:
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason


class _FakeAnthropicStream:
    def __init__(self, texts: list[str], final_message: _FakeAnthropicResponse) -> None:
        self._texts = texts
        self._final = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    @property
    def text_stream(self):
        return self._gen()

    async def _gen(self):
        for text in self._texts:
            yield text

    async def get_final_message(self):
        return self._final


@pytest.mark.asyncio
async def test_anthropic_complete_parses_tool_calls(monkeypatch):
    fake_response = _FakeAnthropicResponse(
        content=[_FakeBlock("tool_use", id="call_1", name="get_weather", input={"city": "NYC"})],
        usage=_FakeAnthropicUsage(10, 5),
    )
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=fake_response))
    )
    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", lambda **kwargs: fake_client)

    provider = AnthropicProvider(api_key="test")
    response = await provider.complete([Message.user("weather?")])

    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "NYC"}


@pytest.mark.asyncio
async def test_anthropic_complete_forces_tool_choice(monkeypatch):
    create = AsyncMock(
        return_value=_FakeAnthropicResponse(content=[], usage=_FakeAnthropicUsage(1, 1))
    )
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", lambda **kwargs: fake_client)

    provider = AnthropicProvider(api_key="test")
    await provider.complete([Message.user("hi")], tool_choice="get_weather")

    assert create.call_args.kwargs["tool_choice"] == {"type": "tool", "name": "get_weather"}


@pytest.mark.asyncio
async def test_anthropic_stream_yields_deltas_and_usage(monkeypatch):
    final_message = _FakeAnthropicResponse(
        content=[], usage=_FakeAnthropicUsage(5, 2), stop_reason="end_turn"
    )
    fake_stream = _FakeAnthropicStream(["Hello", " world"], final_message)
    fake_client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: fake_stream))
    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", lambda **kwargs: fake_client)

    provider = AnthropicProvider(api_key="test")
    deltas = []
    final = None
    async for chunk in provider.stream([Message.user("hi")]):
        if chunk.done:
            final = chunk
        else:
            deltas.append(chunk.delta)

    assert deltas == ["Hello", " world"]
    assert final is not None
    assert final.usage.input_tokens == 5
