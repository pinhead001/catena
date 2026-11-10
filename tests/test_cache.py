"""Tests for the caching middleware."""

import pytest

from catena import Message, Response, Tool, ToolCall, Usage
from catena.middleware import MemoryCache
from catena.middleware.cache import _hash_messages


def _tool(properties: dict) -> Tool:
    return Tool(
        name="Result",
        description="A result.",
        parameters={"type": "object", "properties": properties},
    )


def test_cache_key_differs_for_same_named_tools_with_different_schemas():
    messages = [Message.user("hi")]

    key_a = _hash_messages(messages, "m", tools=[_tool({"a": {"type": "string"}})])
    key_b = _hash_messages(messages, "m", tools=[_tool({"b": {"type": "integer"}})])

    assert key_a != key_b


def test_cache_key_stable_for_identical_tools():
    messages = [Message.user("hi")]
    tool = _tool({"a": {"type": "string"}})

    assert _hash_messages(messages, "m", tools=[tool]) == _hash_messages(
        messages, "m", tools=[tool]
    )


class CountingProvider:
    """Stands in for a provider: counts real calls and echoes its arguments."""

    def __init__(self, default_model: str = "m") -> None:
        self.default_model = default_model
        self.calls = 0

    async def complete(self, messages, *, model=None, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return Response(
            content=f"answer {self.calls}",
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.01),
        )


@pytest.mark.asyncio
async def test_repeat_call_is_served_from_cache_at_zero_cost():
    provider = CountingProvider()
    cached = MemoryCache().wrap(provider.complete)

    first = await cached([Message.user("hi")])
    second = await cached([Message.user("hi")])

    assert provider.calls == 1
    assert second.content == first.content
    assert second.usage.cost_usd == 0
    assert second.metadata["cached"] is True
    assert "cached" not in first.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"system": "Be terse."}, {"system": "Be verbose."}),
        ({"system": "Be terse."}, {}),
        ({"temperature": 0.0}, {"temperature": 1.0}),
        ({"reasoning_effort": "low"}, {"reasoning_effort": "high"}),
    ],
)
async def test_any_changed_argument_misses_the_cache(first, second):
    provider = CountingProvider()
    cached = MemoryCache().wrap(provider.complete)

    await cached([Message.user("hi")], **first)
    await cached([Message.user("hi")], **second)

    assert provider.calls == 2


def test_cache_key_includes_a_pydantic_schema():
    from pydantic import BaseModel

    class A(BaseModel):
        x: int

    class B(BaseModel):
        y: str

    messages = [Message.user("hi")]

    assert _hash_messages(messages, "m", schema=A) != _hash_messages(messages, "m", schema=B)
    assert _hash_messages(messages, "m", schema=A) == _hash_messages(messages, "m", schema=A)


def test_cache_key_ignores_arguments_left_as_none():
    messages = [Message.user("hi")]

    assert _hash_messages(messages, "m", system=None) == _hash_messages(messages, "m")


@pytest.mark.asyncio
async def test_providers_sharing_a_cache_do_not_share_default_model_answers():
    cache = MemoryCache()
    small, large = CountingProvider("small"), CountingProvider("large")

    await cache.wrap(small.complete)([Message.user("hi")])
    await cache.wrap(large.complete)([Message.user("hi")])

    assert (small.calls, large.calls) == (1, 1)


@pytest.mark.asyncio
async def test_wrap_ttl_expires_entries(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("catena.middleware.cache.time.time", lambda: now[0])
    provider = CountingProvider()
    cached = MemoryCache(default_ttl=3600).wrap(provider.complete, ttl=60)

    await cached([Message.user("hi")])
    now[0] += 59
    await cached([Message.user("hi")])
    assert provider.calls == 1

    now[0] += 2
    await cached([Message.user("hi")])
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_default_ttl_applies_and_none_never_expires(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("catena.middleware.cache.time.time", lambda: now[0])
    expiring, forever = MemoryCache(default_ttl=10), MemoryCache(default_ttl=None)
    response = Response(content="x")

    await expiring.set("k", response)
    await forever.set("k", response)
    now[0] += 1_000_000

    assert await expiring.get("k") is None
    assert await forever.get("k") is not None


@pytest.mark.asyncio
async def test_eviction_drops_the_least_recently_used_entry():
    cache = MemoryCache(max_size=2)
    await cache.set("a", Response(content="a"))
    await cache.set("b", Response(content="b"))

    await cache.get("a")  # "b" is now the least recently used
    await cache.set("c", Response(content="c"))

    assert await cache.get("b") is None
    assert await cache.get("a") is not None
    assert await cache.get("c") is not None


@pytest.mark.asyncio
async def test_cached_response_keeps_tool_calls_and_is_a_copy():
    cache = MemoryCache()
    call = ToolCall(id="c1", name="add", arguments={"a": 1})
    await cache.set("k", Response(content="", tool_calls=[call]))

    first = await cache.get("k")
    first.metadata["cached"] = True
    first.tool_calls.clear()
    second = await cache.get("k")

    assert second.tool_calls == [call]
    assert "cached" not in second.metadata


@pytest.mark.asyncio
async def test_stats_and_clear():
    cache = MemoryCache()
    await cache.set("k", Response(content="x"))

    await cache.get("k")
    await cache.get("missing")

    assert cache.stats == {"hits": 1, "misses": 1}
    assert cache.hit_rate == 0.5

    await cache.clear()
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_provider_cached_covers_complete_and_what_is_built_on_it():
    from catena import Provider, tool

    class Echo(Provider):
        default_model = "echo"
        calls = 0

        async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return Response(content="hi", usage=Usage(cost_usd=0.01))

        async def stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError
            yield

    @tool
    def noop() -> str:
        """Do nothing."""
        return ""

    cache = MemoryCache()
    provider = Echo().cached(cache)

    await provider.complete([Message.user("hi")])
    again = await provider.complete([Message.user("hi")])
    run = await provider.run_tools([Message.user("hi")], tools=[noop])
    rerun = await provider.run_tools([Message.user("hi")], tools=[noop])

    assert isinstance(provider, Echo)
    assert provider.calls == 2  # one plain call, one with tools; the repeats were cached
    assert again.usage.cost_usd == 0
    assert (run.usage.cost_usd, rerun.usage.cost_usd) == (0.01, 0)


@pytest.mark.asyncio
async def test_sqlite_cache_survives_reopening(tmp_path):
    from catena.middleware import SQLiteCache

    path = tmp_path / "cache.db"
    call = ToolCall(id="c1", name="add", arguments={"a": 1})
    stored = Response(
        content="hi",
        model="m",
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.01),
        metadata={"finish_reason": "stop"},
        tool_calls=[call],
    )

    first = SQLiteCache(path)
    await first.set("k", stored)
    first.close()

    reopened = SQLiteCache(path)
    hit = await reopened.get("k")

    assert hit.content == "hi"
    assert hit.model == "m"
    assert hit.tool_calls == [call]
    assert hit.metadata == {"finish_reason": "stop"}
    assert (hit.usage.input_tokens, hit.usage.output_tokens, hit.usage.cost_usd) == (10, 5, 0)
    assert await reopened.get("missing") is None
    assert reopened.stats == {"hits": 1, "misses": 1}
    reopened.close()


@pytest.mark.asyncio
async def test_sqlite_cache_ttl_overwrite_and_clear(tmp_path, monkeypatch):
    from catena.middleware import SQLiteCache

    now = [1000.0]
    monkeypatch.setattr("catena.middleware.cache.time.time", lambda: now[0])
    cache = SQLiteCache(tmp_path / "cache.db")

    await cache.set("forever", Response(content="a"))
    await cache.set("brief", Response(content="b"), ttl=60)
    await cache.set("forever", Response(content="a2"))
    now[0] += 61

    assert (await cache.get("forever")).content == "a2"
    assert await cache.get("brief") is None

    await cache.clear()
    assert await cache.get("forever") is None
    cache.close()


@pytest.mark.asyncio
async def test_sqlite_cache_serves_a_wrapped_provider_across_instances(tmp_path):
    from catena.middleware import SQLiteCache

    path = tmp_path / "cache.db"
    provider = CountingProvider()

    first = await SQLiteCache(path).wrap(provider.complete)([Message.user("hi")])
    second = await SQLiteCache(path).wrap(provider.complete)([Message.user("hi")])

    assert provider.calls == 1
    assert second.content == first.content
    assert second.metadata["cached"] is True


def test_response_round_trips_through_a_dict():
    response = Response(
        content="hi",
        model="m",
        usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.5),
        metadata={"stop_reason": "end_turn"},
        tool_calls=[ToolCall(id="c", name="t", arguments={"x": [1, 2]}, error="bad")],
    )

    assert Response.from_dict(response.to_dict()) == response
