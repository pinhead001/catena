"""Tests for the caching middleware."""

from catena import Message, Tool
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
