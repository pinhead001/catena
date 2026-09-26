"""Tests for data models."""

from catena.models import Message, Response, Role, StreamChunk, ToolCall, Usage, calculate_cost


def test_message_creation():
    """Test creating messages."""
    msg = Message(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert msg.content == "Hello"


def test_message_factory_methods():
    """Test convenience factory methods."""
    assert Message.user("Hi").role == Role.USER
    assert Message.assistant("Hello").role == Role.ASSISTANT
    assert Message.system("You are...").role == Role.SYSTEM


def test_message_to_dict():
    """Test converting message to dict."""
    msg = Message.user("Hello")
    d = msg.to_dict()
    assert d == {"role": "user", "content": "Hello"}


def test_usage_addition():
    """Test adding usage stats."""
    u1 = Usage(input_tokens=100, output_tokens=50, cost_usd=0.01)
    u2 = Usage(input_tokens=200, output_tokens=100, cost_usd=0.02)

    combined = u1 + u2

    assert combined.input_tokens == 300
    assert combined.output_tokens == 150
    assert combined.cost_usd == 0.03


def test_calculate_cost_gpt4o():
    """Test cost calculation for GPT-4o."""
    cost = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 12.50  # $2.50 + $10.00


def test_calculate_cost_unknown_model():
    """Test cost calculation for unknown model."""
    cost = calculate_cost("unknown-model", input_tokens=1000, output_tokens=1000)
    assert cost == 0.0


def test_message_tool_factory():
    """Test creating a tool-result message."""
    msg = Message.tool("72F and sunny", tool_call_id="call_1", name="get_weather")
    assert msg.role == Role.TOOL
    assert msg.tool_call_id == "call_1"
    assert msg.name == "get_weather"


def test_message_with_tool_calls_to_dict():
    """Test that tool_calls survive serialization to dict."""
    calls = [ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})]
    msg = Message.assistant("", tool_calls=calls)
    d = msg.to_dict()
    assert d["tool_calls"] == [
        {"id": "call_1", "name": "get_weather", "arguments": {"city": "NYC"}}
    ]


def test_response_message_carries_tool_calls():
    """Test that Response.message includes tool_calls."""
    calls = [ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})]
    response = Response(content="", tool_calls=calls)
    assert response.has_tool_calls
    assert response.message.tool_calls == calls


def test_stream_chunk_defaults():
    """Test StreamChunk default values."""
    chunk = StreamChunk(delta="hello")
    assert chunk.delta == "hello"
    assert chunk.done is False
    assert chunk.usage is None
