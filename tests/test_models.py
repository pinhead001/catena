"""Tests for data models."""

import pytest
from catena.models import Message, Role, Usage, calculate_cost


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
