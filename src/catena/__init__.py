"""Catena - Lightweight AI agent orchestration."""

from catena import providers
from catena.chain import Chain, step
from catena.context import Context
from catena.models import Message, Response, StreamChunk, ToolCall, Usage
from catena.providers.base import Provider
from catena.structured import StructuredOutputError, StructuredResponse
from catena.tools import Tool, tool

__version__ = "0.2.0"
__all__ = [
    "Chain",
    "step",
    "Context",
    "Message",
    "Response",
    "StreamChunk",
    "ToolCall",
    "Usage",
    "Provider",
    "StructuredOutputError",
    "StructuredResponse",
    "Tool",
    "tool",
    "providers",
]
