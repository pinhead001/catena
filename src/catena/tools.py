"""Tool/function-calling support for Catena."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

_JSON_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_for_type(annotation: Any) -> dict[str, Any]:
    return {"type": _JSON_TYPE_MAP.get(annotation, "string")}


def _build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON schema for a function's parameters from its signature."""
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        annotation = hints.get(param_name, str)
        properties[param_name] = _schema_for_type(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


@dataclass
class Tool:
    """A callable tool the model can invoke.

    Example:
        ```python
        from catena.tools import tool

        @tool
        def get_weather(city: str) -> str:
            \"\"\"Get the current weather for a city.\"\"\"
            return f"Sunny in {city}"

        response = await provider.complete(messages, tools=[get_weather])
        ```
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    func: Callable[..., Any] | None = None

    async def call(self, **kwargs: Any) -> Any:
        """Invoke the underlying function, awaiting it if necessary."""
        if self.func is None:
            raise ValueError(f"Tool '{self.name}' has no callable attached")
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def to_openai(self) -> dict[str, Any]:
        """Convert to the OpenAI tool-definition format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        """Convert to the Anthropic tool-definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def tool(func: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Decorator that turns a plain function into a `Tool`.

    The tool's name, description, and parameter schema are derived from the
    function's name, docstring, and type-annotated signature.
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        description = (inspect.getdoc(f) or "").strip()
        return Tool(
            name=name or f.__name__,
            description=description,
            parameters=_build_parameters_schema(f),
            func=f,
        )

    if func is not None:
        return decorator(func)
    return decorator


async def run_tool_calls(tools: list[Tool], tool_calls: list[Any]) -> list[Any]:
    """Execute a batch of tool calls against a list of available tools.

    Returns a list of `Message.tool(...)`-ready (tool_call_id, content) pairs
    in the same order as `tool_calls`.
    """
    by_name = {t.name: t for t in tools}

    async def run_one(call: Any) -> tuple[str, str]:
        matched = by_name.get(call.name)
        if matched is None:
            return call.id, f"Error: unknown tool '{call.name}'"
        try:
            result = await matched.call(**call.arguments)
            return call.id, str(result)
        except Exception as e:
            return call.id, f"Error: {e}"

    return await asyncio.gather(*[run_one(c) for c in tool_calls])
