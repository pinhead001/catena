"""Tool/function-calling support for Catena."""

from __future__ import annotations

import asyncio
import enum
import inspect
import json
import re
import types
from collections import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from catena.models import Message, Response, Usage

_JSON_TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    frozenset: "array",
    dict: "object",
    type(None): "null",
}

_DOC_SECTIONS = ("Args:", "Arguments:", "Parameters:", "Returns:", "Raises:", "Yields:", "Example")


def _schema_for_type(annotation: Any, defs: dict[str, Any]) -> dict[str, Any]:
    """JSON schema for one type annotation; shared definitions go into `defs`."""
    if annotation in _JSON_TYPE_MAP:
        return {"type": _JSON_TYPE_MAP[annotation]}

    origin, args = get_origin(annotation), get_args(annotation)

    if origin is Annotated:
        schema = _schema_for_type(args[0], defs)
        description = next((extra for extra in args[1:] if isinstance(extra, str)), None)
        return {**schema, "description": description} if description else schema
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (Union, types.UnionType):
        options = [_schema_for_type(arg, defs) for arg in args]
        return options[0] if len(options) == 1 else {"anyOf": options}
    if origin in (list, set, frozenset, abc.Sequence, abc.Iterable, abc.Set):
        items = _schema_for_type(args[0], defs) if args else {}
        return {"type": "array", "items": items}
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _schema_for_type(args[0], defs)}
        return {"type": "array", "prefixItems": [_schema_for_type(arg, defs) for arg in args]}
    if origin in (dict, abc.Mapping):
        values = _schema_for_type(args[1], defs) if len(args) == 2 else {}
        return {"type": "object", "additionalProperties": values}

    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            return {"enum": [member.value for member in annotation]}
        if hasattr(annotation, "model_json_schema"):  # a Pydantic model
            schema = dict(annotation.model_json_schema())
            # $ref paths point at the root of the whole schema, so definitions move there.
            defs.update(schema.pop("$defs", {}))
            return schema

    return {}  # Any, or a type with no JSON equivalent: no constraint


def _split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into its summary and its per-argument descriptions.

    Understands the Google style: an `Args:` section of `name: text` or
    `name (type): text` entries, with continuation lines indented further.
    """
    lines = doc.splitlines()
    first_section = next(
        (i for i, line in enumerate(lines) if line.strip().startswith(_DOC_SECTIONS)),
        len(lines),
    )
    summary = "\n".join(lines[:first_section]).strip()

    arguments: dict[str, str] = {}
    current: str | None = None
    arguments_indent = 0
    in_args = False
    for line in lines[first_section:]:
        stripped = line.strip()
        if stripped.startswith(_DOC_SECTIONS):
            in_args = stripped.startswith(("Args:", "Arguments:", "Parameters:"))
            current = None
        elif in_args and stripped:
            entry = re.match(r"(\*{0,2}\w+)\s*(?:\([^)]*\))?:\s*(.*)", stripped)
            if entry and (current is None or _indent(line) <= arguments_indent):
                current = entry.group(1).lstrip("*")
                arguments[current] = entry.group(2)
                arguments_indent = _indent(line)
            elif current:
                arguments[current] = f"{arguments[current]} {stripped}".strip()
    return summary, arguments


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _build_parameters_schema(
    func: Callable[..., Any], descriptions: dict[str, str] | None = None
) -> dict[str, Any]:
    """Derive a JSON schema for a function's parameters from its signature."""
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        # A name in a string annotation could not be resolved (a class defined
        # inside a function, say): keep the annotations that are real types.
        hints = {
            name: param.annotation
            for name, param in signature.parameters.items()
            if param.annotation is not param.empty and not isinstance(param.annotation, str)
        }

    properties: dict[str, Any] = {}
    required: list[str] = []
    defs: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        if param_name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = _schema_for_type(hints.get(param_name, str), defs)
        if descriptions and param_name in descriptions and "description" not in schema:
            schema = {**schema, "description": descriptions[param_name]}
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if defs:
        schema["$defs"] = defs
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
    function's name, docstring, and type-annotated signature. An `Args:`
    section in the docstring (or `Annotated[type, "text"]`) describes each
    parameter to the model.
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        description, arguments = _split_docstring(inspect.getdoc(f) or "")
        return Tool(
            name=name or f.__name__,
            description=description,
            parameters=_build_parameters_schema(f, arguments),
            func=f,
        )

    if func is not None:
        return decorator(func)
    return decorator


@dataclass
class ToolRun:
    """The outcome of `provider.run_tools()`: the answer and how it was reached."""

    response: Response
    """The model's final reply, after it stopped calling tools."""
    messages: list[Message]
    """The whole conversation: the input, every tool call and result, and the answer."""
    usage: Usage = field(default_factory=Usage)
    """Total usage across every model call in the run."""
    turns: int = 1
    """How many times the model was called."""

    @property
    def content(self) -> str:
        return self.response.content


class ToolLoopError(Exception):
    """Raised when the model is still calling tools after `max_turns` model calls.

    `messages` and `usage` hold the conversation and cost so far.
    """

    retryable = False  # a chain step does not retry it: the loop has used its budget

    def __init__(self, message: str, messages: list[Message], usage: Usage) -> None:
        super().__init__(message)
        self.messages = messages
        self.usage = usage


def tool_choice_instruction(tool_choice: str | None) -> str | None:
    """System-prompt wording of a tool_choice, for models that cannot be forced.

    `tool_choice` is "auto", "none", "required", or the name of one tool.
    """
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return "Do not call any tools; answer in text."
    if tool_choice == "required":
        return "Respond by calling one of the available tools."
    return f"Respond by calling the {tool_choice} tool."


def _result_text(result: Any) -> str:
    """A tool's return value as the text the model reads: JSON for structured data."""
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump_json"):
        return str(result.model_dump_json())
    if isinstance(result, (dict, list, tuple, bool, int, float)) or result is None:
        try:
            return json.dumps(result, default=str)
        except (TypeError, ValueError):
            pass
    return str(result)


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
        if getattr(call, "error", None):
            return call.id, f"Error: {call.error}"
        try:
            result = await matched.call(**call.arguments)
        except Exception as e:
            return call.id, f"Error: {e}"
        return call.id, _result_text(result)

    return await asyncio.gather(*[run_one(c) for c in tool_calls])
