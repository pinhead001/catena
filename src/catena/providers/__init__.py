"""LLM provider integrations."""

from importlib import import_module
from typing import Any

from catena.providers.base import Provider

__all__ = ["Provider", "OpenAIProvider", "AnthropicProvider"]

# The implementation modules are private (_openai, not openai): a submodule named
# like a factory below would replace that factory on this package once imported.
_PROVIDER_MODULES = {
    "OpenAIProvider": "_openai",
    "AnthropicProvider": "_anthropic",
}


def __getattr__(name: str) -> Any:
    """Import provider classes on first use, so their SDKs stay optional."""
    if name in _PROVIDER_MODULES:
        module = import_module(f"catena.providers.{_PROVIDER_MODULES[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def openai(**kwargs: Any) -> Provider:
    """Create an OpenAI provider."""
    from catena.providers._openai import OpenAIProvider

    return OpenAIProvider(**kwargs)


def anthropic(**kwargs: Any) -> Provider:
    """Create an Anthropic provider."""
    from catena.providers._anthropic import AnthropicProvider

    return AnthropicProvider(**kwargs)

