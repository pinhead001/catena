"""LLM provider integrations."""

from catena.providers.base import Provider

__all__ = ["Provider"]

# Lazy imports for optional dependencies
def openai(**kwargs):
    """Create an OpenAI provider."""
    from catena.providers.openai import OpenAIProvider
    return OpenAIProvider(**kwargs)


def anthropic(**kwargs):
    """Create an Anthropic provider."""
    from catena.providers.anthropic import AnthropicProvider
    return AnthropicProvider(**kwargs)
