"""Example: Streaming a completion token-by-token."""

import asyncio

from catena import Message, providers


async def main():
    provider = providers.anthropic(model="claude-3-5-haiku-20241022")

    messages = [Message.user("Write a haiku about distributed systems.")]

    async for chunk in provider.stream(messages):
        if chunk.done:
            print(f"\n\nTokens: {chunk.usage.input_tokens + chunk.usage.output_tokens}")
            print(f"Cost: ${chunk.usage.cost_usd:.4f}")
        else:
            print(chunk.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
