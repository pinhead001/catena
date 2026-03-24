"""Example: Using caching to reduce costs."""

import asyncio

from catena import Chain, Context
from catena import providers
from catena.middleware import MemoryCache


async def main():
    # Create provider with caching
    provider = providers.openai(model="gpt-4o-mini")
    cache = MemoryCache(max_size=100, default_ttl=3600)

    # Wrap the provider's complete method with caching
    original_complete = provider.complete
    provider.complete = cache.wrap(original_complete)

    chain = Chain(name="cached-qa")

    @chain.step()
    async def answer(ctx: Context) -> Context:
        response = await provider.complete(ctx.messages)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)

        if response.metadata.get("cached"):
            print("  (served from cache)")

        return ctx

    # First call - will hit the API
    print("First call:")
    result1 = await chain.run("What is 2 + 2?")
    print(f"  Answer: {result1.messages[-1].content}")
    print(f"  Cost: ${result1.usage.cost_usd:.4f}")

    # Second call with same input - served from cache
    print("\nSecond call (same question):")
    result2 = await chain.run("What is 2 + 2?")
    print(f"  Answer: {result2.messages[-1].content}")
    print(f"  Cost: ${result2.usage.cost_usd:.4f}")  # Should be $0.0000

    # Cache stats
    print(f"\nCache hit rate: {cache.hit_rate:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
