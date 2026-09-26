"""Basic example: A simple analysis chain."""

import asyncio

from catena import Chain, Context, Message, providers


async def main():
    # Create a provider (uses OPENAI_API_KEY from env)
    provider = providers.openai(model="gpt-4o-mini")

    # Create a chain
    chain = Chain(name="analyzer")

    @chain.step(name="analyze")
    async def analyze(ctx: Context) -> Context:
        """First step: Analyze the input."""
        response = await provider.complete(
            ctx.messages,
            temperature=0.3,
        )
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        ctx.set("analysis", response.content)
        return ctx

    @chain.step(name="summarize")
    async def summarize(ctx: Context) -> Context:
        """Second step: Summarize the analysis."""
        ctx.add_message(Message.user("Now summarize that in one sentence."))

        response = await provider.complete(ctx.messages)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        return ctx

    # Run the chain
    result = await chain.run("What are the key principles of good API design?")

    # Print results
    print("=== Conversation ===")
    for msg in result.messages:
        role = msg.role.value if hasattr(msg.role, "value") else msg.role
        print(f"{role}: {msg.content[:100]}...")

    print("\n=== Usage ===")
    print(f"Tokens: {result.usage.input_tokens + result.usage.output_tokens}")
    print(f"Cost: ${result.usage.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
