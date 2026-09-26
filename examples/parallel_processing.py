"""Example: Process multiple documents in parallel."""

import asyncio

from catena import Chain, Context, providers
from catena.parallel import map_parallel


async def main():
    provider = providers.openai(model="gpt-4o-mini")

    # Create a summarization chain
    summarize = Chain(name="summarize")

    @summarize.step()
    async def extract_summary(ctx: Context) -> Context:
        ctx.messages[0].content = f"Summarize this in 2 sentences:\n\n{ctx.messages[0].content}"
        response = await provider.complete(ctx.messages, max_tokens=100)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        return ctx

    # Documents to process
    documents = [
        "Python is a high-level programming language known for its readability...",
        "JavaScript was created in 10 days by Brendan Eich at Netscape...",
        "Rust is a systems programming language focused on safety and performance...",
    ]

    # Process all documents in parallel (max 3 concurrent)
    results = await map_parallel(summarize, documents, max_concurrency=3)

    # Print results
    total_cost = 0.0
    for i, result in enumerate(results):
        print(f"Document {i + 1}: {result.messages[-1].content}\n")
        total_cost += result.usage.cost_usd

    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
