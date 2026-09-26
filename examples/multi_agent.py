"""Example: Multi-agent workflow with handoffs."""

import asyncio

from catena import Chain, Context, Message, providers


async def main():
    # Use different models for different roles
    fast = providers.openai(model="gpt-4o-mini")
    smart = providers.openai(model="gpt-4o")

    # Research agent chain
    researcher = Chain(name="researcher")

    @researcher.step()
    async def research(ctx: Context) -> Context:
        ctx.messages.insert(
            0,
            Message.system(
                "You are a research assistant. Gather key facts and data points. "
                "Be thorough but concise."
            ),
        )
        response = await fast.complete(ctx.messages)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        ctx.set("research", response.content)
        return ctx

    # Writer agent chain
    writer = Chain(name="writer")

    @writer.step()
    async def write(ctx: Context) -> Context:
        research = ctx.get("research", "")
        ctx.messages = [
            Message.system(
                "You are an expert writer. Create polished, engaging content "
                "based on the research provided."
            ),
            Message.user(f"Based on this research:\n\n{research}\n\nWrite a brief article."),
        ]
        response = await smart.complete(ctx.messages, max_tokens=500)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        return ctx

    # Compose chains: research >> write
    pipeline = researcher >> writer

    # Run the full pipeline
    result = await pipeline.run("What are the benefits of TypeScript over JavaScript?")

    print("=== Final Article ===")
    print(result.messages[-1].content)
    print(f"\n=== Total Cost: ${result.usage.cost_usd:.4f} ===")


if __name__ == "__main__":
    asyncio.run(main())
