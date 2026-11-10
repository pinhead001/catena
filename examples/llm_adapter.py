"""Example: run a Catena chain on any model the `llm` library can reach.

Setup (once):
    pip install "catena-ai[llm]"
    llm keys set openai            # or: llm install llm-ollama, llm-gemini, ...

Pass any model id from `llm models` on the command line:
    python examples/llm_adapter.py gpt-4o-mini
"""

import asyncio
import sys

from catena import Chain, Context, Message, providers


async def main(model: str | None) -> None:
    # No model given: uses the default set with `llm models default`.
    provider = providers.llm(model)
    chain = Chain(name="draft-and-tighten")

    @chain.step(retries=2, timeout=60)
    async def draft(ctx: Context) -> Context:
        response = await provider.complete(ctx.messages)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        return ctx

    @chain.step(retries=2, timeout=60)
    async def tighten(ctx: Context) -> Context:
        ctx.add_message(Message.user("Now cut that to one sentence."))
        response = await provider.complete(ctx.messages)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        return ctx

    result = await chain.run("Explain what a vector database is.")
    print(result.messages[-1].content)
    print(f"\nTokens: {result.usage.input_tokens} in, {result.usage.output_tokens} out")
    print(f"Cost: ${result.usage.cost_usd:.4f} (0 if the model isn't in Catena's price table)")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
