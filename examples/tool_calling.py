"""Example: Letting the model call tools."""

import asyncio
from typing import Literal

from catena import Message, providers, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. "Tokyo".
    """
    return f"It's 72F and sunny in {city}."


@tool
def convert_currency(amount: float, to: Literal["EUR", "GBP"]) -> str:
    """Convert an amount of USD to another currency.

    Args:
        amount: The amount in US dollars.
        to: The currency to convert to.
    """
    rates = {"eur": 0.92, "gbp": 0.79}
    return f"{amount} USD = {amount * rates.get(to.lower(), 1):.2f} {to.upper()}"


async def main():
    provider = providers.openai(model="gpt-4o-mini")
    tools = [get_weather, convert_currency]

    # run_tools calls the model, runs the tools it asks for, sends the results
    # back, and repeats until the model answers.
    messages = [Message.user("What's the weather in Tokyo, and what's $100 in EUR?")]
    run = await provider.run_tools(messages, tools=tools)

    print(run.content)
    print(f"\n{run.turns} model calls, ${run.usage.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
