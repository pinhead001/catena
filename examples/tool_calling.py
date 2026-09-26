"""Example: Letting the model call tools."""

import asyncio

from catena import Message, providers
from catena.tools import run_tool_calls, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's 72F and sunny in {city}."


@tool
def convert_currency(amount: float, to: str) -> str:
    """Convert an amount of USD to another currency."""
    rates = {"eur": 0.92, "gbp": 0.79}
    return f"{amount} USD = {amount * rates.get(to.lower(), 1):.2f} {to.upper()}"


async def main():
    provider = providers.openai(model="gpt-4o-mini")
    tools = [get_weather, convert_currency]

    messages = [Message.user("What's the weather in Tokyo, and what's $100 in EUR?")]
    response = await provider.complete(messages, tools=tools)
    messages.append(response.message)

    if response.has_tool_calls:
        results = await run_tool_calls(tools, response.tool_calls)
        for tool_call_id, content in results:
            messages.append(Message.tool(content, tool_call_id=tool_call_id))

        # Send tool results back so the model can produce a final answer
        response = await provider.complete(messages, tools=tools)
        messages.append(response.message)

    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
