"""Example: Extracting typed data with structured output."""

import asyncio
from datetime import date

from pydantic import BaseModel

from catena import Message, providers


class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float


class Invoice(BaseModel):
    """Data extracted from an invoice."""

    company: str
    due_date: date
    items: list[LineItem]


INVOICE_TEXT = """
ACME Corp - Invoice #4471
Due: October 1, 2026
3x Widget @ $19.99
1x Setup fee @ $150.00
"""


async def main():
    provider = providers.openai(model="gpt-4o-mini")

    result = await provider.complete_structured(
        [Message.user(f"Extract this invoice:\n{INVOICE_TEXT}")],
        schema=Invoice,
    )

    invoice = result.output
    total = sum(item.quantity * item.unit_price for item in invoice.items)
    print(f"{invoice.company}, due {invoice.due_date:%b %d}: ${total:.2f}")
    print(f"Attempts: {result.attempts}, cost: ${result.usage.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
