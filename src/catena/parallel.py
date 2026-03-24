"""Parallel execution utilities."""

from __future__ import annotations

import asyncio
from typing import Any

from catena.context import Context
from catena.models import Message


async def parallel(*coros: Any) -> list[Context]:
    """Execute multiple chain runs in parallel.

    Example:
        ```python
        results = await parallel(
            chain.run("Analyze revenue"),
            chain.run("Analyze costs"),
            chain.run("Analyze growth"),
        )

        for result in results:
            print(result.messages[-1].content)
        ```
    """
    return await asyncio.gather(*coros)


async def map_parallel(
    chain: Any,  # Chain type, avoiding circular import
    inputs: list[str | list[Message] | Context],
    max_concurrency: int = 5,
) -> list[Context]:
    """Map a chain over multiple inputs with concurrency control.

    Example:
        ```python
        documents = ["doc1.txt content", "doc2.txt content", "doc3.txt content"]
        results = await map_parallel(summarize_chain, documents, max_concurrency=3)
        ```
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_limit(input_data: str | list[Message] | Context) -> Context:
        async with semaphore:
            return await chain.run(input_data)

    return await asyncio.gather(*[run_with_limit(inp) for inp in inputs])
