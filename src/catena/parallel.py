"""Parallel execution utilities."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, cast, overload

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
    return cast(list[Context], await asyncio.gather(*coros))


ChainInput = str | list[Message] | Context


@overload
async def map_parallel(
    chain: Any,
    inputs: Sequence[ChainInput],
    max_concurrency: int = 5,
    return_exceptions: Literal[False] = False,
) -> list[Context]: ...


@overload
async def map_parallel(
    chain: Any,
    inputs: Sequence[ChainInput],
    max_concurrency: int = 5,
    *,
    return_exceptions: Literal[True],
) -> list[Context | BaseException]: ...


async def map_parallel(
    chain: Any,  # Chain type, avoiding circular import
    inputs: Sequence[ChainInput],
    max_concurrency: int = 5,
    return_exceptions: bool = False,
) -> list[Context] | list[Context | BaseException]:
    """Map a chain over multiple inputs with concurrency control.

    Results come back in the order of `inputs`. By default the first failure
    is raised and the other results are lost; with `return_exceptions=True`
    every input finishes and a failed one has its exception in its place.

    Example:
        ```python
        documents = ["doc1.txt content", "doc2.txt content", "doc3.txt content"]
        results = await map_parallel(summarize_chain, documents, max_concurrency=3)
        ```
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_limit(input_data: ChainInput) -> Context:
        async with semaphore:
            result = await chain.run(input_data)
            return cast(Context, result)

    tasks = [asyncio.ensure_future(run_with_limit(inp)) for inp in inputs]
    try:
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    except BaseException:
        # Don't leave the rest running (and spending) after a failure or cancellation.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
