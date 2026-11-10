"""Tests for parallel execution."""

import asyncio

import pytest

from catena import Chain, Context, Message
from catena.parallel import map_parallel, parallel


def _echo_chain(fail_on: str | None = None, delay: float = 0) -> tuple[Chain, dict]:
    chain = Chain()
    state = {"running": 0, "peak": 0, "finished": 0}

    @chain.step(retries=0)
    async def echo(ctx: Context) -> Context:
        state["running"] += 1
        state["peak"] = max(state["peak"], state["running"])
        try:
            await asyncio.sleep(delay)
            if ctx.messages[0].content == fail_on:
                raise RuntimeError(f"cannot process {fail_on}")
            ctx.add_message(Message.assistant(ctx.messages[0].content.upper()))
            state["finished"] += 1
            return ctx
        finally:
            state["running"] -= 1

    return chain, state


@pytest.mark.asyncio
async def test_map_parallel_keeps_input_order_and_limits_concurrency():
    chain, state = _echo_chain(delay=0.01)

    results = await map_parallel(chain, ["a", "b", "c", "d", "e"], max_concurrency=2)

    assert [r.messages[-1].content for r in results] == ["A", "B", "C", "D", "E"]
    assert state["peak"] == 2


@pytest.mark.asyncio
async def test_map_parallel_raises_the_first_failure_and_stops_the_rest():
    chain, state = _echo_chain(fail_on="a", delay=0.01)

    with pytest.raises(RuntimeError, match="cannot process a"):
        await map_parallel(chain, ["a", "b", "c", "d"], max_concurrency=1)

    await asyncio.sleep(0.05)
    assert state["finished"] == 0
    assert state["running"] == 0


@pytest.mark.asyncio
async def test_map_parallel_can_return_failures_in_place():
    chain, _ = _echo_chain(fail_on="b")

    results = await map_parallel(chain, ["a", "b", "c"], return_exceptions=True)

    assert results[0].messages[-1].content == "A"
    assert isinstance(results[1], RuntimeError)
    assert results[2].messages[-1].content == "C"


@pytest.mark.asyncio
async def test_map_parallel_rejects_zero_concurrency():
    chain, _ = _echo_chain()

    with pytest.raises(ValueError, match="max_concurrency"):
        await map_parallel(chain, ["a"], max_concurrency=0)


@pytest.mark.asyncio
async def test_parallel_gathers_runs():
    chain, _ = _echo_chain()

    results = await parallel(chain.run("a"), chain.run("b"))

    assert [r.messages[-1].content for r in results] == ["A", "B"]
