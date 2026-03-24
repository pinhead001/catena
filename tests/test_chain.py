"""Tests for the Chain class."""

import pytest
from catena import Chain, Context, Message


@pytest.mark.asyncio
async def test_basic_chain():
    """Test a simple chain with one step."""
    chain = Chain()

    @chain.step()
    async def add_response(ctx: Context) -> Context:
        ctx.add_message(Message.assistant("Hello!"))
        return ctx

    result = await chain.run("Hi")

    assert len(result.messages) == 2
    assert result.messages[0].content == "Hi"
    assert result.messages[1].content == "Hello!"


@pytest.mark.asyncio
async def test_chain_with_context_data():
    """Test passing data between steps."""
    chain = Chain()

    @chain.step()
    async def step1(ctx: Context) -> Context:
        ctx.set("count", 1)
        return ctx

    @chain.step()
    async def step2(ctx: Context) -> Context:
        count = ctx.get("count", 0)
        ctx.set("count", count + 1)
        return ctx

    result = await chain.run()
    assert result.get("count") == 2


@pytest.mark.asyncio
async def test_chain_composition():
    """Test composing chains with >>."""
    chain1 = Chain(name="first")
    chain2 = Chain(name="second")

    @chain1.step()
    async def s1(ctx: Context) -> Context:
        ctx.set("step", 1)
        return ctx

    @chain2.step()
    async def s2(ctx: Context) -> Context:
        ctx.set("step", ctx.get("step", 0) + 1)
        return ctx

    combined = chain1 >> chain2
    result = await combined.run()

    assert result.get("step") == 2
    assert combined.name == "first>>second"


@pytest.mark.asyncio
async def test_sync_step():
    """Test that sync functions work as steps."""
    chain = Chain()

    @chain.step()
    def sync_step(ctx: Context) -> Context:
        ctx.set("sync", True)
        return ctx

    result = await chain.run()
    assert result.get("sync") is True


@pytest.mark.asyncio
async def test_step_hooks():
    """Test before/after step hooks."""
    chain = Chain()
    events = []

    @chain.step()
    async def my_step(ctx: Context) -> Context:
        events.append("step")
        return ctx

    @chain.on("before_step")
    def before(name, ctx):
        events.append(f"before:{name}")

    @chain.on("after_step")
    def after(name, ctx, duration):
        events.append(f"after:{name}")

    await chain.run()

    assert events == ["before:my_step", "step", "after:my_step"]
