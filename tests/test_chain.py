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


@pytest.mark.asyncio
async def test_retries_zero_runs_the_step_once():
    chain = Chain()
    runs = []

    @chain.step(retries=0)
    async def once(ctx: Context) -> Context:
        runs.append(1)
        return ctx

    await chain.run()

    assert len(runs) == 1


@pytest.mark.asyncio
async def test_retries_counts_attempts_after_the_first():
    chain = Chain()
    runs = []

    @chain.step(retries=2, retry_delay=0)
    async def flaky(ctx: Context) -> Context:
        runs.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await chain.run()

    assert len(runs) == 3


@pytest.mark.asyncio
async def test_step_succeeds_on_a_retry():
    chain = Chain()
    runs = []

    @chain.step(retries=1, retry_delay=0)
    async def flaky(ctx: Context) -> Context:
        runs.append(1)
        if len(runs) == 1:
            raise RuntimeError("boom")
        ctx.set("ok", True)
        return ctx

    result = await chain.run()

    assert result.get("ok") is True


@pytest.mark.asyncio
async def test_failed_attempt_does_not_leak_messages_into_the_retry():
    chain = Chain()
    seen = []

    @chain.step(retries=1, retry_delay=0)
    async def flaky(ctx: Context) -> Context:
        ctx.add_message(Message.user("follow-up"))
        seen.append(len(ctx.messages))
        if len(seen) == 1:
            raise RuntimeError("boom")
        return ctx

    result = await chain.run("hi")

    assert seen == [2, 2]
    assert [m.content for m in result.messages] == ["hi", "follow-up"]


def test_negative_retries_is_rejected():
    chain = Chain()

    with pytest.raises(ValueError, match="retries"):
        chain.step(retries=-1)(lambda ctx: ctx)


@pytest.mark.asyncio
async def test_step_timeout_raises_timeout_error():
    import asyncio

    chain = Chain()

    @chain.step(retries=0, timeout=0.01)
    async def slow(ctx: Context) -> Context:
        await asyncio.sleep(1)
        return ctx

    with pytest.raises(TimeoutError, match="slow"):
        await chain.run()


def test_unknown_hook_event_is_rejected():
    chain = Chain()

    with pytest.raises(ValueError, match="after_step"):
        chain.on("after_steps")


@pytest.mark.asyncio
async def test_on_error_hook_fires_and_error_propagates():
    chain = Chain()
    errors = []

    @chain.step(retries=0)
    async def bad(ctx: Context) -> Context:
        raise RuntimeError("boom")

    @chain.on("on_error")
    def record(name, error, ctx):
        errors.append((name, str(error)))

    with pytest.raises(RuntimeError):
        await chain.run()

    assert errors == [("bad", "boom")]


@pytest.mark.asyncio
async def test_run_adds_input_to_an_existing_context():
    chain = Chain()
    first = Context(messages=[Message.user("one"), Message.assistant("two")])

    result = await chain.run("three", context=first)

    assert [m.content for m in result.messages] == ["one", "two", "three"]
    assert len(first.messages) == 2


@pytest.mark.asyncio
async def test_run_rejects_two_contexts():
    with pytest.raises(ValueError, match="not both"):
        await Chain().run(Context(), context=Context())


@pytest.mark.asyncio
async def test_run_does_not_mutate_the_input_message_list():
    chain = Chain()
    messages = [Message.user("hi")]

    @chain.step()
    async def reply(ctx: Context) -> Context:
        ctx.add_message(Message.assistant("hello"))
        return ctx

    await chain.run(messages)

    assert len(messages) == 1


@pytest.mark.asyncio
async def test_composed_chain_keeps_hooks_from_both_chains():
    first, second = Chain(name="a"), Chain(name="b")
    events = []

    @first.step()
    async def s1(ctx: Context) -> Context:
        return ctx

    @second.step()
    async def s2(ctx: Context) -> Context:
        return ctx

    def shared(name, ctx):
        events.append(f"shared:{name}")

    first.on("before_step")(shared)
    second.on("before_step")(shared)

    @second.on("before_step")
    def only_second(name, ctx):
        events.append(f"second:{name}")

    await (first >> second).run()

    assert events == ["shared:s1", "second:s1", "shared:s2", "second:s2"]
