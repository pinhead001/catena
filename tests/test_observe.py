"""Tests for chain observers."""

import pytest

from catena import Chain, Context, Usage
from catena.middleware import ConsoleObserver, TraceCollector


def _chain(costs: list[float], fail_at: int | None = None) -> Chain:
    chain = Chain(name="demo")
    for index, cost in enumerate(costs):

        def make(index: int = index, cost: float = cost):  # type: ignore[no-untyped-def]
            async def run(ctx: Context) -> Context:
                if index == fail_at:
                    raise RuntimeError("boom")
                ctx.add_usage(Usage(input_tokens=10, output_tokens=5, cost_usd=cost))
                return ctx

            return run

        chain.step(name=f"step{index}", retries=0)(make())
    return chain


@pytest.mark.asyncio
async def test_trace_records_what_each_step_added():
    chain = _chain([0.01, 0.02])
    traces = chain.observe(TraceCollector())

    await chain.run("hi")

    (trace,) = traces.traces
    assert trace.chain_name == "demo"
    assert trace.success
    assert [s.name for s in trace.steps] == ["step0", "step1"]
    assert [s.usage.cost_usd for s in trace.steps] == pytest.approx([0.01, 0.02])
    assert trace.total_usage.cost_usd == pytest.approx(0.03)
    assert trace.total_usage.total_tokens == 30


@pytest.mark.asyncio
async def test_trace_excludes_usage_from_before_the_run():
    chain = _chain([0.01])
    traces = chain.observe(TraceCollector())
    earlier = Context(usage=Usage(input_tokens=100, output_tokens=100, cost_usd=5.0))

    await chain.run("next question", context=earlier)

    assert traces.traces[0].steps[0].usage.cost_usd == pytest.approx(0.01)
    assert traces.traces[0].total_usage.cost_usd == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_failed_run_is_traced_and_still_raises():
    chain = _chain([0.01, 0.02], fail_at=1)
    traces = chain.observe(TraceCollector())

    with pytest.raises(RuntimeError, match="boom"):
        await chain.run("hi")

    (trace,) = traces.traces
    assert not trace.success
    assert [(s.name, s.success, s.error) for s in trace.steps] == [
        ("step0", True, None),
        ("step1", False, "boom"),
    ]
    assert trace.total_usage.cost_usd == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_each_run_gets_its_own_trace_and_exports():
    chain = _chain([0.01])
    traces = chain.observe(TraceCollector())

    await chain.run("one")
    await chain.run("two")

    exported = traces.to_dict()
    assert len(exported) == 2
    assert exported[1]["total_cost_usd"] == pytest.approx(0.01)
    assert exported[1]["steps"][0]["tokens"] == 15


@pytest.mark.asyncio
async def test_console_observer_prints_the_run(capsys):
    chain = _chain([0.01, 0.02])
    chain.observe(ConsoleObserver(verbose=True))

    await chain.run("hi")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[catena] Starting chain: demo"
    assert lines[1] == "[catena]   -> step0"
    assert lines[2].startswith("[catena]   <- step0 (")
    assert lines[2].endswith("15 tokens, $0.0100)")
    assert lines[4].endswith("15 tokens, $0.0200)")
    assert lines[5] == "[catena] Complete: 30 tokens, $0.0300"


@pytest.mark.asyncio
async def test_console_observer_reports_a_failure(capsys):
    chain = _chain([0.01], fail_at=0)
    chain.observe(ConsoleObserver())

    with pytest.raises(RuntimeError):
        await chain.run("hi")

    out = capsys.readouterr().out
    assert "!! step0 failed: boom" in out
    assert "[catena] Failed: 0 tokens" in out


@pytest.mark.asyncio
async def test_composed_chain_keeps_observers():
    first, second = _chain([0.01]), _chain([0.02])
    traces = first.observe(TraceCollector())
    second.observe(traces)

    await (first >> second).run("hi")

    (trace,) = traces.traces
    assert len(trace.steps) == 2
