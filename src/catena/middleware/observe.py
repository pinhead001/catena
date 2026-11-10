"""Observability middleware for chains."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from catena.context import Context
from catena.models import Usage


@dataclass
class StepTrace:
    """Trace data for a single step execution."""

    name: str
    started_at: float
    ended_at: float | None = None
    success: bool = True
    error: str | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at) * 1000


@dataclass
class ChainTrace:
    """Complete trace of a chain execution."""

    chain_name: str
    started_at: float
    ended_at: float | None = None
    steps: list[StepTrace] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at) * 1000

    @property
    def success(self) -> bool:
        return all(s.success for s in self.steps)


class Observer(ABC):
    """Receives the events of a chain run. Attach one with `chain.observe(observer)`."""

    @abstractmethod
    def on_chain_start(self, chain_name: str, ctx: Context | None = None) -> None:
        """Called when a chain starts executing, with its starting context."""
        ...

    @abstractmethod
    def on_step_start(self, step_name: str) -> None:
        """Called when a step starts."""
        ...

    @abstractmethod
    def on_step_end(self, step_name: str, ctx: Context, duration: float) -> None:
        """Called when a step completes successfully."""
        ...

    @abstractmethod
    def on_step_error(self, step_name: str, error: Exception) -> None:
        """Called when a step fails after its retries."""
        ...

    @abstractmethod
    def on_chain_end(self, ctx: Context) -> None:
        """Called when a chain finishes, whether it completed or a step failed."""
        ...


class TraceCollector(Observer):
    """Collects a trace of each chain run for later analysis or export.

    Each step's usage is what that step added, and a trace's total is what
    the run added, even when the run continues an earlier context. An observer
    follows one run at a time: give concurrent runs (`map_parallel`) their own
    chain and observer if you need per-run traces.

    Example:
        ```python
        traces = chain.observe(TraceCollector())
        await chain.run("Hello")
        print(traces.to_dict())
        ```
    """

    def __init__(self) -> None:
        self.traces: list[ChainTrace] = []
        self._current: ChainTrace | None = None
        self._step_start: float = 0
        self._seen = Usage()  # usage already attributed to earlier steps or runs

    def _step_usage(self, ctx: Context) -> Usage:
        added = ctx.usage - self._seen
        self._seen = ctx.usage
        return added

    def on_chain_start(self, chain_name: str, ctx: Context | None = None) -> None:
        self._current = ChainTrace(chain_name=chain_name, started_at=time.time())
        self._seen = ctx.usage if ctx is not None else Usage()

    def on_step_start(self, step_name: str) -> None:
        self._step_start = time.time()

    def on_step_end(self, step_name: str, ctx: Context, duration: float) -> None:
        now = time.time()
        trace = StepTrace(
            name=step_name,
            started_at=now - duration,
            ended_at=now,
            usage=self._step_usage(ctx),
        )
        if self._current:
            self._current.steps.append(trace)
            self._current.total_usage = self._current.total_usage + trace.usage
        self._report_step(trace)

    def on_step_error(self, step_name: str, error: Exception) -> None:
        now = time.time()
        trace = StepTrace(
            name=step_name,
            started_at=self._step_start or now,
            ended_at=now,
            success=False,
            error=str(error),
        )
        if self._current:
            self._current.steps.append(trace)
        self._report_step(trace)

    def on_chain_end(self, ctx: Context) -> None:
        if self._current:
            self._current.ended_at = time.time()
            self.traces.append(self._current)
            self._report_chain(self._current)
            self._current = None

    def _report_step(self, trace: StepTrace) -> None:
        """Hook for subclasses: a step finished or failed."""

    def _report_chain(self, trace: ChainTrace) -> None:
        """Hook for subclasses: the run finished or failed."""

    def to_dict(self) -> list[dict[str, Any]]:
        """Export traces as JSON-serializable dicts."""
        return [
            {
                "chain": t.chain_name,
                "duration_ms": t.duration_ms,
                "success": t.success,
                "total_tokens": t.total_usage.total_tokens,
                "total_cost_usd": t.total_usage.cost_usd,
                "steps": [
                    {
                        "name": s.name,
                        "duration_ms": s.duration_ms,
                        "success": s.success,
                        "error": s.error,
                        "tokens": s.usage.total_tokens,
                        "cost_usd": s.usage.cost_usd,
                    }
                    for s in t.steps
                ],
            }
            for t in self.traces
        ]


class ConsoleObserver(TraceCollector):
    """Prints each chain run to the console as it happens.

    Example:
        ```python
        from catena import Chain
        from catena.middleware import ConsoleObserver

        chain = Chain(name="my-chain")
        chain.observe(ConsoleObserver(verbose=True))

        # [catena] Starting chain: my-chain
        # [catena]   -> step1
        # [catena]   <- step1 (234ms, 150 tokens, $0.0003)
        # [catena] Complete: 150 tokens, $0.0003
        ```
    """

    def __init__(self, verbose: bool = False, prefix: str = "[catena]") -> None:
        super().__init__()
        self.verbose = verbose
        self.prefix = prefix

    def on_chain_start(self, chain_name: str, ctx: Context | None = None) -> None:
        super().on_chain_start(chain_name, ctx)
        print(f"{self.prefix} Starting chain: {chain_name}")

    def on_step_start(self, step_name: str) -> None:
        super().on_step_start(step_name)
        if self.verbose:
            print(f"{self.prefix}   -> {step_name}")

    def _report_step(self, trace: StepTrace) -> None:
        if not trace.success:
            print(f"{self.prefix}   !! {trace.name} failed: {trace.error}")
        elif self.verbose:
            print(
                f"{self.prefix}   <- {trace.name} ({trace.duration_ms:.0f}ms, "
                f"{trace.usage.total_tokens} tokens, ${trace.usage.cost_usd:.4f})"
            )

    def _report_chain(self, trace: ChainTrace) -> None:
        outcome = "Complete" if trace.success else "Failed"
        usage = trace.total_usage
        print(f"{self.prefix} {outcome}: {usage.total_tokens} tokens, ${usage.cost_usd:.4f}")
