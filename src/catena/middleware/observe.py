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
    """Abstract observer for chain execution."""

    @abstractmethod
    def on_chain_start(self, chain_name: str) -> None:
        """Called when a chain starts executing."""
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
        """Called when a step fails."""
        ...

    @abstractmethod
    def on_chain_end(self, ctx: Context) -> None:
        """Called when a chain completes."""
        ...


class ConsoleObserver(Observer):
    """Simple console logger for chain execution.

    Example:
        ```python
        from catena import Chain
        from catena.middleware import ConsoleObserver

        chain = Chain()
        observer = ConsoleObserver(verbose=True)

        @chain.on("before_step")
        def log_start(name, ctx):
            observer.on_step_start(name)

        @chain.on("after_step")
        def log_end(name, ctx, duration):
            observer.on_step_end(name, ctx, duration)
        ```
    """

    def __init__(self, verbose: bool = False, prefix: str = "[catena]") -> None:
        self.verbose = verbose
        self.prefix = prefix
        self._current_trace: ChainTrace | None = None

    def on_chain_start(self, chain_name: str) -> None:
        self._current_trace = ChainTrace(chain_name=chain_name, started_at=time.time())
        print(f"{self.prefix} Starting chain: {chain_name}")

    def on_step_start(self, step_name: str) -> None:
        if self.verbose:
            print(f"{self.prefix}   -> {step_name}")

    def on_step_end(self, step_name: str, ctx: Context, duration: float) -> None:
        if self._current_trace:
            self._current_trace.steps.append(
                StepTrace(
                    name=step_name,
                    started_at=time.time() - duration,
                    ended_at=time.time(),
                    usage=ctx.usage,
                )
            )

        if self.verbose:
            cost = ctx.usage.cost_usd
            tokens = ctx.usage.input_tokens + ctx.usage.output_tokens
            print(f"{self.prefix}   <- {step_name} ({duration*1000:.0f}ms, {tokens} tokens, ${cost:.4f})")

    def on_step_error(self, step_name: str, error: Exception) -> None:
        print(f"{self.prefix}   !! {step_name} failed: {error}")
        if self._current_trace:
            self._current_trace.steps.append(
                StepTrace(
                    name=step_name,
                    started_at=time.time(),
                    ended_at=time.time(),
                    success=False,
                    error=str(error),
                )
            )

    def on_chain_end(self, ctx: Context) -> None:
        if self._current_trace:
            self._current_trace.ended_at = time.time()
            self._current_trace.total_usage = ctx.usage

        usage = ctx.usage
        print(
            f"{self.prefix} Complete: "
            f"{usage.input_tokens + usage.output_tokens} tokens, "
            f"${usage.cost_usd:.4f}"
        )


class TraceCollector(Observer):
    """Collects traces for later analysis or export."""

    def __init__(self) -> None:
        self.traces: list[ChainTrace] = []
        self._current: ChainTrace | None = None
        self._step_start: float = 0

    def on_chain_start(self, chain_name: str) -> None:
        self._current = ChainTrace(chain_name=chain_name, started_at=time.time())

    def on_step_start(self, step_name: str) -> None:
        self._step_start = time.time()

    def on_step_end(self, step_name: str, ctx: Context, duration: float) -> None:
        if self._current:
            self._current.steps.append(
                StepTrace(
                    name=step_name,
                    started_at=self._step_start,
                    ended_at=time.time(),
                    usage=ctx.usage,
                )
            )

    def on_step_error(self, step_name: str, error: Exception) -> None:
        if self._current:
            self._current.steps.append(
                StepTrace(
                    name=step_name,
                    started_at=self._step_start,
                    ended_at=time.time(),
                    success=False,
                    error=str(error),
                )
            )

    def on_chain_end(self, ctx: Context) -> None:
        if self._current:
            self._current.ended_at = time.time()
            self._current.total_usage = ctx.usage
            self.traces.append(self._current)
            self._current = None

    def to_dict(self) -> list[dict[str, Any]]:
        """Export traces as JSON-serializable dicts."""
        return [
            {
                "chain": t.chain_name,
                "duration_ms": t.duration_ms,
                "success": t.success,
                "total_tokens": t.total_usage.input_tokens + t.total_usage.output_tokens,
                "total_cost_usd": t.total_usage.cost_usd,
                "steps": [
                    {
                        "name": s.name,
                        "duration_ms": s.duration_ms,
                        "success": s.success,
                        "error": s.error,
                    }
                    for s in t.steps
                ],
            }
            for t in self.traces
        ]
