"""Chain orchestration - the core of Catena."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from catena.context import Context
from catena.models import Message

T = TypeVar("T")

# Step function types
StepFunc = Callable[[Context], Awaitable[Context]]
SyncStepFunc = Callable[[Context], Context]


@dataclass
class StepConfig:
    """Configuration for a chain step."""

    name: str
    retries: int = 3
    retry_delay: float = 1.0
    timeout: float | None = None

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError(f"retries must be >= 0, got {self.retries}")


@dataclass
class Step:
    """A single step in a chain."""

    func: StepFunc
    config: StepConfig

    async def execute(self, ctx: Context) -> Context:
        """Run the step once, then up to `retries` more times if it fails."""
        attempts = self.config.retries + 1

        for attempt in range(attempts):
            # Each attempt gets its own copy, so a failed attempt's changes
            # (a message it added before the call failed) don't leak into the next.
            attempt_ctx = ctx.clone()
            try:
                if self.config.timeout:
                    return await asyncio.wait_for(
                        self.func(attempt_ctx),
                        timeout=self.config.timeout,
                    )
                return await self.func(attempt_ctx)
            except asyncio.TimeoutError:
                last_error: Exception = TimeoutError(
                    f"Step '{self.config.name}' timed out after {self.config.timeout}s"
                )
            except Exception as e:
                last_error = e

            if attempt < attempts - 1:
                await asyncio.sleep(self.config.retry_delay * (2**attempt))

        raise last_error


class Chain:
    """A chain of LLM steps to execute sequentially.

    Example:
        ```python
        from catena import Chain, step, Context

        chain = Chain()

        @chain.step()
        async def analyze(ctx: Context) -> Context:
            response = await provider.complete(ctx.messages)
            ctx.add_message(response.message)
            return ctx

        result = await chain.run("Analyze this code: ...")
        print(result.messages[-1].content)
        ```
    """

    def __init__(self, name: str = "chain") -> None:
        self.name = name
        self.steps: list[Step] = []
        self._hooks: dict[str, list[Callable[..., Any]]] = {
            "before_step": [],
            "after_step": [],
            "on_error": [],
        }

    def step(
        self,
        name: str | None = None,
        retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float | None = None,
    ) -> Callable[[StepFunc | SyncStepFunc], StepFunc]:
        """Decorator to register a step in the chain.

        `retries` is how many times to try again after a failure, with
        exponential backoff starting at `retry_delay` seconds; 0 runs the step once.
        """

        def decorator(func: StepFunc | SyncStepFunc) -> StepFunc:
            step_name = name or func.__name__

            # Wrap sync functions
            if not asyncio.iscoroutinefunction(func):
                sync_func = func

                @functools.wraps(sync_func)
                async def async_wrapper(ctx: Context) -> Context:
                    return sync_func(ctx)  # type: ignore

                wrapped = async_wrapper
            else:
                wrapped = func  # type: ignore

            config = StepConfig(
                name=step_name,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
            )
            self.steps.append(Step(func=wrapped, config=config))
            return wrapped

        return decorator

    def add_step(self, step: Step) -> None:
        """Add a pre-configured step."""
        self.steps.append(step)

    def on(self, event: str) -> Callable[[T], T]:
        """Register a hook for "before_step", "after_step" or "on_error"."""
        if event not in self._hooks:
            raise ValueError(f"Unknown event {event!r}; choose from {sorted(self._hooks)}")

        def decorator(func: T) -> T:
            self._hooks[event].append(func)  # type: ignore
            return func

        return decorator

    async def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event to all registered hooks."""
        for hook in self._hooks.get(event, []):
            result = hook(*args, **kwargs)
            if asyncio.iscoroutine(result):
                await result

    async def run(
        self,
        input: str | list[Message] | Context | None = None,
        context: Context | None = None,
    ) -> Context:
        """Execute the chain.

        Args:
            input: Initial input - can be a string, list of messages, or context
            context: An existing context to continue from; `input` is added to it

        Returns:
            The final context after all steps complete
        """
        if isinstance(input, Context):
            if context is not None:
                raise ValueError("Pass a Context as input or as context, not both")
            context, input = input, None

        ctx = context.clone() if context is not None else Context()
        if isinstance(input, str):
            ctx.add_message(Message.user(input))
        elif input is not None:
            ctx.messages.extend(input)

        # Execute steps
        for step in self.steps:
            await self._emit("before_step", step.config.name, ctx)
            start = time.perf_counter()

            try:
                ctx = await step.execute(ctx)
                elapsed = time.perf_counter() - start
                await self._emit("after_step", step.config.name, ctx, elapsed)
            except Exception as e:
                await self._emit("on_error", step.config.name, e, ctx)
                raise

        return ctx

    def __rshift__(self, other: Chain) -> Chain:
        """Compose chains with >>: the steps of both, and the hooks of both.

        Hooks apply to every step of the composed chain; one registered on
        both chains is kept once.
        """
        composed = Chain(name=f"{self.name}>>{other.name}")
        composed.steps = self.steps + other.steps
        for event, hooks in composed._hooks.items():
            for hook in self._hooks[event] + other._hooks[event]:
                if not any(hook is kept for kept in hooks):
                    hooks.append(hook)
        return composed


# Standalone step decorator for quick chains
@overload
def step(func: StepFunc) -> StepFunc: ...


@overload
def step(
    *,
    name: str | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float | None = None,
) -> Callable[[StepFunc], StepFunc]: ...


def step(
    func: StepFunc | None = None,
    *,
    name: str | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float | None = None,
) -> StepFunc | Callable[[StepFunc], StepFunc]:
    """Standalone decorator for creating reusable steps."""

    def decorator(f: StepFunc) -> StepFunc:
        @functools.wraps(f)
        async def wrapper(ctx: Context) -> Context:
            return await f(ctx)

        wrapper._step_config = StepConfig(  # type: ignore
            name=name or f.__name__,
            retries=retries,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
