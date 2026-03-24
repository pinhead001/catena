"""Chain orchestration - the core of Catena."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from catena.context import Context
from catena.models import Message, Response

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
    cache: bool = False


@dataclass
class Step:
    """A single step in a chain."""

    func: StepFunc
    config: StepConfig

    async def execute(self, ctx: Context) -> Context:
        """Execute this step with retries."""
        last_error: Exception | None = None

        for attempt in range(self.config.retries):
            try:
                if self.config.timeout:
                    return await asyncio.wait_for(
                        self.func(ctx),
                        timeout=self.config.timeout,
                    )
                return await self.func(ctx)
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Step '{self.config.name}' timed out after {self.config.timeout}s"
                )
            except Exception as e:
                last_error = e

            if attempt < self.config.retries - 1:
                await asyncio.sleep(self.config.retry_delay * (2**attempt))

        raise last_error or RuntimeError(f"Step '{self.config.name}' failed")


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
        cache: bool = False,
    ) -> Callable[[StepFunc | SyncStepFunc], StepFunc]:
        """Decorator to register a step in the chain."""

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
                cache=cache,
            )
            self.steps.append(Step(func=wrapped, config=config))
            return wrapped

        return decorator

    def add_step(self, step: Step) -> None:
        """Add a pre-configured step."""
        self.steps.append(step)

    def on(self, event: str) -> Callable[[T], T]:
        """Register an event hook."""

        def decorator(func: T) -> T:
            if event in self._hooks:
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
            context: Optionally provide an existing context

        Returns:
            The final context after all steps complete
        """
        # Build initial context
        if context is not None:
            ctx = context.clone()
        elif isinstance(input, Context):
            ctx = input.clone()
        elif isinstance(input, list):
            ctx = Context(messages=input.copy())
        elif isinstance(input, str):
            ctx = Context(messages=[Message.user(input)])
        else:
            ctx = Context()

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
        """Compose chains with >> operator."""
        composed = Chain(name=f"{self.name}>>{other.name}")
        composed.steps = self.steps + other.steps
        return composed


# Standalone step decorator for quick chains
@overload
def step(func: StepFunc) -> StepFunc: ...


@overload
def step(
    name: str | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float | None = None,
) -> Callable[[StepFunc], StepFunc]: ...


def step(
    func: StepFunc | None = None,
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
