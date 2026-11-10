"""Chain orchestration - the core of Catena."""

from __future__ import annotations

import asyncio
import functools
import inspect
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, overload

from catena.context import Context
from catena.models import Message

if TYPE_CHECKING:
    from catena.middleware.observe import Observer

T = TypeVar("T")
ObserverT = TypeVar("ObserverT", bound="Observer")

# Step function types
StepFunc = Callable[[Context], Awaitable[Context]]
SyncStepFunc = Callable[[Context], Context]


def should_retry(error: Exception) -> bool:
    """The default retry policy: retry unless trying again cannot help.

    Not retried: errors that say so themselves (`retryable = False`, as on
    `StructuredOutputError` and `ToolLoopError`, which have already used
    their own budget), and HTTP errors that mean the request itself is wrong
    (4xx such as a bad API key or an invalid request), other than timeouts,
    conflicts and rate limits (408, 409, 429).
    """
    if getattr(error, "retryable", True) is False:
        return False
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return status in (408, 409, 429)
    return True


@dataclass
class StepConfig:
    """Configuration for a chain step."""

    name: str
    retries: int = 3
    retry_delay: float = 1.0
    timeout: float | None = None
    retry_on: Callable[[Exception], bool] = should_retry

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
            except Exception as e:
                last_error: Exception = e
                if isinstance(e, asyncio.TimeoutError) and self.config.timeout:
                    last_error = TimeoutError(
                        f"Step '{self.config.name}' timed out after {self.config.timeout}s"
                    )
                if not self.config.retry_on(last_error):
                    break

            if attempt < attempts - 1:
                # Jittered, so parallel runs that fail together don't retry together.
                delay = self.config.retry_delay * (2**attempt)
                await asyncio.sleep(delay * random.uniform(0.75, 1.25))

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
        self._observers: list[Observer] = []

    def step(
        self,
        name: str | None = None,
        retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        retry_on: Callable[[Exception], bool] = should_retry,
    ) -> Callable[[StepFunc | SyncStepFunc], StepFunc]:
        """Decorator to register a step in the chain.

        `retries` is how many times to try again after a failure, with
        exponential backoff starting at `retry_delay` seconds; 0 runs the step once.
        `retry_on` decides whether an error is worth another try; by default
        everything is, except errors a retry cannot fix (see `should_retry`).
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
                retry_on=retry_on,
            )
            self.steps.append(Step(func=wrapped, config=config))
            return wrapped

        return decorator

    def add_step(self, step: Step | StepFunc) -> None:
        """Add a `Step`, or a function made reusable with the standalone `@step`.

        Example:
            ```python
            @step(retries=1)
            async def summarize(ctx: Context) -> Context: ...

            drafts.add_step(summarize)
            reports.add_step(summarize)
            ```
        """
        if not isinstance(step, Step):
            config = getattr(step, "_step_config", None)
            if config is None:
                raise TypeError(
                    f"{getattr(step, '__name__', step)!r} is not a step; "
                    "decorate it with @step or use @chain.step()"
                )
            step = Step(func=step, config=config)
        self.steps.append(step)

    def on(self, event: str) -> Callable[[T], T]:
        """Register a hook for "before_step", "after_step" or "on_error"."""
        if event not in self._hooks:
            raise ValueError(f"Unknown event {event!r}; choose from {sorted(self._hooks)}")

        def decorator(func: T) -> T:
            self._hooks[event].append(func)  # type: ignore
            return func

        return decorator

    def observe(self, observer: ObserverT) -> ObserverT:
        """Report every run of this chain to an observer; returns the observer.

        Example:
            ```python
            traces = chain.observe(TraceCollector())
            await chain.run("Hello")
            print(traces.to_dict())
            ```
        """
        self._observers.append(observer)
        return observer

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

        for observer in self._observers:
            observer.on_chain_start(self.name, ctx)
        try:
            for step in self.steps:
                ctx = await self._run_step(step, ctx)
        finally:
            # Also after a failure, with the context as the last good step left it.
            for observer in self._observers:
                observer.on_chain_end(ctx)

        return ctx

    async def _run_step(self, step: Step, ctx: Context) -> Context:
        name = step.config.name
        await self._emit("before_step", name, ctx)
        for observer in self._observers:
            observer.on_step_start(name)
        start = time.perf_counter()

        try:
            result = await step.execute(ctx)
        except Exception as e:
            await self._emit("on_error", name, e, ctx)
            for observer in self._observers:
                observer.on_step_error(name, e)
            raise

        if not isinstance(result, Context):
            raise TypeError(
                f"Step '{name}' returned {type(result).__name__}, not the Context; "
                "end the step with `return ctx`"
            )

        elapsed = time.perf_counter() - start
        await self._emit("after_step", name, result, elapsed)
        for observer in self._observers:
            observer.on_step_end(name, result, elapsed)
        return result

    def __rshift__(self, other: Chain) -> Chain:
        """Compose chains with >>: the steps, hooks and observers of both.

        Hooks and observers apply to every step of the composed chain; one
        registered on both chains is kept once.
        """
        composed = Chain(name=f"{self.name}>>{other.name}")
        composed.steps = self.steps + other.steps
        for observer in self._observers + other._observers:
            if not any(observer is kept for kept in composed._observers):
                composed._observers.append(observer)
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
    retry_on: Callable[[Exception], bool] = should_retry,
) -> Callable[[StepFunc], StepFunc]: ...


def step(
    func: StepFunc | None = None,
    *,
    name: str | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float | None = None,
    retry_on: Callable[[Exception], bool] = should_retry,
) -> StepFunc | Callable[[StepFunc], StepFunc]:
    """Make a function a reusable step; add it to chains with `chain.add_step()`."""

    def decorator(f: StepFunc) -> StepFunc:
        @functools.wraps(f)
        async def wrapper(ctx: Context) -> Context:
            result = f(ctx)
            return await result if inspect.isawaitable(result) else result  # sync steps too

        wrapper._step_config = StepConfig(  # type: ignore
            name=name or f.__name__,
            retries=retries,
            retry_delay=retry_delay,
            timeout=timeout,
            retry_on=retry_on,
        )
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
