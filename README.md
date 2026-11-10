# Catena

**Lightweight AI agent orchestration for Python.**

Chain LLM calls together with automatic retries, caching, cost tracking, and parallel execution. Think "LangChain but actually simple."

```python
from catena import Chain, Context
from catena import providers

provider = providers.openai(model="gpt-4o")
chain = Chain()

@chain.step()
async def analyze(ctx: Context) -> Context:
    response = await provider.complete(ctx.messages)
    ctx.add_message(response.message)
    ctx.add_usage(response.usage)
    return ctx

result = await chain.run("Explain quantum computing")
print(result.messages[-1].content)
print(f"Cost: ${result.usage.cost_usd:.4f}")
```

## Features

- **Simple API** - Decorator-based chains, no complex abstractions
- **Multi-provider** - OpenAI and Anthropic built in, plus every model the [`llm`](https://llm.datasette.io) library supports
- **Cost tracking** - Know exactly what you're spending
- **Caching** - Stop paying for repeated calls
- **Retries** - Built-in exponential backoff
- **Parallel execution** - Process batches efficiently
- **Composable** - Chain chains together with `>>`
- **Observable** - Trace every step's time, tokens and cost
- **Tools** - Turn a function into a tool; one call runs the whole tool loop
- **Structured output** - Typed responses validated against a Pydantic model
- **Any model** - OpenAI, Anthropic, or anything with an `llm` plugin

## Installation

```bash
pip install catena-ai

# With provider support
pip install catena-ai[openai]
pip install catena-ai[anthropic]
pip install catena-ai[structured]  # Pydantic, for structured output
pip install catena-ai[llm]         # any model with an `llm` plugin
pip install catena-ai[all]
```

## Quick Start

### Basic Chain

```python
import asyncio
from catena import Chain, Context
from catena import providers

async def main():
    provider = providers.openai()
    chain = Chain()

    @chain.step(retries=3, timeout=30.0)
    async def generate(ctx: Context) -> Context:
        response = await provider.complete(ctx.messages)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        return ctx

    result = await chain.run("Hello, world!")
    print(result.messages[-1].content)

asyncio.run(main())
```

### Multi-Agent Pipeline

```python
# Chain composition with >>
researcher = Chain(name="research")
writer = Chain(name="write")
editor = Chain(name="edit")

# Each chain has its own steps...

# Compose into a pipeline
pipeline = researcher >> writer >> editor
result = await pipeline.run("Write about AI safety")
```

### Parallel Processing

```python
from catena.parallel import map_parallel

# Process 100 documents with 10 concurrent workers
results = await map_parallel(
    summarize_chain,
    documents,
    max_concurrency=10
)

total_cost = sum(r.usage.cost_usd for r in results)
print(f"Processed {len(results)} docs for ${total_cost:.2f}")
```

### Caching

```python
from catena.middleware import MemoryCache

cache = MemoryCache(max_size=1000, default_ttl=3600)
provider = providers.openai().cached(cache)

# First call hits API, second call is free
await chain.run("What is 2+2?")  # Costs $0.0001
await chain.run("What is 2+2?")  # Costs $0.0000

cache.hit_rate  # 0.5
```

A call is served from the cache only when everything that shapes the answer matches: messages, model, system prompt, tools, and options. Tool loops and structured output are cached too; streaming is not.

### Cost Tracking

Every response carries its token counts and cost, and `ctx.add_usage()` totals them across a chain. Prices for current OpenAI and Anthropic models are built in; a model with no known price is tracked at $0, so add your own:

```python
from catena.models import set_price

set_price("my-model", 0.50, 1.50)  # USD per million input / output tokens
```

### Context & State

```python
@chain.step()
async def step1(ctx: Context) -> Context:
    # Store data for later steps
    ctx.set("user_id", 123)
    ctx.set("preferences", {"tone": "casual"})
    return ctx

@chain.step()
async def step2(ctx: Context) -> Context:
    # Retrieve data from previous steps
    user_id = ctx.get("user_id")
    prefs = ctx.get("preferences", {})
    return ctx
```

### Observability

```python
from catena.middleware import ConsoleObserver

chain = Chain(name="my-chain")
chain.observe(ConsoleObserver(verbose=True))

# Output:
# [catena] Starting chain: my-chain
# [catena]   -> step1
# [catena]   <- step1 (234ms, 150 tokens, $0.0003)
# [catena] Complete: 150 tokens, $0.0003
```

Collect traces instead of printing them, with what each step took and cost:

```python
from catena.middleware import TraceCollector

traces = chain.observe(TraceCollector())
await chain.run("Hello")
traces.to_dict()
# [{"chain": "my-chain", "duration_ms": 234.1, "success": True,
#   "total_tokens": 150, "total_cost_usd": 0.0003,
#   "steps": [{"name": "step1", "duration_ms": 233.9, "success": True,
#              "error": None, "tokens": 150, "cost_usd": 0.0003}]}]
```

For one-off callbacks, hook an event directly:

```python
@chain.on("on_error")  # also "before_step", "after_step"
def alert(name, error, ctx):
    print(f"{name} failed: {error}")
```

### Tool Calling

Turn any function into a tool. Its type hints become the schema and its docstring tells the model when and how to use it:

```python
from typing import Literal
from catena import Message, providers, tool

@tool
def get_weather(city: str, units: Literal["c", "f"] = "c") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. "Tokyo".
        units: Celsius or Fahrenheit.
    """
    return f"22 degrees and sunny in {city}"

provider = providers.openai()
run = await provider.run_tools(
    [Message.user("What's the weather in Tokyo?")],
    tools=[get_weather],
)

print(run.content)          # "It's 22 degrees and sunny in Tokyo."
print(run.turns)            # 2 model calls: one to ask for the tool, one to answer
print(run.usage.cost_usd)   # total across both
```

`run_tools` calls the model, runs the tools it asks for (in parallel), sends the results back, and repeats until the model answers. A tool that raises is reported to the model so it can recover. `max_turns` (default 10) caps the number of model calls; beyond it `ToolLoopError` is raised, carrying the conversation and cost so far.

Inside a chain step, keep the whole exchange in the context:

```python
@chain.step()
async def research(ctx: Context) -> Context:
    run = await provider.run_tools(ctx.messages, tools=[get_weather])
    ctx.messages = run.messages
    ctx.add_usage(run.usage)
    return ctx
```

To run the loop yourself, pass `tools=` to `provider.complete()` and use `catena.tools.run_tool_calls` on `response.tool_calls`.

### Structured Output

Get validated, typed objects instead of strings. Requires `pip install catena-ai[structured]`.

```python
from datetime import date
from pydantic import BaseModel

class Invoice(BaseModel):
    company: str
    amount: float
    due_date: date

result = await provider.complete_structured(
    [Message.user(f"Extract this invoice: {text}")],
    schema=Invoice,
    retries=2,  # invalid output is sent back to the model with the error
)

result.output.amount      # float
result.output.due_date    # datetime.date
result.usage.cost_usd     # total across all attempts
```

Raises `StructuredOutputError` if the model can't produce valid output within the retry budget; its `usage` attribute holds what the failed attempts cost.

For output that isn't an object, such as a bare list, use a Pydantic `RootModel`:

```python
class Tags(RootModel[list[str]]):
    pass

result = await provider.complete_structured(messages, schema=Tags)
result.output.root        # list[str]
```

### Streaming

```python
provider = providers.anthropic()

async for chunk in provider.stream([Message.user("Write a haiku")]):
    if chunk.done:
        print(f"\nCost: ${chunk.usage.cost_usd:.4f}")
    else:
        print(chunk.delta, end="", flush=True)
```

### Any Model, via `llm`

[`llm`](https://llm.datasette.io) reaches dozens of hosted and local models through plugins. `providers.llm()` puts all of them behind Catena's chains, retries, cost tracking and caching. Requires `pip install catena-ai[llm]`.

```python
# One-time setup, on the command line:
#   llm keys set openai          (or: llm install llm-ollama, llm-gemini, ...)

provider = providers.llm("gpt-4o-mini")   # any id from `llm models`
provider = providers.llm()                # your `llm models default`

response = await provider.complete([Message.user("Hello!")])
```

It supports everything the built-in providers do: tool calling, streaming and `complete_structured` (which uses the model's own schema support where its plugin has it). Model options go in as keyword arguments, e.g. `provider.complete(messages, seed=7)`. Keys are read from `llm`'s own key store.

Cost is reported for models in Catena's price table (matched by name) and is `0` for others, such as local models.

## Why Catena?

| Feature | Catena | LangChain | Raw API |
|---------|--------|-----------|---------|
| Lines to "Hello World" | 8 | 15+ | 12 |
| Built-in cost tracking | Yes | Plugin | No |
| Caching | Yes | Plugin | No |
| Bundle size | ~50KB | ~5MB | ~100KB |
| Learning curve | 10 min | Days | 1 hour |

## API Reference

### Chain

```python
Chain(name: str = "chain")

# Decorator to add steps
@chain.step(
    name: str = None,      # Step name (defaults to function name)
    retries: int = 3,      # Tries after the first failure (0 = run once)
    retry_delay: float = 1.0,  # Seconds before the first retry; doubles each time
    timeout: float = None, # Timeout per attempt, in seconds
)

# Run the chain
await chain.run(
    input: str | list[Message] | Context,
    context: Context = None,  # Continue from an earlier result; input is added to it
) -> Context

# Hooks: "before_step", "after_step", "on_error"
@chain.on("after_step")
def log(name, ctx, duration): ...

# Compose chains (steps and hooks of both)
combined = chain1 >> chain2 >> chain3
```

### Context

```python
Context(
    messages: list[Message] = [],
    data: dict[str, Any] = {},
    usage: Usage = Usage(),
)

ctx.add_message(message: Message)
ctx.add_usage(usage: Usage)
ctx.get(key: str, default: Any = None) -> Any
ctx.set(key: str, value: Any)
ctx.clone() -> Context
```

### Providers

```python
from catena import providers

# OpenAI
openai = providers.openai(
    api_key: str = None,  # Defaults to OPENAI_API_KEY
    model: str = "gpt-4o",
    base_url: str = None,  # For Azure/proxies
)

# Anthropic
anthropic = providers.anthropic(
    api_key: str = None,  # Defaults to ANTHROPIC_API_KEY
    model: str = "claude-opus-5-5",
)

# Any model with an `llm` plugin (keys and default model come from llm's config)
llm = providers.llm(
    model: str = None,    # Defaults to `llm models default`
    key: str = None,      # Defaults to llm's key store
)

# All providers implement:
await provider.complete(
    messages: list[Message],
    model: str = None,
    temperature: float = None,  # sent only when given; some models reject it
    max_tokens: int = None,
    tools: list[Tool] = None,
    tool_choice: str = None,  # "auto", "none", "required", or one tool's name
) -> Response

# Tool loop: call the model and run its tool calls until it answers
await provider.run_tools(
    messages: list[Message],
    tools: list[Tool],
    max_turns: int = 10,      # cap on model calls; ToolLoopError beyond it
    tool_choice: str = None,  # applies to the first model call only
) -> ToolRun  # .content, .response, .messages, .usage, .turns

# Streaming (yields StreamChunk; the last chunk has done=True and `usage`)
async for chunk in provider.stream(
    messages: list[Message],
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
):
    ...
```

## License

MIT
