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
- **Multi-provider** - OpenAI, Anthropic, or bring your own
- **Cost tracking** - Know exactly what you're spending
- **Caching** - Stop paying for repeated calls
- **Retries** - Built-in exponential backoff
- **Parallel execution** - Process batches efficiently
- **Composable** - Chain chains together with `>>`
- **Observable** - Hook into execution events

## Installation

```bash
pip install catena-ai

# With provider support
pip install catena-ai[openai]
pip install catena-ai[anthropic]
pip install catena-ai[structured]  # Pydantic, for structured output
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
provider.complete = cache.wrap(provider.complete)

# First call hits API, second call is free
await chain.run("What is 2+2?")  # Costs $0.0001
await chain.run("What is 2+2?")  # Costs $0.0000
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

chain = Chain()
observer = ConsoleObserver(verbose=True)

@chain.on("before_step")
def on_start(name, ctx):
    observer.on_step_start(name)

@chain.on("after_step")
def on_end(name, ctx, duration):
    observer.on_step_end(name, ctx, duration)

# Output:
# [catena] Starting chain: my-chain
# [catena]   -> step1
# [catena]   <- step1 (234ms, 150 tokens, $0.0003)
# [catena] Complete: 150 tokens, $0.0003
```

### Tool Calling

```python
from catena import Message, providers
from catena.tools import tool, run_tool_calls

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72F and sunny in {city}"

provider = providers.openai()
messages = [Message.user("What's the weather in Tokyo?")]

response = await provider.complete(messages, tools=[get_weather])
messages.append(response.message)

if response.has_tool_calls:
    results = await run_tool_calls([get_weather], response.tool_calls)
    for tool_call_id, content in results:
        messages.append(Message.tool(content, tool_call_id=tool_call_id))

    response = await provider.complete(messages, tools=[get_weather])
```

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

Raises `StructuredOutputError` if the model can't produce valid output within the retry budget.

### Streaming

```python
provider = providers.anthropic()

async for chunk in provider.stream([Message.user("Write a haiku")]):
    if chunk.done:
        print(f"\nCost: ${chunk.usage.cost_usd:.4f}")
    else:
        print(chunk.delta, end="", flush=True)
```

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
    retries: int = 3,      # Retry attempts
    retry_delay: float = 1.0,  # Base delay between retries
    timeout: float = None, # Timeout in seconds
)

# Run the chain
await chain.run(
    input: str | list[Message] | Context,
    context: Context = None,
) -> Context

# Compose chains
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
    model: str = "claude-3-5-sonnet-20241022",
)

# All providers implement:
await provider.complete(
    messages: list[Message],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = None,
    tools: list[Tool] = None,
) -> Response

# Streaming (yields StreamChunk; the last chunk has done=True and `usage`)
async for chunk in provider.stream(
    messages: list[Message],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = None,
):
    ...
```

## License

MIT
