# Changelog

## 0.4.0

### Added

- **Any model through `llm`**: `providers.llm("model-id")` runs Catena on every
  model the [`llm`](https://llm.datasette.io) library can reach. Install with
  `pip install "catena-ai[llm]"` (Python 3.11+).
- **Tool loop**: `provider.run_tools(messages, tools=[...])` calls the model,
  runs the tools it asks for, and repeats until it answers.
- **Better tool schemas**: `@tool` describes lists, dicts, optionals, `Literal`,
  `Enum` and Pydantic models to the model, takes argument descriptions from
  the docstring's `Args:` section, and converts the model's JSON arguments to
  the annotated types before calling your function.
- **Observers**: `chain.observe(TraceCollector())` or `ConsoleObserver()` records
  every run, with the time, tokens and cost of each step.
- **Persistent cache**: `SQLiteCache("file.db")` keeps answers between runs.
  `provider.cached(cache)` attaches any cache.
- **Streaming**: `stream()` accepts `tools=`, and its final chunk carries the
  complete reply as `chunk.response`.
- **Structured output** accepts `RootModel` schemas (for example a bare list).
- `set_price()` to add or correct a model's price; `Context.add_response()` and
  `Context.text`; `map_parallel(..., return_exceptions=True)`; reusable steps
  with `@step` and `chain.add_step()`; `retry_on=` to choose which errors a step
  retries.

### Changed (may affect existing code)

- `retries` now means tries *after* the first failure. `retries=0` runs a step
  once (it used to never run it), and the default of 3 allows 4 attempts.
- Steps no longer retry errors a retry cannot fix: a bad API key or invalid
  request (HTTP 4xx other than 408/409/429), `StructuredOutputError` and
  `ToolLoopError`. Pass `retry_on=lambda error: True` for the old behavior.
- `temperature` is sent only when you pass it (it used to default to 0.7).
  Current Claude models and OpenAI reasoning models reject it.
- `Chain.run(input, context=ctx)` adds `input` to the context instead of
  ignoring it. `chain1 >> chain2` keeps the hooks of both chains.
- `Chain.on()` raises on an unknown event name instead of ignoring the hook.
- `Context.clone()` copies messages, so a step editing a message no longer
  changes the caller's copy.
- Provider classes are imported from `catena.providers`
  (`from catena.providers import OpenAIProvider`); the modules
  `catena.providers.openai` and `catena.providers.anthropic` are gone.
- Trace `usage` per step is what that step added, not the running total.
- The unused `cache=` option of `chain.step()` is removed.

### Fixed

- `providers.openai()` and `providers.anthropic()` failed when called a second
  time in one process.
- Sending tool results back to OpenAI or Anthropic produced requests the APIs
  reject; tool round trips now use each API's own format.
- `system=` only worked on the Anthropic provider; it now works on all three.
- The cache returned the same answer for requests that differed only in system
  prompt, schema or provider options, and ignored a per-call `ttl`.
- `MemoryCache` evicted the oldest entry rather than the least recently used.
- A failed attempt's messages leaked into the retry.
- Prices added to `PRICING` after import were ignored.

## 0.3.0

- Structured output (`complete_structured`), streaming, tool calling, and the
  OpenAI and Anthropic providers.
