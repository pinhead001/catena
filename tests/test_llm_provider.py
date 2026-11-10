"""Tests for the llm-library adapter, run against a fake model registered with llm itself."""

from typing import Any

import pytest
from pydantic import BaseModel, RootModel

from catena import Chain, Context, Message, StructuredOutputError, Tool, ToolCall, providers

llm = pytest.importorskip("llm")


class _ScriptedBase:
    """Plays back scripted steps and records every prompt llm hands the model."""

    can_stream = True
    supports_tools = True

    class Options(llm.Options):
        temperature: float | None = None
        max_output_tokens: int | None = None
        seed: int | None = None

    script: list[dict[str, Any]]
    prompts: list[Any]

    def _play(self, prompt: Any, response: Any) -> list[str]:
        self.prompts.append(prompt)
        step = self.script.pop(0)
        for call in step.get("tool_calls", []):
            response.add_tool_call(llm.ToolCall(**call))
        response.set_usage(input=step.get("input", 3), output=step.get("output", 2))
        return step.get("chunks", [])


class _SyncModel(_ScriptedBase, llm.Model):
    def execute(self, prompt, stream, response, conversation=None):
        yield from self._play(prompt, response)


class _AsyncModel(_ScriptedBase, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation=None):
        for chunk in self._play(prompt, response):
            yield chunk


def _make_model(model_id: str, supports_schema: bool) -> _AsyncModel:
    sync, async_ = _SyncModel(), _AsyncModel()
    for model in (sync, async_):
        model.model_id = model_id
        model.supports_schema = supports_schema
        model.script = []
        model.prompts = []
    return sync, async_


@pytest.fixture
def scripted():
    """Register fake models with llm's plugin manager; yields them by model id."""
    pairs = {
        "catena-fake": _make_model("catena-fake", supports_schema=True),
        "catena-fake-noschema": _make_model("catena-fake-noschema", supports_schema=False),
        # Named so Catena's pricing table recognises it.
        "catena-gpt-4o-mini": _make_model("catena-gpt-4o-mini", supports_schema=True),
    }

    class Plugin:
        @llm.hookimpl
        def register_models(self, register):
            for sync, async_ in pairs.values():
                register(sync, async_)

    llm.pm.register(Plugin(), name="catena-test-models")
    try:
        yield {model_id: async_ for model_id, (_, async_) in pairs.items()}
    finally:
        llm.pm.unregister(name="catena-test-models")


def _roles_and_text(prompt: Any) -> list[tuple[str, str]]:
    return [
        (m.role, "".join(getattr(p, "text", "") for p in m.parts)) for m in prompt.messages
    ]


def test_unknown_model_fails_at_construction(scripted):
    with pytest.raises(llm.UnknownModelError):
        providers.llm("catena-no-such-model")


@pytest.mark.asyncio
async def test_complete_sends_history_and_returns_text_and_usage(scripted):
    model = scripted["catena-gpt-4o-mini"]
    model.script = [{"chunks": ["Hel", "lo"], "input": 1_000_000, "output": 1_000_000}]

    provider = providers.llm("catena-gpt-4o-mini")
    response = await provider.complete(
        [
            Message.system("Be terse."),
            Message.user("hi"),
            Message.assistant("hello"),
            Message.user("how are you?"),
        ]
    )

    assert response.content == "Hello"
    assert response.model == "catena-gpt-4o-mini"
    assert response.usage.input_tokens == 1_000_000
    assert response.usage.cost_usd == 0.75  # priced as gpt-4o-mini

    prompt = model.prompts[0]
    assert _roles_and_text(prompt) == [
        ("system", "Be terse."),
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "how are you?"),
    ]
    # The fields older plugins read are filled in too.
    assert prompt.prompt == "how are you?"
    assert prompt.system == "Be terse."


@pytest.mark.asyncio
async def test_options_are_sent_only_when_given(scripted):
    model = scripted["catena-fake"]
    model.script = [{"chunks": ["a"]}, {"chunks": ["b"]}]
    provider = providers.llm("catena-fake")

    await provider.complete([Message.user("hi")])
    await provider.complete([Message.user("hi")], temperature=0.2, max_tokens=50, seed=7)

    assert model.prompts[0].options.temperature is None
    options = model.prompts[1].options
    assert options.temperature == 0.2
    assert options.max_output_tokens == 50  # this model has no max_tokens option
    assert options.seed == 7


@pytest.mark.asyncio
async def test_tool_call_round_trip(scripted):
    model = scripted["catena-fake"]
    model.script = [
        {"tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 2}, "tool_call_id": "c1"}]},
        {"chunks": ["3"]},
    ]
    add = Tool(
        name="add",
        description="Add two numbers",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    provider = providers.llm("catena-fake")

    messages = [Message.user("1+2?")]
    first = await provider.complete(messages, tools=[add])
    assert first.tool_calls == [ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})]
    assert model.prompts[0].tools[0].input_schema == add.parameters

    messages += [first.message, Message.tool("3", tool_call_id="c1")]
    second = await provider.complete(messages, tools=[add])
    assert second.content == "3"

    sent = model.prompts[1].messages
    assert [m.role for m in sent] == ["user", "assistant", "tool"]
    call_part = sent[1].parts[0]
    assert (call_part.name, call_part.arguments, call_part.tool_call_id) == (
        "add",
        {"a": 1, "b": 2},
        "c1",
    )
    result_part = sent[2].parts[0]
    assert (result_part.name, result_part.output, result_part.tool_call_id) == ("add", "3", "c1")
    assert model.prompts[1].tool_results[0].output == "3"


@pytest.mark.asyncio
async def test_tool_call_without_id_gets_one(scripted):
    model = scripted["catena-fake"]
    model.script = [{"tool_calls": [{"name": "add", "arguments": {}}]}]

    response = await providers.llm("catena-fake").complete([Message.user("x")])

    assert response.tool_calls[0].id


@pytest.mark.asyncio
async def test_tool_choice_becomes_system_instruction(scripted):
    model = scripted["catena-fake"]
    model.script = [{"chunks": ["ok"]}]

    await providers.llm("catena-fake").complete(
        [Message.system("Be terse."), Message.user("hi")], tool_choice="add"
    )

    expected = "Be terse.\n\nRespond by calling the add tool."
    assert model.prompts[0].system == expected
    assert _roles_and_text(model.prompts[0])[0] == ("system", expected)


@pytest.mark.asyncio
async def test_stream_yields_deltas_then_usage(scripted):
    model = scripted["catena-fake"]
    model.script = [{"chunks": ["Hel", "lo"], "input": 4, "output": 2}]

    chunks = [c async for c in providers.llm("catena-fake").stream([Message.user("hi")])]

    assert [c.delta for c in chunks[:-1]] == ["Hel", "lo"]
    assert chunks[-1].done
    assert chunks[-1].usage.input_tokens == 4


class Invoice(BaseModel):
    company: str
    amount: float


@pytest.mark.asyncio
async def test_structured_uses_native_schema_and_retries(scripted):
    model = scripted["catena-fake"]
    model.script = [
        {"chunks": ['{"company": "Acme"}'], "input": 10, "output": 5},
        {"chunks": ['{"company": "Acme", "amount": 12.5}'], "input": 20, "output": 5},
    ]

    result = await providers.llm("catena-fake").complete_structured(
        [Message.user("extract")], schema=Invoice
    )

    assert result.output == Invoice(company="Acme", amount=12.5)
    assert result.attempts == 2
    assert result.usage.input_tokens == 30
    assert model.prompts[0].schema == Invoice.model_json_schema()
    assert model.prompts[0].tools == []
    retry = _roles_and_text(model.prompts[1])
    assert retry[1] == ("assistant", '{"company": "Acme"}')
    assert "failed validation" in retry[2][1]


@pytest.mark.asyncio
async def test_structured_returns_a_root_model(scripted):
    class Tags(RootModel[list[str]]):
        pass

    model = scripted["catena-fake"]
    model.script = [{"chunks": ['{"value": ["a", "b"]}']}]

    result = await providers.llm("catena-fake").complete_structured(
        [Message.user("tag this")], schema=Tags
    )

    assert result.output.root == ["a", "b"]
    assert model.prompts[0].schema["properties"]["value"] == {
        "items": {"type": "string"},
        "title": "Tags",
        "type": "array",
    }


@pytest.mark.asyncio
async def test_structured_raises_with_usage_when_retries_run_out(scripted):
    model = scripted["catena-fake"]
    model.script = [{"chunks": ["not json"], "input": 10, "output": 1} for _ in range(2)]

    with pytest.raises(StructuredOutputError) as exc_info:
        await providers.llm("catena-fake").complete_structured(
            [Message.user("extract")], schema=Invoice, retries=1
        )

    assert exc_info.value.raw_output == "not json"
    assert exc_info.value.usage.input_tokens == 20


@pytest.mark.asyncio
async def test_structured_falls_back_to_tool_when_model_has_no_schema_support(scripted):
    model = scripted["catena-fake-noschema"]
    model.script = [
        {"tool_calls": [{"name": "Invoice", "arguments": {"company": "Acme", "amount": 3}}]}
    ]

    result = await providers.llm("catena-fake-noschema").complete_structured(
        [Message.user("extract")], schema=Invoice
    )

    assert result.output.amount == 3
    prompt = model.prompts[0]
    assert prompt.schema is None
    assert prompt.tools[0].name == "Invoice"
    assert "Respond by calling the Invoice tool." in prompt.system


@pytest.mark.asyncio
async def test_works_inside_a_chain(scripted):
    scripted["catena-fake"].script = [{"chunks": ["draft"]}, {"chunks": ["final"]}]
    provider = providers.llm("catena-fake")
    chain = Chain()

    @chain.step()
    async def draft(ctx: Context) -> Context:
        response = await provider.complete(ctx.messages)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        return ctx

    @chain.step()
    async def revise(ctx: Context) -> Context:
        ctx.add_message(Message.user("Shorter."))
        response = await provider.complete(ctx.messages)
        ctx.add_message(response.message)
        ctx.add_usage(response.usage)
        return ctx

    result = await chain.run("Write a line.")

    assert result.messages[-1].content == "final"
    assert result.usage.input_tokens == 6


@pytest.mark.asyncio
async def test_stream_carries_tools_and_ends_with_the_complete_response(scripted):
    model = scripted["catena-fake"]
    model.script = [
        {
            "chunks": ["Let me ", "check."],
            "tool_calls": [{"name": "add", "arguments": {"a": 2, "b": 40}}],
        }
    ]
    add = Tool(name="add", description="Add.", parameters={"type": "object", "properties": {}})

    provider = providers.llm("catena-fake")
    chunks = [c async for c in provider.stream([Message.user("2 + 40?")], tools=[add])]

    assert [c.delta for c in chunks[:-1]] == ["Let me ", "check."]
    final = chunks[-1].response
    assert final.content == "Let me check."
    assert [(c.name, c.arguments) for c in final.tool_calls] == [("add", {"a": 2, "b": 40})]
    assert final.usage == chunks[-1].usage
    assert [t.name for t in model.prompts[0].tools] == ["add"]
