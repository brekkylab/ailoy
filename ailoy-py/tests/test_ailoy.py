import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from virtx import Client

import ailoy
from ailoy import Agent, AiloyError, LangModel


def content_text(content):
    """A chat-completions `content`, which is a string or a list of text parts, as text."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


# A chat-completions endpoint, streamed as vLLM serves one, that asks for the tool named in the
# first user message, and then answers with whatever that tool returned. Enough to drive a
# whole turn — model, tool, model — without a network or a key.
class FakeModel(BaseHTTPRequestHandler):
    # Every request's model id and `Authorization` header, as sent.
    seen = []
    # The ids `GET /v1/models` lists.
    listed = []

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": id} for id in self.listed]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append((body["model"], self.headers.get("Authorization")))
        messages = body["messages"]
        tool_results = [m for m in messages if m["role"] == "tool"]
        if tool_results:
            text = f"tool said {content_text(tool_results[-1]['content'])}"
            chunks = [{"role": "assistant", "content": text}]
            finish = "stop"
        elif body.get("tools"):
            user = next(m for m in messages if m["role"] == "user")
            name, args = content_text(user["content"]).split(" ", 1)
            call = {"index": 0, "id": "call_1", "type": "function"}
            chunks = [{"role": "assistant", "tool_calls": [{**call, "function": {"name": name, "arguments": args}}]}]
            finish = "tool_calls"
        else:
            chunks = [{"role": "assistant", "content": "hel"}, {"content": "lo"}]
            finish = "stop"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [{"choices": [{"index": 0, "delta": delta}]} for delta in chunks]
        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        events.append({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


MODEL = "fake/model"


@pytest.fixture(scope="module", autouse=True)
def fake_model():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # `fake` runs here, on the backend that reads its server from the environment.
    os.environ["SELF_HOSTED_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
    ailoy.set_lang_model_provider("fake", "self-hosted")
    yield
    ailoy.remove_lang_model_provider("fake")
    server.shutdown()


def tool(name, func):
    return ailoy.register_tool(
        {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
        func,
    )


def texts(outputs):
    return [
        part["text"]
        for output in outputs
        for part in output["message"]["contents"]
        if part["type"] == "text"
    ]


async def run(agent, query):
    return [output async for output in agent.run(query)]


# ---- offline --------------------------------------------------------------------------


def test_sandbox_takes_only_a_virtx_sandbox():
    with pytest.raises(TypeError, match="virtx.Sandbox"):
        Agent.from_spec({"model": MODEL, "use_sandbox": True}, sandbox=object())


def test_register_tool_hands_back_its_name():
    assert tool("noop", lambda: None) == "noop"


def test_register_tool_refuses_what_it_cannot_use():
    with pytest.raises(ValueError):
        ailoy.register_tool({"name": "x", "parameters": {}}, "not callable")
    with pytest.raises(ValueError):
        ailoy.register_tool({"description": "no name"}, lambda: None)


def test_web_search_provider_is_set_by_name():
    assert ailoy.get_web_search_provider() == "web"
    try:
        ailoy.set_web_search_provider("naver")
        assert ailoy.get_web_search_provider() == "naver"
        with pytest.raises(ValueError):
            ailoy.set_web_search_provider("bing")
    finally:
        ailoy.set_web_search_provider("web")


def test_set_lang_model_provider_refuses_an_unknown_backend():
    with pytest.raises(ValueError):
        ailoy.set_lang_model_provider("x", "smoke-signals")


def test_a_server_of_your_own_runs_the_self_hosted_backend():
    base_url = os.environ["SELF_HOSTED_BASE_URL"]
    with pytest.raises(ValueError, match="self-hosted"):
        ailoy.set_lang_model_provider("x", {"backend": "cli", "base_url": base_url})
    with pytest.raises(ValueError, match="base_url"):
        ailoy.set_lang_model_provider("x", {"backend": "self-hosted"})
    with pytest.raises(ValueError, match="baseUrl"):
        ailoy.set_lang_model_provider("x", {"backend": "self-hosted", "baseUrl": base_url})
    with pytest.raises(ValueError, match="base_url"):
        ailoy.set_lang_model_provider("x", {"backend": "self-hosted", "base_url": ""})
    with pytest.raises(ValueError):
        ailoy.set_lang_model_provider("x", {"backend": "self-hosted", "base_url": base_url, "models": ["a"]})
    assert ailoy.get_lang_model_provider("x") is None


def test_thinking_effort_is_checked():
    with pytest.raises(ValueError):
        Agent.from_spec({"model": MODEL, "model_options": {"thinking_effort": "extreme"}})


async def test_an_unserved_model_is_an_error():
    with pytest.raises(AiloyError):
        await Agent.from_spec({"model": "nobody/serves-this"})


async def test_an_unregistered_tool_is_an_error():
    with pytest.raises(AiloyError, match="missing"):
        await Agent.from_spec({"model": MODEL, "tools": ["missing"]})


async def test_a_removed_tool_provider_no_longer_builds():
    name = tool("fleeting", lambda: None)
    await Agent.from_spec({"model": MODEL, "tools": [name]})
    assert ailoy.remove_tool_provider(name)
    assert not ailoy.remove_tool_provider(name)
    with pytest.raises(AiloyError):
        await Agent.from_spec({"model": MODEL, "tools": [name]})


async def test_a_browser_needs_a_sandbox_with_its_abin():
    agent = await Agent.from_spec({"model": MODEL, "use_browser": True})
    with pytest.raises(AiloyError, match="`browser` abin"):
        await run(agent, "hi")
    await agent.close()


# ---- a turn, against the fake model ---------------------------------------------------


async def test_a_turn_without_tools():
    agent = await Agent.from_spec({"model": MODEL, "instruction": "Be brief."})
    outputs = await run(agent, "hi")
    assert texts(outputs) == ["hello"]
    assert outputs[-1]["finish_reason"] == {"type": "stop"}

    history = await agent.history()
    assert [m["role"] for m in history] == ["system", "user", "assistant"]


async def test_a_server_of_your_own_keeps_the_desc():
    base_url = os.environ["SELF_HOSTED_BASE_URL"]
    server = {"backend": "self-hosted", "base_url": base_url}
    ailoy.set_lang_model_provider("meta-llama", {**server, "api_key": "k", "models": {"Llama-3.3-70B-Instruct": "llama3.3:70b"}})
    ailoy.set_lang_model_provider("Qwen", server)
    try:
        assert ailoy.get_lang_model_provider("meta-llama/Llama-3.3-70B-Instruct") == "self-hosted"
        FakeModel.seen.clear()
        FakeModel.listed = ["qwen/qwen3-8b", "qwen3.5-27b"]
        messages = [{"role": "user", "contents": [{"type": "text", "text": "hi"}]}]
        output = await (await LangModel.create("meta-llama/Llama-3.3-70B-Instruct")).infer(messages)
        assert texts([output]) == ["hello"]
        for desc in ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-27B"]:
            await (await LangModel.create(desc)).infer(messages)
        # The server is sent its own id for the model: mapped by `models`, listed by the server
        # for the desc or the name alone, or the desc as it is.
        assert FakeModel.seen == [
            ("llama3.3:70b", "Bearer k"),
            ("meta-llama/Llama-3.1-8B-Instruct", "Bearer k"),
            ("qwen/qwen3-8b", None),
            ("qwen3.5-27b", None),
        ]
    finally:
        FakeModel.listed = []
        ailoy.remove_lang_model_provider("meta-llama")
        ailoy.set_lang_model_provider("Qwen", "api")


async def test_the_self_hosted_backend_reads_its_server_from_the_environment():
    base_url = os.environ["SELF_HOSTED_BASE_URL"]
    ailoy.set_lang_model_provider("env-served", "self-hosted")
    messages = [{"role": "user", "contents": [{"type": "text", "text": "hi"}]}]
    try:
        del os.environ["SELF_HOSTED_BASE_URL"]
        with pytest.raises(AiloyError, match="SELF_HOSTED_BASE_URL"):
            await LangModel.create("env-served/model")

        os.environ["SELF_HOSTED_BASE_URL"] = base_url
        FakeModel.seen.clear()
        await (await LangModel.create("env-served/model")).infer(messages)
        assert FakeModel.seen == [("env-served/model", None)]
    finally:
        os.environ["SELF_HOSTED_BASE_URL"] = base_url
        ailoy.remove_lang_model_provider("env-served")


async def test_a_query_may_be_a_message():
    agent = await Agent.from_spec({"model": MODEL})
    query = {"role": "user", "contents": [{"type": "text", "text": "hi"}]}
    assert texts(await run(agent, query)) == ["hello"]


async def test_a_sync_tool_gets_keyword_arguments():
    seen = {}

    def weather(city):
        seen["city"] = city
        return {"city": city, "celsius": 21.5}

    name = tool("weather", weather)
    agent = await Agent.from_spec({"model": MODEL, "tools": [name]})
    outputs = await run(agent, 'weather {"city": "Seoul"}')

    assert seen == {"city": "Seoul"}
    roles = [o["message"]["role"] for o in outputs]
    assert roles == ["assistant", "tool", "assistant"]
    assert outputs[1]["message"]["contents"][0]["value"] == {"city": "Seoul", "celsius": 21.5}
    assert "21.5" in texts(outputs)[-1]


async def test_an_async_tool_is_awaited_on_the_running_loop():
    loop = asyncio.get_running_loop()

    async def forecast(city):
        assert asyncio.get_running_loop() is loop
        await asyncio.sleep(0)
        return f"sunny in {city}"

    name = tool("forecast", forecast)
    agent = await Agent.from_spec({"model": MODEL, "tools": [name]})
    outputs = await run(agent, 'forecast {"city": "Busan"}')
    assert outputs[1]["message"]["contents"][0]["value"] == "sunny in Busan"


async def test_a_tool_that_raises_answers_with_the_error():
    def broken(city):
        raise RuntimeError(f"no data for {city}")

    name = tool("broken", broken)
    agent = await Agent.from_spec({"model": MODEL, "tools": [name]})
    outputs = await run(agent, 'broken {"city": "Jeju"}')
    value = outputs[1]["message"]["contents"][0]["value"]
    assert value.startswith("error: RuntimeError: no data for Jeju")


async def test_run_stream_yields_deltas():
    agent = await Agent.from_spec({"model": MODEL})
    deltas = [d async for d in agent.run_stream("hi")]
    text = "".join(
        part["text"] for d in deltas for part in d["delta"]["contents"] if part["type"] == "text"
    )
    assert text == "hello"
    assert any(d["finish_reason"] == {"type": "stop"} for d in deltas)
    assert (await agent.history())[-1]["contents"] == [{"type": "text", "text": "hello"}]


async def test_a_turn_left_early_leaves_the_agent_free():
    agent = await Agent.from_spec({"model": MODEL})
    first = agent.run("hi")
    await first.__anext__()
    await first.aclose()
    assert texts(await run(agent, "hi")) == ["hello"]


async def test_from_spec_and_closing():
    spec = {"model": MODEL, "instruction": "From a spec."}
    history = [{"role": "user", "contents": [{"type": "text", "text": "earlier"}]}]
    async with await ailoy.Agent.from_spec(spec, history=history) as agent:
        # The system message goes first as the first turn begins.
        assert [m["role"] for m in await agent.history()] == ["user"]
        await run(agent, "hi")
        roles = [m["role"] for m in await agent.history()]
        assert roles == ["system", "user", "user", "assistant"]
    with pytest.raises(AiloyError):
        await agent.history()
    with pytest.raises(AiloyError):
        await run(agent, "hi")


async def test_a_lang_model_runs_without_an_agent():
    model = await ailoy.LangModel.create(MODEL)
    messages = [{"role": "user", "contents": [{"type": "text", "text": "hi"}]}]

    output = await model.infer(messages)
    assert output["message"]["contents"] == [{"type": "text", "text": "hello"}]
    assert output["finish_reason"] == {"type": "stop"}

    deltas = [d async for d in model.infer_stream(messages)]
    text = "".join(
        part["text"] for d in deltas for part in d["delta"]["contents"] if part["type"] == "text"
    )
    assert text == "hello"


class Echo:
    """A model of the test's own, answering with the last user text."""

    def __init__(self, name):
        self.name = name

    def infer(self, messages, tools, options):
        text = messages[-1]["contents"][0]["text"]
        return {
            "message": {"role": "assistant", "contents": [{"type": "text", "text": f"{self.name}: {text}"}]},
            "finish_reason": {"type": "stop"},
        }


class StreamingEcho(Echo):
    async def infer_stream(self, messages, tools, options):
        for piece in [self.name, ": ", messages[-1]["contents"][0]["text"]]:
            yield {"delta": {"role": "assistant", "contents": [{"type": "text", "text": piece}], "tool_calls": []}}
        yield {"delta": {"contents": [], "tool_calls": []}, "finish_reason": {"type": "stop"}}


async def test_a_custom_provider_serves_an_agent():
    seen = []

    def factory(org, name):
        seen.append((org, name))
        return Echo(name)

    ailoy.set_lang_model_provider("mine", factory)
    try:
        agent = await Agent.from_spec({"model": "mine/echo"})
        assert texts(await run(agent, "hi")) == ["echo: hi"]
        deltas = [d async for d in agent.run_stream("again")]
        assert deltas[-1]["finish_reason"] == {"type": "stop"}
        assert (await agent.history())[-1]["contents"] == [{"type": "text", "text": "echo: again"}]
        # Made for each turn.
        assert seen == [("mine", "echo"), ("mine", "echo")]
    finally:
        assert ailoy.remove_lang_model_provider("mine")


def test_a_provider_reads_back_as_it_was_set():
    assert ailoy.get_lang_model_provider("anthropic") == "cli"
    assert ailoy.get_lang_model_provider("anthropic/claude-sonnet-5-5") == "cli"
    ailoy.set_lang_model_provider("read-back", "bedrock")
    assert ailoy.get_lang_model_provider("read-back/some-model") == "bedrock"
    ailoy.set_lang_model_provider("read-back", lambda org, name: Echo(name))
    assert ailoy.get_lang_model_provider("read-back") == "custom"
    ailoy.remove_lang_model_provider("read-back")
    assert ailoy.get_lang_model_provider("read-back") is None


async def test_a_custom_provider_may_stream_and_be_async():
    async def ainfer(self, messages, tools, options):
        await asyncio.sleep(0)
        return Echo.infer(self, messages, tools, options)

    class Async(StreamingEcho):
        infer = ainfer

    ailoy.set_lang_model_provider("mine-async", lambda org, name: Async(name or org))
    model = await ailoy.LangModel.create("mine-async")
    messages = [{"role": "user", "contents": [{"type": "text", "text": "hi"}]}]
    assert (await model.infer(messages))["message"]["contents"][0]["text"] == "mine-async: hi"
    deltas = [d async for d in model.infer_stream(messages)]
    assert len(deltas) == 4
    ailoy.remove_lang_model_provider("mine-async")


async def test_a_custom_provider_may_hand_back_a_built_in_model():
    built = await ailoy.LangModel.create(MODEL)
    ailoy.set_lang_model_provider("alias", lambda org, name: built)
    agent = await Agent.from_spec({"model": "alias/any"})
    assert texts(await run(agent, "hi")) == ["hello"]
    ailoy.remove_lang_model_provider("alias")


async def test_a_factory_that_raises_fails_the_turn():
    def factory(org, name):
        raise RuntimeError("no models today")

    ailoy.set_lang_model_provider("broken", factory)
    agent = await Agent.from_spec({"model": "broken/x"})
    with pytest.raises(AiloyError, match="no models today"):
        await run(agent, "hi")
    await agent.close()
    ailoy.remove_lang_model_provider("broken")


def test_a_provider_is_a_backend_or_a_factory():
    with pytest.raises(ValueError):
        ailoy.set_lang_model_provider("x", 42)


async def test_an_unknown_org_cannot_make_a_lang_model():
    with pytest.raises(AiloyError):
        await ailoy.LangModel.create("nobody/serves-this")


# ---- against the server virtx starts by default -----------------------------------------


async def test_the_agent_runs_in_the_callers_sandbox():
    with Client.spawn() as client:
        await client.pull_image("python:3.12-slim").complete()
        sandbox = await client.create_sandbox("python:3.12-slim", network=False, memory_mib=1024)
        agent = await Agent.from_spec({"model": MODEL, "use_sandbox": True}, sandbox=sandbox)
        outputs = await run(agent, 'shell {"cmd": "echo shared > /root/out.txt"}')
        assert outputs[1]["message"]["role"] == "tool"

        # The sandbox is still the caller's to use, and removing it removes it for an agent too.
        assert await sandbox.read_text("root/out.txt") == "shared\n"
        later = await Agent.from_spec({"model": MODEL, "use_sandbox": True}, sandbox=sandbox)
        await sandbox.remove()
        with pytest.raises(AiloyError, match="404"):
            await run(later, 'shell {"cmd": "true"}')
