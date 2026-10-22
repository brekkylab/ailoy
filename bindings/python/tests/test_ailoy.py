import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import virtx
from virtx import ConsoleBroken, ConsoleClient, Recipe

import ailoy
from ailoy import AgentBuilder, AiloyError


def content_text(content):
    """A chat-completions `content`, which is a string or a list of text parts, as text."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


# A chat-completions endpoint, streamed as vLLM serves one, that asks for the tool named in the
# first user message, and then answers with whatever that tool returned. Enough to drive a
# whole turn — model, tool, model — without a network or a key.
class FakeModel(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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

    # The `fake` org runs on vLLM, which reads the server from the environment.
    os.environ["VLLM_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
    ailoy.set_lang_model_provider("fake", "vllm")
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


def test_console_takes_only_a_virtx_console():
    with pytest.raises(TypeError, match="virtx.ConsoleClient"):
        AgentBuilder(MODEL).console(object())


def test_register_tool_hands_back_its_desc():
    desc = tool("noop", lambda: None)
    assert desc["name"] == "noop"
    assert desc["parameters"]["type"] == "object"


def test_register_tool_refuses_what_it_cannot_use():
    with pytest.raises(ValueError):
        ailoy.register_tool({"name": "x", "parameters": {}}, "not callable")
    with pytest.raises(ValueError):
        ailoy.register_tool({"description": "no name"}, lambda: None)
    with pytest.raises(ValueError):
        ailoy.set_tool_provider("x", "not callable")


def test_set_lang_model_provider_refuses_an_unknown_backend():
    with pytest.raises(ValueError):
        ailoy.set_lang_model_provider("x", "smoke-signals")


def test_thinking_effort_is_checked():
    with pytest.raises(ValueError):
        AgentBuilder(MODEL).thinking_effort("extreme")


async def test_building_spends_the_builder():
    builder = AgentBuilder(MODEL)
    await builder.build()
    with pytest.raises(ValueError):
        builder.instruction("again")


async def test_an_unserved_model_is_an_error():
    with pytest.raises(AiloyError):
        await AgentBuilder("nobody/serves-this").build()


async def test_an_unregistered_tool_is_an_error():
    missing = {"name": "missing", "parameters": {"type": "object"}}
    with pytest.raises(AiloyError):
        await AgentBuilder(MODEL).tool(missing).build()


async def test_a_removed_tool_provider_no_longer_builds():
    desc = {"name": "fleeting", "parameters": {"type": "object"}}
    ailoy.set_tool_provider("fleeting", lambda: None)
    await AgentBuilder(MODEL).tool(desc).build()
    assert ailoy.remove_tool_provider("fleeting")
    assert not ailoy.remove_tool_provider("fleeting")
    with pytest.raises(AiloyError):
        await AgentBuilder(MODEL).tool(desc).build()


async def test_a_browser_connects_only_when_used_and_needs_no_console():
    agent = await AgentBuilder(MODEL).browser(start_url="https://example.com").build()
    await agent.close()


def test_a_user_browser_is_a_name_or_a_directory(tmp_path):
    AgentBuilder(MODEL).browser(user_browser="edge")
    AgentBuilder(MODEL).browser(user_browser=tmp_path)
    with pytest.raises(ValueError, match="chrome-beta"):
        AgentBuilder(MODEL).browser(user_browser="edgee")


def test_a_browser_can_be_less_careful():
    AgentBuilder(MODEL).browser(careful=False)
    AgentBuilder(MODEL).browser(careful=True, user_browser="/nonexistent")


async def test_an_isolated_browser_has_options_of_its_own(tmp_path):
    agent = await (
        AgentBuilder(MODEL)
        .browser(isolated=True, headless=True, executable="/nonexistent/chrome", profile=tmp_path)
        .build()
    )
    await agent.close()
    with pytest.raises(AiloyError, match="Browser::isolated"):
        await AgentBuilder(MODEL).browser(headless=True).build()


async def test_an_agent_without_a_browser_cannot_start_one():
    agent = await AgentBuilder(MODEL).build()
    with pytest.raises(AiloyError, match="no browser"):
        await agent.start_browser()
    await agent.close()
    with pytest.raises(AiloyError, match="closed"):
        await agent.start_browser()


# ---- a turn, against the fake model ---------------------------------------------------


async def test_a_turn_without_tools():
    agent = await AgentBuilder(MODEL).instruction("Be brief.").build()
    outputs = await run(agent, "hi")
    assert texts(outputs) == ["hello"]
    assert outputs[-1]["finish_reason"] == {"type": "stop"}

    history = agent.history
    assert [m["role"] for m in history] == ["system", "user", "assistant"]


async def test_a_query_may_be_a_message():
    agent = await AgentBuilder(MODEL).build()
    query = {"role": "user", "contents": [{"type": "text", "text": "hi"}]}
    assert texts(await run(agent, query)) == ["hello"]


async def test_a_sync_tool_gets_keyword_arguments():
    seen = {}

    def weather(city):
        seen["city"] = city
        return {"city": city, "celsius": 21.5}

    desc = tool("weather", weather)
    agent = await AgentBuilder(MODEL).tool(desc).build()
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

    desc = tool("forecast", forecast)
    agent = await AgentBuilder(MODEL).tool(desc).build()
    outputs = await run(agent, 'forecast {"city": "Busan"}')
    assert outputs[1]["message"]["contents"][0]["value"] == "sunny in Busan"


async def test_a_tool_that_raises_answers_with_the_error():
    def broken(city):
        raise RuntimeError(f"no data for {city}")

    desc = tool("broken", broken)
    agent = await AgentBuilder(MODEL).tool(desc).build()
    outputs = await run(agent, 'broken {"city": "Jeju"}')
    value = outputs[1]["message"]["contents"][0]["value"]
    assert value.startswith("error: RuntimeError: no data for Jeju")


async def test_run_stream_yields_deltas():
    agent = await AgentBuilder(MODEL).build()
    deltas = [d async for d in agent.run_stream("hi")]
    text = "".join(
        part["text"] for d in deltas for part in d["delta"]["contents"] if part["type"] == "text"
    )
    assert text == "hello"
    assert any(d["finish_reason"] == {"type": "stop"} for d in deltas)
    assert agent.history[-1]["contents"] == [{"type": "text", "text": "hello"}]


async def test_history_is_refused_mid_turn_and_turns_take_turns():
    agent = await AgentBuilder(MODEL).build()
    first = agent.run("hi")
    await first.__anext__()
    with pytest.raises(AiloyError):
        agent.history
    await first.aclose()
    assert texts(await run(agent, "hi")) == ["hello"]


async def test_from_spec_and_closing():
    spec = {"model": MODEL, "instruction": "From a spec."}
    history = [{"role": "user", "contents": [{"type": "text", "text": "earlier"}]}]
    async with await ailoy.Agent.from_spec(spec, history=history) as agent:
        assert [m["role"] for m in agent.history] == ["system", "user"]
    with pytest.raises(AiloyError):
        agent.history
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
        agent = await AgentBuilder("mine/echo").build()
        assert texts(await run(agent, "hi")) == ["echo: hi"]
        deltas = [d async for d in agent.run_stream("again")]
        assert deltas[-1]["finish_reason"] == {"type": "stop"}
        assert agent.history[-1]["contents"] == [{"type": "text", "text": "echo: again"}]
        assert seen == [("mine", "echo")]
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
    agent = await AgentBuilder("alias/any").build()
    assert texts(await run(agent, "hi")) == ["hello"]
    ailoy.remove_lang_model_provider("alias")


async def test_a_factory_that_raises_fails_the_build():
    def factory(org, name):
        raise RuntimeError("no models today")

    ailoy.set_lang_model_provider("broken", factory)
    with pytest.raises(AiloyError, match="no models today"):
        await AgentBuilder("broken/x").build()
    ailoy.remove_lang_model_provider("broken")


def test_a_provider_is_a_backend_or_a_factory():
    with pytest.raises(ValueError):
        ailoy.set_lang_model_provider("x", 42)


async def test_an_unknown_org_cannot_make_a_lang_model():
    with pytest.raises(AiloyError):
        await ailoy.LangModel.create("nobody/serves-this")


# ---- against the console server virtx starts by default ----------------------------


async def test_the_agent_shares_the_console(tmp_path):
    console = await (
        ConsoleClient.builder()
        .image(Recipe("python:3.12-slim-trixie"))
        .mount(tmp_path, "/work")
        .network(False)
        .build()
    )
    agent = await (
        AgentBuilder(MODEL)
        .shell_tool()
        .console(console)
        .build()
    )
    outputs = await run(agent, 'shell {"cmd": "echo shared > /work/out.txt"}')
    assert outputs[1]["message"]["role"] == "tool"
    assert (tmp_path / "out.txt").read_text() == "shared\n"

    # The console is still the caller's to use, and closing it ends it for the agent too.
    await console.start()
    result = await console.exec(["cat", "/work/out.txt"])
    assert result.stdout == b"shared\n"
    await console.stop()

    # The session is the console's: an agent attached to it fails once it is closed.
    later = await (
        AgentBuilder(MODEL)
        .shell_tool()
        .console(console)
        .build()
    )
    await console.close()
    with pytest.raises(virtx.VirtxError):
        await console.exec(["true"])
    with pytest.raises(ConsoleBroken, match="closed"):
        await run(later, 'shell {"cmd": "true"}')
