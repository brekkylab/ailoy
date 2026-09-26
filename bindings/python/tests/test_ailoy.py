import asyncio
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import ailoy
from ailoy import AgentBuilder, AiloyError
from ailoy.cortex import ConsoleClient, CortexError, ErrorCode, NetworkAccess, Recipe


def content_text(content):
    """A chat-completions `content`, which is a string or a list of text parts, as text."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


# A chat-completions endpoint that asks for the tool named in the first user message, and
# then answers with whatever that tool returned. Enough to drive a whole turn — model, tool,
# model — without a network or a key.
class FakeModel(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body["messages"]
        tool_results = [m for m in messages if m["role"] == "tool"]
        if tool_results:
            reply = {"role": "assistant", "content": f"tool said {content_text(tool_results[-1]['content'])}"}
            finish = "stop"
        elif body.get("tools"):
            user = next(m for m in messages if m["role"] == "user")
            name, args = content_text(user["content"]).split(" ", 1)
            reply = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            reply = {"role": "assistant", "content": "hello"}
            finish = "stop"

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hel"}}]},
                {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return

        payload = json.dumps(
            {
                "choices": [{"index": 0, "message": reply, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


PROVIDER = "fake"
MODEL = "fake/model"


@pytest.fixture(scope="module", autouse=True)
def fake_model():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    ailoy.add_lang_model_provider(PROVIDER)
    ailoy.register_lang_model(
        "fake/*",
        "chat_completion",
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        "dummy",
        provider=PROVIDER,
    )
    ailoy.add_tool_provider(PROVIDER)
    ailoy.add_agent_provider(PROVIDER, lang_model_provider=PROVIDER, tool_provider=PROVIDER)
    yield
    server.shutdown()


def tool(name, func):
    return ailoy.register_tool(
        {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
        func,
        provider=PROVIDER,
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


def test_cortex_is_built_in():
    assert ErrorCode.TIMED_OUT == -32000
    assert "python:3.12-slim" in repr(Recipe("python:3.12-slim"))


def test_register_tool_hands_back_its_desc():
    desc = tool("noop", lambda: None)
    assert desc["name"] == "noop"
    assert desc["parameters"]["type"] == "object"


def test_register_tool_refuses_what_it_cannot_use():
    with pytest.raises(ValueError):
        ailoy.register_tool({"name": "x", "parameters": {}}, "not callable", provider=PROVIDER)
    with pytest.raises(ValueError):
        ailoy.register_tool({"description": "no name"}, lambda: None, provider=PROVIDER)
    with pytest.raises(AiloyError):
        ailoy.register_tool({"name": "x", "parameters": {}}, lambda: None, provider="missing")


def test_register_lang_model_refuses_an_unknown_schema():
    with pytest.raises(ValueError):
        ailoy.register_lang_model("x/*", "smoke-signals", "http://localhost", provider=PROVIDER)


def test_response_format_is_checked():
    with pytest.raises(ValueError):
        AgentBuilder(MODEL).response_format({"type": 123})


async def test_building_spends_the_builder():
    builder = AgentBuilder(MODEL).agent_provider(PROVIDER)
    await builder.build()
    with pytest.raises(ValueError):
        builder.instruction("again")


async def test_an_unserved_model_is_an_error():
    with pytest.raises(AiloyError):
        await AgentBuilder("nobody/serves-this").agent_provider(PROVIDER).build()


async def test_an_unregistered_tool_is_an_error():
    missing = {"name": "missing", "parameters": {"type": "object"}}
    with pytest.raises(AiloyError):
        await AgentBuilder(MODEL).agent_provider(PROVIDER).tool(missing).build()


# ---- a turn, against the fake model ---------------------------------------------------


async def test_a_turn_without_tools():
    agent = await AgentBuilder(MODEL).agent_provider(PROVIDER).instruction("Be brief.").build()
    outputs = await run(agent, "hi")
    assert texts(outputs) == ["hello"]
    assert outputs[-1]["finish_reason"] == {"type": "stop"}

    history = agent.history
    assert [m["role"] for m in history] == ["system", "user", "assistant"]


async def test_a_query_may_be_a_message():
    agent = await AgentBuilder(MODEL).agent_provider(PROVIDER).build()
    query = {"role": "user", "contents": [{"type": "text", "text": "hi"}]}
    assert texts(await run(agent, query)) == ["hello"]


async def test_a_sync_tool_gets_keyword_arguments():
    seen = {}

    def weather(city):
        seen["city"] = city
        return {"city": city, "celsius": 21.5}

    desc = tool("weather", weather)
    agent = await AgentBuilder(MODEL).agent_provider(PROVIDER).tool(desc).build()
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
    agent = await AgentBuilder(MODEL).agent_provider(PROVIDER).tool(desc).build()
    outputs = await run(agent, 'forecast {"city": "Busan"}')
    assert outputs[1]["message"]["contents"][0]["value"] == "sunny in Busan"


async def test_a_tool_that_raises_answers_with_the_error():
    def broken(city):
        raise RuntimeError(f"no data for {city}")

    desc = tool("broken", broken)
    agent = await AgentBuilder(MODEL).agent_provider(PROVIDER).tool(desc).build()
    outputs = await run(agent, 'broken {"city": "Jeju"}')
    value = outputs[1]["message"]["contents"][0]["value"]
    assert value.startswith("error: RuntimeError: no data for Jeju")


async def test_run_stream_yields_deltas():
    agent = await AgentBuilder(MODEL).agent_provider(PROVIDER).build()
    deltas = [d async for d in agent.run_stream("hi")]
    text = "".join(
        part["text"] for d in deltas for part in d["delta"]["contents"] if part["type"] == "text"
    )
    assert text == "hello"
    assert deltas[-1]["finish_reason"] == {"type": "stop"}
    assert agent.history[-1]["contents"] == [{"type": "text", "text": "hello"}]


async def test_history_is_refused_mid_turn_and_turns_take_turns():
    agent = await AgentBuilder(MODEL).agent_provider(PROVIDER).build()
    first = agent.run("hi")
    await first.__anext__()
    with pytest.raises(AiloyError):
        agent.history
    await first.aclose()
    assert texts(await run(agent, "hi")) == ["hello"]


async def test_from_spec_and_closing():
    spec = {"model": MODEL, "instruction": "From a spec."}
    history = [{"role": "user", "contents": [{"type": "text", "text": "earlier"}]}]
    async with await ailoy.Agent.from_spec(
        spec, agent_provider=PROVIDER, history=history
    ) as agent:
        assert [m["role"] for m in agent.history] == ["system", "user"]
    with pytest.raises(AiloyError):
        agent.history
    with pytest.raises(AiloyError):
        await run(agent, "hi")


# ---- against a real console server, named by `$AILOY_CORTEX_CONSOLE` ------------------

SERVER = os.environ.get("AILOY_CORTEX_CONSOLE")


@pytest.mark.skipif(
    not SERVER or not shutil.which(SERVER), reason="set $AILOY_CORTEX_CONSOLE"
)
async def test_the_agent_shares_the_console(tmp_path):
    console = await (
        ConsoleClient.builder()
        .cmd([SERVER])
        .image(Recipe("python:3.12-slim-trixie"))
        .mount(tmp_path, "/work")
        .network(NetworkAccess.none())
        .build()
    )
    agent = await (
        AgentBuilder(MODEL)
        .agent_provider(PROVIDER)
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
    await console.close()
    with pytest.raises(CortexError):
        await console.exec(["true"])
