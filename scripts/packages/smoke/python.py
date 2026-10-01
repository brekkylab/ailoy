"""The Python package as a user installs it, on the platform this runs on.

What this platform can do is said in the environment, as for ``node.mjs``:
SMOKE_MOUNT, SMOKE_SERVER, SMOKE_VM. A turn's model is served from this process, the
same one ``fake.mjs`` serves: no network, and no API key.
"""
import asyncio, json, os, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ailoy
from ailoy import virtx

want = lambda name: os.environ.get(name) == "1"
failures = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failures.append(what)


def text(content):
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content or [])


class FakeModel(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body["messages"]
        if messages[-1]["role"] == "tool":
            message = {"role": "assistant", "content": f"the tool said {text(messages[-1]['content'])}"}
        elif body.get("tools"):
            user = next(m for m in messages if m["role"] == "user")
            name, args = text(user["content"]).split(" ", 1)
            call = {"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
        else:
            message = {"role": "assistant", "content": "hello"}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        payload = json.dumps(
            {
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
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


async def run(agent, query):
    return [output async for output in agent.run(query)]


async def main():
    check(hasattr(ailoy, "AgentBuilder"), f"the package loads from {os.path.dirname(ailoy.__file__)}")
    check("echo hi" in repr(virtx.Recipe("alpine:latest").step("echo hi")), "virtx comes built in")

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ailoy.register_lang_model("fake/*", "chat_completion", f"http://127.0.0.1:{server.server_port}/v1/chat/completions")

    # A whole turn -- model, tool, model -- through the extension's bridge to asyncio both ways.
    calls = []

    async def add(a, b):
        calls.append([a, b])
        return a + b

    desc = ailoy.register_tool(
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
        },
        add,
    )
    agent = await ailoy.AgentBuilder("fake/model").tool(desc).build()
    outputs = await run(agent, 'add {"a": 2, "b": 3}')
    roles = [o["message"]["role"] for o in outputs]
    print(f"  turn: {' -> '.join(roles)}")
    check(calls == [[2, 3]], "the model called a Python tool")
    check(roles == ["assistant", "tool", "assistant"], "a turn goes model, tool, model")
    check("5" in json.dumps(outputs[-1]["message"]["contents"]), "the model answered with what the tool said")
    await agent.close()

    # Whether or not this host can mount: without a provider, a mount is an error, not a crash.
    point = tempfile.mkdtemp(prefix="ailoy-smoke-")
    try:
        m = virtx.HostMount(virtx.Directory().with_file("a.txt", "hi"), point)
    except Exception as e:
        m = None
        print(f"  HostMount: {e}")
    check((m is not None) == want("SMOKE_MOUNT"), f"a HostMount is {'' if m else 'not '}made")
    if m is not None:
        check(open(os.path.join(point, "a.txt")).read() == "hi", "HostMount serves its tree")
        del m
        check(not os.path.exists(os.path.join(point, "a.txt")), "dropping the HostMount takes it down")

    if want("SMOKE_SERVER"):
        # The server runs on this machine, and answers -- no VM needed to ask its version.
        images = await virtx.ImageClient.try_new()
        version = await images.version()
        await images.close()
        check(bool(version), f"the console server answers (protocol {version})")

    if want("SMOKE_VM"):
        # An agent's shell tool, in a VM session that sees a host directory.
        host = tempfile.mkdtemp(prefix="ailoy-smoke-host-")
        open(os.path.join(host, "from-host.txt"), "w").write("by path")
        console = await virtx.ConsoleClient.builder().image(virtx.Recipe("alpine:latest")).mount(host, "/host").build()
        agent = await ailoy.AgentBuilder("fake/model").shell_tool().console(console).build()
        cmd = "uname -m; cat /host/from-host.txt; echo written > /host/from-vm.txt"
        outputs = await run(agent, "shell " + json.dumps({"cmd": cmd}))
        said = json.dumps(outputs[1]["message"]["contents"]) if len(outputs) > 1 else ""
        print(f"  vm: {said}")
        check("by path" in said, "the agent's shell tool reads the session's mount")
        check(open(os.path.join(host, "from-vm.txt")).read().strip() == "written", "the host sees the VM's write")
        await agent.close()
        await console.close()

    server.shutdown()
    print("FAILED: " + "; ".join(failures) if failures else "ALL PASS", flush=True)
    raise SystemExit(1 if failures else 0)


asyncio.run(main())
