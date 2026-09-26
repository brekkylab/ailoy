# ailoy for Python

Python bindings for ailoy: the same `AgentBuilder`, `Agent` and provider registries as the
Rust crate, with every call that waits as an awaitable and a turn as an `async for`. cortex
comes built in, as `ailoy.cortex`, for the console an agent's tools run in.

```python
import asyncio

from ailoy import AgentBuilder
from ailoy.cortex import ConsoleClient, NetworkAccess, Recipe


async def main() -> None:
    console = await (
        ConsoleClient.builder()
        .image(Recipe("python:3.12-slim-trixie"))
        .mount("./artifacts", "/artifacts")
        .network(NetworkAccess.none())
        .build()
    )

    agent = await (
        AgentBuilder("anthropic/claude-sonnet-5")
        .instruction("Write what you are asked for into /artifacts.")
        .system_tools()
        .console(console)
        .build()
    )

    async for output in agent.run("Write a haiku about Rust into /artifacts/haiku.txt"):
        message = output["message"]
        for part in message["contents"]:
            if part["type"] == "text":
                print(part["text"])


asyncio.run(main())
```

The model's provider is found by its prefix, from the API keys in the environment —
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`,
`KIMI_API_KEY`, `OPENROUTER_API_KEY` (as `openrouter/<vendor>/<model>`),
`AWS_BEARER_TOKEN_BEDROCK` — read the first time an agent is built.

## Data is dicts

Messages, specs, tool descriptions and what a turn yields are the dicts their JSON form is,
as ailoy writes it: a message is `{"role": "user", "contents": [{"type": "text", "text":
"..."}]}`, and `agent.run` yields `{"message": ..., "finish_reason": ..., "usage": ...}`.
`ailoy.types` has a `TypedDict` for each. A query may also be a plain string, as the user's
text.

`agent.run_stream(query)` is the same turn yielding deltas as the model writes them.

## Tools written in Python

```python
import ailoy

def temperature(city: str) -> float:
    return 21.5

desc = ailoy.register_tool(
    {
        "name": "temperature",
        "description": "The temperature in a city now, in Celsius.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    temperature,
)
agent = await ailoy.AgentBuilder("openai/gpt-5.4-mini").tool(desc).build()
```

The model's arguments are passed as keyword arguments, and the return value — anything JSON
can hold — is the tool's result. An `async def` is awaited on the loop the turn is iterated
from; a plain function runs on a worker thread. An exception becomes the result
(`"error: ..."`) for the model to see, rather than ending the turn.

MCP servers and A2A agents are registered the same way, with `register_mcp_stdio`,
`register_mcp_streamable_http` and `register_a2a`, which answer with the descriptions to
give the builder. Other models are served with `register_lang_model(pattern, schema, url,
api_key)`.

## The console

`AgentBuilder.console(console)` shares the console's session with the agent rather than
taking it: the `ConsoleClient` stays usable, its calls and the agent's tools take turns, and
`console.close()` ends it for both. The agent starts the console's backend for each batch of
tool calls and stops it afterwards, so it is not started beforehand.

Use `ailoy.cortex` rather than the `cortex-py` package for it. Each extension links its own
copy of cortex, and a `ConsoleClient` from another one is not a type this one can hand an agent.

## Building

The extension is built with [maturin](https://www.maturin.rs), against the cortex checkout
beside this one (`../cortex`, as the crate itself is). On macOS, cortex's host mount needs
FUSE-T installed to build and to import: `brew install --cask fuse-t`.

```sh
cd bindings/python
uv sync                        # makes .venv and installs the dev group
uv run maturin develop         # builds the extension into it
uv run pytest
```

`uv run maturin build --release` makes a wheel.

The tests that talk to a model need its API key, and the ones that run a console need the
console server cortex starts by default.
