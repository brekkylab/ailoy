# Using MCP

An agent can use the tools of any [MCP](https://modelcontextprotocol.io) server. Register the server, and you get back a description of each of its tools. Pass those to the `AgentBuilder` with `tools(...)`.

Each tool is registered as `{prefix}__{name}`, so tools from different servers don't clash. For example, the `fetch` tool of a server registered under `fetch` becomes `fetch__fetch`.

## Stdio servers

`register_mcp_stdio` starts the server as a local process and talks to it over stdin and stdout. This runs [`mcp-server-fetch`](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch) with `uvx`:

::: code-group

```python [Python]
import asyncio

from ailoy import AgentBuilder, register_mcp_stdio


async def main() -> None:
    tools = await register_mcp_stdio("fetch", "uvx", ["mcp-server-fetch"])

    agent = await (
        AgentBuilder("anthropic/claude-haiku-4-5")
        .tools(tools)
        .build()
    )

    async for output in agent.run("Summarize https://modelcontextprotocol.io in three sentences."):
        for part in output["message"]["contents"]:
            if part["type"] == "text":
                print(part["text"])


asyncio.run(main())
```

```js [Node.js]
const { AgentBuilder, registerMcpStdio } = require('@brekkylab/ailoy')

const tools = await registerMcpStdio('fetch', 'uvx', ['mcp-server-fetch'])

const agent = await new AgentBuilder('anthropic/claude-haiku-4-5')
  .tools(tools)
  .build()

try {
  for await (const { message } of agent.run('Summarize https://modelcontextprotocol.io in three sentences.')) {
    for (const part of message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
} finally {
  await agent.close()
}
```

```rust [Rust]
use ailoy::{
    agent::AgentBuilder,
    message::{Message, Part, Role},
    tool::register_mcp_stdio,
};
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // "default" is the tool provider to register into.
    let tools = register_mcp_stdio("default", "fetch", "uvx", ["mcp-server-fetch"]).await?;

    let mut agent = AgentBuilder::new("anthropic/claude-haiku-4-5")
        .tools(tools)
        .build()
        .await?;

    let query = Message::new(Role::User)
        .with_contents([Part::text("Summarize https://modelcontextprotocol.io in three sentences.")]);
    let mut stream = agent.run(query);
    while let Some(output) = stream.next().await {
        let message = output?.message;
        if message.role == Role::Assistant {
            for text in message.contents.iter().filter_map(Part::as_text) {
                println!("{text}");
            }
        }
    }
    Ok(())
}
```

:::

::: warning
A stdio server runs on the host, not in the agent's VM, with the same access as your program. Register only servers you trust.
:::

## Streamable HTTP servers

`register_mcp_streamable_http` connects to a remote server by its URL. This registers [DeepWiki](https://deepwiki.com)'s server, which answers questions about GitHub repositories:

::: code-group

```python [Python]
from ailoy import register_mcp_streamable_http

tools = await register_mcp_streamable_http("deepwiki", "https://mcp.deepwiki.com/mcp")
```

```js [Node.js]
const { registerMcpStreamableHttp } = require('@brekkylab/ailoy')

const tools = await registerMcpStreamableHttp('deepwiki', 'https://mcp.deepwiki.com/mcp')
```

```rust [Rust]
use ailoy::tool::register_mcp_streamable_http;

let tools = register_mcp_streamable_http("default", "deepwiki", "https://mcp.deepwiki.com/mcp").await?;
```

:::

Pass `tools` to `AgentBuilder.tools(...)` as above.

## Unregistering

`unregister_mcp` (`unregisterMcp` in Node.js) removes a server's tools by their prefix and closes the connection to the server.

::: code-group

```python [Python]
from ailoy import unregister_mcp

unregister_mcp("fetch")
```

```js [Node.js]
const { unregisterMcp } = require('@brekkylab/ailoy')

unregisterMcp('fetch')
```

```rust [Rust]
ailoy::tool::unregister_mcp("default", "fetch")?;
```

:::
