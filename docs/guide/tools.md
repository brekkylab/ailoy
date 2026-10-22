# Tools

An agent calls the tools you give it on the `AgentBuilder`: the built-in ones on this page, functions of your own, the tools of an [MCP server](./using-mcp), and the [browser](./using-browser).

## Built-in tools

### A shell and files

`system_tools()` (`systemTools()` in Node.js) gives the agent a shell and tools to read and change files in its VM. Which tools come with the shell depends on the model's org:

| Org | Tools |
| --- | --- |
| `openai` | `shell`, `apply_patch`, `imgread` |
| `google` | `shell`, `read_file`, `write`, `edit`, `imgread` |
| `deepseek-ai`, `moonshotai`, `zai-org`, `Qwen` | `shell`, `read`, `write`, `edit` |
| Any other, `anthropic` included | `shell`, `read`, `write`, `edit`, `imgread` |

| Tool | What it does |
| --- | --- |
| `shell` | Runs a command, with an optional timeout in seconds. |
| `read` | Reads a text file with numbered lines, or `limit` lines from line `offset`. |
| `read_file` | Reads a text file, or the lines from `start_line` to `end_line`. |
| `write` | Writes a text file, replacing one already at that path. |
| `edit` | Replaces an exact string in a text file, once or, with `replace_all`, everywhere. |
| `apply_patch` | Adds, deletes, updates and moves files, from a patch in the format Codex uses. |
| `imgread` | Reads a PNG, JPEG, GIF or WebP image of up to 5 MB and gives it to the model as an image. |

`imgread` is left out for `deepseek-ai`, `moonshotai`, `zai-org` and `Qwen` because their backends pass a tool result to the model as text, so the model would get a label in place of the image.

`shell_tool()` (`shellTool()`) adds `shell` alone.

The seven tools in the table run in the agent's VM, so the agent needs one: pass it with `console(...)`, as in [Building VM](./building-vm). Without a console, each call to one of them comes back to the model as an error. The VM is started for each batch of tool calls that has one of these tools in it, and stopped after; stopping keeps its session, so the next batch finds the files as the last one left them.

### The web

`web_search_tool` and `web_fetch_tool` (`webSearchTool`, `webFetchTool`) are not part of `system_tools()`. The two tools run in your program's process, not in the VM, so they work without a console.

- `web_search` sends the query to several search engines at once and merges what they find, one result per URL, best first. It answers with up to 10 results, or up to 30 when the model asks for more. `web_search_tool` takes the engines to use: `Bing`, `Brave`, `DuckDuckGo`, `Google`, `Mojeek`, `Naver`, `Startpage`, `Yahoo` and `Yandex`. None, or an empty list, uses them all.
- `web_fetch` fetches one URL. An HTML page comes back as its visible text (the default), as Markdown, or as the HTML itself; any other body, such as JSON, comes back as it is. A long body comes in pieces of 30,720 characters, or up to 61,440 when the model asks, and the model asks for the next piece with `offset`. It makes at most one request a second to a host, and only to addresses on the public internet: loopback, the local network and cloud metadata addresses are refused.

::: code-group

```python [Python]
agent = await (
    AgentBuilder("anthropic/claude-haiku-4-5")
    .web_search_tool(["DuckDuckGo", "Brave"])
    .web_fetch_tool()
    .build()
)
```

```js [Node.js]
const agent = await new AgentBuilder('anthropic/claude-haiku-4-5')
  .webSearchTool(['DuckDuckGo', 'Brave'])
  .webFetchTool()
  .build()
```

```rust [Rust]
use ailoy::tool::WebSearchEngineKind;

let agent = AgentBuilder::new("anthropic/claude-haiku-4-5")
    .web_search_tool(vec![WebSearchEngineKind::DuckDuckGo, WebSearchEngineKind::Brave])
    .web_fetch_tool()
    .build()
    .await?;
```

:::

## Your own functions

A function of your program becomes a tool once it is registered with a **tool description**: the name the model calls it by, what it does, and a JSON schema for its arguments. Registering returns the description, which you pass to `tool(...)`, or a list of them to `tools(...)`.

This gives the agent `order_status`, which looks an order up by its id:

::: code-group

```python [Python]
import asyncio

from ailoy import AgentBuilder, register_tool

ORDERS = {"A-1001": {"status": "shipped", "eta": "2026-10-24"}}


def order_status(order_id: str) -> dict:
    if order_id not in ORDERS:
        raise KeyError(f"no order {order_id}")
    return ORDERS[order_id]


async def main() -> None:
    desc = register_tool(
        {
            "name": "order_status",
            "description": "Look up an order's status and expected delivery date by its id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "The order id, such as A-1001"}},
                "required": ["order_id"],
            },
        },
        order_status,
    )

    agent = await AgentBuilder("anthropic/claude-haiku-4-5").tool(desc).build()

    async for output in agent.run("When does order A-1001 arrive?"):
        for part in output["message"]["contents"]:
            if part["type"] == "text":
                print(part["text"])


asyncio.run(main())
```

```js [Node.js]
const { AgentBuilder, registerTool } = require('@brekkylab/ailoy')

const orders = { 'A-1001': { status: 'shipped', eta: '2026-10-24' } }

const desc = registerTool(
  {
    name: 'order_status',
    description: "Look up an order's status and expected delivery date by its id.",
    parameters: {
      type: 'object',
      properties: { order_id: { type: 'string', description: 'The order id, such as A-1001' } },
      required: ['order_id'],
    },
  },
  ({ order_id }) => {
    if (!(order_id in orders)) throw new Error(`no order ${order_id}`)
    return orders[order_id]
  },
)

const agent = await new AgentBuilder('anthropic/claude-haiku-4-5').tool(desc).build()

try {
  for await (const { message } of agent.run('When does order A-1001 arrive?')) {
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
    datatype::Value,
    message::{Message, Part, Role},
    to_value, tool_func,
    tool::{ToolDescBuilder, set_tool_provider},
};
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let desc = ToolDescBuilder::new("order_status")
        .description("Look up an order's status and expected delivery date by its id.")
        .parameters(to_value!({
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order id, such as A-1001"}},
            "required": ["order_id"]
        }))
        .build();
    set_tool_provider(
        "order_status",
        tool_func!(|args: Value| -> Value {
            match args.pointer("/order_id").and_then(Value::as_str) {
                Some("A-1001") => to_value!({"status": "shipped", "eta": "2026-10-24"}),
                Some(id) => Value::string(format!("error: no order {id}")),
                None => Value::string("error: no order_id"),
            }
        }),
    );

    let mut agent = AgentBuilder::new("anthropic/claude-haiku-4-5")
        .tool(desc)
        .build()
        .await?;

    let query = Message::new(Role::User).with_contents([Part::text("When does order A-1001 arrive?")]);
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

How the function is called:

- **Python**: the arguments come as keyword arguments, named as in the schema, and the function may be `async`.
- **Node.js**: they come as one object, and the function may return a promise.
- **Rust** registers a `ToolFunc` with `set_tool_provider`, under the name in the description. `tool_func!` builds one from a closure that takes the arguments as a `Value`; with `async` before the closure, its body can `.await`.

The function returns what a message can hold: null, a boolean, a number, a string, or a list or object of those. In Python and Node.js an exception, or a rejected promise, does not end the run: the model gets it as the result, such as `error: KeyError: 'no order B-9'` from the Python function above, and can try something else.

Tool calls the model makes in one step run at the same time, so a function may be called again before its last call has returned.

A Python or Node.js function runs in your program, not in the VM. To run a command in the VM, it holds a `ConsoleClient` of its own. In Rust, a closure that also takes `console: &mut ConsoleClient` runs on the agent's console, as the built-in tools do.

Tools are registered by name for the whole process, the built-in ones included, so registering a name again, `shell` among them, replaces the function every agent built after that gets. `remove_tool_provider` (`removeToolProvider`) takes a name out.
