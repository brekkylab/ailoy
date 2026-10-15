# Quick start

## Set your API key

Set the API key for your model's provider, either in the environment or in a `.env` file:

```sh
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## Install

::: code-group

```sh [Python]
pip install ailoy-py
```

```sh [Node.js]
npm install @brekkylab/ailoy @brekkylab/virtx
```

```toml [Rust (Cargo.toml)]
[dependencies]
ailoy = "0.3"
virtx = "0.1"
```

:::

## Run an agent

This gives an agent a Python machine with `./artifacts` mounted at `/artifacts`, and asks it to draw a chart there.

::: code-group

```python [Python]
import asyncio

from ailoy import AgentBuilder
from virtx import ConsoleClient, Recipe


async def main() -> None:
    console = await (
        ConsoleClient.builder()
        .image(Recipe("python:3.12-slim-trixie").step("pip install matplotlib"))
        .mount("./artifacts", "/artifacts")
        .network(True)
        .build()
    )

    agent = await (
        # For openai, use "openai/gpt-5.6-luna"
        AgentBuilder("anthropic/claude-haiku-4-5")
        .instruction("Write what you are asked for into /artifacts.")
        .system_tools()
        .console(console)
        .build()
    )

    async for output in agent.run("Create a bar chart comparing the populations of European countries and save it to /artifacts/population.png."):
        for part in output["message"]["contents"]:
            if part["type"] == "text":
                print(part["text"])

asyncio.run(main())
```

```js [Node.js]
const { AgentBuilder } = require('@brekkylab/ailoy')
const { ConsoleClient, Recipe } = require('@brekkylab/virtx')

const console_ = await ConsoleClient.builder()
  .image(new Recipe('python:3.12-slim-trixie').step('pip install matplotlib'))
  .mount('./artifacts', '/artifacts')
  .network(true)
  .build()
// For openai, use "openai/gpt-5.6-luna"
const agent = await new AgentBuilder('anthropic/claude-haiku-4-5')
  .instruction('Write what you are asked for into /artifacts.')
  .systemTools()
  .console(console_)
  .build()

try {
  for await (const { message } of agent.run('Create a bar chart comparing the populations of European countries and save it to /artifacts/population.png.')) {
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
    console::ConsoleClient,
    message::{Message, Part, Role},
};
use futures::StreamExt as _;
use virtx::image::Recipe;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let console = ConsoleClient::builder()
        .image(Recipe::new("python:3.12-slim-trixie").step("pip install matplotlib"))
        .mount(std::path::absolute("./artifacts")?, "/artifacts")
        .network(true)
        .build()
        .await?;

    // For openai, use "openai/gpt-5.6-luna"
    let mut agent = AgentBuilder::new("anthropic/claude-haiku-4-5")
        .instruction("Write what you are asked for into /artifacts.")
        .system_tools()
        .console(console)
        .build()
        .await?;

    let query = Message::new(Role::User).with_contents([Part::text("Create a bar chart comparing the populations of European countries and save it to /artifacts/population.png.")]);
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

`agent.run` yields one complete message for each step of the tool loop, and `run_stream` yields token deltas as the model writes them.

