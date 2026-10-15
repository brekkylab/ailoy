# Using a browser

An agent can drive a web browser. It does so from its console with the [`agent-browser`](https://github.com/vercel-labs/agent-browser) CLI, which reads pages as text and acts on their elements, so the agent runs browser commands through its shell like any other command. Give the agent a browser with `browser(...)` on the `AgentBuilder`. The agent's console needs its network on, because the agent reaches the browser over it.

There are two kinds of browser:

- **Sandboxed** (the default) is Chromium in a VM of its own. You watch it and use it from your own browser, through a viewer whose address the agent gives you. It has nothing of yours in it: no cookies, no saved sign-ins.
- **Host** is your own Chrome. The agent works in a window of its own, signed in wherever you are. Chrome asks you before it lets the agent in.

Nothing starts until the agent first runs `agent-browser` or asks you something. To start it sooner, say to have the viewer open before the first request, call `start_browser()` (`startBrowser()` in Node) between turns. It returns the viewer's address, or nothing for your own Chrome, and does nothing once the browser is up. For your own Chrome, this is when Chrome asks you, so call it while you are at your screen. A decline, or remote debugging being off, comes back as an error. A sandboxed browser ends when the agent does. Your own Chrome stays open.

Both need a virtx server whose `/abin` carries `agent-browser`. A sandboxed browser also needs `viewer`.

## A sandboxed browser

::: code-group

```python [Python]
import asyncio

from ailoy import AgentBuilder
from ailoy.virtx import ConsoleClient, Recipe


async def main() -> None:
    console = await ConsoleClient.builder().image(Recipe("debian:trixie-slim")).network(True).build()
    agent = await AgentBuilder("anthropic/claude-sonnet-5").system_tools().console(console).browser().build()

    async for output in agent.run("Open https://example.com and tell me its title."):
        # The viewer's address, once the browser has started.
        print("watch at", agent.browser_url)
        for part in output["message"]["contents"]:
            if part["type"] == "text":
                print(part["text"])

    await agent.close()
    await console.close()


asyncio.run(main())
```

```js [Node.js]
const { AgentBuilder, ConsoleClient, Recipe } = require('@brekkylab/ailoy')

const console_ = await ConsoleClient.builder().image(new Recipe('debian:trixie-slim')).network(true).build()
const agent = await new AgentBuilder('anthropic/claude-sonnet-5').systemTools().console(console_).browser().build()

try {
  for await (const { message } of agent.run('Open https://example.com and tell me its title.')) {
    // The viewer's address, once the browser has started.
    console.log('watch at', agent.browserUrl)
    for (const part of message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
} finally {
  await agent.close()
  await console_.close()
}
```

```rust [Rust]
use ailoy::{agent::AgentBuilder, browser::Browser, console::ConsoleClient};
use virtx::image::Recipe;

let console = ConsoleClient::builder()
    .image(Recipe::new("debian:trixie-slim"))
    .network(true)
    .build()
    .await?;
let mut agent = AgentBuilder::new("anthropic/claude-sonnet-5")
    .system_tools()
    .console(console)
    .browser(Browser::sandboxed())
    .build()
    .await?;
// `agent.browser()` is a handle whose `url()` can be read while a turn runs.
```

:::

`browser_url` (`browserUrl` in Node) is `None` until the browser has started. Its address ends in `#t=` and a token. The viewer lets in only whoever has that token, so do not share the address.

In the viewer you see the tab you choose, and the view follows the agent when it moves to another tab. You can open tabs of your own. If you change the address of the agent's tab or close it, the viewer warns you first, and the agent hears about it on its next turn.

Pages are asked for in your computer's language. To use another, pass `locale`, as in `browser(locale="ko-KR")`.

## Your own Chrome

This needs Chrome 144 or later. In Chrome, open `chrome://inspect/#remote-debugging` and turn remote debugging on. Then:

::: code-group

```python [Python]
agent = await AgentBuilder(model).system_tools().console(console).browser("host").build()
```

```js [Node.js]
const agent = await new AgentBuilder(model).systemTools().console(console_).browser({ mode: 'host' }).build()
```

```rust [Rust]
let mut agent = AgentBuilder::new(model)
    .system_tools()
    .console(console)
    .browser(Browser::host())
    .build()
    .await?;
```

:::

The first time the agent uses the browser, Chrome asks you whether to allow it. While it is allowed, Chrome shows that it is being controlled by automated software. The agent then opens a window of its own and works there. Chrome asks only once for the agent's whole life, because ailoy keeps one connection to Chrome open.

- If you decline, the agent hears so and does not try again. It asks again only through `ask_user` (below).
- If Chrome closes, or you turn remote debugging off, the agent connects again on its next browser command. Chrome asks again, and if the agent's window is gone it gets a new one. The agent is told when this happens.
- If remote debugging is off, the agent is told how to turn it on, and you can pass that on.

By default ailoy looks for this OS's Chrome. For Chrome Beta, Canary or Chromium, pass that Chrome's user data directory as `user_data_dir` (`userDataDir` in Node, `Browser::host().user_data_dir(..)` in Rust). `locale` does not apply: your Chrome keeps its own language.

The agent's console reaches Chrome through ailoy, at a port on your computer's loopback that ailoy opens to consoles for as long as the agent lives. Nothing else on your network can reach it, and a console gets through only with a secret the agent's alone is given.

::: warning
The agent uses your real browser. It is signed in wherever you are, and nothing stops it from reading or using your other tabs. It is told to keep to its own window and to ask you before anything that cannot be undone, such as paying, sending or deleting. That is an instruction, though, not a limit, so use your own Chrome only with an agent and a task you trust.
:::

## Sites that check for bots

Some sites turn away traffic that looks automated, with a robot check or a block. By default the agent is told to use sites as a person would: it starts from a site's home page and goes through its links and forms instead of opening a search URL directly, fills forms one field at a time, pauses between actions, and does not reload or retry in a loop. When a robot check appears anyway, it stops and asks you through `ask_user` rather than trying to get around it, and goes on more slowly once you have cleared it. To leave this out of what the agent is told, pass `careful=False` (`careful: false` in Node, `Browser::careful(false)` in Rust).

A site that keeps turning a sandboxed browser away may let your own Chrome through: it carries your sign-ins, cookies and history, as a fresh browser in a VM does not. The agent suggests this when a site keeps refusing it.

## Asking you to do something

When a site needs something only you can do, such as signing in, a CAPTCHA or a second factor, the agent never asks for your password. It calls its `ask_user` tool. `ask_user` sets how it waits for you:

| `ask_user` | Sandboxed | Host |
|---|---|---|
| `"end_turn"` (default) | The request shows in the viewer and the turn ends. Your next message continues, and the agent hears what you did in the viewer. | The agent tells you in its reply and the turn ends. Your next message continues. |
| `"wait"` | The request shows in the viewer with a Done button, and the turn waits until you press it. | Needs `on_ask`. |

`on_ask` (`onAsk` in Node) lets your app answer instead. It is called with the request and returns `"done"`, `"declined"`, or any other string, which counts as done and is passed to the agent as a note. If it raises, that counts as declined. `ask_timeout` (in seconds; `askTimeoutMs` in Node) bounds how long it may take. This asks on the terminal:

```python
async def ask_here(message: str) -> str:
    answer = await asyncio.to_thread(input, f"{message}\nPress Enter when done, or type no: ")
    return "declined" if answer.strip() == "no" else "done"

agent = await AgentBuilder(model).system_tools().console(console).browser("host", ask_user="wait", on_ask=ask_here).build()
```
