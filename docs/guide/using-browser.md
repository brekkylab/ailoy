# Using a browser

An agent can drive your own browser with its `browser_*` tools. Give it one with `browser(...)` on the `AgentBuilder`. The tools reach the browser from your program over the Chrome DevTools Protocol (CDP), not from the agent's console, so an agent needs no console to browse.

The browser is a Chromium-based one that speaks CDP: Chrome by default. It needs remote debugging on; in Chrome 144 or later, open `chrome://inspect/#remote-debugging` and turn it on. Then:

::: code-group

```python [Python]
import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    agent = await AgentBuilder("anthropic/claude-sonnet-5").browser().build()

    async for output in agent.run("Open https://example.com and tell me its title."):
        for part in output["message"]["contents"]:
            if part["type"] == "text":
                print(part["text"])

    await agent.close()


asyncio.run(main())
```

```js [Node.js]
const { AgentBuilder } = require('@brekkylab/ailoy')

const agent = await new AgentBuilder('anthropic/claude-sonnet-5').browser().build()

try {
  for await (const { message } of agent.run('Open https://example.com and tell me its title.')) {
    for (const part of message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
} finally {
  await agent.close()
}
```

```rust [Rust]
use ailoy::{agent::AgentBuilder, browser::Browser};

let mut agent = AgentBuilder::new("anthropic/claude-sonnet-5")
    .browser(Browser::host())
    .build()
    .await?;
```

:::

Nothing connects until the agent first uses a browser tool or asks you something. Then the browser asks you whether to allow it. While it is allowed, Chrome shows that it is being controlled by automated software. The agent opens a window of its own and works there. The browser asks only once for the agent's whole life, because ailoy keeps one connection to it open.

To connect sooner, say while you are at your screen, call `start_browser()` (`startBrowser()` in Node) between turns. It does nothing once connected. A decline, or remote debugging being off, comes back as an error. `start_url` (`startUrl` in Node, `Browser::start_url` in Rust) is the page the agent's window opens on.

- If you decline, the agent hears so and does not try again. It asks again only through `ask_user` (below).
- If the browser closes, or you turn remote debugging off, the agent connects again on its next browser tool. The browser asks again, and if the agent's window is gone it gets a new one. The agent is told when this happens.
- If remote debugging is off, the agent is told how to turn it on, and you can pass that on.

ailoy finds the browser by the `DevToolsActivePort` file a Chromium-based browser writes in its user data directory while remote debugging is on. By default it looks at Chrome, Chrome Beta, Dev and Canary, Chromium, Edge and Brave, and takes whichever turned remote debugging on last. To name one, pass `host_browser` (`hostBrowser` in Node, `Browser::host_browser` in Rust): `"chrome"`, `"chrome-beta"`, `"chrome-dev"`, `"chrome-canary"`, `"chromium"`, `"edge"` or `"brave"`, or any other by its user data directory (a path in Python, `{ userDataDir }` in Node, `HostBrowser::Custom` in Rust).

::: warning
The agent uses your real browser, signed in wherever you are. Its tools cannot reach your other tabs, but in its own window it can do anything you could. `ask_user`'s description tells it to ask you before anything that cannot be undone, such as paying, sending or deleting. That is an instruction, though, not a limit, so use your own browser only with an agent and a task you trust.
:::

## The tools

Rather than being handed the page at every step, the agent reads it with `browser_snapshot` when it needs to.

| Tool | What it does |
|---|---|
| `browser_navigate` | Opens a URL in the current tab, or in a new one with `new_tab`. |
| `browser_back` | Goes back one page. |
| `browser_snapshot` | The page as text: its accessibility tree in document order, frames included (those of other sites too), with refs such as `[ref=e3]` on what can be acted on (what is hidden or covered is left out, a box that scrolls says how far, a select lists its options), after a line on the page as a whole (links, refs, iframes, how much lies above and below, whether it still loads). `interactive` keeps only what can be acted on, `depth` limits the tree, `selector` keeps one element and what is inside it, `urls` shows where links go. |
| `browser_click` | Clicks an element by ref, at its middle, by script when something covers it; or at `x` and `y` in the last screenshot. A `<select>` gives its options instead, and a download the click starts is waited for and reported. |
| `browser_input` | Types into a field by ref a key at a time, clearing it first unless `clear` is false; a date, time or color field is set outright. Says when the field ends up holding something else. |
| `browser_send_keys` | Presses a key or a shortcut, such as `Enter` or `Control+a`, or types text into what has focus. |
| `browser_scroll` | Scrolls by pages, the page or inside an element by ref (an iframe's document, or the box under it). |
| `browser_find_text` | Scrolls to the first element whose text or an attribute holds some text. |
| `browser_dropdown_options` | Lists a dropdown's options: a `<select>`, an ARIA combobox, menu or listbox, or a Semantic UI dropdown. |
| `browser_select_dropdown` | Picks a dropdown's option by its text or value, case ignored. |
| `browser_search_page` | Searches the page's text, like grep, with the text around each match and the element it is in. |
| `browser_find_elements` | Queries elements by CSS selector, with their text, the attributes asked for and how many children they have. |
| `browser_evaluate` | Runs JavaScript in the page and returns its value; an image it returns as a data URL is shown as one. |
| `browser_screenshot` | A picture of the current tab, at most 1280 pixels wide; `browser_click`'s coordinates are its pixels. |
| `browser_wait` | Waits for text to appear, or for some seconds. |
| `browser_dialog` | Accepts or dismisses a confirm, or answers a prompt. An alert is accepted on its own, and the agent is told. |
| `browser_tabs` | Lists the agent's tabs, by id. |
| `browser_switch_tab`, `browser_close_tab` | Switches to or closes one of them. |
| `ask_user` | Asks you to do something only you can do (below). |

With a console, three more keep or take a file there: `browser_save_screenshot` and `browser_save_as_pdf` save the current tab as a PNG or a PDF in the console, and `browser_upload_file` puts a console file into a file input.

A ref names the same element while the page shows the same document, and one that appeared since the last snapshot is marked `[new]`; after a navigation the agent takes a new snapshot. Each action reports the page's title and address afterwards, and a tab the page opens becomes the agent's current one.

The tools reach only the agent's own tabs: the ones it opened, and those its pages opened.

## Sites that check for bots

Some sites turn away traffic that looks automated, with a robot check or a block. By default the description of `browser_navigate` tells the agent to use sites as a person would: start from a site's home page or search and go through its links and forms rather than typed-in query URLs, act at a person's pace, and never retry in a loop. When a robot check appears anyway, it stops and asks you through `ask_user` rather than trying to get around it. To leave this out, pass `careful=False` (`careful: false` in Node, `Browser::careful(false)` in Rust).

ailoy adds nothing to the agent's system prompt for the browser: what the agent knows of it is in the tools' descriptions.

## Asking you to do something

When a site needs something only you can do, such as signing in, a CAPTCHA or a second factor, the agent never asks for your password. It calls its `ask_user` tool. `ask_user` sets how it waits for you:

- `"end_turn"` (the default): the agent tells you in its reply what to do and the turn ends. Your next message continues.
- `"wait"`: the turn waits for `on_ask` (`onAsk` in Node), which your app provides.

`on_ask` is called with the request and returns `"done"`, `"declined"`, or any other string, which counts as done and is passed to the agent as a note. If it raises, that counts as declined. `ask_timeout` (in seconds; `askTimeoutMs` in Node) bounds how long it may take. This asks on the terminal:

```python
async def ask_here(message: str) -> str:
    answer = await asyncio.to_thread(input, f"{message}\nPress Enter when done, or type no: ")
    return "declined" if answer.strip() == "no" else "done"

agent = await AgentBuilder(model).browser(ask_user="wait", on_ask=ask_here).build()
```
