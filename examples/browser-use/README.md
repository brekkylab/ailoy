# Browser use

An agent with your own browser and no task of its own: it waits for what you ask of it. It drives the browser (Chrome by default, or another Chromium-based one) with its `browser_*` tools, in a window of its own. It also has a console with python and curl (`system_tools()`), for what is easier done there, such as fetching a page's data or computing over it; the browser tools do not go through it. With `--noshell` the agent has the browser alone, and no console server is needed. The app connects before the first request (`start_browser()`), when the browser asks you to allow it, and the agent's window opens on the ailoy docs (`start_url`), so there is a page to ask about. The app then suggests a few requests to try: solving today's Wordle, comparing flights, and comparing wireless mice. Each line you type is the next turn, and `/exit` ends.

When a site needs something only you can do, such as signing in, the agent says so and ends its turn. Do it in its window, then tell the agent to go on.

See [Using a browser](../../docs/guide/using-browser.md) for what this lends the agent.

## Requirements

- A Chromium-based browser with remote debugging on: Chrome 144 or later, at `chrome://inspect/#remote-debugging`.
- A virtx console server for the console, which the example fetches into virtx's cache the first time (`ensure_virtx()`). Not with `--noshell`.
- `AILOY_MODEL` picks the model (`anthropic/claude-sonnet-5` by default); its provider has to be set up.

## Running

Rust (from the repository root):

```sh
cargo run --example browser-use         # or: cargo run --example browser-use -- --noshell
```

Python:

```sh
cd examples/browser-use/python
uv run main.py                          # or: uv run main.py --noshell
```

Node:

```sh
cd examples/browser-use/node
npm install
npm start                               # or: npm start -- --noshell
```
