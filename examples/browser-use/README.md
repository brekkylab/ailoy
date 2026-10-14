# Browser use

An agent with a browser and no task of its own: it waits for what you ask of it. The browser starts before the first request (`start_browser()`), in a VM of its own, which you watch and use from yours at the address printed. The app then opens the ailoy docs in it, with `agent-browser` run in the agent's console, so there is a page to ask about, and suggests a few requests to try: solving today's Wordle, comparing flights, and comparing wireless mice. Each line you type is the next turn, and `/exit` ends.

When a site needs something only you can do, such as signing in, the agent says so and ends its turn. Do it in the viewer, then tell the agent to go on.

With `--host` it uses your own Chrome instead (`Browser::host`), in a window of its own. Turn on remote debugging at `chrome://inspect/#remote-debugging` (Chrome 144 or later), and allow Chrome's prompt, which appears as the example starts. See [Using a browser](../../docs/guide/using-browser.md) for what that lends the agent.

## Requirements

- A virtx console server whose `/abin` carries `agent-browser`, and `viewer` for the sandboxed browser. Point `VIRTX_HOME` at it if it is not the default one.
- `AILOY_MODEL` picks the model (`anthropic/claude-sonnet-5` by default); its provider's API key has to be set.

## Running

Rust (from the repository root):

```sh
cargo run --example browser-use         # or: cargo run --example browser-use -- --host
```

Python:

```sh
cd examples/browser-use/python
uv run main.py                          # or: uv run main.py --host
```

Node:

```sh
cd examples/browser-use/node
npm install
npm start                               # or: npm start -- --host
```
