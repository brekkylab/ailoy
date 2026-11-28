# Browser use

An agent with a browser and no task of its own: it waits for what you ask of it. The browser is virtx's `browser` abin, a headless Chromium in the agent's sandbox, which the agent drives with its `browser` and `browser_screenshot` tools. The sandbox also has python and curl, and the shell and file tools that come with it, for what is easier done without the browser, such as fetching a page's data or computing over it. The app suggests a few requests to try: solving today's Wordle, comparing flights, and comparing wireless mice. Each line you type is the next turn, and `/exit` ends.

See [Using a browser](https://ailoy.co/docs/guides/using-browser/) for what this lends the agent.

## Requirements

- A virtx server with the `browser` abin, which starting the client fetches into virtx's cache the first time.
- `AILOY_MODEL` picks the model (`anthropic/claude-sonnet-5` by default); its provider has to be set up.

## Running

Rust (from the repository root):

```sh
cargo run --example browser-use
```

Python:

```sh
cd examples/browser-use/python
uv run main.py
```

Node:

```sh
cd examples/browser-use/node
npm install
npm start
```
