# AGENTS.md

Ailoy is a library for building AI agents: an `Agent` drives a language model through
tool-augmented turns, and each agent can own a virtual machine (a virtx console) to work in. One
Rust crate carries the behaviour; the Python and Node packages are bindings over it and expose the
same API in their own idiom. Read this before changing anything.

## One behaviour, three surfaces

- The library is `src/`. A feature goes into the crate first and the bindings connect to it;
  `bindings/python/` and `bindings/node/` carry only what one language needs of its own.
- The console is virtx's: an agent takes a `ConsoleClient` built with virtx's own package
  (`virtx` on crates.io and PyPI, `@brekkylab/virtx` on npm). Ailoy adds no console API of its
  own.
- A change to what the library does moves the bindings in the same pull request, and the three
  Quickstart blocks in `README.md` when the public API they show changes.
- Names follow each language's convention and nothing else: `run_stream` in Rust and Python,
  `runStream` in Node. The thing they name is the same.

## Where a change goes

- `src/agent/` — the builder, the run loop (`rt.rs`), subagents, skills, and the spec an agent is
  described by.
- `src/lang_model/` — providers (`impl/api/` has one module per vendor API) and the streaming
  framing under them.
- `src/tool/impl/` — built-in tools in `builtins/`, plus the MCP, A2A and memory tool providers.
- `src/message/` — the message, part and delta types every surface marshals.
- `src/console.rs` — the virtx console an agent is given.
- `bindings/python/src/` and `bindings/node/src/` — the wrappers, in Rust. The Python package
  itself is `bindings/python/python/ailoy/`; the Node package is `bindings/node/index.js` and its
  `.d.ts` files.
- `examples/<name>/` — one folder per language (`rust`, `python`, `node`) and a `shared/` folder for
  what the three have in common. The Rust example is also registered in `Cargo.toml`.
- `docs/guide/` — the VitePress site. `README.md` stays the short version.

## Tests and lint

Run these before pushing. There is no pull-request CI, so what you run locally is the gate.

```bash
cargo fmt --all --check
cargo clippy --workspace
cargo test
cd bindings/python && uv run maturin develop && uv run pytest   # Python
cd bindings/node && npm install && npm run build:debug && npm test   # Node
```

- Tests that need a provider read its key from the environment or a `.env` file
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`) and skip
  without it. A run without keys is quietly narrower than the suite; say which keys were set when
  you report a run.
- Tests that need a console start one on virtx's default server (`test_console` in `src/lib.rs`)
  and fail loudly when it is missing.
- A bug fix brings a test that fails without it. A behaviour change brings the test that pins the
  new behaviour, in the same surface the behaviour lives in: the library's tests are in-module
  under `#[cfg(test)]`, the bindings' are `bindings/python/tests/` and `bindings/node/__test__/`.
- Examples are self-contained: a reader runs one with the command in `README.md` and nothing else
  set up beyond the provider key.

## Comments and docs

- A comment states what the code beside it does now and why, in as few words as that takes. Not
  what it replaced, not the issue it fixed, not the conversation that produced it.
- A comment longer than the code it explains is a comment to cut.
- Every relative link in every markdown file resolves on disk.

## Commits and PRs

- Commit titles are single declarative sentences describing the new state. No trailers.
- A PR's title and description follow Pull requests in `CONTRIBUTING.md`: one sentence saying
  what Ailoy does once it is merged, and a description started from
  `.github/pull_request_template.md`, which `gh pr create --body` bypasses.
- PR and issue bodies reflow paragraphs to one line; no hard wrapping.
