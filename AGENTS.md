# AGENTS.md

Ailoy is a library for building AI agents that each own a virtual machine (a [virtx](https://github.com/brekkylab/virtx) sandbox) to work in; a language model drives the agent through tool-augmented turns. The Rust crates carry the behaviour; the Python and Node packages are bindings over them and expose the same API in their own idiom. Read this before changing anything.

## One behaviour, three surfaces

- The library is `ailoy-runner/`, over the data in `ailoy-core/`; the crate `ailoy` (`ailoy-rs/`) shows both to a Rust program. A feature goes into the crates first and the bindings connect to it; `ailoy-py/` and `ailoy-node/` carry only what one language needs of its own.
- The sandbox is virtx's (`virtx` on crates.io and PyPI, `@brekkylab/virtx` on npm), made for each session where the agent runs and removed with it. The code managing a server makes it with the provider's virtx client, of a package or of a sandbox's args; an app's user, of a package alone, with their own credentials, which virtx lets in, or not, and counts against the package's limits. Ailoy adds no sandbox API of its own. The bindings take the caller's virtx `Client` (`set_virtx`) over the one `libvirtx_core` both packages open (`ailoy-*/src/console.rs`).
- A change to what the library does moves the bindings with it, and, when a public API they show changes, the three Quickstart blocks in `README.md` and the examples under `examples/`.
- That is the standard for a change you make: the crate first, every binding connected, before the work is done. It is not the gate for a person's pull request: `CONTRIBUTING.md` accepts a crate change that reaches only one binding, or a change made in one binding without the crate, and a review holds a PR to that, with the crate-first shape as a suggestion or a follow-up. Code you write for a person who then opens the PR is held to the first; their PR is held to the second.
- Names follow each language's convention and nothing else: `run_stream` in Rust and Python, `runStream` in Node. The thing they name is the same.

## Where a change goes

- `ailoy-core/src/` — what names an agent and what it says, as data that goes wherever an agent runs: the spec, the state an agent carries between turns, and the `AgentProvider` that holds the tools a spec names, the virtx client, and the agents kept by name with each session's sandbox and grant (`agent/`), the message, part and delta types every surface marshals (`message/`), a tool's description, its provider's trait and `ToolFunc` (`tool/`), a model's options (`model.rs`) and `Value` (`datatype/`). It names models and tools and builds none; `ailoy` and `ailoy-runner` show these types at their own paths (`ailoy::message::Message`). The clients of a server are here too (`client/`): `Client` for the code that manages it, `AppClient` for an app's user, who reaches one session with its grant.
- `ailoy-runner/src/agent/` — the run loop (`rt.rs`), how the history is truncated as a turn begins (`state.rs`), and skills.
- `ailoy-runner/src/model/` — the model traits, one module per backend under `impl/` (a vendor API, a CLI, Bedrock, OpenRouter, an OpenAI-compatible server such as vLLM) and the provider registry with its catalogs under `provider/`.
- `ailoy-runner/src/tool/impl/` — the tools this crate provides, one `ToolSet` per group: the shell and file tools in the sandbox (`console/`), `web_search` and `web_fetch` (`web/`), the tools over the sandbox's `browser` abin (`browser.rs`), and an MCP server's (`mcp.rs`).
- `ailoy-runner/src/console.rs` — what the console tools do with the virtx sandbox an agent is given.
- `ailoy-runner/src/router.rs` — the two HTTP APIs that serve an `AgentProvider`: `admin_router` to `ailoy_core::Client`, and `router` to `ailoy_core::AppClient`. The process's own `AgentProvider`, which an agent built here makes its model and tools through, is `agent::default_provider`. Its `tools` (a `ToolProvider`) and `models` (a `LangModelProvider`) map a name to a `ToolSet` or an org to a `LangModelSet`, each a trait in `ailoy-core` that the runner and the bindings implement.
- `ailoy-rs/` — the crate `ailoy`: what a Rust program depends on, and what the bindings build on. Its `Agent` reaches an agent over HTTP from a server in this process that serves the `default_provider`; it re-exports `ailoy-runner`'s modules beside it.
- `ailoy-py/src/` and `ailoy-node/src/` — the wrappers, in Rust. The Python package itself is `ailoy-py/python/ailoy/`; the Node package is `ailoy-node/index.js` and its `.d.ts` files.
- `examples/<name>/` — one folder per language (`rust`, `python`, `node`) and a `shared/` folder for what the three have in common. The Rust example is also registered in `ailoy-rs/Cargo.toml`.
- `docs/` — the Starlight docs site, served at ailoy.co/docs; the pages are in `docs/content/` and `docs/README.md` says what a page may use. Anything longer than a README section goes there. `README.md` stays short: the pitch, the install lines and one example per language.

## Tests and lint

Run these before pushing. Only `docs/` has pull-request CI (its build), so for the code what you run locally is the gate.

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p ailoy -p ailoy-core -p ailoy-runner
(cd ailoy-py && uv run maturin develop && uv run pytest)             # Python
(cd ailoy-node && npm install && npm run build:debug && npm test)    # Node
```

- `git config core.hooksPath .githooks` installs the hooks in `.githooks/`: the format check on every commit, and clippy plus the library's tests on every push. `SKIP_HOOKS=1 git push` skips the hook for that one command.
- Tests that need a provider read its key from the environment or a `.env` file (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`) and skip without it. A run without keys is quietly narrower than the suite; say which keys were set when you report a run.
- Tests that need a sandbox make one on virtx's default server (`test_sandbox` in `ailoy-runner/src/lib.rs`) and fail loudly when it is missing.
- Against a virtx checkout that is not released (path dependencies on `../virtx`: `virtx-rs` in `[workspace.dependencies]`, `virtx-py` in `[tool.uv.sources]`, `virtx-node` as a `file:` dependency), build it with `cargo xtask build client server-stdio` there, which also signs the server with the hypervisor entitlement on macOS, and point the tests at it: `VIRTX_STDIO=../virtx/target/release/virtx-stdio` for the server and `VIRTX_CORE_LIB=../virtx/target/release/libvirtx_core.dylib` for the library the Python and Node packages load (the library's own tests link it in, and need only the server), each as an absolute path, since a test runs in its own crate's or package's directory. Without `VIRTX_STDIO`, the server of the client's version is fetched from virtx's GitHub release, which an unreleased version does not have.
- A bug fix brings a test that fails without it. A behaviour change brings the test that pins the new behaviour, in the same surface the behaviour lives in: the library's tests are in-module under `#[cfg(test)]`, the bindings' are `ailoy-py/tests/` and `ailoy-node/__test__/`.
- Examples are self-contained: a reader runs one with the command in `README.md` and nothing else set up beyond the provider key.

## Docs

- Run the site with `cd docs && npm ci && npm run dev` and open the address it prints, which ends in `/docs/`. `npm run build` is the pull-request check, and a page can render wrong while the build passes, so look at the pages you touched.
- A guide is one `.mdx` file in `docs/content/guides/`, served at `/docs/guides/<file name>/`. Its frontmatter has a `title` and a one-sentence `description`; the body starts at `##`, since the title is the page's `#`.
- A new page is listed by hand in two places: `docs/sidebar.json` and a card in the overview, `docs/content/index.mdx`. A page missing from the sidebar still builds, but nothing leads to it.
- ailoy.co reads `docs/content/` and `docs/sidebar.json` as they are. The header and styles in `docs/preview/` mirror ailoy.co's, so a change to those needs the same change in ailoy.co. The same goes for the `astro` and `@astrojs/starlight` versions in `docs/package.json`: raise them with ailoy.co's, by hand (Dependabot opens only security updates for `docs/`).
- Code that differs by language goes in `<Tabs syncKey="lang">` with one `<TabItem label="...">` per language, labelled exactly `Python`, `Node.js` and `Rust` so a reader's choice carries across pages; import both from `@astrojs/starlight/components` at the top. A file name goes on the fence as `title="Cargo.toml"`.
- Notes are Starlight asides, `:::note`, `:::tip`, `:::caution` or `:::danger`, closed by `:::`. VitePress syntax does not work here and the build does not catch it: `::: code-group` and `::: warning` are printed as text, `:::warning` renders as a plain paragraph, and a `[label]` after a fence is dropped.
- MDX reads `{`, `}` and `<` in prose as code: put them in backticks or escape them (`\{`, `&lt;`).
- Link to another page by its full path with a trailing slash (`/docs/guides/tools/`, `/docs/guides/tools/#a-heading`). A path without `/docs/` breaks on ailoy.co; `npm run build` checks every such link and anchor.

## Comments and docs

- A comment states what the code beside it does now and why, in as few words as that takes. Not what it replaced, not the issue it fixed, not the conversation that produced it.
- A comment longer than the code it explains is a comment to cut.
- Every relative link in every markdown file resolves on disk.

## Commits and PRs

- Commit titles are `type(scope): summary`, as in `fix(examples): give agents room for long replies` or `docs: add CONTRIBUTING.md`: a lower-case imperative summary of the new state, with no issue number (that goes in the PR).
- A PR's title and description follow Pull requests in `CONTRIBUTING.md`: one sentence saying what Ailoy does once it is merged, and a description started from `.github/pull_request_template.md`, which `gh pr create --body` bypasses.
- PR and issue bodies are written one paragraph per line, never wrapped at a column: GitHub renders every newline as a line break, so a wrapped paragraph shows as ragged lines.
