# AGENTS.md

Ailoy is a library for building AI agents that each own a virtual machine (a [virtx](https://github.com/brekkylab/virtx) sandbox) to work in; a language model drives the agent through tool-augmented turns. One Rust crate carries the behaviour; the Python and Node packages are bindings over it and expose the same API in their own idiom. Read this before changing anything.

## One behaviour, three surfaces

- The library is `src/`. A feature goes into the crate first and the bindings connect to it; `bindings/python/` and `bindings/node/` carry only what one language needs of its own.
- The sandbox is virtx's: an agent takes a `Sandbox` made with virtx's own package (`virtx` on crates.io and PyPI, `@brekkylab/virtx` on npm). Ailoy adds no sandbox API of its own. The bindings take the caller's sandbox over the one `libvirtx_core` both packages open (`bindings/*/src/console.rs`).
- A change to what the library does moves the bindings with it, and, when a public API they show changes, the three Quickstart blocks in `README.md` and the examples under `examples/`.
- That is the standard for a change you make: the crate first, every binding connected, before the work is done. It is not the gate for a person's pull request: `CONTRIBUTING.md` accepts a crate change that reaches only one binding, or a change made in one binding without the crate, and a review holds a PR to that, with the crate-first shape as a suggestion or a follow-up. Code you write for a person who then opens the PR is held to the first; their PR is held to the second.
- Names follow each language's convention and nothing else: `run_stream` in Rust and Python, `runStream` in Node. The thing they name is the same.

## Where a change goes

- `src/agent/` — the builder, the run loop (`rt.rs`), the context manager, skills, and the spec an agent is described by.
- `src/model/` — the model traits, one module per backend under `impl/` (a vendor API, a CLI, Bedrock, OpenRouter, vLLM) and the provider registry with its catalogs under `provider/`.
- `src/tool/impl/` — built-in tools in `builtins/`, plus the MCP and memory tool providers.
- `src/message/` — the message, part and delta types every surface marshals.
- `src/console.rs` — the virtx sandbox an agent is given, and what the built-in tools do with it.
- `bindings/python/src/` and `bindings/node/src/` — the wrappers, in Rust. The Python package itself is `bindings/python/python/ailoy/`; the Node package is `bindings/node/index.js` and its `.d.ts` files.
- `examples/<name>/` — one folder per language (`rust`, `python`, `node`) and a `shared/` folder for what the three have in common. The Rust example is also registered in `Cargo.toml`.
- `docs/` — the Starlight docs site, served at ailoy.co/docs; the guides are in `docs/src/content/docs/guide/`. Anything longer than a README section goes there. `README.md` stays short: the pitch, the install lines and one example per language.

## Tests and lint

Run these before pushing. Only `docs/` has pull-request CI (its build), so for the code what you run locally is the gate.

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test
(cd bindings/python && uv run maturin develop && uv run pytest)        # Python
(cd bindings/node && npm install && npm run build:debug && npm test)   # Node
```

- `git config core.hooksPath .githooks` installs the hooks in `.githooks/`: the format check on every commit, and clippy plus the library's tests on every push. `SKIP_HOOKS=1 git push` skips the hook for that one command.
- Tests that need a provider read its key from the environment or a `.env` file (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`) and skip without it. A run without keys is quietly narrower than the suite; say which keys were set when you report a run.
- Tests that need a sandbox make one on virtx's default server (`test_sandbox` in `src/lib.rs`) and fail loudly when it is missing.
- Against a virtx checkout that is not released (the path dependencies on `../virtx`), build it with `cargo xtask build client server-stdio` there, which also signs the server with the hypervisor entitlement on macOS, and point the tests at it: `VIRTX_STDIO=../virtx/target/release/virtx-stdio` for the server and `VIRTX_CORE_LIB=../virtx/target/release/libvirtx_core.dylib` for the library the Python and Node packages load. Without `VIRTX_STDIO`, the server of the client's version is fetched from virtx's GitHub release, which an unreleased version does not have.
- A bug fix brings a test that fails without it. A behaviour change brings the test that pins the new behaviour, in the same surface the behaviour lives in: the library's tests are in-module under `#[cfg(test)]`, the bindings' are `bindings/python/tests/` and `bindings/node/__test__/`.
- Examples are self-contained: a reader runs one with the command in `README.md` and nothing else set up beyond the provider key.

## Docs

- Run the site with `cd docs && npm ci && npm run dev` and open the address it prints, which ends in `/docs/`. `npm run build` is the pull-request check, and a page can render wrong while the build passes, so look at the pages you touched.
- A guide is one `.mdx` file in `docs/src/content/docs/guide/`, served at `/docs/guide/<file name>/`. Its frontmatter has a `title` and a one-sentence `description`; the body starts at `##`, since the title is the page's `#`.
- A new page is listed by hand in two places: the `sidebar` in `docs/astro.config.mjs` and a card in the overview, `docs/src/content/docs/index.mdx`. A page missing from the sidebar still builds, but nothing leads to it.
- ailoy.co serves these files with its own copy of the sidebar, styles and header (`docs/astro.config.mjs`, `docs/src/styles/`, `docs/src/components/` mirror it). A page added, renamed or moved here, or a change to those files, needs the same change in ailoy.co. The same goes for the `astro` and `@astrojs/starlight` versions in `docs/package.json`: raise them with ailoy.co's, by hand (Dependabot opens only security updates for `docs/`).
- Code that differs by language goes in `<Tabs syncKey="lang">` with one `<TabItem label="...">` per language, labelled exactly `Python`, `Node.js` and `Rust` so a reader's choice carries across pages; import both from `@astrojs/starlight/components` at the top. A file name goes on the fence as `title="Cargo.toml"`.
- Notes are Starlight asides, `:::note`, `:::tip`, `:::caution` or `:::danger`, closed by `:::`. VitePress syntax does not work here and the build does not catch it: `::: code-group` and `::: warning` are printed as text, `:::warning` renders as a plain paragraph, and a `[label]` after a fence is dropped.
- MDX reads `{`, `}` and `<` in prose as code: put them in backticks or escape them (`\{`, `&lt;`).
- Link to another guide relatively with a trailing slash (`../tools/`, `../tools/#a-heading`), or by its full path (`/docs/guide/tools/`). A path without `/docs/` breaks on ailoy.co; the build checks absolute links and anchors, not relative ones.

## Comments and docs

- A comment states what the code beside it does now and why, in as few words as that takes. Not what it replaced, not the issue it fixed, not the conversation that produced it.
- A comment longer than the code it explains is a comment to cut.
- Every relative link in every markdown file resolves on disk.

## Commits and PRs

- Commit titles are `type(scope): summary`, as in `fix(examples): give agents room for long replies` or `docs: add CONTRIBUTING.md`: a lower-case imperative summary of the new state, with no issue number (that goes in the PR).
- A PR's title and description follow Pull requests in `CONTRIBUTING.md`: one sentence saying what Ailoy does once it is merged, and a description started from `.github/pull_request_template.md`, which `gh pr create --body` bypasses.
- PR and issue bodies are written one paragraph per line, never wrapped at a column: GitHub renders every newline as a line break, so a wrapped paragraph shows as ragged lines.
