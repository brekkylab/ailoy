# Contributing

Thanks for your interest in improving **Ailoy**. One Rust crate carries the behaviour; the
Python and Node packages are bindings over it, so a change usually touches the crate and then
each binding. [`AGENTS.md`](AGENTS.md) says where each kind of change goes.

## Development setup

```bash
git clone https://github.com/brekkylab/ailoy
cd ailoy
cargo build
cd bindings/python && uv run maturin develop     # Python
cd bindings/node && npm install && npm run build  # Node
```

An agent's console runs on virtx's server, which `ensure_virtx()` (and the tests) fetch into
`$VIRTX_HOME/bin` on first use. Mounting a host folder into it needs FUSE: FUSE-T on macOS,
Dokany on Windows. Tests that drive a provider read its key from the environment or a `.env`
file at the repository root (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`AWS_BEARER_TOKEN_BEDROCK`), and skip without it.

## Running tests and lint

```bash
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
cargo test
cd bindings/python && uv run pytest   # after `maturin develop`
cd bindings/node && npm test          # after `npm run build:debug`
```

There is no pull-request CI, so what you run locally is the gate. A run without provider keys
covers less than the suite: say which keys were set when you report one.

## Pull requests

1. Create a topic branch off `main`.
2. Keep changes focused; one logical change per PR.
3. Bring a test that fails without the change. A bug fix comes with the test that caught it; a
   behaviour change comes with the test that pins the new behaviour, in the surface it lives in.
4. A behaviour the crate gains is reachable from both bindings in the same PR, or the PR says
   which binding it is for and why.
5. Run the checks above before opening the PR.
6. Title the PR with one sentence saying what Ailoy does once it is merged, with no issue
   number and no `fix:`-style prefix. The issue goes in the description as `Closes #N`.
7. Fill in the pull request template: its three headings and three checkboxes, describing the
   finished state rather than the rounds of work behind it. `gh pr create --body` does not load
   the template, so start the description from
   [`.github/pull_request_template.md`](.github/pull_request_template.md).

## Reporting bugs and requesting features

Open an issue at https://github.com/brekkylab/ailoy/issues with what you ran, what you
expected, and what happened, including the language you used Ailoy from. A suspected
vulnerability goes through [SECURITY.md](SECURITY.md) instead, never a public issue.

## Code of Conduct

Taking part in this project means following our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the
[Apache-2.0 License](LICENSE.md).
