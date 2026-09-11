# Desktop development

## The `../cortex` sibling checkout

`ailoy` depends on `cortex` by **path**, not by version: `../cortex`, a sibling of this
checkout. There is no registry copy to fall back on, so a worktree that is not a sibling
of one does not build.

That checkout must be on branch `feat/exec-timeout`, at or after commit **`3dd05ef`**.
Before `9d178c7` the local console accepts the shell tool's `timeout_secs` and ignores it,
so a command that outruns its timeout runs to completion and the timeout tests fail;
`3dd05ef` is what makes the kill reach the whole process group rather than the direct
child alone. `main` has neither.

## Tests

- `$AILOY_CORTEX_CONSOLE` overrides which binary the console-backed tests start. It is
  `cortex-local-console` by default, which lives in the sibling checkout rather than on
  `PATH` during development:

  ```sh
  cargo build --manifest-path ../cortex/Cargo.toml -p cortex-local-console
  AILOY_CORTEX_CONSOLE=$PWD/../cortex/target/debug/cortex-local-console cargo test --lib
  ```

- `cargo test --lib` is offline-safe once `.env` holds no keys: every test that would talk
  to a paid API either skips when its key is absent or is gated by `test_with::env`. With
  keys in `.env` it calls those APIs for real.
- `#[ignore]`d tests are the ones that need something the default run should not assume
  (a network, a console binary). Run one with `cargo test --lib -- --ignored <name>`.

## Formatting and linting

Use `cargo fmt -p ailoy`. Plain `cargo fmt --all` walks into `../cortex` through the path
dependency and reformats a checkout that is not this crate's to reformat. For the same
reason, scope test and clippy runs with `-p ailoy`.
