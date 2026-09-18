# Desktop development

## The `../cortex` sibling checkout

`ailoy` depends on `cortex` by **path**, not by version: `../cortex`, a sibling of this
checkout. There is no registry copy to fall back on, so a worktree that is not a sibling
of one does not build.

That checkout tracks `main`, and must carry **#43** — the split of `WorkFs` into the
three-layer `ContextFs` plus a console with `context`/`artifacts`/`scratch`. The engine's
workspace layer is written against exactly that shape and does not compile without it.

Two older commits matter for the shell tool, and `main` has both: before `9d178c7` the
local console accepts `timeout_secs` and ignores it, so a command that outruns its timeout
runs to completion; `3dd05ef` is what makes the kill reach the whole process group rather
than the direct child alone.

## Tests

- `$AILOY_CORTEX_CONSOLE` overrides which binary the console-backed tests start. It is
  `cortex-local-console` by default, which lives in the sibling checkout rather than on
  `PATH` during development:

  ```sh
  cargo build --manifest-path ../cortex/Cargo.toml -p cortex-local-console
  AILOY_CORTEX_CONSOLE=$PWD/../cortex/target/debug/cortex-local-console cargo test --lib
  ```

- `$AILOY_CORTEX_BIN_DIR` is the **engine's** variable — `ailoy-desktop-core` looks for
  `cortex-local-console` in that directory when the caller passes no explicit path. (The
  Tauri app passes the bundled binary's path and never consults it.) It is distinct from
  `$AILOY_CORTEX_CONSOLE` above, which is the root crate's and names a file, not a
  directory. With neither set, the engine walks up from the working directory into
  `cortex/target/{debug,release}` and warns when that development fallback is what
  answered.

- `cargo test --lib` is offline-safe once `.env` holds no keys: every test that would talk
  to a paid API either skips when its key is absent or is gated by `test_with::env`. With
  keys in `.env` it calls those APIs for real.
- `#[ignore]`d tests are the ones that need something the default run should not assume
  (a network, a console binary). Run one with `cargo test --lib -- --ignored <name>`.

## The desktop engine (`apps/desktop/core`)

`cargo test -p ailoy-desktop-core` runs the whole crate offline. It is **not** in the
workspace's `default-members`: the crate enables cortex's `fuse-t` feature, whose build
script panics without FUSE-T installed, so a bare `cargo test` at the root builds the root
crate alone and the engine is asked for by name.

Its three `#[ignore]`d integration tests each need something real:

```sh
export AILOY_CORTEX_BIN_DIR=$PWD/../cortex/target/debug

# a real cortex-local-console
cargo test -p ailoy-desktop-core --test live_console -- --ignored

# FUSE-T installed: the workspace mounted where the kernel can see it
cargo test -p ailoy-desktop-core --test live_workspace -- --ignored

# both: a fake model's tool call routed through a real console, unmounted and mounted
cargo test -p ailoy-desktop-core --test live_run -- --ignored
```

A `live_workspace` or mounted `live_run` that is killed mid-test can leave a mount behind.
Check with `mount | grep workspace` and clear it with `umount <path>` (or
`diskutil unmount force <path>`).

## Formatting and linting

Use `cargo fmt -p ailoy`. Plain `cargo fmt --all` walks into `../cortex` through the path
dependency and reformats a checkout that is not this crate's to reformat. For the same
reason, scope test and clippy runs with `-p ailoy`.
