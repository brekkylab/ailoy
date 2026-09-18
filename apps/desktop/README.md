# Ailoy Desktop

A macOS desktop app: a Tauri 2 shell (`src-tauri`) around a React webview (`src`). The
real work happens in the engine at `apps/desktop/core`. Its design and testing rules are in
[`docs/desktop-development.md`](../../docs/desktop-development.md).

## Prerequisites

- **macOS.** v1 is macOS-only, because the workspace mount is tied to FUSE-T.
- **FUSE-T** — `brew install --cask fuse-t`. Required: the bundle links `libfuse-t.dylib`
  directly, so on a machine without FUSE-T the app fails at the dyld stage, with no window,
  no log and no error dialog. A development run (`tauri:dev`) behaves the same way.
- **Rust ≥ 1.95** (the workspace `rust-version`) and **Node ≥ 22**.
- **A `../cortex` checkout.** `cortex` is a path dependency, so it has to sit next to this
  repository. Track `main`, at `#43` or later — that is the change that split workfs into
  the context, artifacts and scratch layers, and before it there is no `ContextFs` and no
  three-tree console. For timeouts you also want `#38` or later, which is where the shell
  tool's `timeout_secs` starts being enforced and starts reaping the children a command
  spawned.

## Running it

```sh
cd apps/desktop
npm install
npm run tauri:dev
```

Do not call `tauri dev` directly. The sidecar (`cortex-local-console`) has to be built into
`src-tauri/binaries/` first, and `npm run tauri:dev` is what does that, through
`scripts/build-sidecar.sh`. If `../cortex` lives somewhere else, point `CORTEX_DIR` at it.

For an `.app` bundle, run `npm run tauri:build`. The result lands at
`src-tauri/target/release/bundle/macos/Ailoy.app`.

The first run has no keys. Open **Settings** and enter an API key for the provider you want
before the model list turns `available` and a chat can start. The engine holds the keys and
never hands them back to the webview.

## Data

Everything lives under `~/Library/Application Support/com.brekkylab.ailoy/`.

| Path           | What it holds                                                          |
| -------------- | ---------------------------------------------------------------------- |
| `ailoy.sqlite` | Chats, messages, settings, mounts. Keys too, at file mode 0600          |
| `files/`       | The user's files. The root of the workspace                             |
| `workspace/`   | The FUSE-T mountpoint. The path the agent reads                         |
| `cache/`       | The model catalog cache                                                 |
| `artifacts/`   | Files the agent produced. Inside the workspace these appear at `/artifacts` |
| `scratch/`     | One temporary directory per run. The shell starts here, and it is deleted when the run ends |
| `logs/`        | `ailoy.log.<date>`, rolled daily                                        |
| `engine.lock`  | The lock that allows one instance per data directory                    |

Delete the directory to get back to a first-run state.

## Logs

The default level is `info`, and `RUST_LOG` overrides it
(`RUST_LOG=ailoy_desktop_core=debug npm run tauri:dev`). Logs go to `logs/ailoy.log.<date>`
above, and during development to stderr as well. In the app, **Settings → Open logs folder**
opens that directory in Finder.

## Tests

All from the repository root:

```sh
npm --prefix apps/desktop test       # the webview (vitest)
cargo test -p ailoy-desktop-core     # the engine
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml   # the Tauri commands
```

Do not run a bare `cargo test` at the repository root. The root crate has tests that call
real APIs with the keys in `.env`.

Four tests need real machinery and are `#[ignore]`d: `live_workspace`, which needs FUSE-T,
and `live_console` plus the two `live_run` tests, which need a built console (and, for the
one that goes through a mounted workspace, FUSE-T as well). Again from the repository root:

```sh
cargo build --manifest-path ../cortex/Cargo.toml -p cortex-local-console
AILOY_CORTEX_BIN_DIR=$PWD/../cortex/target/debug \
  cargo test -p ailoy-desktop-core --test live_run -- --ignored
```

## Known limitations

- **macOS only.**
- **No approval UI.** Tools run immediately. The engine knows about the `awaiting_approval`
  event, but the v1 webview never asks.
- **No stdout streaming while a tool runs.** A tool card shows the call and the final
  result, not the output as it arrives.
- **Unsigned bundle.** On another machine it has to be opened around Gatekeeper.
- **One instance per data directory.** A second launch fails to take the lock and exits with
  an error dialog.
- **The agent cannot modify the workspace.** The workspace — the user's files and their
  connectors — is the session's *context*, so it is read-only, and what the agent makes goes
  to `/artifacts`. In v1, a request to change one of the user's files is answered by
  producing a new artifact instead.
- **File previews are read-only (there is no editor).** The workspace panel shows files and
  nothing more.
- **Partial text from a stream that a model error cut short is not saved.** A run the user
  stopped keeps the text it had produced; a run that died on a model or network error does
  not.
- **Without FUSE-T the app does not start at all.** The path that downgrades a mount to
  `degraded` is reachable only when FUSE-T is installed and the mount itself failed. Weak
  linking plus a preflight check is follow-up work.
