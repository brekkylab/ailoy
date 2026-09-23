# Ailoy Desktop

A macOS desktop app: a Tauri 2 shell (`src-tauri`) around a React webview (`src`). The
real work happens in the engine at `apps/desktop/core`.

## Prerequisites

- **macOS.** v1 is macOS-only, because the workspace mount is tied to FUSE-T.
- **FUSE-T** — `brew install --cask fuse-t`. Required: the bundle links `libfuse-t.dylib`
  directly, so on a machine without FUSE-T the app fails at the dyld stage, with no window,
  no log and no error dialog. A development run (`tauri:dev`) behaves the same way.
- **Rust ≥ 1.95** (the workspace `rust-version`) and **Node ≥ 22**.
- **A `../cortex` checkout.** `cortex` is a path dependency, so it has to sit next to this
  repository. It has to carry the console `Backend` (`feat/console-backend`, not yet on
  `main`): the app no longer ships `cortex-local-console` beside itself, and instead runs the
  server cortex builds and embeds — see [the console](#the-console). That branch already
  holds what the app needed from `main` before it: `#43`, which split workfs into the
  context, artifacts and scratch layers, and `#38`, which enforces the shell tool's
  `timeout_secs` and reaps what a command spawned.

## Running it

```sh
cd apps/desktop
npm install
npm run tauri:dev
```

Use `npm run tauri:dev` rather than a bare `tauri dev`: it merges `tauri.dev.conf.json`,
which is what lets the debug build's MCP bridge be reached (see `src-tauri/src/lib.rs`).

### The console

A run's shell tool talks to a console server, and that server comes from cortex itself:
`Backend::local()`, which cortex's `local` feature builds and carries inside the app's own
binary. The first run that needs it writes it out to `cortex/bin/` in the data directory
(below), so there is nothing to build, bundle or keep at the right version by hand.

What it costs is build time. The embedded server is always a release build with fat LTO,
made by cortex's build script the first time and again whenever cortex's sources change.
Working on cortex itself, `CORTEX_EMBED_FAST=1` trades a few megabytes for minutes, and
`CORTEX_LOCAL_CONSOLE_BIN=<path>` runs another build of the server without re-embedding one.

For an `.app` bundle, run `npm run tauri:build`. The result lands at
`src-tauri/target/release/bundle/macos/Ailoy.app`. It fetches the model list from
[models.dev](https://models.dev) first (`npm run catalog`, into the untracked
`core/assets/models.json`) and embeds it, for a first start that cannot reach models.dev. A
dev build embeds none: its model picker is empty until the app's first fetch lands, a
second or two after it starts. From then on the app keeps its own copy under `cache/` and
fetches again every six hours.

The first run has no keys. Open **Settings** and enter an API key for the provider you want
before the model list turns `available` and a chat can start. The engine holds the keys and
never hands them back to the webview.

## Data

Everything lives under `~/Library/Application Support/com.brekkylab.ailoy/`.

| Path           | What it holds                                                          |
| -------------- | ---------------------------------------------------------------------- |
| `ailoy.sqlite` | Chats, messages, settings, mounts. Keys too, at file mode 0600          |
| `files/`       | The workspace root only when the process has no `HOME`; otherwise unused |
| `workspace/`   | The FUSE-T mountpoint. The path the agent reads                         |
| `cache/`       | The model list fetched from models.dev (`models.json`), and remote-source caches |
| `cortex/`      | What cortex writes for itself: the console server, under `bin/`, on the first run that needs it |
| `artifacts/`   | Files the agent produced. Inside the workspace these appear at `/artifacts` |
| `scratch/`     | One temporary directory per run. The shell starts here, and it is deleted when the run ends |
| `logs/`        | `ailoy.log.<date>`, rolled daily                                        |
| `engine.lock`  | The lock that allows one instance per data directory                    |

Delete the directory to get back to a first-run state.

## Logs

The default level is `info`, and `RUST_LOG` overrides it
(`RUST_LOG=ailoy_desktop_core=debug npm run tauri:dev`). Logs go to `logs/ailoy.log.<date>`
above, and during development to stderr as well.

### Filesystem timings

Every store call the window makes is timed onto the `fs_timing` target — one line per
`list`, `stat` and `read`, with how long it took and how much it was for. A remote source
(S3, Notion) charges per call, so this is how a caching decision gets made on numbers rather
than on a hunch:

```sh
npm run tauri:dev                     # drive it: walk the tree, open some files
python3 scripts/fs-timings.py         # what *this run* cost, by operation and by source
python3 scripts/fs-timings.py --all   # every run in the logs instead
```

The newest run only, by default. A log outlives the build that wrote it, and the first
comparison made here mixed two runs of two different binaries and read the average as a
result; the app writes a marker at startup and the summary splits on it.

The summary groups by the mount a path is under, and lists the slowest individual calls. A
read shows as two lines, a `stat` and a body, because which of the two dominates decides
whether a cache should hold metadata, bytes, or both. `RUST_LOG=fs_timing=off` turns the
lines off without a rebuild.

These are the window's numbers only: the agent reads through the FUSE mount, which serves
the same tree without passing through the engine's `fsops`.

## Tests

All from the repository root:

```sh
npm --prefix apps/desktop test       # the webview (vitest)
cargo test -p ailoy-desktop-core     # the engine
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml   # the Tauri commands
```

Do not run a bare `cargo test` at the repository root. The root crate has tests that call
real APIs with the keys in `.env`.

The console-backed tests (`run::tests::the_three_trees_…`, and the unmounted `live_run`) run with the rest:
the console is the one cortex embeds, so they need nothing installed. The two that need
FUSE-T are `#[ignore]`d — `live_workspace`, and the `live_run` test that goes through a
mounted workspace. Again from the repository root:

```sh
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
- **The workspace root is the user's home directory by default.** It is a local source
  like any other, repointed from its sheet in the sidebar, and the agent reads it — so
  out of the box the agent can read everything under `$HOME`. Point it somewhere
  narrower if that is not wanted.
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
