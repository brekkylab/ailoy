# Ailoy Desktop — Plan C: the Tauri app and the frontend

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the `ailoy-desktop-core` engine in a Tauri 2 app, build a chat UI with React + Tailwind + shadcn/ui in a three-column layout (sessions / thread / workspace), and make the E2E scenario work against a real key.

**Architecture:** `src-tauri` holds an `Arc<Engine>` as `State` and exposes nothing but thin `#[tauri::command]`s. Run events flow over a per-run `tauri::ipc::Channel<RunEvent>`, and the frontend builds stream state from `api.ts` (the single path through `invoke`) + `events.ts` (Channel→zustand) + the pure reducer `applyRunEvent`. Lists and settings are cached by TanStack Query, and a `Message` event invalidates that session's message query.

**Tech Stack:** Tauri 2.11(`tauri`, `tauri-build`, `tauri-plugin-dialog`), React 19, Vite 8, TypeScript 5.9+, Tailwind v4(`@tailwindcss/vite`), shadcn/ui, @tanstack/react-query 5, zustand 5, react-markdown 10 + remark-gfm 4, react-shiki, lucide-react, vitest 5

**Spec:** `docs/superpowers/specs/2026-09-11-ailoy-desktop-design.md` §7, §8, §10, §11

## Global Constraints

- The `feat/desktop` branch with Plan A and B done. `../cortex` is on `feat/exec-timeout`.
- `apps/desktop/src-tauri` is **its own cargo workspace** (it sits in the root `exclude`). Run Rust commands from that directory: `cargo check --manifest-path apps/desktop/src-tauri/Cargo.toml`.
- Run npm commands from `apps/desktop`. The package manager is npm (lockfile `package-lock.json`).
- Frontend → Rust calls go only through the functions in `src/api.ts`. Components never call `invoke` directly.
- UI strings live in `src/strings.ts` (in English). No literal sentences in components.
- Credentials and API keys never reach a log or the console.
- Commit prefixes: `feat(desktop-app): …`, `feat(desktop-ui): …`.

---

## File layout

| Path | Responsibility |
|---|---|
| `apps/desktop/package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `components.json` | Frontend scaffold |
| `apps/desktop/src/main.tsx`, `App.tsx`, `index.css` | Entry point, layout, theme |
| `src/types.ts` | Mirror of the engine types |
| `src/api.ts` | invoke wrappers |
| `src/events.ts` | Channel → store |
| `src/store/runs.ts`, `src/store/runs.test.ts` | Live run state and the pure reducer |
| `src/strings.ts` | UI strings |
| `src/lib/toolCall.ts`, `src/lib/toolCall.test.ts` | Parser for displaying tool call arguments and results |
| `src/components/Sidebar.tsx`, `Thread.tsx`, `MessageBubble.tsx`, `ToolCallCard.tsx`, `Composer.tsx`, `UsageBar.tsx`, `Markdown.tsx`, `WorkspacePanel.tsx`, `FileTree.tsx`, `MountDialogs.tsx`, `SettingsDialog.tsx`, `Banner.tsx` | Screens |
| `src/components/ui/*` | shadcn output |
| `apps/desktop/src-tauri/Cargo.toml`, `build.rs`, `tauri.conf.json`, `capabilities/default.json`, `icons/*` | Tauri scaffold |
| `src-tauri/src/main.rs`, `lib.rs`, `sidecar.rs`, `logging.rs`, `commands/{mod,sessions,runs,workspace,settings}.rs` | Rust layer |
| `apps/desktop/scripts/build-sidecar.sh` | Build the cortex console → `src-tauri/binaries/` |

---

### Task C1: frontend scaffold (Vite + React + Tailwind + shadcn + vitest)

**Files:**
- Create: `apps/desktop/package.json`, `vite.config.ts`, `tsconfig.json`, `tsconfig.node.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/index.css`, `src/strings.ts`, `components.json`, `src/lib/utils.ts`, `vitest.config.ts`, `src/smoke.test.ts`

- [ ] **Step 1: create the scaffold**

```bash
mkdir -p apps/desktop && cd apps/desktop
npm create vite@latest . -- --template react-ts
npm install
npm install @tauri-apps/api@^2.11 @tauri-apps/plugin-dialog@^2.7 @tanstack/react-query@^5 zustand@^5 react-markdown@^10 remark-gfm@^4 react-shiki@^0.11 lucide-react clsx tailwind-merge class-variance-authority
npm install -D tailwindcss@^4 @tailwindcss/vite@^4 vitest@^5 @tauri-apps/cli@^2.11 @types/node
```

- [ ] **Step 2: Vite/TS configuration**

`vite.config.ts`:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// Tauri drives this dev server: fixed port, fail if taken.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**"] } },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
});
```

`vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";
import path from "node:path";
export default defineConfig({
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
```

Add `"baseUrl": ".", "paths": { "@/*": ["./src/*"] }` to `compilerOptions` in `tsconfig.json`. `package.json` scripts:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "tauri": "tauri",
    "test": "vitest run",
    "typecheck": "tsc -b --noEmit"
  }
}
```

- [ ] **Step 3: Tailwind v4 + shadcn**

`src/index.css`:

```css
@import "tailwindcss";
@import "tw-animate-css";

@custom-variant dark (&:is(.dark *));

:root { color-scheme: light dark; }
html, body, #root { height: 100%; margin: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
```

```bash
npx shadcn@latest init -d
npx shadcn@latest add button input textarea dialog scroll-area select badge progress tooltip collapsible separator label switch dropdown-menu
```

(`init -d` generates `components.json`, `src/lib/utils.ts` (`cn`) and the CSS tokens. If `tw-animate-css` is not added automatically, `npm i tw-animate-css`.)

- [ ] **Step 4: a minimal app and the strings file**

`src/strings.ts`:

```ts
export const S = {
  appName: "Ailoy",
  newChat: "New chat",
  untitled: "New chat",
  send: "Send",
  stop: "Stop",
  composerPlaceholder: "Type a message. Enter sends it, Shift+Enter starts a new line.",
  thinking: "Thinking",
  running: "Running",
  interrupted: "Interrupted",
  denied: "Denied",
  toolCall: "Tool call",
  workspace: "Workspace",
  mounts: "Connections",
  addMount: "+ Connect",
  connectLocal: "Connect folder",
  connectNotion: "Connect Notion",
  connectS3: "Connect S3",
  remove: "Remove",
  settings: "Settings",
  providers: "Model providers",
  apiKey: "API key",
  saveKey: "Save",
  clearKey: "Clear",
  defaultModel: "Default model",
  maxTokens: "Max response tokens",
  maxTurns: "Turn limit",
  catalogRefresh: "Refresh the model list automatically",
  contextUsage: "Context",
  sessionTokens: "Session tokens",
  estimatedCost: "Estimated cost",
  rateLimit: "Rate limit",
  resetIn: "resets in",
  degraded: "The workspace is not mounted, so the agent cannot see your connections.",
  noKey: "Add an API key for the provider you want to use in Settings first.",
  readOnly: "Read-only",
  rename: "Rename",
  delete: "Delete",
  confirmDelete: "Delete this chat?",
  path: "Path (e.g. /notion)",
  label: "Display name",
  cancel: "Cancel",
  connect: "Connect",
  continueRun: "Continue",
  errorPrefix: "Error",
  fileTooLarge: "Previews stop at 1 MiB.",
  binaryFile: "Not a text file.",
  empty: "Empty",
} as const;
```

`src/App.tsx` (a temporary skeleton, replaced in C5):

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { S } from "@/strings";

const qc = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <div className="h-full grid place-items-center text-muted-foreground">{S.appName}</div>
    </QueryClientProvider>
  );
}
```

`src/smoke.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { S } from "@/strings";
describe("strings", () => { it("has an app name", () => { expect(S.appName).toBe("Ailoy"); }); });
```

- [ ] **Step 5: verify and commit**

```bash
npm run typecheck && npm run build && npm test
cd ../.. && git add apps/desktop && git commit -m "feat(desktop-ui): Vite/React/Tailwind/shadcn scaffold with vitest"
```

(Add `apps/desktop/node_modules`, `apps/desktop/dist`, `apps/desktop/src-tauri/target`, `apps/desktop/src-tauri/gen` and `apps/desktop/src-tauri/binaries` to `.gitignore`.)

---

### Task C2: Tauri scaffold and engine bootstrap

**Files:**
- Create: `apps/desktop/src-tauri/Cargo.toml`, `build.rs`, `tauri.conf.json`, `capabilities/default.json`, `icons/*`, `src/main.rs`, `src/lib.rs`, `src/sidecar.rs`, `src/logging.rs`, `src/commands/mod.rs`

**Interfaces:**
- Produces: an app with `tauri::State<'_, Arc<Engine>>` registered. In `commands::mod`, `pub type Eng<'a> = tauri::State<'a, Arc<Engine>>;`

- [ ] **Step 1: manifest and configuration**

`src-tauri/Cargo.toml`:

```toml
[package]
name = "ailoy-desktop"
version = "0.1.0"
edition = "2024"
publish = false
description = "Ailoy Desktop"

# Its own workspace: Tauri's webview stack must not be charged to the root `cargo test`.
[workspace]

[lib]
name = "ailoy_desktop_lib"
crate-type = ["lib", "cdylib", "staticlib"]

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
tauri-plugin-dialog = "2"
ailoy-desktop-core = { path = "../core" }
ailoy = { path = "../../.." }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-appender = "0.2"
```

`build.rs`: `fn main() { tauri_build::build() }`

`tauri.conf.json`:

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Ailoy",
  "version": "0.1.0",
  "identifier": "com.brekkylab.ailoy",
  "build": {
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:1420",
    "beforeBuildCommand": "npm run build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      { "label": "main", "title": "Ailoy", "width": 1320, "height": 860, "minWidth": 980, "minHeight": 640, "dragDropEnabled": true }
    ],
    "security": {
      "csp": "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: asset: http://asset.localhost; font-src 'self' data:"
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": ["icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png", "icons/icon.icns", "icons/icon.ico"],
    "externalBin": ["binaries/cortex-local-console"]
  }
}
```

`capabilities/default.json`:

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Core surface plus the folder picker used to connect a local directory.",
  "windows": ["main"],
  "permissions": ["core:default", "dialog:default"]
}
```

Icons (reuse the ones from the cortex-gui PoC):

```bash
mkdir -p apps/desktop/src-tauri/icons
for f in 32x32.png 128x128.png 128x128@2x.png icon.icns icon.ico icon.png; do
  git -C ../cortex show origin/cortex-gui:cortex-gui/src-tauri/icons/$f > apps/desktop/src-tauri/icons/$f
done
```

`externalBin` needs a `binaries/cortex-local-console-<target-triple>` file at build time, so until C10 either empty the `externalBin` line in `tauri.conf.json` for now (`[]`) or run C10's script once first.

- [ ] **Step 2: the Rust entry point**

`src/main.rs`:

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    ailoy_desktop_lib::run()
}
```

`src/logging.rs`:

```rust
use std::path::Path;

/// File logging under the app data dir; `RUST_LOG` filters, default `info`.
pub fn init(data_dir: &Path) {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
    let logs = data_dir.join("logs");
    let _ = std::fs::create_dir_all(&logs);
    let file = tracing_appender::rolling::daily(logs, "ailoy.log");
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(file).with_ansi(false))
        .with(fmt::layer().with_writer(std::io::stderr))
        .try_init();
}
```

`src/sidecar.rs`:

```rust
use std::path::PathBuf;

/// The console server beside our own executable (where Tauri puts a sidecar), or wherever
/// `AILOY_CORTEX_BIN_DIR` says. `None` lets the engine search the sibling checkout.
pub fn console_bin() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("AILOY_CORTEX_BIN_DIR") {
        let p = PathBuf::from(dir).join("cortex-local-console");
        if p.is_file() {
            return Some(p);
        }
    }
    let exe = std::env::current_exe().ok()?;
    let beside = exe.parent()?.join("cortex-local-console");
    beside.is_file().then_some(beside)
}
```

`src/commands/mod.rs`:

```rust
use std::sync::Arc;

use ailoy_desktop_core::Engine;

pub mod runs;
pub mod sessions;
pub mod settings;
pub mod workspace;

pub type Eng<'a> = tauri::State<'a, Arc<Engine>>;
```

(The `runs/sessions/settings/workspace` files are written in C3, but create them empty so this Task compiles.)

`src/lib.rs`:

```rust
//! The window over the engine. Commands are thin; everything they answer comes from
//! `ailoy_desktop_core::Engine`.

mod commands;
mod logging;
mod sidecar;

use std::sync::Arc;

use ailoy_desktop_core::{Engine, EngineConfig};
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            logging::init(&data_dir);
            let mut cfg = EngineConfig::new(data_dir);
            cfg.console_bin = sidecar::console_bin();
            // Setup runs on the main thread; the engine's start is short (open the DB,
            // mount, register providers) and must finish before any command can arrive.
            let engine = tauri::async_runtime::block_on(Engine::start(cfg))?;
            app.manage(engine);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::sessions::session_list,
            commands::sessions::session_create,
            commands::sessions::session_rename,
            commands::sessions::session_set_model,
            commands::sessions::session_delete,
            commands::sessions::message_list,
            commands::sessions::session_usage,
            commands::runs::run_start,
            commands::runs::run_attach,
            commands::runs::run_cancel,
            commands::workspace::workspace_info,
            commands::workspace::fs_list,
            commands::workspace::fs_read,
            commands::workspace::fs_write,
            commands::workspace::fs_mkdir,
            commands::workspace::fs_delete,
            commands::workspace::fs_rename,
            commands::workspace::fs_import,
            commands::workspace::mount_list,
            commands::workspace::mount_add,
            commands::workspace::mount_remove,
            commands::settings::settings_get,
            commands::settings::settings_set,
            commands::settings::models_list,
            commands::settings::open_logs,
        ])
        .build(tauri::generate_context!())
        .expect("the window could not be created")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(engine) = app.try_state::<Arc<Engine>>() {
                    let engine = engine.inner().clone();
                    // Unmounting joins a thread; do it before the process goes.
                    tauri::async_runtime::block_on(engine.shutdown());
                }
            }
        });
}
```

The commands listed in `generate_handler!` are implemented in C3. Check that this Task compiles after C3 (the two may go in one commit).

- [ ] **Step 3: commit (scaffold)**

```bash
git add apps/desktop/src-tauri .gitignore && git commit -m "feat(desktop-app): Tauri scaffold, engine bootstrap, logging, sidecar lookup"
```

---

### Task C3: Tauri commands

**Files:**
- Create: `src-tauri/src/commands/sessions.rs`, `runs.rs`, `workspace.rs`, `settings.rs`

**Interfaces:**
- Produces (the names, arguments and return values the frontend calls; argument names arrive as camelCase):
  | command | args | returns |
  |---|---|---|
  | `session_list` | — | `SessionSummary[]` |
  | `session_create` | `{ model?: string }` | `SessionSummary` |
  | `session_rename` | `{ id, title }` | `void` |
  | `session_set_model` | `{ id, model }` | `void` |
  | `session_delete` | `{ id }` | `void` |
  | `message_list` | `{ sessionId }` | `StoredMessage[]` |
  | `session_usage` | `{ sessionId }` | `SessionUsage` |
  | `run_start` | `{ sessionId, text, onEvent: Channel<RunEvent> }` | `string` (run_id) |
  | `run_attach` | `{ sessionId, onEvent }` | `string \| null` |
  | `run_cancel` | `{ sessionId }` | `boolean` |
  | `workspace_info` | — | `WorkspaceInfo` |
  | `fs_list` `{ path }` / `fs_read` `{ path }` / `fs_write` `{ path, text }` / `fs_mkdir` `{ path }` / `fs_delete` `{ path }` / `fs_rename` `{ from, to }` / `fs_import` `{ dest, sources: string[] }` | | `Entry[]` / `FileContent` / `void` / `void` / `void` / `void` / `ImportReport` |
  | `mount_list` | — | `MountInfo[]` |
  | `mount_add` | `{ req: MountRequest }` | `MountInfo` |
  | `mount_remove` | `{ path }` | `void` |
  | `settings_get` | — | `Settings` |
  | `settings_set` | `{ patch: SettingsPatch }` | `Settings` |
  | `models_list` | — | `ModelInfo[]` |
  | `open_logs` | — | `void` (open the log folder in Finder) |

- [ ] **Step 1: test for the event forwarding function** (at the bottom of `runs.rs`)

```rust
#[cfg(test)]
mod tests {
    use ailoy_desktop_core::RunEvent;
    use tokio::sync::broadcast;

    use super::forward;

    #[tokio::test]
    async fn forward_stops_after_a_terminal_event() {
        let (tx, rx) = broadcast::channel(16);
        tx.send(RunEvent::TextDelta { text: "a".into() }).unwrap();
        tx.send(RunEvent::Done).unwrap();
        tx.send(RunEvent::TextDelta { text: "late".into() }).unwrap();
        let mut seen = Vec::new();
        forward(rx, |ev| seen.push(ev)).await;
        assert_eq!(seen.len(), 2);
        assert!(matches!(seen[1], RunEvent::Done));
    }

    #[tokio::test]
    async fn forward_ends_when_the_sender_is_dropped() {
        let (tx, rx) = broadcast::channel(16);
        tx.send(RunEvent::TextDelta { text: "a".into() }).unwrap();
        drop(tx);
        let mut n = 0;
        forward(rx, |_| n += 1).await;
        assert_eq!(n, 1);
    }
}
```

- [ ] **Step 2: implementation**

`commands/runs.rs`:

```rust
use ailoy::message::Part;
use ailoy_desktop_core::{EngineError, RunEvent};
use tauri::ipc::Channel;
use tokio::sync::broadcast;

use super::Eng;

/// Pump a run's broadcast into `sink` until a terminal event or the run is gone. A lagged
/// receiver skips ahead: the UI reconciles from `message_list`, which every completed
/// message already reached.
pub async fn forward(mut rx: broadcast::Receiver<RunEvent>, mut sink: impl FnMut(RunEvent)) {
    loop {
        match rx.recv().await {
            Ok(ev) => {
                let terminal = matches!(ev, RunEvent::Done | RunEvent::Cancelled | RunEvent::Error { .. });
                sink(ev);
                if terminal {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("run event channel lagged by {n}");
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

#[tauri::command]
pub async fn run_start(engine: Eng<'_>, session_id: String, text: String, on_event: Channel<RunEvent>) -> Result<String, EngineError> {
    let handle = engine.run_start(&session_id, vec![Part::text(text)]).await?;
    let run_id = handle.run_id.clone();
    tauri::async_runtime::spawn(forward(handle.events, move |ev| {
        let _ = on_event.send(ev);
    }));
    Ok(run_id)
}

#[tauri::command]
pub async fn run_attach(engine: Eng<'_>, session_id: String, on_event: Channel<RunEvent>) -> Result<Option<String>, EngineError> {
    let Some((handle, partial)) = engine.run_attach(&session_id).await else {
        return Ok(None);
    };
    let run_id = handle.run_id.clone();
    let _ = on_event.send(RunEvent::Started { run_id: run_id.clone() });
    if !partial.is_empty() {
        let _ = on_event.send(RunEvent::TextDelta { text: partial });
    }
    tauri::async_runtime::spawn(forward(handle.events, move |ev| {
        let _ = on_event.send(ev);
    }));
    Ok(Some(run_id))
}

#[tauri::command]
pub async fn run_cancel(engine: Eng<'_>, session_id: String) -> Result<bool, EngineError> {
    Ok(engine.run_cancel(&session_id).await)
}
```

`commands/sessions.rs`:

```rust
use ailoy_desktop_core::{EngineError, SessionSummary, SessionUsage, StoredMessage};

use super::Eng;

#[tauri::command]
pub async fn session_list(engine: Eng<'_>) -> Result<Vec<SessionSummary>, EngineError> { engine.session_list().await }

#[tauri::command]
pub async fn session_create(engine: Eng<'_>, model: Option<String>) -> Result<SessionSummary, EngineError> { engine.session_create(model).await }

#[tauri::command]
pub async fn session_rename(engine: Eng<'_>, id: String, title: String) -> Result<(), EngineError> { engine.session_rename(&id, &title).await }

#[tauri::command]
pub async fn session_set_model(engine: Eng<'_>, id: String, model: String) -> Result<(), EngineError> { engine.session_set_model(&id, &model).await }

#[tauri::command]
pub async fn session_delete(engine: Eng<'_>, id: String) -> Result<(), EngineError> { engine.session_delete(&id).await }

#[tauri::command]
pub async fn message_list(engine: Eng<'_>, session_id: String) -> Result<Vec<StoredMessage>, EngineError> { engine.message_list(&session_id).await }

#[tauri::command]
pub async fn session_usage(engine: Eng<'_>, session_id: String) -> Result<SessionUsage, EngineError> { engine.session_usage(&session_id).await }
```

`commands/workspace.rs`:

```rust
use std::path::PathBuf;

use ailoy_desktop_core::{EngineError, Entry, FileContent, ImportReport, MountInfo, MountRequest, WorkspaceInfo};

use super::Eng;

#[tauri::command]
pub async fn workspace_info(engine: Eng<'_>) -> Result<WorkspaceInfo, EngineError> { Ok(engine.workspace_info()) }
#[tauri::command]
pub async fn fs_list(engine: Eng<'_>, path: String) -> Result<Vec<Entry>, EngineError> { engine.fs_list(&path).await }
#[tauri::command]
pub async fn fs_read(engine: Eng<'_>, path: String) -> Result<FileContent, EngineError> { engine.fs_read(&path).await }
#[tauri::command]
pub async fn fs_write(engine: Eng<'_>, path: String, text: String) -> Result<(), EngineError> { engine.fs_write(&path, &text).await }
#[tauri::command]
pub async fn fs_mkdir(engine: Eng<'_>, path: String) -> Result<(), EngineError> { engine.fs_mkdir(&path).await }
#[tauri::command]
pub async fn fs_delete(engine: Eng<'_>, path: String) -> Result<(), EngineError> { engine.fs_delete(&path).await }
#[tauri::command]
pub async fn fs_rename(engine: Eng<'_>, from: String, to: String) -> Result<(), EngineError> { engine.fs_rename(&from, &to).await }
#[tauri::command]
pub async fn fs_import(engine: Eng<'_>, dest: String, sources: Vec<PathBuf>) -> Result<ImportReport, EngineError> { engine.fs_import(&dest, sources).await }
#[tauri::command]
pub async fn mount_list(engine: Eng<'_>) -> Result<Vec<MountInfo>, EngineError> { Ok(engine.mount_list().await) }
#[tauri::command]
pub async fn mount_add(engine: Eng<'_>, req: MountRequest) -> Result<MountInfo, EngineError> { engine.mount_add(req).await }
#[tauri::command]
pub async fn mount_remove(engine: Eng<'_>, path: String) -> Result<(), EngineError> { engine.mount_remove(&path).await }
```

`commands/settings.rs`:

```rust
use ailoy_desktop_core::{EngineError, ModelInfo, Settings, SettingsPatch};

use super::Eng;

#[tauri::command]
pub async fn settings_get(engine: Eng<'_>) -> Result<Settings, EngineError> { engine.settings_get().await }
#[tauri::command]
pub async fn settings_set(engine: Eng<'_>, patch: SettingsPatch) -> Result<Settings, EngineError> { engine.settings_set(patch).await }
#[tauri::command]
pub fn models_list(engine: Eng<'_>) -> Result<Vec<ModelInfo>, EngineError> { engine.models_list() }

/// Reveal the log directory in Finder.
#[tauri::command]
pub fn open_logs(engine: Eng<'_>) -> Result<(), EngineError> {
    let logs = engine.config().data_dir.join("logs");
    std::process::Command::new("open").arg(logs).spawn().map_err(EngineError::Io)?;
    Ok(())
}
```

- [ ] **Step 3: verify and commit**

```bash
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml 2>&1 | grep -E 'test result|FAILED|error'
git add apps/desktop/src-tauri && git commit -m "feat(desktop-app): Tauri commands over the engine with channel-forwarded run events"
```

(The first compile takes several minutes because of the webview dependency.) If `tauri-build` fails over `externalBin`, do C10 first or empty it to `externalBin: []` for now.

---

### Task C4: types, API layer, events, run store (reducer + tests)

**Files:**
- Create: `src/types.ts`, `src/api.ts`, `src/events.ts`, `src/store/runs.ts`, `src/store/runs.test.ts`

**Interfaces:**
- Produces: the `api.*` functions, `startRun(sessionId, text)`, `attachRun(sessionId)`, `cancelRun(sessionId)`, `useRunStore` (a `LiveRun` per session), the pure `applyRunEvent(state, ev): LiveRun`, `emptyRun(): LiveRun`

- [ ] **Step 1: `src/types.ts`** (mirrors the engine types; includes the ailoy Message JSON)

```ts
export type Role = "system" | "user" | "assistant" | "tool";

export type Part =
  | { type: "text"; text: string }
  | { type: "function"; id: string; function: { name: string; arguments: unknown } }
  | { type: "value"; value: unknown }
  | { type: "image"; image: { type: "embedded"; mime_type: string; data: string } | { type: "url"; url: string } };

export interface Message {
  role: Role;
  contents: Part[];
  thinking?: string;
  tool_calls?: Part[];
  id?: string;
  signature?: string;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
}

export interface RateLimitWindow { limit: number | null; remaining: number | null; reset_at_ms: number | null }
export interface RateLimitInfo {
  requests?: RateLimitWindow | null;
  tokens?: RateLimitWindow | null;
  input_tokens?: RateLimitWindow | null;
  output_tokens?: RateLimitWindow | null;
}

export interface SessionSummary { id: string; title: string; model: string; created_at: number; updated_at: number; running: boolean }
export interface StoredMessage { seq: number; depth: number; source_agent: string | null; message: Message; usage: TokenUsage | null; created_at: number }
export interface SessionUsage {
  input_tokens: number; output_tokens: number; cache_read_tokens: number; cache_write_tokens: number;
  estimated_cost_usd: number | null; context_used: number | null; context_limit: number | null;
}

export type RunEvent =
  | { type: "started"; run_id: string }
  | { type: "text_delta"; text: string }
  | { type: "thinking_delta"; text: string }
  | { type: "tool_call_started"; id: string; name: string; arguments: unknown }
  | { type: "message"; seq: number; depth: number; source_agent: string | null; message: Message; usage: TokenUsage | null }
  | { type: "usage"; usage: TokenUsage | null; rate_limit: RateLimitInfo | null; context_used: number | null; context_limit: number | null }
  | { type: "awaiting_approval"; id: string; name: string; arguments: unknown }
  | { type: "done" }
  | { type: "cancelled" }
  | { type: "error"; kind: string; message: string };

export type MountKind = "root" | "local" | "notion" | "s3";
export type MountStatus = { status: "ok" } | { status: "error"; message: string };
export interface MountInfo { id: string; path: string; kind: MountKind; label: string; detail: string; writable: boolean; status: MountStatus }
export type MountConfig =
  | { kind: "local"; host_root: string }
  | { kind: "notion"; api_key: string }
  | { kind: "s3"; bucket: string; region: string; access_key_id: string; secret_access_key: string; endpoint: string | null; key_prefix: string | null };
export interface MountRequest { path: string; label: string | null; config: MountConfig }

export type WorkspaceStatus = { status: "mounted" } | { status: "degraded"; reason: string };
export interface WorkspaceInfo { mountpoint: string; files_root: string; status: WorkspaceStatus }

export interface Entry { name: string; path: string; kind: "dir" | "file"; size: number | null; mtime_ms: number | null }
export interface FileContent { path: string; text: string | null; size: number; truncated: boolean }
export interface ImportReport { files: number; bytes: number; skipped: string[] }

export interface ProviderSetting { key: string; label: string; has_key: boolean; key_hint: string; region: string | null }
export interface Settings { providers: ProviderSetting[]; default_model: string; max_tokens: number; max_turns: number; catalog_refresh: boolean }
export interface SettingsPatch {
  provider_keys?: Record<string, string | null>;
  bedrock_region?: string;
  default_model?: string;
  max_tokens?: number;
  max_turns?: number;
  catalog_refresh?: boolean;
}
export interface ModelCost { input: number | null; output: number | null; cache_read: number | null; cache_write: number | null }
export interface ModelInfo { id: string; provider: string; name: string; context: number | null; output: number | null; cost: ModelCost | null; reasoning: boolean; tool_call: boolean; available: boolean }
```

- [ ] **Step 2: `src/api.ts`**

```ts
// Every call into Rust, in one place and typed once.
import { Channel, invoke } from "@tauri-apps/api/core";
import type * as T from "./types";

export const sessionList = () => invoke<T.SessionSummary[]>("session_list");
export const sessionCreate = (model?: string) => invoke<T.SessionSummary>("session_create", { model: model ?? null });
export const sessionRename = (id: string, title: string) => invoke<void>("session_rename", { id, title });
export const sessionSetModel = (id: string, model: string) => invoke<void>("session_set_model", { id, model });
export const sessionDelete = (id: string) => invoke<void>("session_delete", { id });
export const messageList = (sessionId: string) => invoke<T.StoredMessage[]>("message_list", { sessionId });
export const sessionUsage = (sessionId: string) => invoke<T.SessionUsage>("session_usage", { sessionId });

export const runStart = (sessionId: string, text: string, onEvent: Channel<T.RunEvent>) =>
  invoke<string>("run_start", { sessionId, text, onEvent });
export const runAttach = (sessionId: string, onEvent: Channel<T.RunEvent>) =>
  invoke<string | null>("run_attach", { sessionId, onEvent });
export const runCancel = (sessionId: string) => invoke<boolean>("run_cancel", { sessionId });

export const workspaceInfo = () => invoke<T.WorkspaceInfo>("workspace_info");
export const fsList = (path: string) => invoke<T.Entry[]>("fs_list", { path });
export const fsRead = (path: string) => invoke<T.FileContent>("fs_read", { path });
export const fsWrite = (path: string, text: string) => invoke<void>("fs_write", { path, text });
export const fsMkdir = (path: string) => invoke<void>("fs_mkdir", { path });
export const fsDelete = (path: string) => invoke<void>("fs_delete", { path });
export const fsRename = (from: string, to: string) => invoke<void>("fs_rename", { from, to });
export const fsImport = (dest: string, sources: string[]) => invoke<T.ImportReport>("fs_import", { dest, sources });
export const mountList = () => invoke<T.MountInfo[]>("mount_list");
export const mountAdd = (req: T.MountRequest) => invoke<T.MountInfo>("mount_add", { req });
export const mountRemove = (path: string) => invoke<void>("mount_remove", { path });

export const settingsGet = () => invoke<T.Settings>("settings_get");
export const settingsSet = (patch: T.SettingsPatch) => invoke<T.Settings>("settings_set", { patch });
export const modelsList = () => invoke<T.ModelInfo[]>("models_list");
export const openLogs = () => invoke<void>("open_logs");

export function messageOf(err: unknown): string {
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;
  return String(err);
}
```

- [ ] **Step 3: reducer tests** — `src/store/runs.test.ts`

```ts
import { describe, expect, it } from "vitest";
import { applyRunEvent, emptyRun } from "./runs";
import type { Message, RunEvent } from "@/types";

const asst = (text: string, toolCalls?: Message["tool_calls"]): Message => ({ role: "assistant", contents: [{ type: "text", text }], tool_calls: toolCalls });
const tool = (id: string, value: unknown): Message => ({ role: "tool", id, contents: [{ type: "value", value }] });

function run(events: RunEvent[]) {
  return events.reduce(applyRunEvent, emptyRun());
}

describe("applyRunEvent", () => {
  it("accumulates text and thinking while running", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "thinking_delta", text: "hmm " },
      { type: "text_delta", text: "Hel" },
      { type: "text_delta", text: "lo" },
    ]);
    expect(s.status).toBe("running");
    expect(s.runId).toBe("r1");
    expect(s.text).toBe("Hello");
    expect(s.thinking).toBe("hmm ");
  });

  it("clears live text when the assistant message is persisted and flags a refetch", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "Hello" },
      { type: "message", seq: 2, depth: 0, source_agent: null, message: asst("Hello"), usage: null },
    ]);
    expect(s.text).toBe("");
    expect(s.messagesDirty).toBe(true);
  });

  it("tracks tool calls from started to done with the tool result", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: { cmd: "ls" } },
      { type: "message", seq: 3, depth: 0, source_agent: null, message: tool("c1", { stdout: "a\n" }), usage: null },
    ]);
    expect(s.toolOrder).toEqual(["c1"]);
    expect(s.toolCalls.c1.status).toBe("done");
    expect(s.toolCalls.c1.result).toEqual({ stdout: "a\n" });
  });

  it("marks running tools interrupted on cancel and keeps text", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "partial" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: {} },
      { type: "cancelled" },
    ]);
    expect(s.status).toBe("cancelled");
    expect(s.toolCalls.c1.status).toBe("interrupted");
    expect(s.text).toBe("partial");
  });

  it("records usage and errors", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "usage", usage: { input_tokens: 10, output_tokens: 2 }, rate_limit: { requests: { limit: 100, remaining: 99, reset_at_ms: null } }, context_used: 10, context_limit: 1000 },
      { type: "error", kind: "model", message: "401" },
    ]);
    expect(s.status).toBe("error");
    expect(s.error).toEqual({ kind: "model", message: "401" });
    expect(s.contextUsed).toBe(10);
    expect(s.rateLimit?.requests?.remaining).toBe(99);
  });

  it("a new start resets the previous run's live state", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "old" },
      { type: "done" },
      { type: "started", run_id: "r2" },
    ]);
    expect(s.runId).toBe("r2");
    expect(s.text).toBe("");
    expect(s.toolOrder).toEqual([]);
  });
});
```

- [ ] **Step 4: store implementation** — `src/store/runs.ts`

```ts
import { create } from "zustand";
import type { RateLimitInfo, RunEvent, TokenUsage } from "@/types";

export type ToolStatus = "running" | "done" | "error" | "interrupted";
export interface ToolCallState { id: string; name: string; arguments: unknown; status: ToolStatus; result?: unknown; startedAt: number; finishedAt?: number }

export interface LiveRun {
  runId: string | null;
  status: "idle" | "running" | "done" | "cancelled" | "error";
  text: string;
  thinking: string;
  toolCalls: Record<string, ToolCallState>;
  toolOrder: string[];
  usage?: TokenUsage | null;
  rateLimit?: RateLimitInfo | null;
  contextUsed?: number | null;
  contextLimit?: number | null;
  error?: { kind: string; message: string };
  /** Set when a persisted message arrived; the thread refetches `message_list` and clears it. */
  messagesDirty: boolean;
}

export const emptyRun = (): LiveRun => ({ runId: null, status: "idle", text: "", thinking: "", toolCalls: {}, toolOrder: [], messagesDirty: false });

function toolResultValue(msg: { contents: { type: string; [k: string]: unknown }[] }): unknown {
  const first = msg.contents[0];
  if (!first) return null;
  if (first.type === "value") return first.value;
  if (first.type === "text") return first.text;
  return first;
}

export function applyRunEvent(s: LiveRun, ev: RunEvent): LiveRun {
  switch (ev.type) {
    case "started":
      return { ...emptyRun(), runId: ev.run_id, status: "running" };
    case "text_delta":
      return { ...s, text: s.text + ev.text };
    case "thinking_delta":
      return { ...s, thinking: s.thinking + ev.text };
    case "tool_call_started":
      return {
        ...s,
        toolCalls: { ...s.toolCalls, [ev.id]: { id: ev.id, name: ev.name, arguments: ev.arguments, status: "running", startedAt: Date.now() } },
        toolOrder: s.toolOrder.includes(ev.id) ? s.toolOrder : [...s.toolOrder, ev.id],
      };
    case "message": {
      const m = ev.message;
      if (m.role === "tool" && m.id && s.toolCalls[m.id]) {
        const value = toolResultValue(m);
        const isError = typeof value === "object" && value !== null && "error" in (value as Record<string, unknown>);
        return { ...s, messagesDirty: true, toolCalls: { ...s.toolCalls, [m.id]: { ...s.toolCalls[m.id], status: isError ? "error" : "done", result: value, finishedAt: Date.now() } } };
      }
      if (m.role === "assistant" && ev.depth === 0) {
        return { ...s, text: "", thinking: "", messagesDirty: true };
      }
      return { ...s, messagesDirty: true };
    }
    case "usage":
      return { ...s, usage: ev.usage ?? s.usage, rateLimit: ev.rate_limit ?? s.rateLimit, contextUsed: ev.context_used ?? s.contextUsed, contextLimit: ev.context_limit ?? s.contextLimit };
    case "awaiting_approval":
      return s;
    case "done":
      return { ...s, status: "done" };
    case "cancelled":
      return { ...s, status: "cancelled", toolCalls: interruptRunning(s.toolCalls) };
    case "error":
      return { ...s, status: "error", error: { kind: ev.kind, message: ev.message }, toolCalls: interruptRunning(s.toolCalls) };
  }
}

function interruptRunning(calls: Record<string, ToolCallState>): Record<string, ToolCallState> {
  const out: Record<string, ToolCallState> = {};
  for (const [id, c] of Object.entries(calls)) out[id] = c.status === "running" ? { ...c, status: "interrupted", finishedAt: Date.now() } : c;
  return out;
}

interface RunStore {
  runs: Record<string, LiveRun>;
  apply: (sessionId: string, ev: RunEvent) => void;
  clearDirty: (sessionId: string) => void;
  reset: (sessionId: string) => void;
}

export const useRunStore = create<RunStore>((set) => ({
  runs: {},
  apply: (sessionId, ev) => set((st) => ({ runs: { ...st.runs, [sessionId]: applyRunEvent(st.runs[sessionId] ?? emptyRun(), ev) } })),
  clearDirty: (sessionId) => set((st) => (st.runs[sessionId] ? { runs: { ...st.runs, [sessionId]: { ...st.runs[sessionId], messagesDirty: false } } } : st)),
  reset: (sessionId) => set((st) => ({ runs: { ...st.runs, [sessionId]: emptyRun() } })),
}));

export const selectRun = (sessionId: string | null) => (st: RunStore) => (sessionId ? st.runs[sessionId] ?? emptyRun() : emptyRun());
```

- [ ] **Step 5: `src/events.ts`**

```ts
import { Channel } from "@tauri-apps/api/core";
import * as api from "./api";
import { useRunStore } from "./store/runs";
import type { RunEvent } from "./types";

function channelFor(sessionId: string): Channel<RunEvent> {
  const ch = new Channel<RunEvent>();
  ch.onmessage = (ev) => useRunStore.getState().apply(sessionId, ev);
  return ch;
}

export async function startRun(sessionId: string, text: string): Promise<string> {
  return api.runStart(sessionId, text, channelFor(sessionId));
}

/** After a reload: re-subscribe if a run is still going. Resolves to the run id or null. */
export async function attachRun(sessionId: string): Promise<string | null> {
  return api.runAttach(sessionId, channelFor(sessionId));
}

export const cancelRun = (sessionId: string) => api.runCancel(sessionId);
```

- [ ] **Step 6: verify and commit**

```bash
npm run typecheck && npm test
git add apps/desktop/src && git commit -m "feat(desktop-ui): types, api layer, run event channel and reducer with tests"
```

---

### Task C5: app shell and session sidebar

**Files:**
- Create: `src/components/Sidebar.tsx`, `src/components/Banner.tsx`
- Modify: `src/App.tsx`, `src/main.tsx`

**Interfaces:**
- Produces: `App` — a three-column grid (`Sidebar` 260px / `Thread` / `WorkspacePanel` 320px). The selected session id lives in `useState` + `localStorage("ailoy.session")`. `Sidebar` props `{ selected: string | null; onSelect(id): void; onOpenSettings(): void }`.

- [ ] **Step 1: `Sidebar.tsx`**

```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { MessageSquarePlus, Settings as SettingsIcon, Trash2, Pencil } from "lucide-react";
import * as api from "@/api";
import { S } from "@/strings";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export function Sidebar({ selected, onSelect, onOpenSettings }: { selected: string | null; onSelect: (id: string) => void; onOpenSettings: () => void }) {
  const qc = useQueryClient();
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessionList, refetchInterval: 5000 });
  const create = useMutation({
    mutationFn: () => api.sessionCreate(),
    onSuccess: (s) => { qc.invalidateQueries({ queryKey: ["sessions"] }); onSelect(s.id); },
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.sessionDelete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
  const rename = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => api.sessionRename(id, title),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });

  return (
    <aside className="flex h-full flex-col border-r bg-muted/30">
      <div className="flex items-center gap-2 p-3">
        <Button className="flex-1" onClick={() => create.mutate()} disabled={create.isPending}>
          <MessageSquarePlus className="mr-2 size-4" /> {S.newChat}
        </Button>
        <Button variant="ghost" size="icon" onClick={onOpenSettings} aria-label={S.settings}><SettingsIcon className="size-4" /></Button>
      </div>
      <ScrollArea className="flex-1 px-2">
        {(sessions.data ?? []).map((s) => (
          <div key={s.id} className={cn("group flex items-center gap-1 rounded-md px-2 py-1.5 text-sm hover:bg-accent", selected === s.id && "bg-accent")}>
            <button className="flex-1 truncate text-left" onClick={() => onSelect(s.id)} title={s.model}>
              {s.running && <span className="mr-1 inline-block size-2 animate-pulse rounded-full bg-emerald-500" />}
              {s.title}
            </button>
            <button className="hidden group-hover:block" aria-label={S.rename} onClick={() => { const t = window.prompt(S.rename, s.title); if (t && t.trim()) rename.mutate({ id: s.id, title: t.trim() }); }}><Pencil className="size-3.5" /></button>
            <button className="hidden group-hover:block" aria-label={S.delete} onClick={() => { if (window.confirm(S.confirmDelete)) remove.mutate(s.id); }}><Trash2 className="size-3.5" /></button>
          </div>
        ))}
        {sessions.data?.length === 0 && <p className="p-3 text-xs text-muted-foreground">{S.empty}</p>}
      </ScrollArea>
    </aside>
  );
}
```

- [ ] **Step 2: `Banner.tsx`** (the degraded / no-key notices)

```tsx
import { AlertTriangle } from "lucide-react";

export function Banner({ text, tone = "warn" }: { text: string; tone?: "warn" | "error" }) {
  return (
    <div className={tone === "error" ? "flex items-center gap-2 border-b bg-destructive/10 px-3 py-1.5 text-xs text-destructive" : "flex items-center gap-2 border-b bg-amber-500/10 px-3 py-1.5 text-xs text-amber-700 dark:text-amber-300"}>
      <AlertTriangle className="size-3.5" /> {text}
    </div>
  );
}
```

- [ ] **Step 3: `App.tsx`**

```tsx
import { useEffect, useState } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import * as api from "@/api";
import { S } from "@/strings";
import { Sidebar } from "@/components/Sidebar";
import { Thread } from "@/components/Thread";
import { WorkspacePanel } from "@/components/WorkspacePanel";
import { SettingsDialog } from "@/components/SettingsDialog";
import { Banner } from "@/components/Banner";

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });
const KEY = "ailoy.session";

function Shell() {
  const [selected, setSelected] = useState<string | null>(() => { try { return localStorage.getItem(KEY); } catch { return null; } });
  const [settingsOpen, setSettingsOpen] = useState(false);
  useEffect(() => { try { if (selected) localStorage.setItem(KEY, selected); } catch { /* storage may be unavailable */ } }, [selected]);
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo });
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
  const noKey = settings.data && !settings.data.providers.some((p) => p.has_key);

  return (
    <div className="grid h-full grid-cols-[260px_minmax(0,1fr)_320px]">
      <Sidebar selected={selected} onSelect={setSelected} onOpenSettings={() => setSettingsOpen(true)} />
      <main className="flex h-full min-w-0 flex-col">
        {ws.data?.status.status === "degraded" && <Banner text={`${S.degraded} (${ws.data.status.reason})`} />}
        {noKey && <Banner text={S.noKey} tone="error" />}
        <Thread sessionId={selected} />
      </main>
      <WorkspacePanel />
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <Shell />
    </QueryClientProvider>
  );
}
```

`Thread`, `WorkspacePanel` and `SettingsDialog` are written in C6–C9. To compile this Task, drop in an `export function X() { return null; }` stub for each first.

- [ ] **Step 4: verify and commit**

```bash
npm run typecheck
git add apps/desktop/src && git commit -m "feat(desktop-ui): app shell and session sidebar"
```

---

### Task C6: thread — messages, markdown, tool call cards, streaming

**Files:**
- Create: `src/components/Thread.tsx`, `MessageBubble.tsx`, `ToolCallCard.tsx`, `Markdown.tsx`, `src/lib/toolCall.ts`, `src/lib/toolCall.test.ts`

**Interfaces:**
- Produces: `Thread({ sessionId })` — renders stored messages plus live state, invalidates `["messages", id]` on every new `message` event, and calls `attachRun` on mount. `summarizeToolCall(name, args): string`, `fieldsOf(value): {key, value, kind}[]`.

- [ ] **Step 1: tool call parser tests** — `src/lib/toolCall.test.ts`

```ts
import { describe, expect, it } from "vitest";
import { fieldsOf, summarizeToolCall } from "./toolCall";

describe("toolCall helpers", () => {
  it("summarizes known tools by their primary argument", () => {
    expect(summarizeToolCall("shell", { cmd: "ls -la" })).toBe("ls -la");
    expect(summarizeToolCall("read", { path: "/a.txt" })).toBe("/a.txt");
    expect(summarizeToolCall("web_search", { query: "rust" })).toBe("rust");
    expect(summarizeToolCall("unknown", { a: 1 })).toBe('{"a":1}');
  });
  it("splits an object result into displayable fields", () => {
    const f = fieldsOf({ stdout: "line1\nline2", exit_code: 0, truncated: false });
    expect(f.map((x) => x.key)).toEqual(["stdout", "exit_code", "truncated"]);
    expect(f[0].kind).toBe("block");
    expect(f[1].kind).toBe("inline");
    expect(fieldsOf("plain")).toEqual([{ key: "", value: "plain", kind: "block" }]);
  });
});
```

- [ ] **Step 2: `src/lib/toolCall.ts`**

```ts
export type FieldKind = "inline" | "block";
export interface Field { key: string; value: string; kind: FieldKind }

const PRIMARY: Record<string, string> = { shell: "cmd", read: "path", write: "path", edit: "path", glob: "pattern", grep: "pattern", web_search: "query", web_fetch: "url" };

export function summarizeToolCall(name: string, args: unknown): string {
  if (args && typeof args === "object") {
    const key = PRIMARY[name];
    const v = key ? (args as Record<string, unknown>)[key] : undefined;
    if (typeof v === "string") return v;
  }
  try { return JSON.stringify(args); } catch { return String(args); }
}

export function fieldsOf(value: unknown): Field[] {
  if (value === null || value === undefined) return [];
  if (typeof value !== "object") return [{ key: "", value: String(value), kind: "block" }];
  if (Array.isArray(value)) return [{ key: "", value: JSON.stringify(value, null, 2), kind: "block" }];
  return Object.entries(value as Record<string, unknown>).map(([key, v]) => {
    if (typeof v === "string") return { key, value: v, kind: v.includes("\n") || v.length > 80 ? "block" : "inline" };
    if (typeof v === "number" || typeof v === "boolean") return { key, value: String(v), kind: "inline" };
    return { key, value: JSON.stringify(v, null, 2), kind: "block" };
  });
}
```

- [ ] **Step 3: `Markdown.tsx`**

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ShikiHighlighter from "react-shiki";

export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children }) {
            const lang = /language-(\w+)/.exec(className ?? "")?.[1];
            const code = String(children).replace(/\n$/, "");
            if (!lang) return <code className="rounded bg-muted px-1 py-0.5">{children}</code>;
            return <ShikiHighlighter language={lang} theme="github-dark" showLanguage={false}>{code}</ShikiHighlighter>;
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
```

(Check `react-shiki`'s default export name with `npm view react-shiki readme`. To use the `prose` classes, `npm i -D @tailwindcss/typography` and then `@plugin "@tailwindcss/typography";` in `index.css`.)

- [ ] **Step 4: `ToolCallCard.tsx`**

```tsx
import { ChevronRight, Loader2, CheckCircle2, XCircle, Ban } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { fieldsOf, summarizeToolCall } from "@/lib/toolCall";
import type { ToolStatus } from "@/store/runs";
import { S } from "@/strings";

export function ToolCallCard({ name, args, status, result, elapsedMs }: { name: string; args: unknown; status: ToolStatus; result?: unknown; elapsedMs?: number }) {
  const Icon = status === "running" ? Loader2 : status === "done" ? CheckCircle2 : status === "error" ? XCircle : Ban;
  return (
    <Collapsible className="my-1 rounded-md border bg-muted/40 text-sm">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left">
        <ChevronRight className="size-3.5 transition-transform group-data-[state=open]:rotate-90" />
        <Icon className={status === "running" ? "size-4 animate-spin" : "size-4"} />
        <span className="font-mono text-xs text-muted-foreground">{name}</span>
        <span className="flex-1 truncate font-mono text-xs">{summarizeToolCall(name, args)}</span>
        {status === "running" && <span className="text-xs text-muted-foreground">{S.running}{elapsedMs ? ` · ${Math.round(elapsedMs / 1000)}s` : ""}</span>}
        {status === "interrupted" && <span className="text-xs text-muted-foreground">{S.interrupted}</span>}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-2 border-t px-3 py-2">
        <pre className="max-h-40 overflow-auto rounded bg-background p-2 font-mono text-xs">{JSON.stringify(args, null, 2)}</pre>
        {result !== undefined && fieldsOf(result).map((f) => (
          f.kind === "inline"
            ? <div key={f.key} className="font-mono text-xs"><span className="text-muted-foreground">{f.key}: </span>{f.value}</div>
            : <div key={f.key}>{f.key && <div className="font-mono text-xs text-muted-foreground">{f.key}</div>}<pre className="max-h-64 overflow-auto rounded bg-background p-2 font-mono text-xs whitespace-pre-wrap">{f.value}</pre></div>
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}
```

- [ ] **Step 5: `MessageBubble.tsx`** (one stored turn: a user message, or an assistant message plus its tool calls)

```tsx
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Markdown } from "@/components/Markdown";
import { ToolCallCard } from "@/components/ToolCallCard";
import type { Message, StoredMessage } from "@/types";
import { S } from "@/strings";

function textOf(m: Message): string {
  return m.contents.map((p) => (p.type === "text" ? p.text : "")).join("");
}

export function UserBubble({ message }: { message: Message }) {
  return <div className="ml-auto max-w-[80%] rounded-2xl bg-primary px-4 py-2 text-primary-foreground whitespace-pre-wrap">{textOf(message)}</div>;
}

export function AssistantBubble({ message, toolResults }: { message: Message; toolResults: Map<string, StoredMessage> }) {
  const text = textOf(message);
  return (
    <div className="max-w-[92%] space-y-1">
      {message.thinking && (
        <Collapsible>
          <CollapsibleTrigger className="text-xs text-muted-foreground hover:underline">{S.thinking}</CollapsibleTrigger>
          <CollapsibleContent><pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted/40 p-2 text-xs">{message.thinking}</pre></CollapsibleContent>
        </Collapsible>
      )}
      {text && <Markdown text={text} />}
      {(message.tool_calls ?? []).map((p) => {
        if (p.type !== "function") return null;
        const res = toolResults.get(p.id);
        const value = res?.message.contents[0];
        const result = value?.type === "value" ? value.value : value?.type === "text" ? value.text : undefined;
        const isError = typeof result === "object" && result !== null && "error" in (result as Record<string, unknown>);
        return <ToolCallCard key={p.id} name={p.function.name} args={p.function.arguments} status={res ? (isError ? "error" : "done") : "interrupted"} result={result} />;
      })}
    </div>
  );
}
```

- [ ] **Step 6: `Thread.tsx`**

```tsx
import { useEffect, useMemo, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "@/api";
import { attachRun } from "@/events";
import { selectRun, useRunStore } from "@/store/runs";
import { AssistantBubble, UserBubble } from "@/components/MessageBubble";
import { ToolCallCard } from "@/components/ToolCallCard";
import { Markdown } from "@/components/Markdown";
import { Composer } from "@/components/Composer";
import { S } from "@/strings";
import type { StoredMessage } from "@/types";

export function Thread({ sessionId }: { sessionId: string | null }) {
  const qc = useQueryClient();
  const live = useRunStore(selectRun(sessionId));
  const clearDirty = useRunStore((s) => s.clearDirty);
  const messages = useQuery({ queryKey: ["messages", sessionId], queryFn: () => api.messageList(sessionId!), enabled: !!sessionId });
  const bottom = useRef<HTMLDivElement>(null);

  // Re-subscribe after a reload while a run is still going.
  useEffect(() => { if (sessionId) attachRun(sessionId).catch(() => {}); }, [sessionId]);

  // A persisted message invalidates the list; the live bubble already cleared itself.
  useEffect(() => {
    if (sessionId && live.messagesDirty) {
      qc.invalidateQueries({ queryKey: ["messages", sessionId] });
      qc.invalidateQueries({ queryKey: ["usage", sessionId] });
      qc.invalidateQueries({ queryKey: ["sessions"] });
      clearDirty(sessionId);
    }
  }, [sessionId, live.messagesDirty, qc, clearDirty]);

  useEffect(() => { bottom.current?.scrollIntoView({ block: "end" }); }, [messages.data?.length, live.text, live.thinking, live.toolOrder.length]);

  const { turns, toolResults } = useMemo(() => {
    const toolResults = new Map<string, StoredMessage>();
    const turns: StoredMessage[] = [];
    for (const m of messages.data ?? []) {
      if (m.depth !== 0) continue;                      // sub-agent internals stay hidden in v1
      if (m.message.role === "tool" && m.message.id) { toolResults.set(m.message.id, m); continue; }
      if (m.message.role === "system") continue;
      turns.push(m);
    }
    return { turns, toolResults };
  }, [messages.data]);

  if (!sessionId) return <div className="grid flex-1 place-items-center text-muted-foreground">{S.newChat}</div>;

  const streaming = live.status === "running";
  return (
    <>
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {turns.map((m) => m.message.role === "user"
            ? <UserBubble key={m.seq} message={m.message} />
            : <AssistantBubble key={m.seq} message={m.message} toolResults={toolResults} />)}
          {(streaming || live.text || live.toolOrder.length > 0) && (
            <div className="max-w-[92%] space-y-1">
              {live.thinking && <pre className="max-h-32 overflow-auto whitespace-pre-wrap rounded bg-muted/40 p-2 text-xs text-muted-foreground">{live.thinking}</pre>}
              {live.text && <Markdown text={live.text} />}
              {live.toolOrder.map((id) => { const c = live.toolCalls[id]; return <ToolCallCard key={id} name={c.name} args={c.arguments} status={c.status} result={c.result} elapsedMs={(c.finishedAt ?? Date.now()) - c.startedAt} />; })}
              {streaming && !live.text && live.toolOrder.length === 0 && <span className="text-sm text-muted-foreground animate-pulse">…</span>}
            </div>
          )}
          {live.status === "error" && live.error && <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-sm text-destructive">{S.errorPrefix} ({live.error.kind}): {live.error.message}</div>}
          <div ref={bottom} />
        </div>
      </div>
      <Composer sessionId={sessionId} />
    </>
  );
}
```

- [ ] **Step 7: verify and commit**

```bash
npm run typecheck && npm test
git add apps/desktop/src && git commit -m "feat(desktop-ui): thread with markdown, tool call cards and live streaming"
```

---

### Task C7: composer, model picker, usage bar

**Files:**
- Create: `src/components/Composer.tsx`, `src/components/UsageBar.tsx`

**Interfaces:**
- Produces: `Composer({ sessionId })` — text input, Enter sends, a stop button, the model picker (`session_set_model`), and `UsageBar` above it. `UsageBar({ sessionId })` — the `session_usage` query merged with the live `contextUsed/contextLimit/rateLimit`.

- [ ] **Step 1: `UsageBar.tsx`**

```tsx
import { useQuery } from "@tanstack/react-query";
import * as api from "@/api";
import { selectRun, useRunStore } from "@/store/runs";
import { Progress } from "@/components/ui/progress";
import { S } from "@/strings";
import type { RateLimitWindow } from "@/types";

const fmt = (n: number) => n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);

function Window({ label, w }: { label: string; w: RateLimitWindow | null | undefined }) {
  if (!w || w.limit == null || w.remaining == null) return null;
  const pct = Math.round((w.remaining / w.limit) * 100);
  const resetIn = w.reset_at_ms ? Math.max(0, Math.round((w.reset_at_ms - Date.now()) / 1000)) : null;
  return <span title={`${w.remaining}/${w.limit}${resetIn != null ? ` · ${S.resetIn} ${resetIn}s` : ""}`} className={pct < 10 ? "text-destructive" : ""}>{label} {pct}%</span>;
}

export function UsageBar({ sessionId }: { sessionId: string }) {
  const usage = useQuery({ queryKey: ["usage", sessionId], queryFn: () => api.sessionUsage(sessionId) });
  const live = useRunStore(selectRun(sessionId));
  const used = live.contextUsed ?? usage.data?.context_used ?? null;
  const limit = live.contextLimit ?? usage.data?.context_limit ?? null;
  const pct = used != null && limit ? Math.min(100, (used / limit) * 100) : null;
  const u = usage.data;
  return (
    <div className="flex items-center gap-4 px-1 pb-1 text-xs text-muted-foreground">
      {pct != null && (
        <div className="flex items-center gap-2" title={`${used} / ${limit}`}>
          <span>{S.contextUsage}</span>
          <Progress value={pct} className="h-1.5 w-28" />
          <span>{pct.toFixed(0)}%</span>
        </div>
      )}
      {u && <span title={`in ${u.input_tokens} · out ${u.output_tokens} · cache r ${u.cache_read_tokens} w ${u.cache_write_tokens}`}>{S.sessionTokens} {fmt(u.input_tokens + u.output_tokens)}</span>}
      {u?.estimated_cost_usd != null && <span>{S.estimatedCost} ${u.estimated_cost_usd.toFixed(4)}</span>}
      {live.rateLimit && (
        <span className="flex gap-2">
          <span>{S.rateLimit}</span>
          <Window label="req" w={live.rateLimit.requests} />
          <Window label="tok" w={live.rateLimit.tokens} />
          <Window label="in" w={live.rateLimit.input_tokens} />
          <Window label="out" w={live.rateLimit.output_tokens} />
        </span>
      )}
    </div>
  );
}
```

- [ ] **Step 2: `Composer.tsx`**

```tsx
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { SendHorizontal, Square } from "lucide-react";
import * as api from "@/api";
import { cancelRun, startRun } from "@/events";
import { selectRun, useRunStore } from "@/store/runs";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UsageBar } from "@/components/UsageBar";
import { S } from "@/strings";

export function Composer({ sessionId }: { sessionId: string }) {
  const qc = useQueryClient();
  const [text, setText] = useState("");
  const live = useRunStore(selectRun(sessionId));
  const running = live.status === "running";
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessionList });
  const models = useQuery({ queryKey: ["models"], queryFn: api.modelsList });
  const session = sessions.data?.find((s) => s.id === sessionId);
  const setModel = useMutation({ mutationFn: (m: string) => api.sessionSetModel(sessionId, m), onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }) });
  const send = useMutation({
    mutationFn: () => startRun(sessionId, text.trim()),
    onSuccess: () => { setText(""); qc.invalidateQueries({ queryKey: ["sessions"] }); },
  });

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (text.trim() && !running) send.mutate(); }
  };

  return (
    <div className="border-t px-6 py-3">
      <div className="mx-auto max-w-3xl">
        <UsageBar sessionId={sessionId} />
        <div className="flex items-end gap-2 rounded-xl border bg-background p-2">
          <Textarea value={text} onChange={(e) => setText(e.target.value)} onKeyDown={onKey} placeholder={S.composerPlaceholder} rows={2} className="min-h-10 flex-1 resize-none border-0 shadow-none focus-visible:ring-0" />
          {running
            ? <Button variant="destructive" size="icon" onClick={() => cancelRun(sessionId)} aria-label={S.stop}><Square className="size-4" /></Button>
            : <Button size="icon" onClick={() => send.mutate()} disabled={!text.trim() || send.isPending} aria-label={S.send}><SendHorizontal className="size-4" /></Button>}
        </div>
        <div className="mt-1 flex items-center justify-between">
          <Select value={session?.model ?? ""} onValueChange={(m) => setModel.mutate(m)} disabled={running}>
            <SelectTrigger className="h-7 w-72 text-xs"><SelectValue placeholder={session?.model} /></SelectTrigger>
            <SelectContent>
              {(models.data ?? []).filter((m) => m.available).map((m) => <SelectItem key={m.id} value={m.id} className="text-xs">{m.provider} · {m.name}</SelectItem>)}
            </SelectContent>
          </Select>
          {send.isError && <span className="text-xs text-destructive">{api.messageOf(send.error)}</span>}
          {live.status === "error" && live.error?.kind === "max_turns" && <Button size="sm" variant="outline" onClick={() => startRun(sessionId, "continue")}>{S.continueRun}</Button>}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: verify and commit**

```bash
npm run typecheck
git add apps/desktop/src && git commit -m "feat(desktop-ui): composer with model picker, stop, and usage bar"
```

---

### Task C8: workspace panel — file tree, preview, mounts, connect dialogs

**Files:**
- Create: `src/components/WorkspacePanel.tsx`, `FileTree.tsx`, `MountDialogs.tsx`

- [ ] **Step 1: `FileTree.tsx`** (a lazily loaded tree; clicking a directory calls `fs_list`)

```tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronRight, File, Folder } from "lucide-react";
import * as api from "@/api";
import { cn } from "@/lib/utils";

export function FileTree({ path, depth = 0, onOpen, selected }: { path: string; depth?: number; onOpen: (path: string) => void; selected: string | null }) {
  const entries = useQuery({ queryKey: ["fs", path], queryFn: () => api.fsList(path) });
  const [open, setOpen] = useState<Record<string, boolean>>({});
  return (
    <ul>
      {(entries.data ?? []).map((e) => (
        <li key={e.path}>
          <button
            className={cn("flex w-full items-center gap-1 truncate rounded px-1 py-0.5 text-left text-xs hover:bg-accent", selected === e.path && "bg-accent")}
            style={{ paddingLeft: 4 + depth * 12 }}
            onClick={() => (e.kind === "dir" ? setOpen((o) => ({ ...o, [e.path]: !o[e.path] })) : onOpen(e.path))}
            title={e.size != null ? `${e.size} bytes` : undefined}
          >
            {e.kind === "dir" ? <ChevronRight className={cn("size-3 transition-transform", open[e.path] && "rotate-90")} /> : <span className="w-3" />}
            {e.kind === "dir" ? <Folder className="size-3.5" /> : <File className="size-3.5" />}
            <span className="truncate">{e.name}</span>
          </button>
          {e.kind === "dir" && open[e.path] && <FileTree path={e.path} depth={depth + 1} onOpen={onOpen} selected={selected} />}
        </li>
      ))}
      {entries.isError && <li className="px-2 text-xs text-destructive">{api.messageOf(entries.error)}</li>}
    </ul>
  );
}
```

- [ ] **Step 2: `MountDialogs.tsx`** (three forms — local/Notion/S3; local uses `open({ directory: true })` from `@tauri-apps/plugin-dialog`)

```tsx
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import * as api from "@/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { S } from "@/strings";
import type { MountConfig } from "@/types";

type Kind = "local" | "notion" | "s3";

export function MountDialog({ kind, onClose }: { kind: Kind | null; onClose: () => void }) {
  const qc = useQueryClient();
  const [path, setPath] = useState("");
  const [label, setLabel] = useState("");
  const [hostRoot, setHostRoot] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [s3, setS3] = useState({ bucket: "", region: "us-east-1", access_key_id: "", secret_access_key: "", endpoint: "", key_prefix: "" });

  const add = useMutation({
    mutationFn: (config: MountConfig) => api.mountAdd({ path, label: label || null, config }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["mounts"] }); qc.invalidateQueries({ queryKey: ["fs"] }); onClose(); },
  });

  const pickFolder = async () => {
    const dir = await open({ directory: true, multiple: false });
    if (typeof dir === "string") { setHostRoot(dir); if (!path) setPath("/" + dir.split("/").filter(Boolean).pop()); }
  };

  const submit = () => {
    if (kind === "local") add.mutate({ kind: "local", host_root: hostRoot });
    if (kind === "notion") add.mutate({ kind: "notion", api_key: apiKey });
    if (kind === "s3") add.mutate({ kind: "s3", ...s3, endpoint: s3.endpoint || null, key_prefix: s3.key_prefix || null });
  };

  const title = kind === "local" ? S.connectLocal : kind === "notion" ? S.connectNotion : S.connectS3;
  return (
    <Dialog open={kind !== null} onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader><DialogTitle>{title}</DialogTitle></DialogHeader>
        <div className="space-y-3">
          <div><Label>{S.path}</Label><Input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/docs" /></div>
          <div><Label>{S.label}</Label><Input value={label} onChange={(e) => setLabel(e.target.value)} /></div>
          {kind === "local" && <div className="flex gap-2"><Input value={hostRoot} readOnly placeholder="~/Documents/project" /><Button variant="outline" onClick={pickFolder}>…</Button></div>}
          {kind === "notion" && <div><Label>Notion Integration Token</Label><Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} /></div>}
          {kind === "s3" && (
            <div className="grid grid-cols-2 gap-2">
              <div><Label>bucket</Label><Input value={s3.bucket} onChange={(e) => setS3({ ...s3, bucket: e.target.value })} /></div>
              <div><Label>region</Label><Input value={s3.region} onChange={(e) => setS3({ ...s3, region: e.target.value })} /></div>
              <div><Label>access key id</Label><Input value={s3.access_key_id} onChange={(e) => setS3({ ...s3, access_key_id: e.target.value })} /></div>
              <div><Label>secret access key</Label><Input type="password" value={s3.secret_access_key} onChange={(e) => setS3({ ...s3, secret_access_key: e.target.value })} /></div>
              <div><Label>endpoint (optional)</Label><Input value={s3.endpoint} onChange={(e) => setS3({ ...s3, endpoint: e.target.value })} /></div>
              <div><Label>key prefix (optional)</Label><Input value={s3.key_prefix} onChange={(e) => setS3({ ...s3, key_prefix: e.target.value })} /></div>
            </div>
          )}
          {add.isError && <p className="text-sm text-destructive">{api.messageOf(add.error)}</p>}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>{S.cancel}</Button>
          <Button onClick={submit} disabled={add.isPending || !path.trim()}>{S.connect}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
```

- [ ] **Step 3: `WorkspacePanel.tsx`**

```tsx
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderPlus, Globe, HardDrive, Trash2, Cloud } from "lucide-react";
import * as api from "@/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FileTree } from "@/components/FileTree";
import { MountDialog } from "@/components/MountDialogs";
import { S } from "@/strings";

export function WorkspacePanel() {
  const qc = useQueryClient();
  const mounts = useQuery({ queryKey: ["mounts"], queryFn: api.mountList });
  const [dialog, setDialog] = useState<"local" | "notion" | "s3" | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const file = useQuery({ queryKey: ["file", selected], queryFn: () => api.fsRead(selected!), enabled: !!selected });
  const remove = useMutation({ mutationFn: (p: string) => api.mountRemove(p), onSuccess: () => { qc.invalidateQueries({ queryKey: ["mounts"] }); qc.invalidateQueries({ queryKey: ["fs"] }); } });

  return (
    <aside className="flex h-full min-w-0 flex-col border-l">
      <div className="flex items-center justify-between p-3">
        <h2 className="text-sm font-medium">{S.workspace}</h2>
        <DropdownMenu>
          <DropdownMenuTrigger asChild><Button size="sm" variant="outline"><FolderPlus className="mr-1 size-3.5" />{S.addMount}</Button></DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => setDialog("local")}><HardDrive className="mr-2 size-4" />{S.connectLocal}</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setDialog("notion")}><Globe className="mr-2 size-4" />{S.connectNotion}</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setDialog("s3")}><Cloud className="mr-2 size-4" />{S.connectS3}</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="space-y-1 px-3 pb-2">
        {(mounts.data ?? []).map((m) => (
          <div key={m.path} className="flex items-center gap-2 text-xs">
            <Badge variant={m.status.status === "error" ? "destructive" : "secondary"}>{m.kind}</Badge>
            <span className="truncate font-mono" title={m.detail}>{m.path}</span>
            {!m.writable && <span className="text-muted-foreground">{S.readOnly}</span>}
            {m.status.status === "error" && <span className="truncate text-destructive" title={m.status.message}>!</span>}
            {m.kind !== "root" && <button className="ml-auto" aria-label={S.remove} onClick={() => remove.mutate(m.path)}><Trash2 className="size-3.5" /></button>}
          </div>
        ))}
      </div>
      <ScrollArea className="flex-1 border-t px-2 py-1">
        <FileTree path="/" onOpen={setSelected} selected={selected} />
      </ScrollArea>
      {selected && (
        <div className="max-h-[40%] overflow-auto border-t p-2">
          <div className="mb-1 truncate font-mono text-xs text-muted-foreground">{selected}</div>
          {file.data?.text != null
            ? <pre className="whitespace-pre-wrap font-mono text-xs">{file.data.text}{file.data.truncated && `\n… ${S.fileTooLarge}`}</pre>
            : file.data ? <p className="text-xs text-muted-foreground">{S.binaryFile}</p> : null}
          {file.isError && <p className="text-xs text-destructive">{api.messageOf(file.error)}</p>}
        </div>
      )}
      <MountDialog kind={dialog} onClose={() => setDialog(null)} />
    </aside>
  );
}
```

- [ ] **Step 4: verify and commit**

```bash
npm run typecheck
git add apps/desktop/src && git commit -m "feat(desktop-ui): workspace panel with file tree, preview, mounts and connect dialogs"
```

---

### Task C9: settings dialog

**Files:**
- Create: `src/components/SettingsDialog.tsx`

- [ ] **Step 1: implementation**

```tsx
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "@/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { S } from "@/strings";
import type { SettingsPatch } from "@/types";

export function SettingsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const qc = useQueryClient();
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet, enabled: open });
  const models = useQuery({ queryKey: ["models"], queryFn: api.modelsList, enabled: open });
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo, enabled: open });
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const save = useMutation({
    mutationFn: (patch: SettingsPatch) => api.settingsSet(patch),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["settings"] }); qc.invalidateQueries({ queryKey: ["models"] }); setDrafts({}); },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-2xl overflow-auto">
        <DialogHeader><DialogTitle>{S.settings}</DialogTitle></DialogHeader>
        <section className="space-y-3">
          <h3 className="text-sm font-medium">{S.providers}</h3>
          {(settings.data?.providers ?? []).map((p) => (
            <div key={p.key} className="grid grid-cols-[140px_1fr_auto_auto] items-center gap-2">
              <Label>{p.label}</Label>
              <Input type="password" placeholder={p.has_key ? `${S.apiKey} ${p.key_hint}` : S.apiKey} value={drafts[p.key] ?? ""} onChange={(e) => setDrafts({ ...drafts, [p.key]: e.target.value })} autoComplete="off" />
              <Button size="sm" disabled={!drafts[p.key]} onClick={() => save.mutate({ provider_keys: { [p.key]: drafts[p.key] }, ...(p.key === "bedrock" && drafts["bedrock.region"] ? { bedrock_region: drafts["bedrock.region"] } : {}) })}>{S.saveKey}</Button>
              <Button size="sm" variant="ghost" disabled={!p.has_key} onClick={() => save.mutate({ provider_keys: { [p.key]: null } })}>{S.clearKey}</Button>
              {p.key === "bedrock" && <Input className="col-span-4" placeholder={`region (${p.region ?? "us-east-1"})`} value={drafts["bedrock.region"] ?? ""} onChange={(e) => setDrafts({ ...drafts, "bedrock.region": e.target.value })} />}
            </div>
          ))}
        </section>
        <section className="grid grid-cols-2 gap-3 pt-3">
          <div>
            <Label>{S.defaultModel}</Label>
            <Select value={settings.data?.default_model} onValueChange={(m) => save.mutate({ default_model: m })}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>{(models.data ?? []).map((m) => <SelectItem key={m.id} value={m.id} disabled={!m.available}>{m.provider} · {m.name}</SelectItem>)}</SelectContent>
            </Select>
          </div>
          <div><Label>{S.maxTokens}</Label><Input type="number" defaultValue={settings.data?.max_tokens} onBlur={(e) => save.mutate({ max_tokens: Number(e.target.value) })} /></div>
          <div><Label>{S.maxTurns}</Label><Input type="number" defaultValue={settings.data?.max_turns} onBlur={(e) => save.mutate({ max_turns: Number(e.target.value) })} /></div>
          <div className="flex items-center gap-2 pt-5"><Switch checked={settings.data?.catalog_refresh ?? true} onCheckedChange={(c) => save.mutate({ catalog_refresh: c })} /><Label>{S.catalogRefresh}</Label></div>
        </section>
        <section className="pt-3 text-xs text-muted-foreground">
          <div>{S.workspace}: <span className="font-mono">{ws.data?.mountpoint}</span> · {ws.data?.status.status}</div>
          <Button size="sm" variant="link" onClick={() => api.openLogs()}>Open logs folder</Button>
        </section>
        {save.isError && <p className="text-sm text-destructive">{api.messageOf(save.error)}</p>}
      </DialogContent>
    </Dialog>
  );
}
```

("Open logs folder" goes into `strings.ts` as `S.openLogs`.)

- [ ] **Step 2: verify and commit**

```bash
npm run typecheck && npm test
git add apps/desktop/src && git commit -m "feat(desktop-ui): settings dialog for provider keys, defaults and workspace status"
```

---

### Task C10: sidecar build and bundling

**Files:**
- Create: `apps/desktop/scripts/build-sidecar.sh`
- Modify: `apps/desktop/package.json` (scripts), `src-tauri/tauri.conf.json` (check `externalBin`)

- [ ] **Step 1: the script**

```bash
#!/usr/bin/env bash
# Build cortex-local-console from the sibling checkout and place it where Tauri expects a
# sidecar: src-tauri/binaries/<name>-<target-triple>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CORTEX="${CORTEX_DIR:-$HERE/../../../cortex}"
PROFILE="${1:-release}"
TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
( cd "$CORTEX" && cargo build -p cortex-local-console --"$PROFILE" 2>/dev/null || cargo build -p cortex-local-console $( [ "$PROFILE" = release ] && echo --release ) )
mkdir -p "$HERE/src-tauri/binaries"
cp "$CORTEX/target/$PROFILE/cortex-local-console" "$HERE/src-tauri/binaries/cortex-local-console-$TRIPLE"
echo "sidecar: $HERE/src-tauri/binaries/cortex-local-console-$TRIPLE"
```

`chmod +x apps/desktop/scripts/build-sidecar.sh`. Add to `package.json` scripts:

```json
"sidecar": "bash scripts/build-sidecar.sh release",
"sidecar:debug": "bash scripts/build-sidecar.sh debug",
"tauri:dev": "npm run sidecar:debug && tauri dev",
"tauri:build": "npm run sidecar && tauri build"
```

- [ ] **Step 2: check the dev run**

```bash
cd apps/desktop && npm run tauri:dev
```

Expected: the window opens with a "New chat" button in the sidebar, the workspace on the right (mount status `mounted` — when FUSE-T is installed), and a settings dialog that accepts a key. `workspace mounted at …` in the terminal log.

- [ ] **Step 3: check the bundle**

```bash
npm run tauri:build 2>&1 | tail -5
ls src-tauri/target/release/bundle/macos/
```

Expected: `Ailoy.app` is produced. `cortex-local-console` must sit alongside it inside `Ailoy.app/Contents/MacOS/`.

- [ ] **Step 4: commit**

```bash
cd ../.. && git add apps/desktop/scripts apps/desktop/package.json apps/desktop/src-tauri/tauri.conf.json && git commit -m "feat(desktop-app): sidecar build script and bundle wiring"
```

---

### Task C11: manual E2E and wrap-up

- [ ] **Step 1: run the E2E checklist** (a real key; paste `ANTHROPIC_API_KEY` from `.env` into the settings dialog)

1. Start the app → workspace `mounted`, `~/Library/Application Support/com.brekkylab.ailoy/workspace` opens in Finder.
2. Settings → save the Anthropic key → Anthropic models show as `available` in the model list.
3. New chat → "List the files in the workspace" → streaming text, a `shell` tool card (`ls`) → the result on the card once it finishes.
4. Connect a local folder (`/docs`) → "Read me the first line of the README in the docs folder" → a `read` or `cat` call and the contents.
5. Connect Notion (token) → check the page directories in the tree → "Tell me the title of the first page in notion" → reads `page.json`.
6. Stop during a long task → `cancelled` state, partial text kept, unfinished tool cards "Interrupted".
7. Quit the app → restart → chats and mounts restored, the mountpoint healthy.
8. Check that the context gauge %, session tokens, estimated cost and the Anthropic rate limit % are shown.
9. Save a bad key → the 401 error shows in the thread.

Fix any defect you find inside this Task and leave each fix as its own commit.

- [ ] **Step 2: cleanup**

```bash
cd apps/desktop && npm run typecheck && npm test && npm run build
cargo test --manifest-path src-tauri/Cargo.toml 2>&1 | grep -E 'test result|FAILED'
cd ../.. && cargo test -p ailoy-desktop-core 2>&1 | grep -E 'test result|FAILED'
git add -A && git commit -m "chore(desktop): e2e pass, fixes and polish"
```

- [ ] **Step 3: README** — in `apps/desktop/README.md`, write down how to run it (install FUSE-T, `../cortex` on branch `feat/exec-timeout`, `npm run tauri:dev`), where the data directory is, where the logs are, and the known limits (no stdout streaming, no approval UI, macOS only), then commit.

---

## Self-Review checklist (for the author)

- Spec §7 command table → all of C3 (added: `session_set_model`, `open_logs`) ✓; Channel per run ✓; sidecar path resolution ✓ (C2 `sidecar.rs`); `shutdown` on exit ✓; CSP ✓
- §8 stack, three columns, reducer, tool call cards, settings dialog, strings file → C1, C4–C9 ✓
- §10 error UX: degraded banner (C5), no-key banner (C5), model error display (C6), `max_turns` "Continue" (C7), connector error badge (C8), cancellation display (C6) ✓
- §11 E2E checklist → C11 ✓
- Type agreement: the `RunEvent` tags (`started, text_delta, thinking_delta, tool_call_started, message, usage, awaiting_approval, done, cancelled, error`) match the engine's `#[serde(tag="type", rename_all="snake_case")]` ✓; `MountConfig` tag `kind` lowercase ✓; `MountStatus`/`WorkspaceStatus` tag `status` lowercase ✓; command arguments camelCase (`sessionId`, `onEvent`) ✓
