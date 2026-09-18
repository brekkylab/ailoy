# Ailoy Desktop — design (v1)

- Written: 2026-09-11
- Status: awaiting review
- Scope: the v1 (minimal) design for turning ailoy from a developer library into a Claude/ChatGPT-style desktop app (Tauri)
- Related repositories: `brekkylab/ailoy` (this repository), `brekkylab/cortex` (sibling checkout `../cortex`, a path dependency for now)

---

## 1. Goals and scope

### 1.1 Goal

Build a desktop app with a session-based conversational UI. The agent runs shell commands and file I/O through `cortex-local-console` inside a **cortex workspace** (a directory where WorkFs is mounted over FUSE-T), and the user can attach local folders, Notion, and S3 to a single filesystem tree to show them to the agent (the Rambox/Ferdium style of "attach services to my FS"). The agentic loop has the baseline polish of Claude/ChatGPT: cancellation, a turn limit, an approval hook, failure recovery, and usage display.

### 1.2 In v1

- Session CRUD and persistence (SQLite), conversations restored after an app restart
- Streaming conversation (text and thinking deltas), tool call cards (name, arguments, result, status)
- The system tools `shell/read/write/edit/glob/grep` plus `web_search/web_fetch`
- Cancelling a run, a turn limit (50 by default), a tool approval hook (the v1 policy is auto-approve)
- Workspace: a FUSE-T mount held for the app's lifetime, a persistent local directory as the root, and local folder/Notion/S3 connectors
- File browser (tree and text preview), mount management UI
- Token usage: per-message and per-session totals, context window utilization, estimated cost, and the remaining Anthropic/OpenAI (xAI conditionally) rate limit
- Settings: provider API keys, default model
- Target platform: macOS (Apple Silicon first). FUSE-T must be installed.

### 1.3 Not in v1 (the follow-up roadmap is §12)

The approval UI and policy storage, wiring up the `mem` memory tool, the micro-VM console, the GDrive connector (OAuth), stdout streaming, context summarization, MCP, Keychain, Windows/Linux, continuing after `Length`, automatic session titles, and account balance and monthly usage lookups.

---

## 2. Current state and reuse decisions

### 2.1 ailoy branches

| Branch | Gist | Decision |
|---|---|---|
| `develop` (`60716d12`) | A pure Rust library. Five API models (Anthropic/OpenAI Responses/ChatCompletion/Gemini/Bedrock). Its own `runenv` (Local, microsandbox), `skill`, `python_repl`. MCP is `todo!()`. | Not used as the base. We do take #448 (Bedrock wire) from it. |
| `origin/mem-applied` (`8d3f0238`, 9/7) ⊃ `cortex-applied` | `runenv`, `skill`, and `python_repl` removed. Re-exports `cortex::console::Console` (`src/console.rs`). The tools call cortex `exec/read/write` directly. `console.start/stop` around every tool batch. `src/memory` (built on the `mem` executable) plus the `mem_search/mem_insert` tools. `cortex = { path = "../cortex/cortex" }`. **Confirmed to compile against cortex main.** | **The base for this work.** |
| `origin/feat/krun-sandbox` (8/6) | The base of agent-k. ailoy brings up a libkrun VM itself. | Not adopted, because cortex absorbed the VM console. |
| `origin/feat/vfs-provider-mounts` and others (June–July) | `src/vfs` (S3/Notion/GDrive plus in-guest FUSE). | Superseded by cortex `fs`. For reference only. |

`mem-applied` branched off before #448, so it has no `bedrock.rs`. The working branch is cut from `mem-applied` and merges `develop` (#448 as is, #443 resolved by the deletion of the sandbox code, the `web_fetch.rs` change kept).

### 2.2 cortex (`../cortex` main `dfabb34`)

- `console`: BSON JSON-RPC over stdio. The methods `init/exec/read/write/commit/start/stop/quit`. The servers are `cortex-local-console` (host) and `cortex-uvm-console` (micro-VM). **Constraints**: no stdout streaming (by design), no directory listing, one request per console (`&mut self`), `timeout_ms` is accepted but **unimplemented** in both servers, and a 64 MiB frame cap (`ExecResp::truncated`).
- `fs`: the `FileSystem` trait (`stat/list/read_at` required, the rest read-only by default), the `WorkFs` longest-prefix mount table (itself a `FileSystem`), the backends `InMemFs/PassthroughFs/S3Fs/NotionFs/GdriveFs`, and the host mounts `FuseMount` (macFUSE) and `FuseTMount` (FUSE-T, no kext needed). The `Mount` trait: `mountpoint()`, unmounts on drop.
- `rootfs`/image layers, the `mem`/`index` executables (SQLite+vec0).
- **`origin/cortex-gui`** (jhlee525, 9/7): a Tauri 2 + React PoC. A WorkFs file browser, local/Notion/S3 connectors, `SharedFs` (a handle that does not lose WorkFs when you hand it to a mount), and `mem init` over a temporary FUSE-T mount. No session storage. From the README: "agent execution — where ailoy plugs in".

### 2.3 agent-k (`../agent-k`)

A server-style full stack (axum + SQLite + SSE) on top of ailoy (`feat/krun-sandbox`) and cortex. The frontend is tied to the old backend and cannot be reused. **The pieces worth porting**: `backend/src/agent_stream.rs` (the `MessageAssembler` that reassembles deltas into messages, tests included), the interruption handling in `state/session.rs` (drain mode, `completed_naturally`, stubs for unfinished tool_calls), the `SessionMessage{depth, source_agent, message}` storage shape and its "only depth 0 goes back into the model" rule, and `lib/toolCallFormat.ts` (one parser for rendering and copying tool calls). There was no tool approval.

### 2.4 Reuse decisions at a glance

| Source | Use as is | Port (copy and adapt code) | Ideas only |
|---|---|---|---|
| ailoy `mem-applied` | The whole core (the message model, the five LM wires, the tools, the agent loop, memory) | — | — |
| cortex main | `Console`, `WorkFs`, the FS backends, `FuseTMount` | — | — |
| cortex-gui | — | `SharedFs`, `fsops` (list/read/write/import), `mounts` (connector pre-validation), the file tree UI logic | The three-column layout |
| agent-k | — | `MessageAssembler`, interruption handling, the `SessionMessage` shape, the tool call formatter | `.ref` knowledge indexing (follow-up) |

---

## 3. Decisions (Q&A record)

| Item | Decision |
|---|---|
| Core base branch | `mem-applied` (+ the develop #448 merge) |
| v1 console backend | `cortex-local-console` (runs on the host). The micro-VM becomes a later toggle |
| v1 scope | Chat + local folders + external connectors (Notion/S3) |
| Tool approval | The hook and the events now, the policy stays auto-approve |
| Code location | `apps/desktop/` in the ailoy repository. The branch comes from `mem-applied`. Merging `mem-applied→develop` is the branch owner's call |
| cortex changes | Create a branch on cortex when needed (referenced through the path dependency) |
| Workspace model | One workspace per app, N sessions (the schema keeps `workspace_id`) |
| UI stack | React 19 + Vite + TS + Tailwind v4 + shadcn/ui, with the chat UI built by hand |
| Runtime layout | A headless session engine crate embedded in Tauri |
| Mount lifetime | The FUSE-T mount is held for the app's lifetime |
| Usage display | Tier 1 (context utilization, totals, and cost across every vendor) plus tier 2 (rate limit headers) |
| Model metadata | A bundled models.dev (`https://models.dev/api.json`) snapshot refreshed at runtime |

Conventional defaults (decided without asking): SQLite is `rusqlite` (bundled), API keys are stored in the settings DB in the app data directory (Keychain later), the default model is `anthropic/claude-opus-5`, `max_tokens` defaults to 32,000, and the turn limit is 50.

---

## 4. Overall architecture

### 4.1 Repository layout

```
ailoy/                         cargo workspace root (the `ailoy` package stays a library)
├─ Cargo.toml                  members = ["./", "apps/desktop/core"], exclude += ["apps/desktop/src-tauri"]
├─ src/                        the ailoy core (§5)
├─ apps/desktop/
│  ├─ package.json, vite.config.ts, tailwind, src/   the React frontend (§8)
│  ├─ core/                    crate `ailoy-desktop-core` — the headless session engine (§6)
│  ├─ src-tauri/               crate `ailoy-desktop` — its own [workspace] (§7)
│  └─ scripts/                 generating the models.dev snapshot, etc.
└─ ../cortex/cortex            path dependency
```

`src-tauri` is kept out of the root workspace's members for the same reason as in cortex-gui: the hundreds of webview dependencies Tauri pulls in should not weigh on the root's `cargo test`. `core` is light (rusqlite, tokio, cortex, ailoy), so it stays a member.

### 4.2 Processes and data flow

```
┌──────────────── Tauri app process ────────────────┐      stdio(BSON JSON-RPC)   ┌──────────────────────┐
│ WebView(React) ⇄ invoke/Channel ⇄ src-tauri       │ ───────────────────────────▶ │ cortex-local-console │ (one per run)
│                    └── ailoy-desktop-core          │                              │  cwd = workfs mount  │
│                          ├─ Agent(ailoy) ─ LLM API │                              └──────────┬───────────┘
│                          ├─ WorkFs ── FuseTMount ──┼── <appdata>/workspace ◀── kernel FUSE ──┘
│                          └─ SQLite(<appdata>/db)   │
└───────────────────────────────────────────────────┘
```

- The file tree is seen through two paths. The agent's shell gets the FUSE-T mount path as its workfs, and the UI's file browser reads the same `WorkFs` in process through `FileSystem::list/read` (no FUSE round trip).
- There is **one console per run**. A cortex console takes one request at a time, so concurrent runs across sessions fall out naturally, and the console is cleaned up with `quit` when the run ends. A session has at most one run at a time.
- LLM calls are made by the ailoy core inside the app process (the keys exist only in the core process).

### 4.3 App data directory

`~/Library/Application Support/com.brekkylab.ailoy/`
- `ailoy.sqlite` — sessions, messages, mounts, settings
- `files/` — the workspace root (`PassthroughFs`)
- `workspace/` — the FUSE-T mountpoint (must be empty)
- `cache/models.json` — the models.dev refresh cache

---

## 5. Changes to the ailoy core

The principle: keep the existing `run`/`run_stream` and the `anyhow`-based public API, and **add** a controllable entry point and typed errors. Loop consistency (cancellation and recovery) lives in the core so that every library consumer gets the same level of polish.

### 5.1 `RunControl` and `run_stream_controlled`

```rust
// src/agent/control.rs (new)
pub struct RunControl {
    pub cancel: tokio_util::sync::CancellationToken,
    pub max_turns: Option<u32>,            // cap on model calls. None = unlimited (the existing behavior)
    pub tool_gate: Arc<dyn ToolGate>,      // AllowAll by default
}

pub struct ToolCallRequest<'a> { pub id: &'a str, pub name: &'a str, pub arguments: &'a Value }
pub enum ToolDecision { Allow, Deny { reason: String } }

#[async_trait]
pub trait ToolGate: Send + Sync {
    async fn review(&self, call: ToolCallRequest<'_>) -> ToolDecision;
}

impl Agent {
    pub fn run_stream_controlled(&mut self, query: Message, ctl: RunControl)
        -> BoxStream<'_, Result<MessageDeltaOutput, AgentError>>;
}
```

- The existing `run_stream` delegates with `RunControl::default()` (no cancellation, no limit, AllowAll) and wraps `AgentError` in `anyhow`.
- The `MessageDeltaOutput` item type does not change. UI events such as "awaiting approval" are emitted on the gate implementation (the engine) side.

### 5.2 Loop consistency rules

- **Turn limit**: checked right before a model call (the point where every tool result from the previous turn has been committed). Over the limit it is `Err(AgentError::MaxTurns { turns })`. The history at that point is always safe to resend. `max_turns` only exists for the streaming `run_stream_controlled`, and `max_turns: Some(0)` never calls the model at all, so it pops the pending user message (the same as the rollback rule).
- **Cancellation — during the model response**: race the stream's `next()` against `cancel.cancelled()` with `select!`. On cancellation, commit the partial assistant message accumulated so far (text and thinking as is, unfinished tool_call fragments dropped, `finish_reason = Stop`). If there is neither text nor a tool_call, commit nothing and pop the pending user message (the existing rollback rule).
- **Cancellation — during tool execution**: drop the tool stream (aborting the futures in flight). For every tool_call whose result was not committed, put a `Role::Tool` stub `"[Interrupted: cancelled before this tool call completed]"` into the history under the same `id`.
- **Approval denied**: a call that comes back `Deny{reason}` is not executed and is recorded as a `Role::Tool` result `{"error":"denied by user: <reason>","phase":"policy"}`. The allowed calls in the same batch run normally.
- **Model call failure (turn 2 onward)**: previously the history could be left with an assistant `tool_calls` and no results after it. If there are unfinished tool_calls at the point of failure, close the history with the same stubs as above and then return the error.
- **The blocking `run` behaves the same**: when a tool batch is aborted by a console startup failure or an unknown tool, put the same stub in for every tool_call without a result and return the error (`run` has no `max_turns` — see the turn limit item above).
- No matter which path ends the run, **no unpaired `tool_use` is left in the history** (this avoids Anthropic 400s). Tests pin this down.

### 5.3 `AgentError`

```rust
pub enum AgentError {
    Cancelled,
    MaxTurns { turns: u32 },
    Model(ModelError),                 // status: Option<u16>, retryable: bool, provider_message: String
    Tool(anyhow::Error),               // tool execution failed. No name field; it carries the source error as is
    Console(anyhow::Error),            // no console / startup failed
    Other(anyhow::Error),
}
```

`ModelError` is introduced at the `LangModel` layer too, carrying the HTTP status, the body, and whether a retry is worthwhile. The existing `anyhow` path stays compatible through `From<AgentError> for anyhow::Error`.

### 5.4 Retries and the client

- `send_with_retry`: **5xx and transport errors** join 429 as subjects of exponential backoff (at most 3 attempts, capped at 10 seconds). 4xx (other than 429) fails immediately. The permanent quota error classification (`is_permanent_quota_error`) stays.
- Cache the `reqwest::Client` on `LangModel` to remove the TLS handshake from every call.

### 5.5 Usage and rate limit information

- `TokenUsage` is filled in by all five wires, and the cache fields are all parsed too (Gemini reads `usageMetadata.cachedContentTokenCount` and OpenAI Responses reads `usage.input_tokens_details.cached_tokens` into `cache_read_input_tokens`, then normalizes `input_tokens` to the total prompt minus that. Neither vendor reports cache write numbers, so `cache_creation_input_tokens` is `None`).
- New types and fields:

```rust
pub struct RateLimitWindow { pub limit: Option<u64>, pub remaining: Option<u64>, pub reset_at_ms: Option<u64> }
// reset_at_ms: Unix epoch milliseconds. Whether it arrives as RFC 3339 (Anthropic) or a duration (OpenAI), it is normalized to ms at parse time.
pub struct RateLimitInfo {
    pub requests: Option<RateLimitWindow>,
    pub tokens: Option<RateLimitWindow>,        // combined tokens (only the vendors that have it)
    pub input_tokens: Option<RateLimitWindow>,
    pub output_tokens: Option<RateLimitWindow>,
}
// Add `pub rate_limit: Option<RateLimitInfo>` (serde skip_if_none) to MessageOutput / MessageDeltaOutput.
// run_stream: the response headers ride on **the first delta only** (rate_limit is always None on later deltas). run: they ride on MessageOutput.
```

**Defining what `TokenUsage` means**: `input_tokens` counts **only uncached input**. `cache_read_input_tokens` and `cache_creation_input_tokens` are non-overlapping **additive components**, so the total prompt is the sum of the three (we take the Anthropic wire's meaning as the standard). Providers that report cache tokens **included** in the prompt number — OpenAI Responses `usage.input_tokens`, OpenAI ChatCompletion `usage.prompt_tokens` (+`prompt_tokens_details.cached_tokens`), Gemini `usageMetadata.promptTokenCount` (+`cachedContentTokenCount`) — are normalized in the parser with `input_tokens = total_prompt.saturating_sub(cached)`. The `context_used` formula in §6.6 (the sum of the three) therefore holds as written for every provider, with no double counting.

| Wire schema | Header mapping |
|---|---|
| Anthropic | `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}` (reset is RFC 3339) |
| OpenAI Responses / ChatCompletion | `x-ratelimit-{limit,remaining,reset}-{requests,tokens}` (reset is a `1m2s`-style duration → added to the current time). xAI, DeepSeek, and Moonshot go through the same parser and come back `None` when the headers are absent |
| Gemini, Bedrock | no headers → `None` |

### 5.6 Sorting out the base branch

The `feat/desktop` branch = `origin/mem-applied` + a `develop` merge (#448 kept, #443's `sandbox.rs` change resolved by deleting the file). Merging back into `develop` later is coordinated with the owner of `mem-applied`.

---

## 6. The session engine `ailoy-desktop-core`

A library crate with no Tauri dependency. Every public type is `serde`-serializable and doubles as the IPC payload.

### 6.1 Modules

| Module | Responsibility |
|---|---|
| `engine` | `Engine::start(config) -> Engine`: open and migrate the DB → clean up stale mounts → assemble and mount WorkFs → restore connectors → register providers → load the model catalog. `Engine::shutdown()`: cancel every run → quit the consoles → drop the mount. |
| `workspace` | Owns `Arc<RwLock<WorkFs>>`. Hands `SharedFs` (ported from cortex-gui) to `FuseTMount::try_new` and holds it for the app's lifetime. The root `""` = `PassthroughFs(<appdata>/files)`. Connector `mount_add/remove` updates the DB and calls `WorkFs::mount/unmount`. `list/read/write/mkdir/delete/rename/import` for the file browser (ported from cortex-gui `fsops`). A `WorkspaceMount: cortex::fs::Mount` implementation passes the mountpoint to the console. |
| `console` | Resolve the sidecar path → `tokio::process::Command` → `Console::builder().client(StdioClient::new(cmd)).mount(WorkspaceMount).build()`. Prepend the sidecar directory to the `PATH` environment variable (in preparation for `mem` later). One per run, dropped (`quit`) when the run ends. A spawn failure is `EngineError::ConsoleUnavailable`. |
| `store` | `rusqlite` (bundled, WAL). The schema is in §6.2. A `Message` is stored as the ailoy serde JSON wrapped in `{version, depth, source_agent, message}`. |
| `catalog` | The models.dev snapshot (§6.5). Supplies the model list, context windows, prices, and capability flags. |
| `providers` | Register and update the API keys from settings on ailoy's global `"default"` `LangModelProvider` (`get_lm_providers_mut`). Key changes take effect immediately. |
| `prompt` | Builds the system preamble (§6.4). |
| `run` | At most one actor task per session (§6.3). |
| `usage` | Computes context utilization, session totals, estimated cost, and the remaining rate limit from `TokenUsage`, `RateLimitInfo`, and the catalog (§6.6). |
| `events` | `RunEvent` (§6.3), `EngineEvent` (mount changes and errors). |

### 6.2 Data model (SQLite)

```sql
CREATE TABLE workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at INTEGER NOT NULL);
-- v1 creates a single 'default' row

CREATE TABLE mounts (
  id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  path TEXT NOT NULL,                 -- the path inside WorkFs. The root row is '' (displayed as '/')
  kind TEXT NOT NULL,                 -- 'root' | 'local' | 'notion' | 's3'
  label TEXT NOT NULL,
  config TEXT NOT NULL,               -- JSON. Includes credentials (plaintext in v1, file mode 0600)
  writable INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE(workspace_id, path)
);

CREATE TABLE sessions (
  id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  title TEXT NOT NULL, model TEXT NOT NULL,
  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);

CREATE TABLE messages (
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  depth INTEGER NOT NULL DEFAULT 0,   -- 0 = top level, ≥1 = inside a subagent
  source_agent TEXT,
  role TEXT NOT NULL,                 -- duplicated for easy querying
  content TEXT NOT NULL,              -- {"version":1,"message":<ailoy Message JSON>}
  usage TEXT,                         -- TokenUsage JSON (assistant messages only)
  created_at INTEGER NOT NULL,
  PRIMARY KEY(session_id, seq)
);

CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- provider.<name>.api_key, default_model, catalog.last_refreshed, and so on
```

- The history fed back into the model is only the rows with `depth = 0` (the agent-k rule). Messages from inside a subagent are stored purely for display.
- `messages.usage` is the basis for session totals and context utilization.
- Migrations are sequential SQL files keyed off `PRAGMA user_version` (`core/migrations/NNNN_*.sql`).

### 6.3 Run lifetime and events

```
run_start(session, parts)
 ├─ EngineError::AlreadyRunning if the session already has an active run
 ├─ write the user message to the DB immediately (seq N) → RunEvent::Message
 ├─ spawn the console → assemble the Agent:
 │     AgentBuilder::new(model).instruction(preamble).system_tools()
 │        .web_search_tool(vec![]).web_fetch_tool()
 │        .history(the depth-0 messages).console(console).build()
 │     spec.max_tokens = settings or 32_000
 ├─ tokio::spawn(actor):
 │     stream = agent.run_stream_controlled(user_msg, RunControl{cancel, max_turns: 50, tool_gate: AllowAll})
 │     classify deltas into (TextDelta|ThinkingDelta|Completed) with MessageAssembler (ported from agent-k)
 │     a Completed message goes to the DB at once (seq++) → RunEvent::Message
 │     a delta carrying usage/rate_limit → RunEvent::Usage
 │     ending: Ok → Done, Err(Cancelled) → Cancelled, Err(MaxTurns) → Error{kind:"max_turns"}, anything else → Error
 └─ drop the console (quit), update the session's updated_at
```

```rust
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunEvent {
    Started { run_id: String },
    TextDelta { text: String },
    ThinkingDelta { text: String },
    ToolCallStarted { id: String, name: String, arguments: Value },
    Message { seq: i64, depth: u8, source_agent: Option<String>, message: Message, usage: Option<TokenUsage> },
    Usage { usage: Option<TokenUsage>, rate_limit: Option<RateLimitInfo>, context_used: Option<u64>, context_limit: Option<u64> },
    AwaitingApproval { id: String, name: String, arguments: Value },   // never fires in v1
    Done, Cancelled,
    Error { kind: String, message: String },
}
```

- Events go out over a per-run `tokio::sync::broadcast`. A refreshed window resubscribes with `run_attach`, which sends the partial text so far as a single `TextDelta`.
- `run_cancel` calls `CancellationToken::cancel()` and returns without waiting for the actor to finish. The final state arrives as an event.
- Partial assistant text is committed by the core (§5.2), so it survives in the DB after a cancellation.

### 6.4 The system preamble

ailoy puts nothing in beyond `instruction`, so the engine assembles it. The parts: (1) identity and role, (2) the date and the OS, (3) the working directory = the workfs path plus the rule that "every path lives inside it", (4) the mount table: path, kind, whether it is read-only, and how to use each connector (Notion renders `page.json`, S3 has object key rules), (5) tool usage guidance (`shell` is `sh -c`, results are truncated at 30k characters and what the `truncated` flag means), (6) extra user-configured instructions (optional). A mount change takes effect from the next run (a change mid-session does not replace the system message; it shows up in the first message of the next run).

### 6.5 The model catalog (models.dev)

- Source: `https://models.dev/api.json` (MIT, TOML sources plus PRs, 213 providers and 7,677 models, 4.5 MB). Schema: `provider.models[id] = { name, family, limit: {context, output}, cost: {input, output, cache_read, cache_write} (USD/1M), reasoning, tool_call, structured_output, modalities, release_date, ... }`.
- **At build time**: the `apps/desktop/scripts/gen-catalog` script fetches api.json and generates the snapshot `core/assets/models.json` (a few tens of KB) keeping only the chat models of seven providers (`anthropic, openai, google, amazon-bedrock, xai, deepseek, moonshotai`), which is committed. It is embedded with `include_str!`.
- **At runtime**: on startup, try a background refresh against a 24-hour cache (`<appdata>/cache/models.json`). A failure quietly falls back to the embedded snapshot. The refresh can be turned off in settings.
- **ID mapping**: ailoy `provider/model` → models.dev `provider.models[model]`. The prefix mapping is `anthropic→anthropic, openai→openai, google→google, x-ai→xai, deepseek→deepseek, moonshotai→moonshotai, bedrock→amazon-bedrock` (for Bedrock the model ID itself matches, in the form `anthropic.claude-...`). A model missing from the catalog omits the context and cost display and shows "unknown".
- The model picker's list is "providers with a registered key × catalog models with `tool_call: true`", and the user can also type an arbitrary ID directly.

### 6.6 Usage calculations

- Context utilization: from the `usage` of the session's last assistant message, `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` (each `None` counts as 0) ÷ the catalog's `limit.context`. Shown as an approximation of the input size of the next call (Anthropic's `input_tokens` covers only the tokens after the last cache breakpoint, so the three have to be summed to get the total input).
- Session totals: the sum of every assistant `usage`. Cost = Σ(input·cost.input + output·cost.output + cache_read·cost.cache_read + cache_write·cost.cache_write) / 1e6. If the catalog has no prices, the cost is omitted.
- Remaining rate limit: `remaining / limit` for each window of `RateLimitInfo`. Counts down to `reset_at_ms`. Providers without the headers are not shown.

### 6.7 Public API (the engine)

```rust
impl Engine {
  pub async fn start(cfg: EngineConfig) -> Result<Arc<Engine>, EngineError>;
  pub async fn shutdown(&self);
  // sessions
  pub async fn session_list(&self) -> Vec<SessionSummary>;
  pub async fn session_create(&self, model: Option<String>) -> Session;
  pub async fn session_rename(&self, id, title); pub async fn session_delete(&self, id);
  pub async fn message_list(&self, session_id) -> Vec<StoredMessage>;
  // runs
  pub async fn run_start(&self, session_id, parts: Vec<Part>) -> Result<(RunId, broadcast::Receiver<RunEvent>), EngineError>;
  pub async fn run_attach(&self, session_id) -> Option<(RunId, broadcast::Receiver<RunEvent>, String /*partial*/)>;
  pub async fn run_cancel(&self, session_id) -> bool;
  // workspace
  pub async fn fs_list(&self, path) / fs_read / fs_write / fs_mkdir / fs_delete / fs_rename / fs_import;
  pub async fn mount_list(&self) -> Vec<MountInfo>;
  pub async fn mount_add(&self, req: MountRequest) -> Result<MountInfo, EngineError>;   // Local{host_root} | Notion{api_key} | S3{S3Form}
  pub async fn mount_remove(&self, path);
  pub fn workspace_info(&self) -> WorkspaceInfo;   // the mountpoint and the status
  // settings & catalog
  pub async fn settings_get(&self) -> Settings /*keys are masked*/; pub async fn settings_set(&self, patch: SettingsPatch);
  pub fn models_list(&self) -> Vec<ModelInfo>;
  pub async fn session_usage(&self, session_id) -> SessionUsage;
}
```

---

## 7. The Tauri layer `ailoy-desktop`

- In the `tauri::Builder` setup, `Engine::start` → `app.manage(engine)`. On exit (`tauri::RunEvent::ExitRequested`, a different type from the engine's `RunEvent`), wait for `engine.shutdown()` on `spawn_blocking` before quitting (unmounting FUSE needs a thread join).
- Commands (all `async`, thin wrappers around engine calls): `session_list/create/rename/delete`, `message_list`, `run_start(session_id, parts, on_event: Channel<RunEvent>) -> run_id`, `run_attach(session_id, on_event)`, `run_cancel(session_id)`, `fs_*`, `mount_list/add_local/add_notion/add_s3/remove`, `workspace_info`, `settings_get/set`, `models_list`, `session_usage`.
- Events: run events go over `tauri::ipc::Channel<RunEvent>` (typed and ordered, made for streaming). Only global changes (`workspace_changed`, `mount_error`, `catalog_refreshed`) use `AppHandle::emit`.
- Sidecar: `tauri.conf.json` `bundle.externalBin: ["binaries/cortex-local-console"]`, capability `shell:allow-execute` (sidecar). cortex's `StdioClient` wants a `tokio::process::Command`, so we resolve the path ourselves instead of using the plugin: in a bundle it sits next to the executable (`current_exe().parent()`), and in dev it is `AILOY_CORTEX_BIN_DIR` with a `../../../cortex/target/{debug,release}` fallback. The build script runs `cargo build -p cortex-local-console` in `../cortex` and copies the result into `binaries/` with the target-triple suffix.
- Security: CSP `default-src 'self'`, no remote content. Credentials only reach the frontend masked. `fs_read` has a text cap (1 MiB) and binary detection.
- Logs: `tracing` → `logs/` in the app data directory. "Open logs folder" from the frontend.

---

## 8. Frontend

- Stack: React 19, Vite, TypeScript, Tailwind v4, shadcn/ui (Radix), TanStack Query (lists and settings), Zustand (stream and run state), react-markdown + remark-gfm + shiki, lucide-react. `src/api.ts` is the only route to `invoke`, and `src/events.ts` wires `Channel` into the stores (the cortex-gui pattern).
- Layout (three columns): left, the session list (+ New chat, a settings button, and each session's title, model, and last activity); center, the thread (user/assistant bubbles, collapsible thinking, tool call cards with the name, an argument summary, the result, the status running/done/error, and the elapsed time, and a composer with model selection, send, and stop, a context utilization gauge above the composer next to the session token and cost summary, and a provider badge with the remaining rate limit and a reset countdown); right, the workspace panel (the file tree, the mount list with kind badges and read-only markers, the "+ Connect" dialog for local/Notion/S3 — one validation request before connecting — and a text preview with light editing).
- Settings dialog: per-provider API keys (masked, registered as soon as they are saved), the default model, `max_tokens`, the turn limit, and the catalog refresh toggle.
- Stream reducer: `RunEvent` → accumulate `TextDelta`/`ThinkingDelta` into the current assistant bubble, create a card on `ToolCallStarted`, replace the optimistic state with the stored message on `Message` (tool results are attached to the card), refresh the gauges on `Usage`, and mark the end on `Done/Cancelled/Error`. On a window refresh, `message_list` + `run_attach`.
- Tool call rendering: port agent-k's `toolCallFormat.ts` (rendering and copying share one parser).
- Strings are collected in the single file `src/strings.ts` (Korean by default) in preparation for i18n later.

---

## 9. Changes to cortex

| Item | How v1 handles it |
|---|---|
| `timeout_ms` unimplemented | **the cortex branch `feat/exec-timeout`**: in `cortex-local-console`'s `execute`, kill the child after a `tokio::time::timeout` and answer with `Error::TIMED_OUT`. The ailoy `shell` tool passes its `timeout_secs` argument through as `timeout_ms` (600 seconds by default). The app assumes `../cortex` is on this branch and says so in the README. |
| No stdout streaming | v1 substitutes a "running + elapsed time" display. The requirement (a server→client `exec.output` notification, or a chunked response) is filed as a cortex issue. |
| No console listing | Use the in-process `FileSystem::list`. No change needed. |
| `Drop for Console` needs a runtime | The engine drops it explicitly when a run ends (inside the runtime), and uses `spawn_blocking` at shutdown. |

`../cortex` is a path dependency, so whichever branch is checked out is what gets built. Check out `feat/exec-timeout` during development, and go back to main once it is merged.

**Implementation result**: the changes above landed as cortex branch `feat/exec-timeout` commit `9d178c7` (including adding the tokio `time` feature to the `cortex-local-console` crate). The shell timeout tests on the ailoy side need a `cortex-local-console` binary built from this branch, so CI has to pin that commit.

---

## 10. Error handling

| Situation | Behavior |
|---|---|
| FUSE-T missing / mount failure | The engine still starts, but with `WorkspaceInfo.status = Degraded{reason}`. When a run starts, hand the console the `<appdata>/files` directory itself through a `Mount` implementation (one that only returns the mountpoint) without FUSE, and show a warning banner in the UI ("the agent cannot see your connections"). |
| Stale mount (after a crash) | On startup, if the mountpoint is still mounted or is not empty, `umount` it (falling back to `diskutil unmount force`) and retry. If that still fails, Degraded. |
| Console spawn failure | An `Error{kind:"console_unavailable"}` event, plus guidance about the sidecar path in settings. |
| Connector validation failure (bad key or bucket) | `mount_add` returns the 400-class error message as is and does not mount. A restore failure at startup raises the global `mount_error` event and an error badge in the list. |
| Model error | `ModelError{status, retryable}` becomes `Error{kind:"model", message}`. A 401 offers to open the settings dialog, and a 429 spend cap (`enforced_spend_limit_reached`) gets its own wording. |
| Turn limit | `Error{kind:"max_turns"}` plus a "Continue" button (which sends a new user message, "continue"). |
| Cancellation | Partial text is kept, and unfinished tool call cards read "Interrupted". |
| DB error | Fatal. A startup failure dialog (showing the file path). |

---

## 11. Testing and verification

- **The ailoy core**: bring up a fake OpenAI-compatible server with `axum` from dev-deps and verify (a) the final shape of the history per cancellation point (during the model, during a tool), (b) the turn limit, (c) a `ToolGate` denial, (d) 5xx retries, and (e) rate limit header parsing (the Anthropic and OpenAI formats). The existing live tests stay.
- **The engine**: store and migration tests on in-memory SQLite, the ported `MessageAssembler` tests, and unit tests for catalog mapping and usage calculations. The run tests that need a real `cortex-local-console` and the FUSE-T mount tests are `#[ignore]` (they run when `AILOY_CORTEX_BIN_DIR` is set).
- **The app**: `cargo check` (src-tauri), `tsc --noEmit`, and vitest for the stream reducer and the tool call formatter.
- **Manual E2E checklist** (with real keys): create a session → send a message → `ls` the workfs with `shell` → connect a local folder and `cat` a file inside it → connect Notion and read `page.json` → cancel a run in progress → restart the app and confirm the conversation and mounts are restored → confirm the context gauge, cost, and remaining rate limit are displayed.

---

## 12. Follow-up roadmap (in priority order)

1. The approval UI and policy storage (swap in a different `ToolGate` implementation, turn on the `AwaitingApproval` event, per-session and per-workspace allow rules)
2. Wiring up the `mem` memory tool (the `mem` sidecar plus `mem init`, cortex-gui's `+ memory` UI)
3. A micro-VM console toggle (`cortex-uvm-console`, image and network policy settings)
4. The GDrive connector (the OAuth desktop flow)
5. stdout streaming (a cortex protocol extension)
6. Context summarization/compaction (including Anthropic's server-side compaction)
7. An MCP client (implementing ailoy's `ToolProviderElem::MCP`)
8. Account balance and monthly usage: the DeepSeek/Moonshot balance APIs, opt-in Anthropic/OpenAI Admin keys
9. Keychain storage, multiple workspaces, Windows/Linux, continuing after `Length`, automatic session titles

---

## 13. Risks and open issues

- **The FUSE-T dependency**: shipping to end users needs installation guidance. The macFUSE alternative (the `fuse` feature) carries the burden of kext approval.
- **Stale mounts**: if the cleanup routine for a leftover `<appdata>/workspace` after a crash fails, the app only runs in Degraded mode.
- **The path dependency**: whichever branch is checked out in `../cortex` determines the build. Until it is merged, state the required branch in the README. Longer term, move to a pinned git rev or a crates.io release.
- **Team coordination**: `mem-applied` and `cortex-gui` are jhlee525's in-progress branches. Record where ported code came from in the commit messages, and agree on when `mem-applied→develop` lands.
- **models.dev availability**: an outage of the external service falls back to the embedded snapshot, so the feature keeps working, but information about new models can lag.
- **cortex protocol constraints**: one request per console and no streaming are accepted for v1. Parallel tool calls run serially as far as the console is concerned (the model asks for them in parallel, but they execute in sequence).
- **Anthropic thinking display**: the newest models default to `omitted`, so thinking deltas can arrive empty. The display option (`display: summarized`) needs a new ailoy marshalling option, so it is left for later.
