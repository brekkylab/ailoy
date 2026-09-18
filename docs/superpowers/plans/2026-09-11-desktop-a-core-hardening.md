# Ailoy Desktop — Plan A: harden the ailoy core + cortex exec timeout

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cancellation, a turn bound, a tool-approval hook, history consistency on failure, typed errors, 5xx retries and rate-limit header parsing to the ailoy agent loop, and enforce `timeout_ms` in the cortex local console.

**Architecture:** Add `Agent::run_stream_controlled(query, RunControl)` as the new entry point and let the existing `run_stream` delegate to it with the default controls. So that no unmatched `tool_use` is ever left in the history on any exit path, `close_dangling_tool_calls` inserts stub `Role::Tool` messages. `LangModel` raises a `ModelError` (HTTP status, whether it is retryable) and parses the response headers into a `RateLimitInfo` that rides on the first delta. The cortex local console server kills the process group when `exec.timeout_ms` elapses and answers `TIMED_OUT`.

**Tech Stack:** Rust 1.97, tokio, tokio-util (`CancellationToken`), thiserror, reqwest 0.13, humantime, axum 0.8 (fake server for tests), cortex (`../cortex/cortex` path dependency)

**Spec:** `docs/superpowers/specs/2026-09-11-ailoy-desktop-design.md` §5, §9

## Global Constraints

- Working branch: ailoy `feat/desktop` (based on `origin/mem-applied`, including the design-doc commit `b7c845a8`). cortex must sit at `../cortex` with the `feat/exec-timeout` branch (created in Task A2) checked out.
- The working tree must sit where `../cortex` is a sibling directory (`Cargo.toml`'s `cortex = { path = "../cortex/cortex" }`). Recommended: `git switch feat/desktop` in the main checkout, or `git worktree add ../ailoy-desktop feat/desktop`.
- The existing public API (`Agent::run`, `Agent::run_stream`, returning `anyhow`) stays. New types are only added.
- Leave the live tests that call external APIs (they need keys in `.env`) alone. Every new test must be offline (axum fake server). The one exception is the `shell` tool test, which needs the `cortex-local-console` binary (point at it with the `AILOY_CORTEX_CONSOLE` environment variable; the default is `cortex-local-console`).
- Commit messages are conventional commits (`feat(agent): …`). Commit at the end of each Task.
- Commit after `cargo fmt`. Follow the existing files' editing style (comments explain "why").

---

## File layout

| Path | Responsibility |
|---|---|
| `src/agent/control.rs` (new) | `RunControl`, `ToolGate`, `ToolCallRequest`, `ToolDecision`, `AllowAll` |
| `src/agent/error.rs` (new) | `AgentError` |
| `src/agent/rt.rs` (modified) | `run_stream_controlled`, `close_dangling_tool_calls`, `run_stream` delegation |
| `src/agent/builder.rs` (modified) | `AgentBuilder::max_tokens` |
| `src/agent/mod.rs` (modified) | Re-exports |
| `src/agent/test_support.rs` (new, `cfg(test)`) | Fake ChatCompletion SSE server, helper that registers a fake provider |
| `src/lang_model/error.rs` (new) | `ModelError` |
| `src/lang_model/rate_limit.rs` (new) | Headers → `RateLimitInfo` parser |
| `src/lang_model/rt.rs` (modified) | Retry policy, client cache, wiring up header parsing |
| `src/lang_model/mod.rs` (modified) | Re-exports |
| `src/message/rate_limit.rs` (new) | `RateLimitInfo`, `RateLimitWindow` |
| `src/message/message.rs`, `message_delta.rs`, `mod.rs` (modified) | Add the `rate_limit` field, and accumulate it |
| `src/lang_model/impl/api/openai.rs`, `gemini.rs` (modified) | Cached-token parsing |
| `src/tool/impl/builtins/shell.rs` (modified) | Pass `timeout_secs` → `timeout_ms` |
| `Cargo.toml` (modified) | Add `tokio-util`, `humantime` |
| `../cortex/cortex-console-servers/local/src/server/mod.rs` (modified) | Enforce the timeout in `execute` |
| `../cortex/cortex-console-servers/local/tests/exec_timeout.rs` (new) | timeout E2E |

---

### Task A1: Merge develop (#448 Bedrock) into the working branch

**Files:**
- Modify: `Cargo.toml`, `src/lang_model/impl/api/mod.rs`, `src/lang_model/provider.rs` (merge result)
- Delete (conflict resolution): `src/runenv/sandbox.rs` and the rest of `src/runenv/*`

**Interfaces:**
- Produces: the `LangModelAPISchema::Bedrock` variant and `src/lang_model/impl/api/bedrock.rs` exist on the branch. Later Tasks must include a `Bedrock` arm in `match schema`.

- [ ] **Step 1: Check the branch and start the merge**

```bash
git switch feat/desktop
git log --oneline -1          # must read b7c845a8 docs(design): ...
git merge develop
```

Expected: conflicts reported. The conflicting files are usually `Cargo.toml`, `Cargo.lock`, `src/runenv/sandbox.rs` (deleted in ours, modified in theirs), and sometimes `src/tool/impl/builtins/web_fetch.rs`.

- [ ] **Step 2: Apply the conflict-resolution rules**

```bash
# runenv is a module cortex-applied removed: resolve it by deleting
git rm -q src/runenv/sandbox.rs 2>/dev/null || true
git status --short | grep '^UD\|^DU\|^AA\|^UU'
```

- `Cargo.toml`: take ours (mem-applied). Do not bring in the `microsandbox*` dependencies or the `sandbox` feature. Take only the dependencies develop added (the Bedrock-related `sha2`/`hmac` sort, if any). Leave `[features]` with `default = []` alone.
- `src/lang_model/impl/api/mod.rs`: combine both sides — the `LangModelAPISchema::Bedrock` variant, the `BedrockRegion` re-export, `provider_api`'s `Bedrock => Box::new(BedrockUnmarshal)` arm, and `mod bedrock;`.
- `src/lang_model/provider.rs`: take develop's `bedrock()` constructor and the `AWS_BEARER_TOKEN_BEDROCK` block in `Default`.
- `web_fetch.rs`: accept develop's (#443) changes, and in the conflicting parts prefer develop, going with whichever side compiles.
- `Cargo.lock`: run `git checkout --theirs Cargo.lock` and let Step 3's `cargo check` refresh it.

- [ ] **Step 3: Check that it compiles**

```bash
cargo check --all-targets 2>&1 | tail -5
```

Expected: `Finished`. If an error is a reference to `runenv`/`Sandbox`, delete that reference (develop-only code).

- [ ] **Step 4: Check the offline tests pass**

```bash
cargo test --lib lang_model::provider 2>&1 | tail -3
cargo test --lib message 2>&1 | tail -3
```

Expected: `test result: ok` everywhere.

- [ ] **Step 5: Merge commit**

```bash
git add -A
git commit -m "merge: develop into feat/desktop (Bedrock wire #448; drop microsandbox-only sandbox changes)"
```

---

### Task A2: Enforce `timeout_ms` in the cortex local console

**Files:**
- Modify: `../cortex/cortex-console-servers/local/src/server/mod.rs:587-630` (`execute`)
- Create: `../cortex/cortex-console-servers/local/tests/exec_timeout.rs`

**Interfaces:**
- Consumes: `ExecCall { cmd: Vec<String>, timeout_ms: Option<u64> }`, `Error::TIMED_OUT: i64 = -32000`, `refused(code, msg) -> Error`, `finished(io::Result<Output>) -> Response`
- Produces: when `exec` carries a `timeout_ms`, expiry answers `Response::Error(Error{code: TIMED_OUT, message: "killed after {ms}ms"})`. The console session stays alive and takes the next request.

- [ ] **Step 1: Create the cortex branch**

```bash
cd ../cortex && git switch -c feat/exec-timeout main && cd -
```

- [ ] **Step 2: Write the failing E2E test**

`../cortex/cortex-console-servers/local/tests/exec_timeout.rs`:

```rust
//! `timeout_ms` is enforced: a command that outlives it is killed and answered with
//! `TIMED_OUT`, and the session goes on answering afterwards.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};

use cortex::{
    console::{Console, Error, stdio::StdioClient},
    fs::Mount,
};
use tokio::process::Command;

struct Mounted(PathBuf);

impl Mount for Mounted {
    fn mountpoint(&self) -> &Path {
        &self.0
    }
}

async fn console_over(root: &Path) -> anyhow::Result<Console> {
    let mut server = Command::new(env!("CARGO_BIN_EXE_cortex-local-console"));
    server.stderr(Stdio::inherit());
    let client = StdioClient::new(server)?;
    Console::builder()
        .client(client)
        .mount(Mounted(root.to_path_buf()))
        .build()
        .await
}

#[tokio::test]
async fn a_command_past_its_timeout_is_killed_and_reported() {
    let dir = tempfile::tempdir().unwrap();
    let mut console = console_over(dir.path()).await.unwrap();

    let started = Instant::now();
    let err = console
        .exec(["sh", "-c", "sleep 10"], Some(300))
        .await
        .expect_err("a 10s sleep under a 300ms timeout must be refused");
    assert_eq!(err.code(), Some(Error::TIMED_OUT), "{err:?}");
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "the kill must not wait for the command: {:?}",
        started.elapsed()
    );

    // The session survives the kill.
    let ok = console.exec(["echo", "still-here"], Some(5_000)).await.unwrap();
    assert_eq!(ok.stdout, b"still-here\n");
}

#[tokio::test]
async fn a_command_within_its_timeout_is_answered_normally() {
    let dir = tempfile::tempdir().unwrap();
    let mut console = console_over(dir.path()).await.unwrap();
    let ok = console.exec(["sh", "-c", "sleep 0.1; echo done"], Some(5_000)).await.unwrap();
    assert_eq!(ok.code, 0);
    assert_eq!(ok.stdout, b"done\n");
}

#[tokio::test]
async fn no_timeout_means_no_limit() {
    let dir = tempfile::tempdir().unwrap();
    let mut console = console_over(dir.path()).await.unwrap();
    let ok = console.exec(["sh", "-c", "sleep 0.2; echo ok"], None).await.unwrap();
    assert_eq!(ok.stdout, b"ok\n");
}
```

- [ ] **Step 3: Confirm it fails**

```bash
cd ../cortex && cargo test -p cortex-local-console --test exec_timeout 2>&1 | tail -15; cd -
```

Expected: `a_command_past_its_timeout_is_killed_and_reported` FAIL (today an `Ok` response arrives 10 seconds later, or the `elapsed` assertion fails).

- [ ] **Step 4: Implement `execute`**

Replace `execute` in `../cortex/cortex-console-servers/local/src/server/mod.rs` with the following (keep and extend the existing comments). Add `std::time::Duration` to the `use` at the top of the file.

```rust
/// Run one command, and answer with everything it produced.
///
/// One request, one answer: the command is spawned, both of its pipes are drained, and
/// what comes back is how it ended. Nothing arrives on the channel in between, which is
/// why this is a function of the request rather than something threaded through the
/// server.
///
/// `timeout_ms` is a kill, as the protocol promises: the command runs in a process group
/// of its own so that what a shell spawned goes with it, and expiry answers
/// [`TIMED_OUT`](Error::TIMED_OUT) with nothing of the partial output — a killed command
/// has no result to report.
async fn execute(exec: &ExecCall, session: &Session) -> Response {
    let Some((program, args)) = exec.split() else {
        return Response::Error(refused(Error::INVALID_PARAMS, "an empty command"));
    };

    let mut cmd = Command::new(program);
    cmd.args(args)
        .envs(environment(session))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // The future below owns the child; dropping it on expiry kills the direct child.
        // The group kill after it reaches whatever that child spawned.
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);

    if let Some(dir) = session.cwd() {
        cmd.current_dir(dir);
    }

    let child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            let code = match e.kind() {
                io::ErrorKind::NotFound => NOT_FOUND,
                _ => NOT_EXECUTABLE,
            };
            return Response::Error(refused(
                Error::NOT_EXECUTABLE,
                format!("{program}: {e} (a shell would report {code})"),
            ));
        }
    };
    let pid = child.id();

    let waited = match exec.timeout_ms {
        None => child.wait_with_output().await,
        Some(ms) => match tokio::time::timeout(Duration::from_millis(ms), child.wait_with_output()).await {
            Ok(output) => output,
            Err(_elapsed) => {
                // The child went with the dropped future (`kill_on_drop`); this reaches
                // its descendants, which are in the group the spawn put it in.
                #[cfg(unix)]
                if let Some(pid) = pid {
                    // SAFETY: plain libc call on a pid we spawned; a stale pid is a no-op
                    // (ESRCH), never a wrong target, because a group id is not reused
                    // while any member lives.
                    unsafe {
                        libc::killpg(pid as libc::pid_t, libc::SIGKILL);
                    }
                }
                return Response::Error(refused(Error::TIMED_OUT, format!("killed after {ms}ms")));
            }
        },
    };

    finished(waited)
}
```

Change the "Neither timeout is enforced" sentence on lines 76-78 of the module's top doc comment to "`exec.timeout_ms` is enforced by a kill; `init` carries no default."

- [ ] **Step 5: Confirm the tests pass**

```bash
cd ../cortex && cargo test -p cortex-local-console --test exec_timeout 2>&1 | tail -8 && cargo test -p cortex-local-console 2>&1 | grep -E 'test result|FAILED' ; cd -
```

Expected: 3 passed, and the existing `files`/`workfs` tests ok too.

- [ ] **Step 6: Commit (cortex)**

```bash
cd ../cortex && git add -A && git commit -m "feat(local-console): enforce exec timeout_ms with a process-group kill

A command that outlives its timeout is killed and answered TIMED_OUT; the
session keeps answering. Needed by ailoy-desktop, whose shell tool sets a
default timeout." && cd -
```

---

### Task A3: The `shell` tool passes `timeout_secs` to the console

**Files:**
- Modify: `src/tool/impl/builtins/shell.rs`

**Interfaces:**
- Produces: the `shell` argument `timeout_secs` (0 or omitted = the default 600 seconds) is passed to `Console::exec(.., Some(ms))`. On expiry the result is `{"timed_out": true, "exit_code": -1}` (the existing branch stays).

- [ ] **Step 1: Add the failing test** (inside `shell.rs`'s `mod tests`, using the existing `provider()` helper)

```rust
    #[tokio::test]
    async fn test_timeout_secs_kills_and_reports_timed_out() {
        let provider = provider().await;
        let funcs = provider.provide(&[get_shell_tool_desc()]).unwrap();
        let f = funcs.get("shell").unwrap();
        let mut console = test_console().await;
        let started = std::time::Instant::now();
        let msg = f
            .call(to_value!({ "cmd": "sleep 10", "timeout_secs": 1 }), "", &mut console)
            .next()
            .await
            .unwrap()
            .message;
        let v = msg.contents[0].as_value().unwrap();
        assert_eq!(v.pointer("/timed_out").and_then(|b| b.as_bool()), Some(true), "{v:?}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }
```

- [ ] **Step 2: Confirm it fails**

```bash
AILOY_CORTEX_CONSOLE=../cortex/target/debug/cortex-local-console cargo test --lib tool::impl::builtins::shell::tests::test_timeout_secs 2>&1 | tail -5
```

Expected: FAIL (`timed_out` is `false`, and it takes 10 seconds). If the binary is missing, run `(cd ../cortex && cargo build -p cortex-local-console)` first.

- [ ] **Step 3: Implement** — inside `get_shell_tool_func`, compute the timeout right before the `console.exec(...)` call and pass it instead of `None`.

```rust
        // 0 or absent means the default. The protocol's expiry is a kill with no output,
        // so a bound has to exist: an agent that hangs a shell forever hangs the run.
        const DEFAULT_TIMEOUT_SECS: u64 = 600;
        let timeout_ms = args
            .pointer("/timeout_secs")
            .and_then(|v| v.as_integer())
            .filter(|s| *s > 0)
            .map(|s| s as u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .saturating_mul(1000);

        let out = match console.exec(["sh", "-c", cmd.as_str()], Some(timeout_ms)).await {
```

Change the tool description's `timeout_secs` wording to `"Timeout in seconds. 0 or omitted means the default (600)."`

- [ ] **Step 4: Confirm it passes**

```bash
AILOY_CORTEX_CONSOLE=../cortex/target/debug/cortex-local-console cargo test --lib tool::impl::builtins::shell 2>&1 | grep -E 'test result|FAILED'
```

Expected: all ok.

- [ ] **Step 5: Commit**

```bash
git add src/tool/impl/builtins/shell.rs
git commit -m "feat(tool): shell passes timeout_secs to the console (default 600s)"
```

---

### Task A4: `ModelError`, a wider retry policy, and a cached client

**Files:**
- Create: `src/lang_model/error.rs`
- Modify: `src/lang_model/rt.rs` (`LangModel` fields, `send_with_retry`), `src/lang_model/mod.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct ModelError { pub status: Option<u16>, pub retryable: bool, pub message: String, pub attempts: u32 }
  // std::error::Error + Display. It goes out wrapped in an `anyhow::Error`, and `downcast_ref::<ModelError>()` works.
  ```
  Retries: 429 (except a permanent quota), 408, 5xx and transport failures (connect/timeout/request), up to 3 times (4 attempts in all) with exponential backoff (1, 2, 4 seconds; `retry-after` wins; capped at 10 seconds). Every other 4xx fails immediately.

- [ ] **Step 1: Write the failing tests** (added to `src/lang_model/rt.rs`'s `mod tests`; the same axum pattern as the existing 429 test)

```rust
    /// 503 twice then 200: transient server errors are retried like 429.
    #[tokio::test]
    async fn test_retries_5xx_then_succeeds() {
        use std::sync::{Arc, Mutex};
        use axum::{Router, body::Body, response::Response, routing::post};

        let count = Arc::new(Mutex::new(0u32));
        let c = count.clone();
        let app = Router::new().route("/", post(move || {
            let c = c.clone();
            async move {
                let n = { let mut g = c.lock().unwrap(); *g += 1; *g };
                if n <= 2 {
                    Response::builder().status(503).body(Body::from("overloaded")).unwrap()
                } else {
                    Response::builder().status(200).header("content-type", "application/json")
                        .body(Body::from(r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#))
                        .unwrap()
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });

        let lm = LangModel::from_elem("m".into(), LangModelProviderElem::API {
            schema: LangModelAPISchema::ChatCompletion,
            url: format!("http://{addr}/").parse().unwrap(),
            api_key: None,
        });
        // Backoff would sleep 1s+2s; keep the test fast by overriding the base wait.
        let out = lm.run_with_backoff_base(&[Message::new(Role::User).with_contents([crate::message::Part::text("hi")])], &[], &LangModelOptions::default(), std::time::Duration::from_millis(1)).await.unwrap();
        assert_eq!(out.message.contents[0].as_text(), Some("ok"));
        assert_eq!(*count.lock().unwrap(), 3);
    }

    /// 400 is not retried and surfaces as a typed, non-retryable ModelError.
    #[tokio::test]
    async fn test_400_is_typed_and_not_retried() {
        use std::sync::{Arc, Mutex};
        use axum::{Router, body::Body, response::Response, routing::post};

        let count = Arc::new(Mutex::new(0u32));
        let c = count.clone();
        let app = Router::new().route("/", post(move || {
            let c = c.clone();
            async move {
                *c.lock().unwrap() += 1;
                Response::builder().status(400).body(Body::from(r#"{"error":"bad request"}"#)).unwrap()
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });

        let lm = LangModel::from_elem("m".into(), LangModelProviderElem::API {
            schema: LangModelAPISchema::ChatCompletion,
            url: format!("http://{addr}/").parse().unwrap(),
            api_key: None,
        });
        let err = lm.run(&[Message::new(Role::User).with_contents([crate::message::Part::text("hi")])], &[], &LangModelOptions::default()).await.unwrap_err();
        let me = err.downcast_ref::<ModelError>().expect("a ModelError inside the anyhow chain");
        assert_eq!(me.status, Some(400));
        assert!(!me.retryable);
        assert_eq!(me.attempts, 1);
        assert_eq!(*count.lock().unwrap(), 1);
    }
```

If the existing tests build a `LangModel { model, provider }` literal, switch them to `LangModel::from_elem(model, elem)` (added in Step 3).

- [ ] **Step 2: Confirm it fails**

```bash
cargo test --lib lang_model::rt::tests::test_retries_5xx 2>&1 | tail -5
```

Expected: compile failure (`from_elem`, `run_with_backoff_base` and `ModelError` are undefined).

- [ ] **Step 3: Define `ModelError`**

`src/lang_model/error.rs`:

```rust
//! The error a model request ends in, with what a caller can act on.

use thiserror::Error;

/// A request to the model API that did not produce a response.
///
/// `status` is the HTTP status when a response arrived, `None` for a transport failure.
/// `retryable` says whether the same request may succeed later — 429/408/5xx/transport —
/// which is what a UI uses to offer "retry" and what the retry loop already acted on
/// (`attempts` is how many times it tried). `message` is the provider's body, verbatim.
#[derive(Debug, Clone, Error)]
#[error("model request failed{}: {message}", status.map(|s| format!(" (HTTP {s})")).unwrap_or_default())]
pub struct ModelError {
    pub status: Option<u16>,
    pub retryable: bool,
    pub message: String,
    pub attempts: u32,
}
```

Add `mod error; pub use error::ModelError;` to `src/lang_model/mod.rs`.

- [ ] **Step 4: Add a cached client and a constructor to `LangModel`, and rewrite the retry**

`src/lang_model/rt.rs`:

```rust
pub struct LangModel {
    model: String,
    provider: LangModelProviderElem,
    /// One connection pool per model, not per call: a fresh `Client` per request was a
    /// TLS handshake per turn.
    client: reqwest::Client,
}

impl LangModel {
    /// Build directly from a resolved endpoint. `model` is the API-side id.
    pub fn from_elem(model: String, provider: LangModelProviderElem) -> Self {
        Self { model, provider, client: reqwest::Client::new() }
    }
    // Change try_from_provider's `Ok(Self { model: api_model_id, provider: provider_elem })`
    // to `Ok(Self::from_elem(api_model_id, provider_elem))`.
}
```

`run` delegates to `run_with_backoff_base(messages, tools, options, Duration::from_secs(1))`, and the new function takes the existing `run` body but uses `&self.client` instead of `reqwest::Client::new()` and `send_with_retry(&self.client, &url, header_map, &body, provider.as_ref(), backoff_base)` instead of `send_with_retry(...)`. `run_stream` likewise captures `let client = self.client.clone();` outside the stream and passes `Duration::from_secs(1)`.

Replace `send_with_retry`:

```rust
/// POSTs the request, retrying what may recover — 429 (unless the body says the quota is
/// gone for good), 408, 5xx, and transport failures — with exponential backoff from
/// `backoff_base` (1s in production; tests pass ~1ms) capped at 10s, honouring
/// `retry-after` when present. Returns the 2xx response **unconsumed**. Anything else is a
/// [`ModelError`] carrying the status and whether it was retryable.
async fn send_with_retry(
    client: &reqwest::Client,
    url: &str,
    headers: HeaderMap,
    body: &serde_json::Value,
    provider: &(dyn api::ProviderApi + Send + Sync),
    backoff_base: std::time::Duration,
) -> Result<reqwest::Response, ModelError> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(10);
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let response = match client.post(url).headers(headers.clone()).json(body).send().await {
            Ok(r) => r,
            Err(e) => {
                let transient = e.is_connect() || e.is_timeout() || e.is_request();
                if transient && attempt <= MAX_RETRIES {
                    log::warn!("transport error, retrying (attempt {attempt}/{MAX_RETRIES}): {e}");
                    tokio::time::sleep((backoff_base * (1u32 << (attempt - 1))).min(MAX_WAIT)).await;
                    continue;
                }
                return Err(ModelError { status: None, retryable: transient, message: e.to_string(), attempts: attempt });
            }
        };
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map(std::time::Duration::from_secs);
        let text = response.text().await.unwrap_or_default();
        let code = status.as_u16();
        let permanent_quota = code == 429 && provider.is_permanent_quota_error(&text);
        let retryable = !permanent_quota && (code == 429 || code == 408 || (500..600).contains(&code));
        if retryable && attempt <= MAX_RETRIES {
            let wait = retry_after.unwrap_or(backoff_base * (1u32 << (attempt - 1))).min(MAX_WAIT);
            log::warn!("HTTP {code}, retrying after {wait:?} (attempt {attempt}/{MAX_RETRIES}): {text}");
            tokio::time::sleep(wait).await;
            continue;
        }
        if permanent_quota {
            log::warn!("Quota exhausted (429), not retrying: {text}");
        }
        return Err(ModelError { status: Some(code), retryable, message: text, attempts: attempt });
    }
}
```

The `?` at the call sites in `run`/`run_stream` keeps working (`ModelError: std::error::Error` → `anyhow`). Existing tests of the `test_retries_429_then_succeeds` sort use `retry-after: 0`, so they pass unchanged.

- [ ] **Step 5: Confirm it passes**

```bash
cargo test --lib lang_model::rt 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: all ok (live tests follow the skip/ignore rule when there is no key).

- [ ] **Step 6: Commit**

```bash
git add src/lang_model
git commit -m "feat(lang_model): typed ModelError, retry 5xx/transport, cached reqwest client"
```

---

### Task A5: The `RateLimitInfo` type and the header parser

**Files:**
- Create: `src/message/rate_limit.rs`, `src/lang_model/rate_limit.rs`
- Modify: `src/message/mod.rs`, `src/message/message.rs`, `src/message/message_delta.rs`, `src/lang_model/mod.rs`, `src/lang_model/rt.rs`, `Cargo.toml`, and every file holding a `MessageOutput { .. }` literal

**Interfaces:**
- Produces:
  ```rust
  pub struct RateLimitWindow { pub limit: Option<u64>, pub remaining: Option<u64>, pub reset_at_ms: Option<u64> }
  pub struct RateLimitInfo { pub requests: Option<RateLimitWindow>, pub tokens: Option<RateLimitWindow>, pub input_tokens: Option<RateLimitWindow>, pub output_tokens: Option<RateLimitWindow> }
  impl RateLimitInfo { pub fn is_empty(&self) -> bool }
  // MessageOutput / MessageDeltaOutput: pub rate_limit: Option<RateLimitInfo>  (serde skip_if_none)
  pub(crate) fn lang_model::rate_limit::parse_rate_limit(schema: &LangModelAPISchema, headers: &HeaderMap, now: SystemTime) -> Option<RateLimitInfo>
  ```
  It rides on the first delta when streaming, and on the `MessageOutput` in the blocking `run`. Accumulation is `other.or(self)`.

- [ ] **Step 1: Add the dependency**

Add `humantime = "2"` to `Cargo.toml`'s `[dependencies]`.

- [ ] **Step 2: Define the types**

`src/message/rate_limit.rs`:

```rust
use serde::{Deserialize, Serialize};

/// One rate-limit window as a provider reports it in response headers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RateLimitWindow {
    pub limit: Option<u64>,
    pub remaining: Option<u64>,
    /// When the window is fully replenished, as Unix epoch milliseconds.
    pub reset_at_ms: Option<u64>,
}

impl RateLimitWindow {
    fn is_empty(&self) -> bool {
        self.limit.is_none() && self.remaining.is_none() && self.reset_at_ms.is_none()
    }
}

/// Rate-limit headroom read off one response. Anthropic reports requests, unified tokens,
/// input tokens and output tokens; OpenAI-shaped APIs report requests and tokens; Gemini
/// and Bedrock report nothing, and are `None` upstream rather than an empty value here.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RateLimitInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests: Option<RateLimitWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<RateLimitWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<RateLimitWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<RateLimitWindow>,
}

impl RateLimitInfo {
    pub fn is_empty(&self) -> bool {
        [&self.requests, &self.tokens, &self.input_tokens, &self.output_tokens]
            .iter()
            .all(|w| w.as_ref().is_none_or(|w| w.is_empty()))
    }
}
```

`src/message/mod.rs`: `mod rate_limit; pub use rate_limit::{RateLimitInfo, RateLimitWindow};`

Add a field to `MessageOutput` in `src/message/message.rs` (at the end):

```rust
    /// Rate-limit headroom the provider reported with this response, when it reports any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimitInfo>,
```

Add the same field to `MessageDeltaOutput` in `src/message/message_delta.rs`; `rate_limit: None` in `new()`; `let rate_limit = other.rate_limit.or(self.rate_limit);` in `accumulate`, included in the resulting struct; `rate_limit: self.rate_limit` in `finish`'s `MessageOutput { .. }`; `rate_limit: out.rate_limit` in `From<MessageOutput> for MessageDeltaOutput`.

- [ ] **Step 3: Fix up the remaining literals**

```bash
grep -rn 'source_agent: None' src --include=*.rs | grep -v 'rate_limit' | cut -d: -f1 | sort -u
```

In each file listed (`src/tool/func.rs`, `src/agent/rt.rs`, `src/agent/subagent.rs`, `src/lang_model/impl/api/*.rs` and so on), add `rate_limit: None,` to every `MessageOutput { ... source_agent: None }` / `MessageDeltaOutput { ... }` literal. `cargo check --all-targets` points out the ones you missed.

- [ ] **Step 4: Write the parser tests** (`mod tests` in `src/lang_model/rate_limit.rs`)

```rust
#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use reqwest::header::{HeaderMap, HeaderValue};

    use super::*;
    use crate::lang_model::LangModelAPISchema;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn anthropic_headers_fill_all_four_windows() {
        let h = headers(&[
            ("anthropic-ratelimit-requests-limit", "4000"),
            ("anthropic-ratelimit-requests-remaining", "3999"),
            ("anthropic-ratelimit-requests-reset", "2026-09-11T10:00:00Z"),
            ("anthropic-ratelimit-tokens-limit", "10400000"),
            ("anthropic-ratelimit-tokens-remaining", "10399000"),
            ("anthropic-ratelimit-input-tokens-limit", "10000000"),
            ("anthropic-ratelimit-input-tokens-remaining", "9999000"),
            ("anthropic-ratelimit-output-tokens-limit", "400000"),
            ("anthropic-ratelimit-output-tokens-remaining", "399000"),
            ("anthropic-ratelimit-output-tokens-reset", "2026-09-11T10:00:30Z"),
        ]);
        let info = parse_rate_limit(&LangModelAPISchema::Anthropic, &h, UNIX_EPOCH).unwrap();
        assert_eq!(info.requests.as_ref().unwrap().limit, Some(4000));
        assert_eq!(info.requests.as_ref().unwrap().remaining, Some(3999));
        // 2026-09-11T10:00:00Z
        assert_eq!(info.requests.as_ref().unwrap().reset_at_ms, Some(1_789_120_800_000));
        assert_eq!(info.tokens.as_ref().unwrap().remaining, Some(10_399_000));
        assert_eq!(info.input_tokens.as_ref().unwrap().limit, Some(10_000_000));
        assert_eq!(info.output_tokens.as_ref().unwrap().reset_at_ms, Some(1_789_120_830_000));
    }

    #[test]
    fn openai_headers_fill_requests_and_tokens_with_relative_reset() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000);
        let h = headers(&[
            ("x-ratelimit-limit-requests", "10000"),
            ("x-ratelimit-remaining-requests", "9999"),
            ("x-ratelimit-reset-requests", "6m0s"),
            ("x-ratelimit-limit-tokens", "30000000"),
            ("x-ratelimit-remaining-tokens", "29999500"),
            ("x-ratelimit-reset-tokens", "1.5s"),
        ]);
        for schema in [LangModelAPISchema::OpenAI, LangModelAPISchema::ChatCompletion] {
            let info = parse_rate_limit(&schema, &h, now).unwrap();
            assert_eq!(info.requests.as_ref().unwrap().reset_at_ms, Some(1_000_000 + 360_000));
            assert_eq!(info.tokens.as_ref().unwrap().reset_at_ms, Some(1_000_000 + 1_500));
            assert_eq!(info.tokens.as_ref().unwrap().remaining, Some(29_999_500));
            assert!(info.input_tokens.is_none());
        }
    }

    #[test]
    fn no_headers_is_none_not_empty() {
        assert!(parse_rate_limit(&LangModelAPISchema::OpenAI, &HeaderMap::new(), UNIX_EPOCH).is_none());
        assert!(parse_rate_limit(&LangModelAPISchema::Gemini, &headers(&[("x-ratelimit-limit-requests", "1")]), UNIX_EPOCH).is_none());
    }

    #[test]
    fn duration_parser_handles_openai_shapes() {
        assert_eq!(parse_reset_duration("6m0s"), Some(Duration::from_secs(360)));
        assert_eq!(parse_reset_duration("1.5s"), Some(Duration::from_millis(1500)));
        assert_eq!(parse_reset_duration("120ms"), Some(Duration::from_millis(120)));
        assert_eq!(parse_reset_duration("1h2m3s"), Some(Duration::from_secs(3723)));
        assert_eq!(parse_reset_duration("abc"), None);
    }
}
```

- [ ] **Step 5: Confirm it fails**

```bash
cargo test --lib lang_model::rate_limit 2>&1 | tail -5
```

Expected: compile failure (no such module).

- [ ] **Step 6: Implement the parser** — `src/lang_model/rate_limit.rs`

```rust
//! Rate-limit headroom, read off response headers per wire schema.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::HeaderMap;

use crate::{
    lang_model::LangModelAPISchema,
    message::{RateLimitInfo, RateLimitWindow},
};

/// The headroom `headers` report, or `None` when the schema reports none (Gemini, Bedrock)
/// or the headers are absent. `now` anchors relative resets (OpenAI's `6m0s`).
pub(crate) fn parse_rate_limit(
    schema: &LangModelAPISchema,
    headers: &HeaderMap,
    now: SystemTime,
) -> Option<RateLimitInfo> {
    let info = match schema {
        LangModelAPISchema::Anthropic => parse_anthropic(headers),
        LangModelAPISchema::OpenAI | LangModelAPISchema::ChatCompletion => parse_openai(headers, now),
        LangModelAPISchema::Gemini | LangModelAPISchema::Bedrock => return None,
    };
    (!info.is_empty()).then_some(info)
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    header_str(headers, name).and_then(|s| s.trim().parse().ok())
}

fn epoch_ms(t: SystemTime) -> Option<u64> {
    t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_millis() as u64)
}

fn window(limit: Option<u64>, remaining: Option<u64>, reset_at_ms: Option<u64>) -> Option<RateLimitWindow> {
    let w = RateLimitWindow { limit, remaining, reset_at_ms };
    (w.limit.is_some() || w.remaining.is_some() || w.reset_at_ms.is_some()).then_some(w)
}

fn parse_anthropic(headers: &HeaderMap) -> RateLimitInfo {
    // `anthropic-ratelimit-<kind>-{limit,remaining,reset}`, reset in RFC 3339.
    let win = |kind: &str| {
        let get = |suffix: &str| format!("anthropic-ratelimit-{kind}-{suffix}");
        let reset = header_str(headers, &get("reset"))
            .and_then(|s| humantime::parse_rfc3339_weak(s.trim()).ok())
            .and_then(epoch_ms);
        window(header_u64(headers, &get("limit")), header_u64(headers, &get("remaining")), reset)
    };
    RateLimitInfo {
        requests: win("requests"),
        tokens: win("tokens"),
        input_tokens: win("input-tokens"),
        output_tokens: win("output-tokens"),
    }
}

fn parse_openai(headers: &HeaderMap, now: SystemTime) -> RateLimitInfo {
    // `x-ratelimit-{limit,remaining,reset}-<kind>`, reset as a Go-style duration ("6m0s").
    let win = |kind: &str| {
        let reset = header_str(headers, &format!("x-ratelimit-reset-{kind}"))
            .and_then(parse_reset_duration)
            .and_then(|d| epoch_ms(now + d));
        window(
            header_u64(headers, &format!("x-ratelimit-limit-{kind}")),
            header_u64(headers, &format!("x-ratelimit-remaining-{kind}")),
            reset,
        )
    };
    RateLimitInfo {
        requests: win("requests"),
        tokens: win("tokens"),
        input_tokens: None,
        output_tokens: None,
    }
}

/// `"1h2m3s"`, `"6m0s"`, `"1.5s"`, `"120ms"` — number/unit pairs, concatenated.
pub(crate) fn parse_reset_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut total = Duration::ZERO;
    let mut rest = s;
    while !rest.is_empty() {
        let num_end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(rest.len());
        let (num, tail) = rest.split_at(num_end);
        let value: f64 = num.parse().ok()?;
        let unit_end = tail.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(tail.len());
        let (unit, tail) = tail.split_at(unit_end);
        let secs = match unit {
            "h" => value * 3600.0,
            "m" => value * 60.0,
            "s" => value,
            "ms" => value / 1000.0,
            _ => return None,
        };
        total += Duration::from_secs_f64(secs);
        rest = tail;
    }
    Some(total)
}
```

Add `pub(crate) mod rate_limit;` to `src/lang_model/mod.rs`. If `LangModelAPISchema` has no `Bedrock` (check the Task A1 merge), drop that arm.

- [ ] **Step 7: Wire it into the response** — `src/lang_model/rt.rs`

In `run_with_backoff_base`:

```rust
                let response = send_with_retry(&self.client, &url, header_map, &body, provider.as_ref(), backoff_base).await?;
                let rate_limit = crate::lang_model::rate_limit::parse_rate_limit(schema, response.headers(), std::time::SystemTime::now());
                let response_text = response.text().await?;
                // ...
                let mut out = delta_output.finish()?;
                out.rate_limit = rate_limit;
                Ok(out)
```

In `run_stream`: make a `let schema = schema.clone();` outside the stream, and inside the stream, right after the response arrives, put

```rust
            let mut rate_limit = crate::lang_model::rate_limit::parse_rate_limit(&schema, response.headers(), std::time::SystemTime::now());
```

and immediately before each of the two `yield output;` sites, put

```rust
                        if let Some(rl) = rate_limit.take() { output.rate_limit = Some(rl); }
```

(Change the `output` binding in both places to `let mut output`/`Some(mut output)`.)

- [ ] **Step 8: Test the stream wiring** (`src/lang_model/rt.rs` `mod tests`)

```rust
    /// Headers on the SSE response land on the first delta only.
    #[tokio::test]
    async fn test_stream_carries_rate_limit_on_first_delta() {
        use axum::{Router, body::Body, response::Response, routing::post};
        let app = Router::new().route("/", post(|| async {
            let sse = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"}}]}\n\n\
                       data: {\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n\
                       data: [DONE]\n\n";
            Response::builder().status(200)
                .header("content-type", "text/event-stream")
                .header("x-ratelimit-limit-requests", "100")
                .header("x-ratelimit-remaining-requests", "99")
                .body(Body::from(sse)).unwrap()
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let lm = LangModel::from_elem("m".into(), LangModelProviderElem::API {
            schema: LangModelAPISchema::ChatCompletion,
            url: format!("http://{addr}/").parse().unwrap(),
            api_key: None,
        });
        let deltas: Vec<_> = lm.run_stream(&[Message::new(Role::User).with_contents([crate::message::Part::text("hi")])], &[], &LangModelOptions::default())
            .collect::<Vec<_>>().await.into_iter().map(|r| r.unwrap()).collect();
        assert!(deltas.len() >= 2);
        assert_eq!(deltas[0].rate_limit.as_ref().unwrap().requests.as_ref().unwrap().remaining, Some(99));
        assert!(deltas[1..].iter().all(|d| d.rate_limit.is_none()));
    }
```

- [ ] **Step 9: Check everything**

```bash
cargo check --all-targets 2>&1 | tail -3
cargo test --lib lang_model::rate_limit lang_model::rt::tests::test_stream_carries message 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: it compiles, tests ok.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(lang_model,message): RateLimitInfo parsed from Anthropic/OpenAI headers onto outputs"
```

---

### Task A6: Fill in cached-token parsing (OpenAI Responses, Gemini)

**Files:**
- Modify: `src/lang_model/impl/api/openai.rs:521-536`, `src/lang_model/impl/api/gemini.rs:380-396`

**Interfaces:**
- Produces: `TokenUsage.cache_read_input_tokens` is filled from OpenAI Responses' `usage.input_tokens_details.cached_tokens` and Gemini's `usageMetadata.cachedContentTokenCount`.

- [ ] **Step 1: The failing tests** — `openai.rs` `mod tests`:

```rust
    #[test]
    fn test_unmarshal_usage_cached_tokens() {
        let response = to_value!({
            "status": "completed",
            "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"x"}]}],
            "usage": {"input_tokens": 200, "output_tokens": 75, "input_tokens_details": {"cached_tokens": 150}}
        });
        let usage = OpenAIUnmarshal.unmarshal(response).unwrap().usage.unwrap();
        assert_eq!(usage.cache_read_input_tokens, Some(150));
    }
```

`gemini.rs` `mod tests`:

```rust
    #[test]
    fn test_unmarshal_usage_cached_content_tokens() {
        let val = to_value!({
            "candidates": [{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"STOP"}],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 6, "cachedContentTokenCount": 10}
        });
        let usage = GeminiUnmarshal.unmarshal(val).unwrap().usage.unwrap();
        assert_eq!(usage.cache_read_input_tokens, Some(10));
    }
```

(Match how the existing tests call the `Unmarshal` trait — `T::default().unmarshal(val)` and the like — by following `test_unmarshal_usage` in the same file.)

- [ ] **Step 2: Confirm it fails**

```bash
cargo test --lib 'lang_model::impl::api::openai::tests::test_unmarshal_usage_cached' 'lang_model::impl::api::gemini::tests::test_unmarshal_usage_cached' 2>&1 | grep -E 'test result|panicked'
```

Expected: FAIL(`None`).

- [ ] **Step 3: Implement** — in `openai.rs`'s usage parsing, `cache_read_input_tokens: u.pointer("/input_tokens_details/cached_tokens").and_then(|v| v.as_u64())` (reach the value with the same helper the file's existing code uses — `.get("input_tokens")` and so on). In `gemini.rs`'s `parse_usage`, `cache_read_input_tokens: u.get("cachedContentTokenCount").and_then(as_u64)`. `cache_creation_input_tokens` stays `None` in both places.

- [ ] **Step 4: Confirm it passes, then commit**

```bash
cargo test --lib lang_model::impl::api 2>&1 | grep -E 'test result|FAILED'
git add src/lang_model/impl/api && git commit -m "fix(lang_model): report cached input tokens for OpenAI Responses and Gemini"
```

---

### Task A7: The `AgentError`, `RunControl` and `ToolGate` types

**Files:**
- Create: `src/agent/error.rs`, `src/agent/control.rs`
- Modify: `src/agent/mod.rs`, `Cargo.toml`

**Interfaces:**
- Produces:
  ```rust
  pub enum AgentError { Cancelled, MaxTurns { turns: u32 }, Model(ModelError), Tool(anyhow::Error), Console(anyhow::Error), Other(anyhow::Error) }
  impl AgentError { pub fn from_anyhow(e: anyhow::Error) -> Self }   // finds a ModelError and classifies it as Model
  pub struct RunControl { pub cancel: CancellationToken, pub max_turns: Option<u32>, pub tool_gate: Arc<dyn ToolGate> }  // Default = unbounded/AllowAll
  pub struct ToolCallRequest<'a> { pub id: &'a str, pub name: &'a str, pub arguments: &'a Value }
  pub enum ToolDecision { Allow, Deny { reason: String } }
  #[async_trait] pub trait ToolGate: Send + Sync { async fn review(&self, call: ToolCallRequest<'_>) -> ToolDecision; }
  pub struct AllowAll;
  ```

- [ ] **Step 1: Add the dependency** — add `tokio-util = "0.7"` to `Cargo.toml`'s `[dependencies]`.

- [ ] **Step 2: Write the type tests** — at the bottom of `src/agent/control.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn allow_all_allows() {
        let gate = AllowAll;
        let args = crate::datatype::Value::object_empty();
        let d = gate.review(ToolCallRequest { id: "c1", name: "shell", arguments: &args }).await;
        assert!(matches!(d, ToolDecision::Allow));
    }

    #[test]
    fn default_control_is_unbounded_and_open() {
        let ctl = RunControl::default();
        assert!(ctl.max_turns.is_none());
        assert!(!ctl.cancel.is_cancelled());
    }

    #[test]
    fn from_anyhow_classifies_model_errors() {
        let me = crate::lang_model::ModelError { status: Some(401), retryable: false, message: "no".into(), attempts: 1 };
        let e: anyhow::Error = me.into();
        assert!(matches!(crate::agent::AgentError::from_anyhow(e), crate::agent::AgentError::Model(m) if m.status == Some(401)));
        assert!(matches!(crate::agent::AgentError::from_anyhow(anyhow::anyhow!("x")), crate::agent::AgentError::Other(_)));
    }
}
```

- [ ] **Step 3: Implement**

`src/agent/error.rs`:

```rust
//! How a controlled run ends when it does not end with a message.

use thiserror::Error;

use crate::lang_model::ModelError;

/// The reasons a run stops short. `Cancelled` and `MaxTurns` leave the history
/// consistent (every tool call answered, a partial answer committed); the rest report
/// the failing layer so a caller can decide whether retrying makes sense.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("run cancelled")]
    Cancelled,
    #[error("turn limit reached after {turns} model calls")]
    MaxTurns { turns: u32 },
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("tool execution failed: {0}")]
    Tool(#[source] anyhow::Error),
    #[error("console unavailable: {0}")]
    Console(#[source] anyhow::Error),
    #[error(transparent)]
    Other(anyhow::Error),
}

impl AgentError {
    /// Classify an `anyhow` error from the model layer: a [`ModelError`] inside becomes
    /// [`AgentError::Model`]; anything else is [`AgentError::Other`].
    pub fn from_anyhow(e: anyhow::Error) -> Self {
        match e.downcast::<ModelError>() {
            Ok(m) => AgentError::Model(m),
            Err(e) => AgentError::Other(e),
        }
    }
}
```

`src/agent/control.rs`:

```rust
//! What a caller can do to a run while it runs: stop it, bound it, and vet its tool calls.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::datatype::Value;

/// One tool call the model asked for, before it runs.
pub struct ToolCallRequest<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub arguments: &'a Value,
}

/// A gate's answer. `Deny` becomes the tool's result — the model reads the reason and
/// continues — rather than an error, so a refusal never wedges the history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolDecision {
    Allow,
    Deny { reason: String },
}

/// Reviews each tool call between the model asking and the runtime executing. An
/// implementation may await a person; the run waits with it.
#[async_trait]
pub trait ToolGate: Send + Sync {
    async fn review(&self, call: ToolCallRequest<'_>) -> ToolDecision;
}

/// The default gate: everything runs.
pub struct AllowAll;

#[async_trait]
impl ToolGate for AllowAll {
    async fn review(&self, _call: ToolCallRequest<'_>) -> ToolDecision {
        ToolDecision::Allow
    }
}

/// Controls for one `run_stream_controlled` call.
#[derive(Clone)]
pub struct RunControl {
    /// Cancel at any await point. The runtime commits what it has and answers pending
    /// tool calls with stubs before returning `AgentError::Cancelled`.
    pub cancel: CancellationToken,
    /// Upper bound on model calls in this run. `None` is unbounded (the pre-existing
    /// behaviour of `run_stream`).
    pub max_turns: Option<u32>,
    pub tool_gate: Arc<dyn ToolGate>,
}

impl Default for RunControl {
    fn default() -> Self {
        Self {
            cancel: CancellationToken::new(),
            max_turns: None,
            tool_gate: Arc::new(AllowAll),
        }
    }
}
```

Add `mod control; mod error; pub use control::*; pub use error::AgentError;` to `src/agent/mod.rs`.

- [ ] **Step 4: Confirm it passes and commit**

```bash
cargo test --lib agent::control 2>&1 | grep -E 'test result|FAILED'
git add Cargo.toml Cargo.lock src/agent && git commit -m "feat(agent): AgentError, RunControl and ToolGate types"
```

---

### Task A8: Test support — a fake ChatCompletion SSE server and provider registration

**Files:**
- Create: `src/agent/test_support.rs`
- Modify: `src/agent/mod.rs` (`#[cfg(test)] pub(crate) mod test_support;`)

**Interfaces:**
- Produces (test-only):
  ```rust
  pub(crate) async fn spawn_sse_server(bodies: Vec<String>, chunk_delay: Option<Duration>) -> (SocketAddr, Arc<AtomicU32> /*call count*/)
  pub(crate) fn sse_text(text_chunks: &[&str]) -> String                       // assistant text → stop
  pub(crate) fn sse_tool_call(id: &str, name: &str, args_json: &str) -> String  // tool_calls → tool_calls finish
  pub(crate) fn register_fake_provider(name: &'static str, addr: SocketAddr, tools: Vec<(&str, ToolDesc, ToolFunc)>) -> &'static str
  pub(crate) fn user(text: &str) -> Message
  ```
  On its i-th call the fake server answers `bodies[min(i, len-1)]` as `text/event-stream`. With a `chunk_delay`, it waits that long before each `\n\n`-delimited event.

- [ ] **Step 1: Implement**

```rust
//! Offline scaffolding for agent-loop tests: a scripted ChatCompletion SSE server and a
//! provider bundle pointing at it. No network beyond loopback, no keys.

use std::{
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use axum::{Router, body::Body, response::Response, routing::post};
use futures::StreamExt as _;

use crate::{
    agent::{AgentProvider, get_agent_providers_mut},
    lang_model::{LangModelProvider, get_lm_providers_mut},
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolFunc, ToolProvider, get_tool_providers_mut},
};

pub(crate) async fn spawn_sse_server(
    bodies: Vec<String>,
    chunk_delay: Option<Duration>,
) -> (SocketAddr, Arc<AtomicU32>) {
    let calls = Arc::new(AtomicU32::new(0));
    let bodies = Arc::new(bodies);
    let counter = calls.clone();
    let app = Router::new().route(
        "/",
        post(move || {
            let bodies = bodies.clone();
            let counter = counter.clone();
            async move {
                let i = counter.fetch_add(1, Ordering::SeqCst) as usize;
                let body = bodies[i.min(bodies.len() - 1)].clone();
                let events: Vec<String> = body
                    .split("\n\n")
                    .filter(|e| !e.trim().is_empty())
                    .map(|e| format!("{e}\n\n"))
                    .collect();
                let stream = futures::stream::iter(events).then(move |e| async move {
                    if let Some(d) = chunk_delay {
                        tokio::time::sleep(d).await;
                    }
                    Ok::<_, std::io::Error>(axum::body::Bytes::from(e))
                });
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, calls)
}

pub(crate) fn sse_text(text_chunks: &[&str]) -> String {
    let mut s = String::new();
    s.push_str("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n");
    for c in text_chunks {
        let c = c.replace('"', "\\\"");
        s.push_str(&format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{c}\"}}}}]}}\n\n"));
    }
    s.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
    s.push_str("data: [DONE]\n\n");
    s
}

pub(crate) fn sse_tool_call(id: &str, name: &str, args_json: &str) -> String {
    let args = args_json.replace('"', "\\\"");
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"role\":\"assistant\",\"tool_calls\":[{{\"index\":0,\"id\":\"{id}\",\"type\":\"function\",\"function\":{{\"name\":\"{name}\",\"arguments\":\"{args}\"}}}}]}}}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n\
         data: [DONE]\n\n"
    )
}

/// Register `name` as a lang-model provider (pattern `fake/*` → the server), a tool
/// provider holding exactly `tools`, and an agent-provider bundle. Idempotent per name;
/// the registries are process-global, so use a distinct name per test.
pub(crate) fn register_fake_provider(
    name: &'static str,
    addr: SocketAddr,
    tools: Vec<(&str, ToolDesc, ToolFunc)>,
) -> &'static str {
    {
        let mut lmps = get_lm_providers_mut();
        let mut lmp = LangModelProvider::new();
        lmp.insert(
            "fake/*".into(),
            LangModelProvider::chat_completion(&format!("http://{addr}/"), None).unwrap(),
        );
        lmps.insert(name.to_string(), lmp);
    }
    {
        let mut tps = get_tool_providers_mut();
        let mut tp = ToolProvider::empty();
        for (tool_name, _desc, func) in &tools {
            tp.insert_func(*tool_name, func.clone());
        }
        tps.insert(name.to_string(), tp);
    }
    get_agent_providers_mut().insert(name.to_string(), AgentProvider::new(name, name));
    name
}

pub(crate) fn user(text: &str) -> Message {
    Message::new(Role::User).with_contents([Part::text(text)])
}
```

- [ ] **Step 2: Check that it compiles**

```bash
cargo check --all-targets 2>&1 | tail -3
```

Check in `src/lang_model/impl/api/chat_completion.rs` whether `LangModelProvider::chat_completion`'s signature is `(url: &str, api_key: Option<String>) -> anyhow::Result<LangModelProviderElem>`, and match it if it differs.

- [ ] **Step 3: Commit**

```bash
git add src/agent && git commit -m "test(agent): scripted SSE server and fake provider helpers"
```

---

### Task A9: `close_dangling_tool_calls` and `run_stream_controlled`

**Files:**
- Modify: `src/agent/rt.rs`

**Interfaces:**
- Consumes: the Task A7 types, the A8 helpers, `MessageOutput.rate_limit` (A5)
- Produces:
  ```rust
  impl Agent {
      pub fn run_stream_controlled(&mut self, query: Message, ctl: RunControl)
          -> Pin<Box<impl Stream<Item = Result<MessageDeltaOutput, AgentError>> + Send + '_>>;
      pub(crate) fn close_dangling_tool_calls(history: &mut Vec<Message>, note: &str);
  }
  pub const INTERRUPTED_BY_CANCEL: &str = "[Interrupted: cancelled before this tool call completed]";
  pub const INTERRUPTED_BY_FAILURE: &str = "[Interrupted: tool execution failed before this tool call completed]";
  ```
  `run_stream` becomes `run_stream_controlled(query, RunControl::default())` with an `anyhow` conversion on top.

- [ ] **Step 1: Unit test — stub insertion** (`rt.rs` `mod tests`)

```rust
    #[test]
    fn close_dangling_stubs_only_unanswered_calls_of_the_last_batch() {
        let mut h = vec![
            Message::new(Role::User).with_contents([Part::text("q")]),
            Message::new(Role::Assistant).with_tool_calls([
                Part::function("c1", "shell", to_value!({})),
                Part::function("c2", "shell", to_value!({})),
            ]),
            Message::new(Role::Tool).with_id("c1").with_contents([Part::text("done")]),
        ];
        Agent::close_dangling_tool_calls(&mut h, INTERRUPTED_BY_CANCEL);
        assert_eq!(h.len(), 4);
        assert_eq!(h[3].role, Role::Tool);
        assert_eq!(h[3].id.as_deref(), Some("c2"));
        assert_eq!(h[3].contents[0].as_text(), Some(INTERRUPTED_BY_CANCEL));
        // Idempotent.
        Agent::close_dangling_tool_calls(&mut h, INTERRUPTED_BY_CANCEL);
        assert_eq!(h.len(), 4);
    }
```

- [ ] **Step 2: Four integration tests** (`rt.rs` `mod tests`; `use crate::agent::test_support::*;`, `use crate::agent::{RunControl, ToolGate, ToolCallRequest, ToolDecision, AgentError, INTERRUPTED_BY_CANCEL};`)

```rust
    fn slow_tool(secs: u64) -> (&'static str, ToolDesc, ToolFunc) {
        let desc = ToolDescBuilder::new("slow").description("sleeps").parameters(to_value!({"type":"object","properties":{}})).build();
        let func = crate::tool_func!(async |_args: Value| -> Value {
            tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
            Value::string("slept")
        });
        ("slow", desc, func)
    }

    fn fast_tool() -> (&'static str, ToolDesc, ToolFunc) {
        let desc = ToolDescBuilder::new("fast").description("returns").parameters(to_value!({"type":"object","properties":{}})).build();
        let func = crate::tool_func!(|_args: Value| -> Value { Value::string("ok") });
        ("fast", desc, func)
    }

    async fn drain(stream: impl Stream<Item = Result<MessageDeltaOutput, AgentError>>) -> Result<Vec<MessageDeltaOutput>, AgentError> {
        let mut out = Vec::new();
        let mut s = std::pin::pin!(stream);
        while let Some(item) = s.next().await {
            out.push(item?);
        }
        Ok(out)
    }

    #[tokio::test]
    async fn cancel_during_model_stream_commits_partial_text() {
        let (addr, _) = spawn_sse_server(vec![sse_text(&["Hel", "lo", " world"])], Some(std::time::Duration::from_millis(150))).await;
        let p = register_fake_provider("ctl_cancel_model", addr, vec![]);
        let mut agent = Agent::try_with_provider(AgentSpec::new("fake/m"), p).unwrap();
        let ctl = RunControl::default();
        let cancel = ctl.cancel.clone();
        let mut stream = agent.run_stream_controlled(user("hi"), ctl);
        let mut saw_text = false;
        let mut ended = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(d) => {
                    if d.delta.contents.iter().any(|p| matches!(p, crate::message::PartDelta::Text { .. })) && !saw_text {
                        saw_text = true;
                        cancel.cancel();
                    }
                }
                Err(e) => { ended = Some(e); break; }
            }
        }
        drop(stream);
        assert!(matches!(ended, Some(AgentError::Cancelled)), "{ended:?}");
        let h = agent.get_history();
        assert_eq!(h.len(), 2, "{h:?}");
        assert_eq!(h[1].role, Role::Assistant);
        let text = h[1].contents[0].as_text().unwrap();
        assert!(text.starts_with("Hel") && text.len() < "Hello world".len(), "{text:?}");
    }

    #[tokio::test]
    async fn cancel_during_tool_execution_stubs_the_pending_call() {
        let (addr, _) = spawn_sse_server(vec![sse_tool_call("call_1", "slow", "{}"), sse_text(&["never"])], None).await;
        let p = register_fake_provider("ctl_cancel_tool", addr, vec![slow_tool(30)]);
        let spec = AgentSpec::new("fake/m").tool(slow_tool(30).1);
        let mut agent = Agent::try_with_provider(spec, p).unwrap();
        let ctl = RunControl::default();
        let cancel = ctl.cancel.clone();
        tokio::spawn(async move { tokio::time::sleep(std::time::Duration::from_millis(500)).await; cancel.cancel(); });
        let started = std::time::Instant::now();
        let err = drain(agent.run_stream_controlled(user("go"), ctl)).await.unwrap_err();
        assert!(matches!(err, AgentError::Cancelled));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        let h = agent.get_history();
        let last = h.last().unwrap();
        assert_eq!(last.role, Role::Tool);
        assert_eq!(last.id.as_deref(), Some("call_1"));
        assert_eq!(last.contents[0].as_text(), Some(INTERRUPTED_BY_CANCEL));
    }

    #[tokio::test]
    async fn max_turns_stops_with_a_consistent_history() {
        let (addr, calls) = spawn_sse_server(vec![sse_tool_call("call_x", "fast", "{}")], None).await;
        let p = register_fake_provider("ctl_max_turns", addr, vec![fast_tool()]);
        let spec = AgentSpec::new("fake/m").tool(fast_tool().1);
        let mut agent = Agent::try_with_provider(spec, p).unwrap();
        let ctl = RunControl { max_turns: Some(2), ..Default::default() };
        let err = drain(agent.run_stream_controlled(user("loop"), ctl)).await.unwrap_err();
        assert!(matches!(err, AgentError::MaxTurns { turns: 2 }), "{err:?}");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        let h = agent.get_history();
        // user, assistant(tool_call), tool, assistant(tool_call), tool
        assert_eq!(h.len(), 5, "{h:?}");
        assert_eq!(h.last().unwrap().role, Role::Tool);
    }

    struct DenyAll;
    #[async_trait::async_trait]
    impl ToolGate for DenyAll {
        async fn review(&self, _c: ToolCallRequest<'_>) -> ToolDecision {
            ToolDecision::Deny { reason: "policy says no".into() }
        }
    }

    #[tokio::test]
    async fn denied_tool_call_becomes_a_tool_result_and_the_run_continues() {
        let (addr, calls) = spawn_sse_server(vec![sse_tool_call("call_d", "fast", "{}"), sse_text(&["fine"])], None).await;
        let p = register_fake_provider("ctl_deny", addr, vec![fast_tool()]);
        let spec = AgentSpec::new("fake/m").tool(fast_tool().1);
        let mut agent = Agent::try_with_provider(spec, p).unwrap();
        let ctl = RunControl { tool_gate: std::sync::Arc::new(DenyAll), ..Default::default() };
        let deltas = drain(agent.run_stream_controlled(user("try"), ctl)).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        let h = agent.get_history();
        assert_eq!(h[2].role, Role::Tool);
        assert_eq!(h[2].id.as_deref(), Some("call_d"));
        let v = h[2].contents[0].as_value().unwrap();
        assert!(v.pointer("/error").and_then(|e| e.as_str()).unwrap().contains("policy says no"));
        assert_eq!(h[3].role, Role::Assistant);
        assert!(deltas.iter().any(|d| d.delta.role == Some(Role::Tool)), "the denial is emitted on the stream too");
    }
```

- [ ] **Step 3: Confirm it fails**

```bash
cargo test --lib agent::rt::tests::close_dangling agent::rt::tests::cancel_during agent::rt::tests::max_turns agent::rt::tests::denied 2>&1 | tail -5
```

Expected: compile failure (undefined items).

- [ ] **Step 4: Implement** — `src/agent/rt.rs`

Add to the `use`: `use crate::agent::{AgentError, RunControl, ToolCallRequest, ToolDecision};`

Constants and helper (outside or inside `impl Agent`, as fits):

```rust
/// What a tool call that never got to answer is answered with, so the history it sits in
/// stays one a provider accepts (every `tool_use` matched by a `tool_result`).
pub const INTERRUPTED_BY_CANCEL: &str = "[Interrupted: cancelled before this tool call completed]";
pub const INTERRUPTED_BY_FAILURE: &str = "[Interrupted: tool execution failed before this tool call completed]";

impl Agent {
    /// Answer every tool call of the last assistant message that has no `Role::Tool`
    /// result after it, with a stub carrying `note`. A no-op when nothing is pending, so
    /// it is safe to call on every exit path.
    pub(crate) fn close_dangling_tool_calls(history: &mut Vec<Message>, note: &str) {
        let Some(pos) = history.iter().rposition(|m| {
            m.role == Role::Assistant && m.tool_calls.as_ref().is_some_and(|c| !c.is_empty())
        }) else {
            return;
        };
        let answered: std::collections::HashSet<String> = history[pos + 1..]
            .iter()
            .filter(|m| m.role == Role::Tool)
            .filter_map(|m| m.id.clone())
            .collect();
        let pending: Vec<String> = history[pos]
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|p| p.as_function().map(|(id, _, _)| id.to_string()))
                    .filter(|id| !answered.contains(id))
                    .collect()
            })
            .unwrap_or_default();
        for id in pending {
            history.push(Message::new(Role::Tool).with_id(id).with_contents([Part::text(note)]));
        }
    }
}
```

Replace `run_stream`:

```rust
    /// Token-streaming counterpart to [`run`](Self::run) with the default
    /// [`RunControl`] — no cancellation, no turn bound, every tool call allowed.
    /// See [`run_stream_controlled`](Self::run_stream_controlled).
    pub fn run_stream(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<MessageDeltaOutput>> + Send + '_>> {
        Box::pin(
            self.run_stream_controlled(query, RunControl::default())
                .map(|item| item.map_err(anyhow::Error::from)),
        )
    }
```

`run_stream_controlled` (move the existing `run_stream` body into this shape):

```rust
    /// Drive one agent turn as a stream of deltas under `ctl`.
    ///
    /// Invariants on every exit, success or not: the history never ends with a tool call
    /// nobody answered, and a cancelled run keeps whatever answer text had arrived.
    /// Ends in `Err(AgentError::Cancelled)` / `MaxTurns` / `Model` / `Tool` / `Console`.
    pub fn run_stream_controlled(
        &mut self,
        query: Message,
        ctl: RunControl,
    ) -> Pin<Box<impl Stream<Item = Result<MessageDeltaOutput, AgentError>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {
            self.state.history.push(query);
            // If a turn fails before its assistant message commits, pop the dangling user
            // query so a reused agent's next run doesn't send two consecutive User messages.
            let mut committed = false;
            let mut turns: u32 = 0;

            loop {
                // Checked here — after the previous batch's tool results committed — so a
                // run that stops on the bound stops on a history a provider will accept.
                if let Some(max) = ctl.max_turns
                    && turns >= max
                {
                    Err(AgentError::MaxTurns { turns })?;
                }
                turns += 1;

                if let Some(cm) = &self.context_manager
                    && self.state.last_input_tokens.unwrap_or(0) > cm.max_input_tokens
                {
                    cm.truncate_history(&mut self.state.history);
                }

                // ── model phase ─────────────────────────────────────────────
                let mut acc = MessageDeltaOutput::new();
                let mut cancelled = false;
                {
                    let mut delta_stream = self.model.run_stream(
                        &self.state.history,
                        &self.tool_descs,
                        &self.model_options,
                    );
                    loop {
                        let next = tokio::select! {
                            biased;
                            _ = ctl.cancel.cancelled() => { cancelled = true; break; }
                            item = delta_stream.next() => item,
                        };
                        let Some(item) = next else { break };
                        let mut delta = match item {
                            Ok(d) => d,
                            Err(e) => {
                                if !committed { self.state.history.pop(); }
                                Err(AgentError::from_anyhow(e))?
                            }
                        };
                        acc = match acc.accumulate(delta.clone()) {
                            Ok(a) => a,
                            Err(e) => {
                                if !committed { self.state.history.pop(); }
                                Err(AgentError::Other(e))?
                            }
                        };
                        delta.depth = Some(0);
                        self.stamp_source_agent(&mut delta.source_agent);
                        yield delta;
                    }
                }

                if cancelled {
                    // Keep the words that arrived; a half-built tool call is not a call.
                    acc.delta.tool_calls.clear();
                    acc.finish_reason = Some(FinishReason::Stop {});
                    let partial = if acc.delta.role.is_some() { acc.finish().ok() } else { None };
                    match partial {
                        Some(out) if !out.message.contents.is_empty() || out.message.thinking.is_some() => {
                            self.state.history.push(out.message);
                        }
                        _ => {
                            if !committed { self.state.history.pop(); }
                        }
                    }
                    Err(AgentError::Cancelled)?;
                }

                let mut output = match acc.finish() {
                    Ok(o) => o,
                    Err(e) => {
                        if !committed { self.state.history.pop(); }
                        Err(AgentError::Other(e))?
                    }
                };
                if let Some(u) = &output.usage {
                    self.state.last_input_tokens = Some(u.input_tokens);
                }
                output.depth = Some(0);
                self.state.history.push(output.message.clone());
                committed = true;

                let tool_calls = match &output.finish_reason {
                    FinishReason::ToolCall {} => output.message.tool_calls.clone().unwrap_or_default(),
                    _ => break,
                };

                // ── gate ────────────────────────────────────────────────────
                let mut allowed = Vec::with_capacity(tool_calls.len());
                for call in tool_calls {
                    let Some((id, name, args)) = call.as_function() else { continue };
                    match ctl.tool_gate.review(ToolCallRequest { id, name, arguments: args }).await {
                        ToolDecision::Allow => allowed.push(call.clone()),
                        ToolDecision::Deny { reason } => {
                            // A refusal is a result the model reads, not an error the run dies of.
                            let denied = Message::new(Role::Tool).with_id(id).with_contents([Part::value(
                                crate::to_value!({ "error": format!("denied by user: {reason}"), "phase": "policy" }),
                            )]);
                            self.state.history.push(denied.clone());
                            let mut out = MessageOutput {
                                message: denied,
                                finish_reason: FinishReason::Stop {},
                                usage: None,
                                depth: Some(0),
                                source_agent: None,
                                rate_limit: None,
                            };
                            self.stamp_source_agent(&mut out.source_agent);
                            yield out.into();
                        }
                    }
                }
                if allowed.is_empty() {
                    continue;
                }

                // ── tool phase ──────────────────────────────────────────────
                if let Err(e) = self.start_console().await {
                    Self::close_dangling_tool_calls(&mut self.state.history, INTERRUPTED_BY_FAILURE);
                    Err(AgentError::Console(e))?;
                }
                let mut tool_stream = match self.execute_tool_calls(allowed) {
                    Ok(s) => s,
                    Err(e) => {
                        Self::close_dangling_tool_calls(&mut self.state.history, INTERRUPTED_BY_FAILURE);
                        let _ = self.stop_console().await;
                        Err(AgentError::Tool(e))?
                    }
                };
                let mut failure: Option<AgentError> = None;
                loop {
                    let next = tokio::select! {
                        biased;
                        _ = ctl.cancel.cancelled() => { cancelled = true; break; }
                        ev = tool_stream.next() => ev,
                    };
                    let Some(event) = next else { break };
                    match event {
                        Err(e) => {
                            failure = Some(AgentError::Tool(e));
                            break;
                        }
                        Ok(mut out) => {
                            if out.message.role == Role::Tool && out.depth == Some(0) {
                                out.message = Self::cap_tool_result(out.message);
                                self.state.history.push(out.message.clone());
                            }
                            self.stamp_source_agent(&mut out.source_agent);
                            yield out.into();
                        }
                    }
                }
                // Dropping the stream aborts whatever tool futures are still running.
                drop(tool_stream);

                if cancelled {
                    Self::close_dangling_tool_calls(&mut self.state.history, INTERRUPTED_BY_CANCEL);
                    let _ = self.stop_console().await;
                    Err(AgentError::Cancelled)?;
                }
                let stopped = self.stop_console().await;
                if let Some(e) = failure {
                    Self::close_dangling_tool_calls(&mut self.state.history, INTERRUPTED_BY_FAILURE);
                    Err(e)?;
                }
                if let Err(e) = stopped {
                    Err(AgentError::Console(e))?;
                }
            }
        })
    }
```

Put the same guard on the tool-failure path of `run` (the blocking one): `if let Some(e) = failure { Self::close_dangling_tool_calls(&mut self.state.history, INTERRUPTED_BY_FAILURE); Err(e)?; }`.

`map` from `futures::StreamExt` is needed, so `use futures::{FutureExt as _, Stream, StreamExt as _, ...}` stays as it is (it already has `StreamExt as _`).

- [ ] **Step 5: Confirm it passes**

```bash
cargo test --lib agent 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: all ok, including the 5 new tests. The existing `run_stream` tests must pass exactly as before (behaviour unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/agent && git commit -m "feat(agent): run_stream_controlled with cancel, turn bound, tool gate and dangling-call repair"
```

---

### Task A10: `AgentBuilder::max_tokens` and the docs

**Files:**
- Modify: `src/agent/builder.rs`, `src/agent/rt.rs` (module doc), `README.md`

- [ ] **Step 1: The test** (`builder.rs` `mod tests`)

```rust
    #[tokio::test]
    async fn test_builder_max_tokens_reaches_the_spec() {
        ensure_dummy_provider();
        let agent = AgentBuilder::new(TEST_MODEL)
            .agent_provider(TEST_PROVIDER_NAME)
            .max_tokens(32_000)
            .build()
            .unwrap();
        assert_eq!(agent.model_options().max_tokens, Some(32_000));
    }
```

- [ ] **Step 2: Implement** — `AgentBuilder`:

```rust
    /// Cap on tokens per model response. The provider default (8192 on Anthropic) is too
    /// low for answers that ride inside tool-call arguments — a written file is one.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.spec = self.spec.max_tokens(max_tokens);
        self
    }
```

Add a read accessor to `Agent` (`rt.rs`): `pub fn model_options(&self) -> &LangModelOptions { &self.model_options }`.

- [ ] **Step 3: Confirm and commit**

```bash
cargo test --lib agent::builder 2>&1 | grep -E 'test result|FAILED'
git add src/agent && git commit -m "feat(agent): AgentBuilder::max_tokens and Agent::model_options"
```

---

### Task A11: Full verification and cleanup

- [ ] **Step 1: Format, clippy, and the whole offline test suite**

```bash
cargo fmt --all
cargo clippy --all-targets 2>&1 | grep -E '^(warning|error)' | sort | uniq -c | sort -rn | head
cargo test --lib 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: no newly added clippy warnings (the existing warning count holds), and every offline test ok. If a live test fails for a missing key, check that its name is one of the pre-existing live tests and move on.

- [ ] **Step 2: Design-doc consistency note** — in `docs/superpowers/specs/2026-09-11-ailoy-desktop-design.md`, change §5.3's `Tool { name, source }` to `Tool(anyhow::Error)` and §5.5's `reset_at: Option<SystemTime>` to `reset_at_ms: Option<u64>` (matching the implementation).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "chore(core): fmt, clippy, align spec with implemented types"
```

---

## Self-review checklist (for the author)

- Spec §5.1 RunControl/ToolGate/run_stream_controlled → A7, A9 ✓
- §5.2 the cancel, turn-bound, denial and failure stub rules → A9 ✓ (a failure after the second model turn happens once the tool results have committed, so no stub is needed — spelled out in A9's comments)
- §5.3 AgentError/ModelError → A4, A7 ✓ (the Tool variant is unnamed, `Tool(anyhow::Error)`; the spec catches up in A11)
- §5.4 wider retries, cached client → A4 ✓
- §5.5 usage cache fields, RateLimitInfo, header mapping → A5, A6 ✓ (`reset_at_ms`)
- §5.6 base-branch cleanup → A1 ✓
- §9 cortex timeout → A2, A3 ✓
- Type agreement: `RateLimitInfo/RateLimitWindow` (message) · `parse_rate_limit` (lang_model::rate_limit) · `AgentError::{Cancelled, MaxTurns{turns}, Model, Tool, Console, Other}` · `RunControl{cancel, max_turns, tool_gate}` · `ToolCallRequest{id,name,arguments}` · `ToolDecision::{Allow, Deny{reason}}` · `INTERRUPTED_BY_CANCEL/FAILURE` — spelled identically throughout A7–A9 ✓
