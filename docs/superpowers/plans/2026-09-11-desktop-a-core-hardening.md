# Ailoy Desktop — Plan A: ailoy 코어 강화 + cortex exec timeout

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ailoy 에이전트 루프에 취소·턴 상한·툴 승인 훅·실패 시 history 정합성·타입 있는 에러·5xx 재시도·rate-limit 헤더 파싱을 추가하고, cortex 로컬 콘솔에 `timeout_ms` 강제를 구현한다.

**Architecture:** `Agent::run_stream_controlled(query, RunControl)`를 새 진입점으로 추가하고 기존 `run_stream`은 기본 제어로 위임한다. 모든 종료 경로에서 짝 없는 `tool_use`가 history에 남지 않도록 `close_dangling_tool_calls`가 stub `Role::Tool` 메시지를 넣는다. `LangModel`은 `ModelError`(HTTP 상태·재시도 가능 여부)를 내고 응답 헤더를 `RateLimitInfo`로 파싱해 첫 델타에 실어 보낸다. cortex 로컬 콘솔 서버는 `exec.timeout_ms` 경과 시 프로세스 그룹을 kill하고 `TIMED_OUT`을 답한다.

**Tech Stack:** Rust 1.97, tokio, tokio-util(`CancellationToken`), thiserror, reqwest 0.13, humantime, axum 0.8(테스트용 가짜 서버), cortex(`../cortex/cortex` path 의존)

**Spec:** `docs/superpowers/specs/2026-09-11-ailoy-desktop-design.md` §5, §9

## Global Constraints

- 작업 브랜치: ailoy `feat/desktop`(`origin/mem-applied` 기반, 설계서 커밋 `b7c845a8` 포함). cortex는 `../cortex`에 `feat/exec-timeout` 브랜치(Task A2에서 생성) 체크아웃 상태여야 한다.
- 작업 트리는 `../cortex`가 형제 디렉터리로 보이는 위치여야 한다(`Cargo.toml`의 `cortex = { path = "../cortex/cortex" }`). 권장: 메인 체크아웃에서 `git switch feat/desktop`, 또는 `git worktree add ../ailoy-desktop feat/desktop`.
- 기존 공개 API(`Agent::run`, `Agent::run_stream`, `anyhow` 반환)는 유지한다. 새 타입은 추가만 한다.
- 테스트 중 외부 API를 호출하는 라이브 테스트(`.env` 키 필요)는 건드리지 않는다. 새 테스트는 모두 오프라인(axum 가짜 서버)이어야 한다. 단 `shell` 툴 테스트는 `cortex-local-console` 바이너리가 필요하다(`AILOY_CORTEX_CONSOLE` 환경 변수로 경로 지정, 기본 `cortex-local-console`).
- 커밋 메시지는 conventional commits(`feat(agent): …`). 각 Task 끝에 커밋한다.
- `cargo fmt` 후 커밋. 편집 스타일은 기존 파일(주석은 "왜"를 설명)을 따른다.

---

## 파일 구조

| 경로 | 책임 |
|---|---|
| `src/agent/control.rs` (신규) | `RunControl`, `ToolGate`, `ToolCallRequest`, `ToolDecision`, `AllowAll` |
| `src/agent/error.rs` (신규) | `AgentError` |
| `src/agent/rt.rs` (수정) | `run_stream_controlled`, `close_dangling_tool_calls`, `run_stream` 위임 |
| `src/agent/builder.rs` (수정) | `AgentBuilder::max_tokens` |
| `src/agent/mod.rs` (수정) | 재수출 |
| `src/agent/test_support.rs` (신규, `cfg(test)`) | 가짜 ChatCompletion SSE 서버, 가짜 프로바이더 등록 헬퍼 |
| `src/lang_model/error.rs` (신규) | `ModelError` |
| `src/lang_model/rate_limit.rs` (신규) | 헤더 → `RateLimitInfo` 파서 |
| `src/lang_model/rt.rs` (수정) | 재시도 정책, 클라이언트 캐시, 헤더 파싱 연결 |
| `src/lang_model/mod.rs` (수정) | 재수출 |
| `src/message/rate_limit.rs` (신규) | `RateLimitInfo`, `RateLimitWindow` |
| `src/message/message.rs`, `message_delta.rs`, `mod.rs` (수정) | `rate_limit` 필드 추가와 누적 |
| `src/lang_model/impl/api/openai.rs`, `gemini.rs` (수정) | 캐시 토큰 파싱 |
| `src/tool/impl/builtins/shell.rs` (수정) | `timeout_secs` → `timeout_ms` 전달 |
| `Cargo.toml` (수정) | `tokio-util`, `humantime` 추가 |
| `../cortex/cortex-console-servers/local/src/server/mod.rs` (수정) | `execute`에 timeout 강제 |
| `../cortex/cortex-console-servers/local/tests/exec_timeout.rs` (신규) | timeout E2E |

---

### Task A1: 작업 브랜치에 develop(#448 Bedrock) 머지

**Files:**
- Modify: `Cargo.toml`, `src/lang_model/impl/api/mod.rs`, `src/lang_model/provider.rs` (머지 결과)
- Delete(충돌 해소): `src/runenv/sandbox.rs` 등 `src/runenv/*`

**Interfaces:**
- Produces: `LangModelAPISchema::Bedrock` 변형과 `src/lang_model/impl/api/bedrock.rs`가 브랜치에 존재. 이후 Task는 `match schema` 에 `Bedrock` 가지를 포함해야 한다.

- [ ] **Step 1: 브랜치 확인 및 머지 시작**

```bash
git switch feat/desktop
git log --oneline -1          # b7c845a8 docs(design): ... 이어야 함
git merge develop
```

Expected: 충돌 보고. 충돌 파일은 대체로 `Cargo.toml`, `Cargo.lock`, `src/runenv/sandbox.rs`(ours에서 삭제됨, theirs에서 수정됨), 경우에 따라 `src/tool/impl/builtins/web_fetch.rs`.

- [ ] **Step 2: 충돌 해소 규칙 적용**

```bash
# runenv는 cortex-applied에서 제거된 모듈: 삭제로 해소
git rm -q src/runenv/sandbox.rs 2>/dev/null || true
git status --short | grep '^UD\|^DU\|^AA\|^UU'
```

- `Cargo.toml`: ours(mem-applied) 기준. `microsandbox*` 의존과 `sandbox` feature는 넣지 않는다. develop이 추가한 의존(있다면 Bedrock 관련 `sha2`/`hmac` 류)만 가져온다. `[features]`는 `default = []`만 남긴다.
- `src/lang_model/impl/api/mod.rs`: 두 쪽을 합친다 — `LangModelAPISchema::Bedrock` 변형, `BedrockRegion` 재수출, `provider_api`의 `Bedrock => Box::new(BedrockUnmarshal)` 가지, `mod bedrock;`.
- `src/lang_model/provider.rs`: develop 쪽 `bedrock()` 생성자와 `Default`의 `AWS_BEARER_TOKEN_BEDROCK` 블록을 가져온다.
- `web_fetch.rs`: develop(#443)의 변경을 받아들이되, 충돌 부분은 컴파일이 되는 쪽으로 develop 우선.
- `Cargo.lock`: `git checkout --theirs Cargo.lock` 후 Step 3의 `cargo check`가 갱신하게 둔다.

- [ ] **Step 3: 컴파일 확인**

```bash
cargo check --all-targets 2>&1 | tail -5
```

Expected: `Finished`. 오류가 `runenv`/`Sandbox` 참조라면 그 참조를 지운다(develop 전용 코드).

- [ ] **Step 4: 오프라인 테스트 통과 확인**

```bash
cargo test --lib lang_model::provider 2>&1 | tail -3
cargo test --lib message 2>&1 | tail -3
```

Expected: 모두 `test result: ok`.

- [ ] **Step 5: 머지 커밋**

```bash
git add -A
git commit -m "merge: develop into feat/desktop (Bedrock wire #448; drop microsandbox-only sandbox changes)"
```

---

### Task A2: cortex 로컬 콘솔 `timeout_ms` 강제

**Files:**
- Modify: `../cortex/cortex-console-servers/local/src/server/mod.rs:587-630` (`execute`)
- Create: `../cortex/cortex-console-servers/local/tests/exec_timeout.rs`

**Interfaces:**
- Consumes: `ExecCall { cmd: Vec<String>, timeout_ms: Option<u64> }`, `Error::TIMED_OUT: i64 = -32000`, `refused(code, msg) -> Error`, `finished(io::Result<Output>) -> Response`
- Produces: `exec` 에 `timeout_ms` 가 있으면 경과 시 `Response::Error(Error{code: TIMED_OUT, message: "killed after {ms}ms"})`. 콘솔 세션은 살아 있어 다음 요청을 받는다.

- [ ] **Step 1: cortex 브랜치 생성**

```bash
cd ../cortex && git switch -c feat/exec-timeout main && cd -
```

- [ ] **Step 2: 실패하는 E2E 테스트 작성**

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

- [ ] **Step 3: 실패 확인**

```bash
cd ../cortex && cargo test -p cortex-local-console --test exec_timeout 2>&1 | tail -15; cd -
```

Expected: `a_command_past_its_timeout_is_killed_and_reported` FAIL (현재는 10초 뒤 `Ok` 응답이 오거나 `elapsed` 단언 실패).

- [ ] **Step 4: `execute` 구현**

`../cortex/cortex-console-servers/local/src/server/mod.rs` 의 `execute` 를 다음으로 교체한다(기존 주석은 유지·보강). 파일 상단 `use` 에 `std::time::Duration` 을 추가한다.

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

모듈 상단 doc 주석 76-78행의 "Neither timeout is enforced" 문장을 "`exec.timeout_ms` is enforced by a kill; `init` carries no default." 로 고친다.

- [ ] **Step 5: 테스트 통과 확인**

```bash
cd ../cortex && cargo test -p cortex-local-console --test exec_timeout 2>&1 | tail -8 && cargo test -p cortex-local-console 2>&1 | grep -E 'test result|FAILED' ; cd -
```

Expected: 3 passed, 기존 `files`/`workfs` 테스트도 ok.

- [ ] **Step 6: 커밋(cortex)**

```bash
cd ../cortex && git add -A && git commit -m "feat(local-console): enforce exec timeout_ms with a process-group kill

A command that outlives its timeout is killed and answered TIMED_OUT; the
session keeps answering. Needed by ailoy-desktop, whose shell tool sets a
default timeout." && cd -
```

---

### Task A3: `shell` 툴이 `timeout_secs`를 콘솔에 전달

**Files:**
- Modify: `src/tool/impl/builtins/shell.rs`

**Interfaces:**
- Produces: `shell` 인자 `timeout_secs`(0 또는 생략 = 기본 600초)를 `Console::exec(.., Some(ms))` 로 전달. 만료 시 결과 `{"timed_out": true, "exit_code": -1}` (기존 분기 유지).

- [ ] **Step 1: 실패하는 테스트 추가** (`shell.rs` `mod tests` 안, 기존 헬퍼 `provider()` 사용)

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

- [ ] **Step 2: 실패 확인**

```bash
AILOY_CORTEX_CONSOLE=../cortex/target/debug/cortex-local-console cargo test --lib tool::impl::builtins::shell::tests::test_timeout_secs 2>&1 | tail -5
```

Expected: FAIL (`timed_out` 가 `false`, 10초 소요). 바이너리가 없으면 먼저 `(cd ../cortex && cargo build -p cortex-local-console)`.

- [ ] **Step 3: 구현** — `get_shell_tool_func` 안, `console.exec(...)` 호출 직전에 timeout 계산을 넣고 `None` 대신 전달한다.

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

툴 설명의 `timeout_secs` 문구를 `"Timeout in seconds. 0 or omitted means the default (600)."` 로 고친다.

- [ ] **Step 4: 통과 확인**

```bash
AILOY_CORTEX_CONSOLE=../cortex/target/debug/cortex-local-console cargo test --lib tool::impl::builtins::shell 2>&1 | grep -E 'test result|FAILED'
```

Expected: 모두 ok.

- [ ] **Step 5: 커밋**

```bash
git add src/tool/impl/builtins/shell.rs
git commit -m "feat(tool): shell passes timeout_secs to the console (default 600s)"
```

---

### Task A4: `ModelError`와 재시도 정책 확대, 클라이언트 캐시

**Files:**
- Create: `src/lang_model/error.rs`
- Modify: `src/lang_model/rt.rs` (`LangModel` 필드, `send_with_retry`), `src/lang_model/mod.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct ModelError { pub status: Option<u16>, pub retryable: bool, pub message: String, pub attempts: u32 }
  // std::error::Error + Display. `anyhow::Error` 로 감싸져 나가며 `downcast_ref::<ModelError>()` 가능.
  ```
  재시도: 429(영구 quota 제외)·408·5xx·transport(connect/timeout/request) 를 최대 3회(총 4시도) 지수 백오프(1,2,4초, `retry-after` 우선, 상한 10초). 그 외 4xx는 즉시 실패.

- [ ] **Step 1: 실패하는 테스트 작성** (`src/lang_model/rt.rs` `mod tests`에 추가; 기존 429 테스트와 같은 axum 패턴)

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

기존 테스트들이 `LangModel { model, provider }` 리터럴로 구성한다면 `LangModel::from_elem(model, elem)` 로 바꾼다(Step 3에서 추가).

- [ ] **Step 2: 실패 확인**

```bash
cargo test --lib lang_model::rt::tests::test_retries_5xx 2>&1 | tail -5
```

Expected: 컴파일 실패(`from_elem`, `run_with_backoff_base`, `ModelError` 미정의).

- [ ] **Step 3: `ModelError` 정의**

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

`src/lang_model/mod.rs` 에 `mod error; pub use error::ModelError;` 추가.

- [ ] **Step 4: `LangModel` 에 클라이언트 캐시와 생성자 추가, 재시도 재작성**

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
    // try_from_provider 의 `Ok(Self { model: api_model_id, provider: provider_elem })` 를
    // `Ok(Self::from_elem(api_model_id, provider_elem))` 로 바꾼다.
}
```

`run` 은 `run_with_backoff_base(messages, tools, options, Duration::from_secs(1))` 로 위임하고, 새 함수는 기존 `run` 본문에서 `reqwest::Client::new()` 대신 `&self.client` 를, `send_with_retry(...)` 대신 `send_with_retry(&self.client, &url, header_map, &body, provider.as_ref(), backoff_base)` 를 쓴다. `run_stream` 도 `let client = self.client.clone();` 을 스트림 밖에서 캡처해 사용하고 `Duration::from_secs(1)` 을 넘긴다.

`send_with_retry` 교체:

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

`run`/`run_stream` 의 호출부 `?` 는 그대로 동작한다(`ModelError: std::error::Error` → `anyhow`). 기존 `test_retries_429_then_succeeds` 류 테스트는 `retry-after: 0` 을 쓰므로 그대로 통과한다.

- [ ] **Step 5: 통과 확인**

```bash
cargo test --lib lang_model::rt 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: 모두 ok(라이브 테스트는 키가 없으면 skip/ignore 규칙에 따름).

- [ ] **Step 6: 커밋**

```bash
git add src/lang_model
git commit -m "feat(lang_model): typed ModelError, retry 5xx/transport, cached reqwest client"
```

---

### Task A5: `RateLimitInfo` 타입과 헤더 파서

**Files:**
- Create: `src/message/rate_limit.rs`, `src/lang_model/rate_limit.rs`
- Modify: `src/message/mod.rs`, `src/message/message.rs`, `src/message/message_delta.rs`, `src/lang_model/mod.rs`, `src/lang_model/rt.rs`, `Cargo.toml`, 그리고 `MessageOutput { .. }` 리터럴이 있는 모든 파일

**Interfaces:**
- Produces:
  ```rust
  pub struct RateLimitWindow { pub limit: Option<u64>, pub remaining: Option<u64>, pub reset_at_ms: Option<u64> }
  pub struct RateLimitInfo { pub requests: Option<RateLimitWindow>, pub tokens: Option<RateLimitWindow>, pub input_tokens: Option<RateLimitWindow>, pub output_tokens: Option<RateLimitWindow> }
  impl RateLimitInfo { pub fn is_empty(&self) -> bool }
  // MessageOutput / MessageDeltaOutput: pub rate_limit: Option<RateLimitInfo>  (serde skip_if_none)
  pub(crate) fn lang_model::rate_limit::parse_rate_limit(schema: &LangModelAPISchema, headers: &HeaderMap, now: SystemTime) -> Option<RateLimitInfo>
  ```
  스트리밍에서는 첫 델타에, 블로킹 `run` 에서는 `MessageOutput` 에 실린다. 누적 시 `other.or(self)`.

- [ ] **Step 1: 의존 추가**

`Cargo.toml` `[dependencies]` 에 `humantime = "2"` 추가.

- [ ] **Step 2: 타입 정의**

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

`src/message/message.rs` `MessageOutput` 에 필드 추가(마지막에):

```rust
    /// Rate-limit headroom the provider reported with this response, when it reports any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimitInfo>,
```

`src/message/message_delta.rs` `MessageDeltaOutput` 에 같은 필드 추가; `new()` 에 `rate_limit: None`; `accumulate` 에 `let rate_limit = other.rate_limit.or(self.rate_limit);` 와 결과 구조체에 포함; `finish` 의 `MessageOutput { .. }` 에 `rate_limit: self.rate_limit`; `From<MessageOutput> for MessageDeltaOutput` 에 `rate_limit: out.rate_limit`.

- [ ] **Step 3: 나머지 리터럴 보정**

```bash
grep -rn 'source_agent: None' src --include=*.rs | grep -v 'rate_limit' | cut -d: -f1 | sort -u
```

나열된 파일(`src/tool/func.rs`, `src/agent/rt.rs`, `src/agent/subagent.rs`, `src/lang_model/impl/api/*.rs` 등)의 `MessageOutput { ... source_agent: None }` / `MessageDeltaOutput { ... }` 리터럴마다 `rate_limit: None,` 을 추가한다. `cargo check --all-targets` 가 빠진 곳을 알려준다.

- [ ] **Step 4: 파서 테스트 작성** (`src/lang_model/rate_limit.rs` 의 `mod tests`)

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

- [ ] **Step 5: 실패 확인**

```bash
cargo test --lib lang_model::rate_limit 2>&1 | tail -5
```

Expected: 컴파일 실패(모듈 없음).

- [ ] **Step 6: 파서 구현** — `src/lang_model/rate_limit.rs`

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

`src/lang_model/mod.rs` 에 `pub(crate) mod rate_limit;` 추가. `LangModelAPISchema` 에 `Bedrock` 이 없다면(Task A1 머지 확인) 그 가지를 제거한다.

- [ ] **Step 7: 응답에 연결** — `src/lang_model/rt.rs`

`run_with_backoff_base` 에서:

```rust
                let response = send_with_retry(&self.client, &url, header_map, &body, provider.as_ref(), backoff_base).await?;
                let rate_limit = crate::lang_model::rate_limit::parse_rate_limit(schema, response.headers(), std::time::SystemTime::now());
                let response_text = response.text().await?;
                // ...
                let mut out = delta_output.finish()?;
                out.rate_limit = rate_limit;
                Ok(out)
```

`run_stream` 에서: 스트림 밖에서 `let schema = schema.clone();` 을 만들고, 스트림 안에서 응답을 받은 직후

```rust
            let mut rate_limit = crate::lang_model::rate_limit::parse_rate_limit(&schema, response.headers(), std::time::SystemTime::now());
```

를 두고, `yield output;` 두 곳 모두 직전에

```rust
                        if let Some(rl) = rate_limit.take() { output.rate_limit = Some(rl); }
```

(두 곳의 `output` 바인딩을 `let mut output`/`Some(mut output)` 으로 바꾼다.)

- [ ] **Step 8: 스트림 연결 테스트** (`src/lang_model/rt.rs` `mod tests`)

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

- [ ] **Step 9: 전체 확인**

```bash
cargo check --all-targets 2>&1 | tail -3
cargo test --lib lang_model::rate_limit lang_model::rt::tests::test_stream_carries message 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: 컴파일 성공, 테스트 ok.

- [ ] **Step 10: 커밋**

```bash
git add -A
git commit -m "feat(lang_model,message): RateLimitInfo parsed from Anthropic/OpenAI headers onto outputs"
```

---

### Task A6: 캐시 토큰 파싱 보강 (OpenAI Responses, Gemini)

**Files:**
- Modify: `src/lang_model/impl/api/openai.rs:521-536`, `src/lang_model/impl/api/gemini.rs:380-396`

**Interfaces:**
- Produces: `TokenUsage.cache_read_input_tokens` 가 OpenAI Responses `usage.input_tokens_details.cached_tokens`, Gemini `usageMetadata.cachedContentTokenCount` 에서 채워진다.

- [ ] **Step 1: 실패하는 테스트** — `openai.rs` `mod tests`:

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

(기존 테스트가 `Unmarshal` 트레이트를 어떻게 호출하는지 — `T::default().unmarshal(val)` 등 — 같은 파일의 `test_unmarshal_usage` 를 따라 맞춘다.)

- [ ] **Step 2: 실패 확인**

```bash
cargo test --lib 'lang_model::impl::api::openai::tests::test_unmarshal_usage_cached' 'lang_model::impl::api::gemini::tests::test_unmarshal_usage_cached' 2>&1 | grep -E 'test result|panicked'
```

Expected: FAIL(`None`).

- [ ] **Step 3: 구현** — `openai.rs` usage 파싱에서 `cache_read_input_tokens: u.pointer("/input_tokens_details/cached_tokens").and_then(|v| v.as_u64())`(값 접근 방식은 파일의 기존 코드 — `.get("input_tokens")` 등 — 와 동일한 헬퍼를 사용). `gemini.rs` `parse_usage` 에서 `cache_read_input_tokens: u.get("cachedContentTokenCount").and_then(as_u64)`. `cache_creation_input_tokens` 는 두 곳 모두 `None` 유지.

- [ ] **Step 4: 통과 확인 후 커밋**

```bash
cargo test --lib lang_model::impl::api 2>&1 | grep -E 'test result|FAILED'
git add src/lang_model/impl/api && git commit -m "fix(lang_model): report cached input tokens for OpenAI Responses and Gemini"
```

---

### Task A7: `AgentError`, `RunControl`, `ToolGate` 타입

**Files:**
- Create: `src/agent/error.rs`, `src/agent/control.rs`
- Modify: `src/agent/mod.rs`, `Cargo.toml`

**Interfaces:**
- Produces:
  ```rust
  pub enum AgentError { Cancelled, MaxTurns { turns: u32 }, Model(ModelError), Tool(anyhow::Error), Console(anyhow::Error), Other(anyhow::Error) }
  impl AgentError { pub fn from_anyhow(e: anyhow::Error) -> Self }   // ModelError 를 찾아 Model 로 분류
  pub struct RunControl { pub cancel: CancellationToken, pub max_turns: Option<u32>, pub tool_gate: Arc<dyn ToolGate> }  // Default = 무제한/AllowAll
  pub struct ToolCallRequest<'a> { pub id: &'a str, pub name: &'a str, pub arguments: &'a Value }
  pub enum ToolDecision { Allow, Deny { reason: String } }
  #[async_trait] pub trait ToolGate: Send + Sync { async fn review(&self, call: ToolCallRequest<'_>) -> ToolDecision; }
  pub struct AllowAll;
  ```

- [ ] **Step 1: 의존 추가** — `Cargo.toml` `[dependencies]` 에 `tokio-util = "0.7"` 추가.

- [ ] **Step 2: 타입 테스트 작성** — `src/agent/control.rs` 하단:

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

- [ ] **Step 3: 구현**

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

`src/agent/mod.rs` 에 `mod control; mod error; pub use control::*; pub use error::AgentError;` 추가.

- [ ] **Step 4: 통과 확인 및 커밋**

```bash
cargo test --lib agent::control 2>&1 | grep -E 'test result|FAILED'
git add Cargo.toml Cargo.lock src/agent && git commit -m "feat(agent): AgentError, RunControl and ToolGate types"
```

---

### Task A8: 테스트 지원 — 가짜 ChatCompletion SSE 서버와 프로바이더 등록

**Files:**
- Create: `src/agent/test_support.rs`
- Modify: `src/agent/mod.rs` (`#[cfg(test)] pub(crate) mod test_support;`)

**Interfaces:**
- Produces(테스트 전용):
  ```rust
  pub(crate) async fn spawn_sse_server(bodies: Vec<String>, chunk_delay: Option<Duration>) -> (SocketAddr, Arc<AtomicU32> /*call count*/)
  pub(crate) fn sse_text(text_chunks: &[&str]) -> String                       // assistant 텍스트 → stop
  pub(crate) fn sse_tool_call(id: &str, name: &str, args_json: &str) -> String  // tool_calls → tool_calls finish
  pub(crate) fn register_fake_provider(name: &'static str, addr: SocketAddr, tools: Vec<(&str, ToolDesc, ToolFunc)>) -> &'static str
  pub(crate) fn user(text: &str) -> Message
  ```
  가짜 서버는 i번째 호출에 `bodies[min(i, len-1)]` 를 `text/event-stream` 으로 답한다. `chunk_delay` 가 있으면 `\n\n` 단위 이벤트마다 그만큼 기다린다.

- [ ] **Step 1: 구현**

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

- [ ] **Step 2: 컴파일 확인**

```bash
cargo check --all-targets 2>&1 | tail -3
```

`LangModelProvider::chat_completion` 시그니처가 `(url: &str, api_key: Option<String>) -> anyhow::Result<LangModelProviderElem>` 인지 `src/lang_model/impl/api/chat_completion.rs` 에서 확인하고 다르면 맞춘다.

- [ ] **Step 3: 커밋**

```bash
git add src/agent && git commit -m "test(agent): scripted SSE server and fake provider helpers"
```

---

### Task A9: `close_dangling_tool_calls`와 `run_stream_controlled`

**Files:**
- Modify: `src/agent/rt.rs`

**Interfaces:**
- Consumes: Task A7 타입, A8 헬퍼, `MessageOutput.rate_limit`(A5)
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
  `run_stream` 은 `run_stream_controlled(query, RunControl::default())` 에 `anyhow` 변환을 얹은 것이 된다.

- [ ] **Step 1: 단위 테스트 — stub 삽입** (`rt.rs` `mod tests`)

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

- [ ] **Step 2: 통합 테스트 4종** (`rt.rs` `mod tests`; `use crate::agent::test_support::*;`, `use crate::agent::{RunControl, ToolGate, ToolCallRequest, ToolDecision, AgentError, INTERRUPTED_BY_CANCEL};`)

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

- [ ] **Step 3: 실패 확인**

```bash
cargo test --lib agent::rt::tests::close_dangling agent::rt::tests::cancel_during agent::rt::tests::max_turns agent::rt::tests::denied 2>&1 | tail -5
```

Expected: 컴파일 실패(미정의 항목).

- [ ] **Step 4: 구현** — `src/agent/rt.rs`

`use` 에 추가: `use crate::agent::{AgentError, RunControl, ToolCallRequest, ToolDecision};`

상수와 헬퍼(`impl Agent` 바깥/안 적절히):

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

`run_stream` 교체:

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

`run_stream_controlled` (기존 `run_stream` 본문을 이 형태로 옮긴다):

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

`run`(블로킹) 의 툴 실패 경로에도 같은 보호를 넣는다: `if let Some(e) = failure { Self::close_dangling_tool_calls(&mut self.state.history, INTERRUPTED_BY_FAILURE); Err(e)?; }`.

`futures::StreamExt` 의 `map` 이 필요하므로 `use futures::{FutureExt as _, Stream, StreamExt as _, ...}` 는 그대로(이미 `StreamExt as _`).

- [ ] **Step 5: 통과 확인**

```bash
cargo test --lib agent 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: 새 테스트 5개 포함 모두 ok. 기존 `run_stream` 계열 테스트도 동일하게 통과해야 한다(동작 불변).

- [ ] **Step 6: 커밋**

```bash
git add src/agent && git commit -m "feat(agent): run_stream_controlled with cancel, turn bound, tool gate and dangling-call repair"
```

---

### Task A10: `AgentBuilder::max_tokens` 와 문서

**Files:**
- Modify: `src/agent/builder.rs`, `src/agent/rt.rs`(모듈 doc), `README.md`

- [ ] **Step 1: 테스트** (`builder.rs` `mod tests`)

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

- [ ] **Step 2: 구현** — `AgentBuilder`:

```rust
    /// Cap on tokens per model response. The provider default (8192 on Anthropic) is too
    /// low for answers that ride inside tool-call arguments — a written file is one.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.spec = self.spec.max_tokens(max_tokens);
        self
    }
```

`Agent` 에 읽기 접근자 추가(`rt.rs`): `pub fn model_options(&self) -> &LangModelOptions { &self.model_options }`.

- [ ] **Step 3: 확인·커밋**

```bash
cargo test --lib agent::builder 2>&1 | grep -E 'test result|FAILED'
git add src/agent && git commit -m "feat(agent): AgentBuilder::max_tokens and Agent::model_options"
```

---

### Task A11: 전체 검증과 정리

- [ ] **Step 1: 포맷·클리피·전체 오프라인 테스트**

```bash
cargo fmt --all
cargo clippy --all-targets 2>&1 | grep -E '^(warning|error)' | sort | uniq -c | sort -rn | head
cargo test --lib 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: 새로 추가된 clippy 경고 없음(기존 경고 수 유지), 오프라인 테스트 전부 ok. 라이브 테스트가 키 부재로 실패하면 그 테스트 이름이 기존 라이브 테스트인지 확인하고 넘어간다.

- [ ] **Step 2: 설계서 정합성 메모** — `docs/superpowers/specs/2026-09-11-ailoy-desktop-design.md` §5.3 의 `Tool { name, source }` 를 `Tool(anyhow::Error)` 로, §5.5 의 `reset_at: Option<SystemTime>` 를 `reset_at_ms: Option<u64>` 로 고친다(구현과 일치).

- [ ] **Step 3: 커밋**

```bash
git add -A && git commit -m "chore(core): fmt, clippy, align spec with implemented types"
```

---

## Self-Review 체크리스트 (작성자용)

- 스펙 §5.1 RunControl/ToolGate/run_stream_controlled → A7, A9 ✓
- §5.2 취소·턴 상한·거부·실패 stub 규칙 → A9 ✓ (모델 2턴 이후 실패는 툴 결과 커밋 뒤에 발생하므로 stub 불필요 — A9 주석에 명시)
- §5.3 AgentError/ModelError → A4, A7 ✓ (Tool 변형은 이름 없이 `Tool(anyhow::Error)`; A11에서 스펙 반영)
- §5.4 재시도 확대·클라이언트 캐시 → A4 ✓
- §5.5 사용량 캐시 필드·RateLimitInfo·헤더 매핑 → A5, A6 ✓ (`reset_at_ms`)
- §5.6 기반 브랜치 정리 → A1 ✓
- §9 cortex timeout → A2, A3 ✓
- 타입 일치: `RateLimitInfo/RateLimitWindow`(message) · `parse_rate_limit`(lang_model::rate_limit) · `AgentError::{Cancelled, MaxTurns{turns}, Model, Tool, Console, Other}` · `RunControl{cancel, max_turns, tool_gate}` · `ToolCallRequest{id,name,arguments}` · `ToolDecision::{Allow, Deny{reason}}` · `INTERRUPTED_BY_CANCEL/FAILURE` — A7~A9 전체에서 동일 철자 사용 ✓
