# Ailoy Desktop — Plan B: 세션 엔진 `ailoy-desktop-core`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tauri 없이 테스트 가능한 헤드리스 세션 엔진 크레이트를 만든다 — SQLite 영속, 워크스페이스(WorkFs + FUSE-T 마운트 + 커넥터), 콘솔 스폰, 모델 카탈로그(models.dev), 프로바이더 등록, run actor(취소·이벤트·재조립·사용량).

**Architecture:** `Engine`이 `Store`(rusqlite), `WorkspaceManager`(WorkFs·FuseTMount·fsops), `ConsoleFactory`, `Catalog`, `RunManager`를 소유한다. run마다 콘솔 하나를 스폰해 `AgentBuilder`로 `Agent`를 조립하고 `run_stream_controlled`를 actor 태스크에서 구동한다. 델타는 `MessageAssembler`(agent-k 이식)가 `RunEvent`로 바꿔 `broadcast` 채널로 내보내고, 완성 메시지는 즉시 DB에 기록한다.

**Tech Stack:** Rust 1.97, tokio, rusqlite 0.40(bundled), cortex(`fuse-t`,`notion`,`s3`), ailoy(Plan A 완료 상태), serde/serde_json, reqwest(카탈로그 갱신), chrono, tracing, uuid

**Spec:** `docs/superpowers/specs/2026-09-11-ailoy-desktop-design.md` §4, §6, §10

## Global Constraints

- Plan A가 완료된 `feat/desktop` 브랜치 위에서 작업한다. `../cortex`는 `feat/exec-timeout` 체크아웃.
- 크레이트 경로 `apps/desktop/core`, 이름 `ailoy-desktop-core`, 루트 workspace member. cortex path 의존은 `../../../../cortex/cortex`.
- 모든 공개 타입은 `serde::{Serialize, Deserialize}` 가능해야 한다(Tauri IPC 페이로드). 자격 증명은 `Settings`/`MountInfo`로 나갈 때 마스킹한다.
- 단위 테스트는 오프라인·in-memory SQLite. 실제 `cortex-local-console`·FUSE-T가 필요한 테스트는 `#[ignore]`로 표시하고 환경 변수 `AILOY_CORTEX_BIN_DIR`(콘솔 바이너리 디렉터리)로 켠다.
- 앱 데이터 디렉터리 구조: `<data_dir>/ailoy.sqlite`, `<data_dir>/files/`, `<data_dir>/workspace/`(마운트포인트), `<data_dir>/cache/models.json`.
- 커밋 메시지 접두: `feat(desktop-core): …`.

---

## 파일 구조

| 경로 | 책임 |
|---|---|
| `Cargo.toml` (루트, 수정) | members에 `apps/desktop/core`, exclude에 `apps/desktop/src-tauri` |
| `apps/desktop/core/Cargo.toml` | 크레이트 정의 |
| `src/lib.rs` | 모듈 선언·재수출 |
| `src/error.rs` | `EngineError`, `Result<T>` |
| `src/types.rs` | IPC 페이로드 타입 전부 |
| `src/config.rs` | `EngineConfig`와 경로 헬퍼 |
| `src/store/mod.rs`, `src/store/migrations/0001_init.sql` | SQLite 저장소 |
| `src/catalog.rs`, `assets/models.json`, `src/bin/gen_catalog.rs` | 모델 카탈로그 |
| `src/providers.rs` | 설정 → ailoy 프로바이더 등록 |
| `src/prompt.rs` | 시스템 프리앰블 |
| `src/workspace/mod.rs`, `shared.rs`, `fsops.rs`, `connectors.rs`, `mount.rs` | 워크스페이스 |
| `src/console.rs` | 콘솔 바이너리 탐색·스폰 |
| `src/assembler.rs` | 델타 → 텍스트/thinking/완성 메시지 |
| `src/events.rs` | `RunEvent` |
| `src/usage.rs` | 사용량·비용·컨텍스트 계산 |
| `src/run.rs` | `RunManager`, actor |
| `src/engine.rs` | `Engine` 파사드 |
| `tests/live_console.rs` | `#[ignore]` 통합 테스트 |

---

### Task B1: 크레이트 스캐폴드, 에러, 타입, 설정

**Files:**
- Modify: `Cargo.toml` (루트)
- Create: `apps/desktop/core/Cargo.toml`, `src/lib.rs`, `src/error.rs`, `src/types.rs`, `src/config.rs`

**Interfaces:**
- Produces(이후 모든 Task가 사용):
  ```rust
  pub enum EngineError { NotFound(String), AlreadyRunning, Invalid(String), ConsoleUnavailable(String), Workspace(String), Storage(rusqlite::Error), Io(std::io::Error), Other(anyhow::Error) }  // Serialize → 문자열
  pub type Result<T> = std::result::Result<T, EngineError>;
  pub struct EngineConfig { pub data_dir: PathBuf, pub console_bin: Option<PathBuf>, pub catalog_refresh: bool, pub mount_workspace: bool }
  impl EngineConfig { pub fn new(data_dir: impl Into<PathBuf>) -> Self; pub fn db_path(&self) -> PathBuf; pub fn files_root(&self) -> PathBuf; pub fn mountpoint(&self) -> PathBuf; pub fn cache_dir(&self) -> PathBuf }
  pub fn now_ms() -> i64;
  // types.rs
  pub enum MountKind { Root, Local, Notion, S3 }                          // serde lowercase
  #[serde(tag = "kind", rename_all = "lowercase")]
  pub enum MountConfig { Root, Local { host_root: PathBuf }, Notion { api_key: String }, S3 { bucket: String, region: String, access_key_id: String, secret_access_key: String, endpoint: Option<String>, key_prefix: Option<String> } }
  pub enum MountStatus { Ok, Error { message: String } }                 // serde tag "status"
  pub struct MountInfo { pub id: String, pub path: String, pub kind: MountKind, pub label: String, pub detail: String, pub writable: bool, pub status: MountStatus }
  pub struct MountRequest { pub path: String, pub label: Option<String>, pub config: MountConfig }
  pub struct SessionSummary { pub id: String, pub title: String, pub model: String, pub created_at: i64, pub updated_at: i64, pub running: bool }
  pub struct StoredMessage { pub seq: i64, pub depth: u8, pub source_agent: Option<String>, pub message: ailoy::message::Message, pub usage: Option<ailoy::message::TokenUsage>, pub created_at: i64 }
  pub struct ProviderSetting { pub key: String, pub label: String, pub has_key: bool, pub key_hint: String, pub region: Option<String> }
  pub struct Settings { pub providers: Vec<ProviderSetting>, pub default_model: String, pub max_tokens: u64, pub max_turns: u32, pub catalog_refresh: bool }
  pub struct SettingsPatch { pub provider_keys: BTreeMap<String, Option<String>>, pub bedrock_region: Option<String>, pub default_model: Option<String>, pub max_tokens: Option<u64>, pub max_turns: Option<u32>, pub catalog_refresh: Option<bool> }
  pub struct ModelCost { pub input: Option<f64>, pub output: Option<f64>, pub cache_read: Option<f64>, pub cache_write: Option<f64> }   // USD / 1M tokens
  pub struct ModelInfo { pub id: String, pub provider: String, pub name: String, pub context: Option<u64>, pub output: Option<u64>, pub cost: Option<ModelCost>, pub reasoning: bool, pub tool_call: bool, pub available: bool }
  pub struct SessionUsage { pub input_tokens: u64, pub output_tokens: u64, pub cache_read_tokens: u64, pub cache_write_tokens: u64, pub estimated_cost_usd: Option<f64>, pub context_used: Option<u64>, pub context_limit: Option<u64> }
  pub enum WorkspaceStatus { Mounted, Degraded { reason: String } }      // serde tag "status"
  pub struct WorkspaceInfo { pub mountpoint: PathBuf, pub files_root: PathBuf, pub status: WorkspaceStatus }
  pub struct Entry { pub name: String, pub path: String, pub kind: String, pub size: Option<u64>, pub mtime_ms: Option<u64> }
  pub struct FileContent { pub path: String, pub text: Option<String>, pub size: u64, pub truncated: bool }
  pub struct ImportReport { pub files: usize, pub bytes: u64, pub skipped: Vec<String> }
  ```

- [ ] **Step 1: 루트 workspace 갱신** — `Cargo.toml`(루트):

```toml
[workspace]
members = [
    "./",
    "apps/desktop/core",
]
exclude = [
    ".worktrees",
    # Tauri pulls the whole webview stack; it is its own workspace so a root `cargo test`
    # does not pay for it. See apps/desktop/src-tauri/Cargo.toml.
    "apps/desktop/src-tauri",
]
```

- [ ] **Step 2: 크레이트 매니페스트** — `apps/desktop/core/Cargo.toml`:

```toml
[package]
name = "ailoy-desktop-core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false
description = "Headless session engine for Ailoy Desktop: sessions, workspace, runs"

[[bin]]
name = "gen-catalog"
path = "src/bin/gen_catalog.rs"

[dependencies]
ailoy = { path = "../../.." }
# The workspace, its stores, and the FUSE-T binding that puts it where a kernel answers.
cortex = { path = "../../../../cortex/cortex", features = ["fuse-t", "notion", "s3"] }
anyhow = "1"
thiserror = "2"
serde = { version = "1", features = ["derive"] }
serde_json = { version = "1", features = ["preserve_order"] }
rusqlite = { version = "0.40", features = ["bundled"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync", "process", "fs", "time"] }
tokio-util = "0.7"
futures = "0.3"
reqwest = { version = "0.13", features = ["json"] }
chrono = "0.4"
tracing = "0.1"
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: `src/error.rs`**

```rust
//! The one error the engine answers in. Serialized as its message, because a webview
//! shows a string and nothing else.

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("session is already running")]
    AlreadyRunning,
    #[error("{0}")]
    Invalid(String),
    #[error("console unavailable: {0}")]
    ConsoleUnavailable(String),
    #[error("workspace: {0}")]
    Workspace(String),
    #[error("storage: {0}")]
    Storage(#[from] rusqlite::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl serde::Serialize for EngineError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

pub type Result<T> = std::result::Result<T, EngineError>;
```

- [ ] **Step 4: `src/types.rs`** — 위 Interfaces 블록의 모든 타입을 `#[derive(Clone, Debug, Serialize, Deserialize)]`로 정의한다. 열거형 serde 속성: `MountKind`/`MountStatus`/`WorkspaceStatus`는 `#[serde(rename_all = "lowercase")]`, `MountStatus`·`WorkspaceStatus`는 추가로 `#[serde(tag = "status")]`, `MountConfig`는 `#[serde(tag = "kind", rename_all = "lowercase")]`. `SettingsPatch`는 `Default`도 파생하고 `#[serde(default)]` 를 구조체에 붙여 부분 패치가 가능하게 한다(테스트가 `SettingsPatch::default()`를 쓴다). `MountKind`·`MountStatus`·`WorkspaceStatus`는 `PartialEq`도 파생한다. 끝에:

```rust
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}

impl MountConfig {
    /// A copy safe to show: secrets replaced by their last four characters.
    pub fn masked(&self) -> MountConfig {
        fn hint(s: &str) -> String {
            let tail: String = s.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
            format!("…{tail}")
        }
        match self {
            MountConfig::Notion { api_key } => MountConfig::Notion { api_key: hint(api_key) },
            MountConfig::S3 { bucket, region, access_key_id, secret_access_key, endpoint, key_prefix } => MountConfig::S3 {
                bucket: bucket.clone(), region: region.clone(),
                access_key_id: hint(access_key_id), secret_access_key: hint(secret_access_key),
                endpoint: endpoint.clone(), key_prefix: key_prefix.clone(),
            },
            other => other.clone(),
        }
    }
}
```

- [ ] **Step 5: `src/config.rs`**

```rust
use std::path::PathBuf;

/// Where the engine keeps everything, and what it may start.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub data_dir: PathBuf,
    /// The `cortex-local-console` binary. `None` searches `AILOY_CORTEX_BIN_DIR`, then the
    /// sibling `cortex` checkout's `target/` — see `console::resolve_console_bin`.
    pub console_bin: Option<PathBuf>,
    pub catalog_refresh: bool,
    /// `false` skips the FUSE-T mount (tests, machines without FUSE-T). The console then
    /// stands directly in `files_root()`.
    pub mount_workspace: bool,
}

impl EngineConfig {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self { data_dir: data_dir.into(), console_bin: None, catalog_refresh: true, mount_workspace: true }
    }
    pub fn db_path(&self) -> PathBuf { self.data_dir.join("ailoy.sqlite") }
    pub fn files_root(&self) -> PathBuf { self.data_dir.join("files") }
    pub fn mountpoint(&self) -> PathBuf { self.data_dir.join("workspace") }
    pub fn cache_dir(&self) -> PathBuf { self.data_dir.join("cache") }
}
```

- [ ] **Step 6: `src/lib.rs`** (아직 없는 모듈은 이 Task에서는 선언하지 않고, 각 Task가 추가한다)

```rust
//! Ailoy Desktop's session engine: everything the window does, without the window.

pub mod config;
pub mod error;
pub mod types;

pub use config::EngineConfig;
pub use error::{EngineError, Result};
pub use types::*;
```

- [ ] **Step 7: 컴파일과 커밋**

```bash
cargo check -p ailoy-desktop-core 2>&1 | tail -3
git add Cargo.toml apps/desktop/core && git commit -m "feat(desktop-core): crate scaffold, error and IPC types, config"
```

`gen-catalog` bin이 아직 없어 실패하면 `src/bin/gen_catalog.rs` 에 `fn main() {}` 스텁을 두고 Task B3에서 채운다.

---

### Task B2: SQLite 저장소

**Files:**
- Create: `src/store/mod.rs`, `src/store/migrations/0001_init.sql`
- Modify: `src/lib.rs` (`pub mod store;`)

**Interfaces:**
- Produces:
  ```rust
  pub struct Store { /* Mutex<Connection> */ }
  pub struct SessionRow { pub id: String, pub title: String, pub model: String, pub created_at: i64, pub updated_at: i64 }
  pub struct MountRow { pub id: String, pub path: String, pub kind: MountKind, pub label: String, pub config: MountConfig, pub writable: bool, pub created_at: i64 }
  pub struct NewMessage<'a> { pub depth: u8, pub source_agent: Option<&'a str>, pub message: &'a Message, pub usage: Option<&'a TokenUsage> }
  impl Store {
    pub fn open(path: &Path) -> Result<Store>; pub fn open_in_memory() -> Result<Store>;
    pub fn session_list(&self) -> Result<Vec<SessionRow>>; pub fn session_get(&self, id: &str) -> Result<SessionRow>;
    pub fn session_create(&self, id: &str, title: &str, model: &str) -> Result<SessionRow>;
    pub fn session_rename(&self, id: &str, title: &str) -> Result<()>; pub fn session_set_model(&self, id: &str, model: &str) -> Result<()>;
    pub fn session_touch(&self, id: &str) -> Result<()>; pub fn session_delete(&self, id: &str) -> Result<()>;
    pub fn message_append(&self, session_id: &str, m: NewMessage<'_>) -> Result<i64>;   // 반환: seq
    pub fn message_list(&self, session_id: &str) -> Result<Vec<StoredMessage>>;
    pub fn message_history(&self, session_id: &str) -> Result<Vec<Message>>;            // depth 0만, seq 순
    pub fn message_usages(&self, session_id: &str) -> Result<Vec<TokenUsage>>;         // assistant usage 모두, seq 순
    pub fn mount_list(&self) -> Result<Vec<MountRow>>; pub fn mount_insert(&self, row: &MountRow) -> Result<()>; pub fn mount_delete(&self, path: &str) -> Result<()>;
    pub fn setting_get(&self, key: &str) -> Result<Option<String>>; pub fn setting_set(&self, key: &str, value: &str) -> Result<()>; pub fn setting_delete(&self, key: &str) -> Result<()>;
  }
  ```

- [ ] **Step 1: 마이그레이션 SQL** — `src/store/migrations/0001_init.sql`

```sql
CREATE TABLE workspaces (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
INSERT INTO workspaces (id, name, created_at) VALUES ('default', 'Workspace', 0);

CREATE TABLE mounts (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL DEFAULT 'default' REFERENCES workspaces(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  kind TEXT NOT NULL,
  label TEXT NOT NULL,
  config TEXT NOT NULL,
  writable INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE(workspace_id, path)
);

CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL DEFAULT 'default' REFERENCES workspaces(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  model TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE messages (
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  depth INTEGER NOT NULL DEFAULT 0,
  source_agent TEXT,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  usage TEXT,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (session_id, seq)
);

CREATE TABLE settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
```

- [ ] **Step 2: 실패하는 테스트** — `src/store/mod.rs` 하단 `mod tests`:

```rust
#[cfg(test)]
mod tests {
    use ailoy::message::{Message, Part, Role, TokenUsage};

    use super::*;

    fn text(role: Role, t: &str) -> Message {
        Message::new(role).with_contents([Part::text(t)])
    }

    #[test]
    fn sessions_round_trip() {
        let s = Store::open_in_memory().unwrap();
        let row = s.session_create("s1", "첫 대화", "anthropic/claude-opus-5").unwrap();
        assert_eq!(row.title, "첫 대화");
        s.session_rename("s1", "이름 변경").unwrap();
        assert_eq!(s.session_get("s1").unwrap().title, "이름 변경");
        assert_eq!(s.session_list().unwrap().len(), 1);
        s.session_delete("s1").unwrap();
        assert!(matches!(s.session_get("s1"), Err(EngineError::NotFound(_))));
    }

    #[test]
    fn messages_keep_order_depth_and_usage() {
        let s = Store::open_in_memory().unwrap();
        s.session_create("s1", "t", "m").unwrap();
        let u = TokenUsage { input_tokens: 10, output_tokens: 5, cache_creation_input_tokens: None, cache_read_input_tokens: Some(3) };
        let seq1 = s.message_append("s1", NewMessage { depth: 0, source_agent: None, message: &text(Role::User, "hi"), usage: None }).unwrap();
        let seq2 = s.message_append("s1", NewMessage { depth: 0, source_agent: None, message: &text(Role::Assistant, "hello"), usage: Some(&u) }).unwrap();
        let seq3 = s.message_append("s1", NewMessage { depth: 1, source_agent: Some("sub"), message: &text(Role::Assistant, "inner"), usage: None }).unwrap();
        assert_eq!((seq1, seq2, seq3), (1, 2, 3));

        let all = s.message_list("s1").unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[2].depth, 1);
        assert_eq!(all[2].source_agent.as_deref(), Some("sub"));
        assert_eq!(all[1].usage.as_ref().unwrap().cache_read_input_tokens, Some(3));

        let history = s.message_history("s1").unwrap();
        assert_eq!(history.len(), 2, "depth 1 is not replayed to the model");
        assert_eq!(history[1].contents[0].as_text(), Some("hello"));

        assert_eq!(s.message_usages("s1").unwrap().len(), 1);
        s.session_delete("s1").unwrap();
        assert!(s.message_list("s1").unwrap().is_empty(), "cascade");
    }

    #[test]
    fn mounts_and_settings_round_trip() {
        let s = Store::open_in_memory().unwrap();
        let row = MountRow {
            id: "m1".into(), path: "/notion".into(), kind: MountKind::Notion, label: "notion".into(),
            config: MountConfig::Notion { api_key: "secret_x".into() }, writable: false, created_at: 1,
        };
        s.mount_insert(&row).unwrap();
        assert!(s.mount_insert(&row).is_err(), "unique path");
        let rows = s.mount_list().unwrap();
        assert!(matches!(&rows[0].config, MountConfig::Notion { api_key } if api_key == "secret_x"));
        s.mount_delete("/notion").unwrap();
        assert!(s.mount_list().unwrap().is_empty());

        assert_eq!(s.setting_get("default_model").unwrap(), None);
        s.setting_set("default_model", "openai/gpt-5").unwrap();
        s.setting_set("default_model", "anthropic/claude-opus-5").unwrap();
        assert_eq!(s.setting_get("default_model").unwrap().as_deref(), Some("anthropic/claude-opus-5"));
        s.setting_delete("default_model").unwrap();
        assert_eq!(s.setting_get("default_model").unwrap(), None);
    }
}
```

- [ ] **Step 3: 실패 확인**

```bash
cargo test -p ailoy-desktop-core store 2>&1 | tail -5
```

Expected: 컴파일 실패.

- [ ] **Step 4: 구현** — `src/store/mod.rs`

```rust
//! SQLite persistence. One connection behind a mutex: every call is a few local
//! statements, so a lock is cheaper than a pool and keeps writes ordered.

use std::{
    path::Path,
    sync::Mutex,
};

use ailoy::message::{Message, TokenUsage};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::{
    error::{EngineError, Result},
    types::{MountConfig, MountKind, StoredMessage, now_ms},
};

const MIGRATIONS: &[&str] = &[include_str!("migrations/0001_init.sql")];

/// The content column: the ailoy message, versioned so a later shape can be read back.
#[derive(Serialize, Deserialize)]
struct Content {
    version: u32,
    message: Message,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRow {
    pub id: String,
    pub title: String,
    pub model: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MountRow {
    pub id: String,
    pub path: String,
    pub kind: MountKind,
    pub label: String,
    pub config: MountConfig,
    pub writable: bool,
    pub created_at: i64,
}

pub struct NewMessage<'a> {
    pub depth: u8,
    pub source_agent: Option<&'a str>,
    pub message: &'a Message,
    pub usage: Option<&'a TokenUsage>,
}

pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Store> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;")?;
        Self::migrate(&conn)?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    pub fn open_in_memory() -> Result<Store> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        Self::migrate(&conn)?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    fn migrate(conn: &Connection) -> Result<()> {
        let current: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        for (i, sql) in MIGRATIONS.iter().enumerate() {
            let version = i as u32 + 1;
            if version > current {
                conn.execute_batch(&format!("BEGIN; {sql} PRAGMA user_version = {version}; COMMIT;"))?;
            }
        }
        Ok(())
    }

    fn with<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> Result<T> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        Ok(f(&conn)?)
    }

    // ── sessions ────────────────────────────────────────────────────────────

    fn session_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRow> {
        Ok(SessionRow { id: r.get(0)?, title: r.get(1)?, model: r.get(2)?, created_at: r.get(3)?, updated_at: r.get(4)? })
    }

    pub fn session_list(&self) -> Result<Vec<SessionRow>> {
        self.with(|c| {
            let mut st = c.prepare("SELECT id, title, model, created_at, updated_at FROM sessions ORDER BY updated_at DESC")?;
            st.query_map([], Self::session_from_row)?.collect()
        })
    }

    pub fn session_get(&self, id: &str) -> Result<SessionRow> {
        self.with(|c| {
            c.query_row("SELECT id, title, model, created_at, updated_at FROM sessions WHERE id = ?1", params![id], Self::session_from_row).optional()
        })?
        .ok_or_else(|| EngineError::NotFound(format!("session {id}")))
    }

    pub fn session_create(&self, id: &str, title: &str, model: &str) -> Result<SessionRow> {
        let now = now_ms();
        self.with(|c| {
            c.execute(
                "INSERT INTO sessions (id, title, model, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?4)",
                params![id, title, model, now],
            )
        })?;
        Ok(SessionRow { id: id.into(), title: title.into(), model: model.into(), created_at: now, updated_at: now })
    }

    pub fn session_rename(&self, id: &str, title: &str) -> Result<()> {
        let n = self.with(|c| c.execute("UPDATE sessions SET title = ?2, updated_at = ?3 WHERE id = ?1", params![id, title, now_ms()]))?;
        if n == 0 { return Err(EngineError::NotFound(format!("session {id}"))); }
        Ok(())
    }

    pub fn session_set_model(&self, id: &str, model: &str) -> Result<()> {
        let n = self.with(|c| c.execute("UPDATE sessions SET model = ?2 WHERE id = ?1", params![id, model]))?;
        if n == 0 { return Err(EngineError::NotFound(format!("session {id}"))); }
        Ok(())
    }

    pub fn session_touch(&self, id: &str) -> Result<()> {
        self.with(|c| c.execute("UPDATE sessions SET updated_at = ?2 WHERE id = ?1", params![id, now_ms()]))?;
        Ok(())
    }

    pub fn session_delete(&self, id: &str) -> Result<()> {
        let n = self.with(|c| c.execute("DELETE FROM sessions WHERE id = ?1", params![id]))?;
        if n == 0 { return Err(EngineError::NotFound(format!("session {id}"))); }
        Ok(())
    }

    // ── messages ────────────────────────────────────────────────────────────

    pub fn message_append(&self, session_id: &str, m: NewMessage<'_>) -> Result<i64> {
        let content = serde_json::to_string(&Content { version: 1, message: m.message.clone() })
            .map_err(|e| EngineError::Other(e.into()))?;
        let usage = m.usage.map(serde_json::to_string).transpose().map_err(|e| EngineError::Other(e.into()))?;
        let role = m.message.role.to_string();
        self.with(|c| {
            let seq: i64 = c.query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE session_id = ?1",
                params![session_id],
                |r| r.get(0),
            )?;
            c.execute(
                "INSERT INTO messages (session_id, seq, depth, source_agent, role, content, usage, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![session_id, seq, m.depth as i64, m.source_agent, role, content, usage, now_ms()],
            )?;
            Ok(seq)
        })
    }

    fn stored_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<StoredMessage> {
        let content: String = r.get(3)?;
        let usage: Option<String> = r.get(4)?;
        let parsed: Content = serde_json::from_str(&content)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e)))?;
        let usage = usage
            .map(|u| serde_json::from_str::<TokenUsage>(&u))
            .transpose()
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e)))?;
        Ok(StoredMessage {
            seq: r.get(0)?,
            depth: r.get::<_, i64>(1)? as u8,
            source_agent: r.get(2)?,
            message: parsed.message,
            usage,
            created_at: r.get(5)?,
        })
    }

    pub fn message_list(&self, session_id: &str) -> Result<Vec<StoredMessage>> {
        self.with(|c| {
            let mut st = c.prepare("SELECT seq, depth, source_agent, content, usage, created_at FROM messages WHERE session_id = ?1 ORDER BY seq")?;
            st.query_map(params![session_id], Self::stored_from_row)?.collect()
        })
    }

    pub fn message_history(&self, session_id: &str) -> Result<Vec<Message>> {
        Ok(self.message_list(session_id)?.into_iter().filter(|m| m.depth == 0).map(|m| m.message).collect())
    }

    pub fn message_usages(&self, session_id: &str) -> Result<Vec<TokenUsage>> {
        Ok(self.message_list(session_id)?.into_iter().filter_map(|m| m.usage).collect())
    }

    // ── mounts ──────────────────────────────────────────────────────────────

    pub fn mount_list(&self) -> Result<Vec<MountRow>> {
        self.with(|c| {
            let mut st = c.prepare("SELECT id, path, kind, label, config, writable, created_at FROM mounts ORDER BY path")?;
            st.query_map([], |r| {
                let kind: String = r.get(2)?;
                let config: String = r.get(4)?;
                Ok(MountRow {
                    id: r.get(0)?,
                    path: r.get(1)?,
                    kind: serde_json::from_value(serde_json::Value::String(kind))
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(2, rusqlite::types::Type::Text, Box::new(e)))?,
                    label: r.get(3)?,
                    config: serde_json::from_str(&config)
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e)))?,
                    writable: r.get::<_, i64>(5)? != 0,
                    created_at: r.get(6)?,
                })
            })?
            .collect()
        })
    }

    pub fn mount_insert(&self, row: &MountRow) -> Result<()> {
        let kind = serde_json::to_value(row.kind.clone()).map_err(|e| EngineError::Other(e.into()))?;
        let kind = kind.as_str().unwrap_or("local").to_string();
        let config = serde_json::to_string(&row.config).map_err(|e| EngineError::Other(e.into()))?;
        self.with(|c| {
            c.execute(
                "INSERT INTO mounts (id, path, kind, label, config, writable, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![row.id, row.path, kind, row.label, config, row.writable as i64, row.created_at],
            )
        })?;
        Ok(())
    }

    pub fn mount_delete(&self, path: &str) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM mounts WHERE path = ?1", params![path]))?;
        Ok(())
    }

    // ── settings ────────────────────────────────────────────────────────────

    pub fn setting_get(&self, key: &str) -> Result<Option<String>> {
        self.with(|c| c.query_row("SELECT value FROM settings WHERE key = ?1", params![key], |r| r.get(0)).optional())
    }

    pub fn setting_set(&self, key: &str, value: &str) -> Result<()> {
        self.with(|c| c.execute("INSERT INTO settings (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value", params![key, value]))?;
        Ok(())
    }

    pub fn setting_delete(&self, key: &str) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM settings WHERE key = ?1", params![key]))?;
        Ok(())
    }
}
```

`src/lib.rs`에 `pub mod store;` 추가.

- [ ] **Step 5: 통과 확인·커밋**

```bash
cargo test -p ailoy-desktop-core store 2>&1 | grep -E 'test result|FAILED|panicked'
git add apps/desktop/core && git commit -m "feat(desktop-core): SQLite store for sessions, messages, mounts, settings"
```

---

### Task B3: 모델 카탈로그 (models.dev)

**Files:**
- Create: `src/catalog.rs`, `src/bin/gen_catalog.rs`, `assets/models.json`(생성물)
- Modify: `src/lib.rs`

**Interfaces:**
- Produces:
  ```rust
  pub const PROVIDER_MAP: &[(&str /*ailoy prefix*/, &str /*models.dev id*/)];
  pub struct CatalogModel { pub id: String, pub name: String, pub reasoning: bool, pub tool_call: bool, pub context: Option<u64>, pub output: Option<u64>, pub cost: Option<ModelCost> }
  pub struct CatalogData { pub providers: BTreeMap<String /*models.dev id*/, CatalogProvider { pub name: String, pub models: BTreeMap<String, CatalogModel> }> }
  pub fn filter_models_dev(full: &serde_json::Value) -> CatalogData;
  pub fn split_model_id(ailoy_model: &str) -> Option<(&str, &str)>;
  pub fn models_dev_provider(ailoy_prefix: &str) -> Option<&'static str>;
  pub struct Catalog { .. }
  impl Catalog {
    pub fn load(cache: Option<&Path>) -> Catalog;             // 내장 스냅샷 → 캐시 파일이 있으면 덮어씀
    pub fn lookup(&self, ailoy_model: &str) -> Option<CatalogModel>;
    pub fn models_for(&self, ailoy_prefix: &str) -> Vec<CatalogModel>;
    pub async fn refresh(&self, cache: &Path) -> anyhow::Result<()>;   // GET https://models.dev/api.json → filter → 캐시 저장 → 메모리 교체
  }
  ```

- [ ] **Step 1: 실패하는 테스트** (`catalog.rs` 하단)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> serde_json::Value {
        serde_json::json!({
            "anthropic": { "id": "anthropic", "name": "Anthropic", "models": {
                "claude-opus-5": { "id": "claude-opus-5", "name": "Claude Opus 5", "reasoning": true, "tool_call": true,
                    "modalities": {"input": ["text","image"], "output": ["text"]},
                    "limit": {"context": 1000000, "output": 128000},
                    "cost": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25} },
                "claude-3-embed": { "id": "claude-3-embed", "name": "Embed", "tool_call": false,
                    "modalities": {"input": ["text"], "output": ["embedding"]}, "limit": {"context": 8000} }
            }},
            "xai": { "id": "xai", "name": "xAI", "models": {
                "grok-4.6": { "id": "grok-4.6", "name": "Grok 4.6", "tool_call": true,
                    "modalities": {"input": ["text"], "output": ["text"]}, "limit": {"context": 256000, "output": 32000}, "cost": {"input": 3, "output": 15} }
            }},
            "someone-else": { "id": "someone-else", "name": "X", "models": {
                "m": { "id": "m", "name": "m", "tool_call": true, "modalities": {"input":["text"],"output":["text"]}, "limit": {"context": 1} }
            }}
        })
    }

    #[test]
    fn filter_keeps_mapped_providers_and_chat_tool_models_only() {
        let data = filter_models_dev(&sample());
        assert!(data.providers.contains_key("anthropic"));
        assert!(data.providers.contains_key("xai"));
        assert!(!data.providers.contains_key("someone-else"));
        let a = &data.providers["anthropic"];
        assert_eq!(a.name, "Anthropic");
        assert!(a.models.contains_key("claude-opus-5"));
        assert!(!a.models.contains_key("claude-3-embed"), "no tool_call / non-text output");
        let m = &a.models["claude-opus-5"];
        assert_eq!(m.context, Some(1_000_000));
        assert_eq!(m.cost.as_ref().unwrap().cache_write, Some(6.25));
    }

    #[test]
    fn lookup_maps_ailoy_ids_to_models_dev_providers() {
        let cat = Catalog::from_data(filter_models_dev(&sample()));
        assert_eq!(cat.lookup("anthropic/claude-opus-5").unwrap().name, "Claude Opus 5");
        assert_eq!(cat.lookup("x-ai/grok-4.6").unwrap().context, Some(256_000));
        assert!(cat.lookup("google/gemini-2.5-pro").is_none());
        assert!(cat.lookup("no-slash").is_none());
        assert_eq!(cat.models_for("anthropic").len(), 1);
        assert_eq!(split_model_id("bedrock/anthropic.claude-opus-5"), Some(("bedrock", "anthropic.claude-opus-5")));
        assert_eq!(models_dev_provider("bedrock"), Some("amazon-bedrock"));
    }

    #[test]
    fn embedded_snapshot_parses_and_has_anthropic() {
        let cat = Catalog::load(None);
        assert!(cat.lookup("anthropic/claude-opus-5").is_some(), "regenerate assets/models.json with `cargo run -p ailoy-desktop-core --bin gen-catalog`");
    }
}
```

- [ ] **Step 2: 실패 확인**

```bash
cargo test -p ailoy-desktop-core catalog 2>&1 | tail -5
```

- [ ] **Step 3: 구현** — `src/catalog.rs`

```rust
//! Model metadata — context window, output cap, prices, capabilities — from models.dev.
//!
//! A snapshot filtered to the providers ailoy speaks is embedded at build time
//! (`assets/models.json`, regenerated by the `gen-catalog` binary); at runtime the same
//! filter runs over a fresh `https://models.dev/api.json` and the result is cached under the
//! app's data directory. A failed refresh leaves the embedded snapshot in place.

use std::{
    collections::BTreeMap,
    path::Path,
    sync::RwLock,
};

use serde::{Deserialize, Serialize};

use crate::types::ModelCost;

pub const MODELS_DEV_URL: &str = "https://models.dev/api.json";

/// ailoy model-id prefix → models.dev provider id.
pub const PROVIDER_MAP: &[(&str, &str)] = &[
    ("anthropic", "anthropic"),
    ("openai", "openai"),
    ("google", "google"),
    ("x-ai", "xai"),
    ("deepseek", "deepseek"),
    ("moonshotai", "moonshotai"),
    ("bedrock", "amazon-bedrock"),
];

pub fn models_dev_provider(ailoy_prefix: &str) -> Option<&'static str> {
    PROVIDER_MAP.iter().find(|(p, _)| *p == ailoy_prefix).map(|(_, id)| *id)
}

/// `"anthropic/claude-opus-5"` → `("anthropic", "claude-opus-5")`.
pub fn split_model_id(ailoy_model: &str) -> Option<(&str, &str)> {
    ailoy_model.split_once('/').filter(|(p, m)| !p.is_empty() && !m.is_empty())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub tool_call: bool,
    pub context: Option<u64>,
    pub output: Option<u64>,
    pub cost: Option<ModelCost>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CatalogProvider {
    pub name: String,
    pub models: BTreeMap<String, CatalogModel>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CatalogData {
    pub providers: BTreeMap<String, CatalogProvider>,
}

/// The raw models.dev shape, only the fields read here. Everything is optional so a new
/// field upstream never breaks the parse.
#[derive(Deserialize)]
struct RawProvider {
    #[serde(default)]
    name: String,
    #[serde(default)]
    models: BTreeMap<String, RawModel>,
}

#[derive(Deserialize, Default)]
struct RawModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    modalities: RawModalities,
    #[serde(default)]
    limit: RawLimit,
    cost: Option<ModelCost>,
}

#[derive(Deserialize, Default)]
struct RawModalities {
    #[serde(default)]
    output: Vec<String>,
}

#[derive(Deserialize, Default)]
struct RawLimit {
    context: Option<u64>,
    output: Option<u64>,
}

/// Keep the providers ailoy can call and the models an agent can drive: tool-calling, text
/// out, a known context window.
pub fn filter_models_dev(full: &serde_json::Value) -> CatalogData {
    let mut data = CatalogData::default();
    for (_, md_id) in PROVIDER_MAP {
        let Some(raw) = full.get(*md_id) else { continue };
        let Ok(provider) = serde_json::from_value::<RawProvider>(raw.clone()) else { continue };
        let models: BTreeMap<String, CatalogModel> = provider
            .models
            .into_iter()
            .filter(|(_, m)| m.tool_call && m.modalities.output.iter().any(|o| o == "text") && m.limit.context.is_some())
            .map(|(key, m)| {
                let id = if m.id.is_empty() { key.clone() } else { m.id };
                let name = if m.name.is_empty() { id.clone() } else { m.name };
                (key, CatalogModel { id, name, reasoning: m.reasoning, tool_call: m.tool_call, context: m.limit.context, output: m.limit.output, cost: m.cost })
            })
            .collect();
        if !models.is_empty() {
            data.providers.insert(md_id.to_string(), CatalogProvider { name: provider.name, models });
        }
    }
    data
}

pub struct Catalog {
    data: RwLock<CatalogData>,
}

impl Catalog {
    pub fn from_data(data: CatalogData) -> Catalog {
        Catalog { data: RwLock::new(data) }
    }

    /// The embedded snapshot, replaced by `cache` when it exists and parses.
    pub fn load(cache: Option<&Path>) -> Catalog {
        let embedded: CatalogData = serde_json::from_str(include_str!("../assets/models.json"))
            .expect("the embedded catalog is generated by gen-catalog and must parse");
        let data = cache
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<CatalogData>(&s).ok())
            .filter(|d| !d.providers.is_empty())
            .unwrap_or(embedded);
        Catalog::from_data(data)
    }

    pub fn lookup(&self, ailoy_model: &str) -> Option<CatalogModel> {
        let (prefix, model) = split_model_id(ailoy_model)?;
        let md = models_dev_provider(prefix)?;
        self.data.read().ok()?.providers.get(md)?.models.get(model).cloned()
    }

    pub fn models_for(&self, ailoy_prefix: &str) -> Vec<CatalogModel> {
        let Some(md) = models_dev_provider(ailoy_prefix) else { return vec![] };
        self.data
            .read()
            .ok()
            .and_then(|d| d.providers.get(md).map(|p| p.models.values().cloned().collect()))
            .unwrap_or_default()
    }

    pub async fn refresh(&self, cache: &Path) -> anyhow::Result<()> {
        let full: serde_json::Value = reqwest::Client::new()
            .get(MODELS_DEV_URL)
            .timeout(std::time::Duration::from_secs(20))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let data = filter_models_dev(&full);
        anyhow::ensure!(!data.providers.is_empty(), "models.dev answered with none of our providers");
        if let Some(parent) = cache.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(cache, serde_json::to_vec_pretty(&data)?).await?;
        *self.data.write().map_err(|_| anyhow::anyhow!("catalog lock poisoned"))? = data;
        Ok(())
    }
}
```

- [ ] **Step 4: 생성기** — `src/bin/gen_catalog.rs`

```rust
//! Regenerate `assets/models.json` from models.dev. Run from the repository root:
//! `cargo run -p ailoy-desktop-core --bin gen-catalog`.

use ailoy_desktop_core::catalog::{MODELS_DEV_URL, filter_models_dev};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let full: serde_json::Value = reqwest::get(MODELS_DEV_URL).await?.error_for_status()?.json().await?;
    let data = filter_models_dev(&full);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models.json");
    std::fs::create_dir_all(out.parent().unwrap())?;
    std::fs::write(&out, serde_json::to_vec_pretty(&data)?)?;
    let n: usize = data.providers.values().map(|p| p.models.len()).sum();
    println!("wrote {} ({} providers, {n} models)", out.display(), data.providers.len());
    Ok(())
}
```

`src/lib.rs`에 `pub mod catalog;` 추가. 스냅샷 생성(첫 컴파일 전에 `include_str!` 대상이 있어야 하므로 먼저 빈 파일을 만든 뒤 생성기를 실행):

```bash
mkdir -p apps/desktop/core/assets && echo '{"providers":{}}' > apps/desktop/core/assets/models.json
cargo run -p ailoy-desktop-core --bin gen-catalog
```

Expected: `wrote .../assets/models.json (7 providers, N models)`.

- [ ] **Step 5: 통과 확인·커밋**

```bash
cargo test -p ailoy-desktop-core catalog 2>&1 | grep -E 'test result|FAILED|panicked'
git add apps/desktop/core && git commit -m "feat(desktop-core): models.dev catalog with embedded snapshot and refresh"
```

---

### Task B4: 프로바이더 등록과 설정

**Files:**
- Create: `src/providers.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct ProviderDef { pub key: &'static str, pub label: &'static str, pub ailoy_prefix: &'static str, pub pattern: &'static str }
  pub const PROVIDERS: &[ProviderDef];   // anthropic, openai, google, xai, deepseek, moonshotai, bedrock
  pub const KEY_ANTHROPIC.. 대신: pub fn setting_key(provider_key: &str) -> String  // "provider.<key>.api_key"
  pub const BEDROCK_REGION_KEY: &str = "provider.bedrock.region";
  pub fn apply(store: &Store) -> Result<Vec<&'static str>>;      // 키 있는 프로바이더를 ailoy "default" 레지스트리에 등록, 없는 것은 제거. 반환: 활성 key 목록
  pub fn key_hint(key: &str) -> String;                          // "…abcd"
  pub fn read_settings(store: &Store) -> Result<Settings>;
  pub fn write_settings(store: &Store, patch: &SettingsPatch) -> Result<()>;
  pub const DEFAULT_MODEL: &str = "anthropic/claude-opus-5"; pub const DEFAULT_MAX_TOKENS: u64 = 32_000; pub const DEFAULT_MAX_TURNS: u32 = 50;
  ```

- [ ] **Step 1: 테스트** (`providers.rs` 하단)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    #[test]
    fn settings_default_then_patch() {
        let s = Store::open_in_memory().unwrap();
        let settings = read_settings(&s).unwrap();
        assert_eq!(settings.default_model, DEFAULT_MODEL);
        assert_eq!(settings.max_turns, DEFAULT_MAX_TURNS);
        assert!(settings.providers.iter().all(|p| !p.has_key));

        let mut patch = SettingsPatch::default();
        patch.provider_keys.insert("anthropic".into(), Some("sk-ant-abcdefgh1234".into()));
        patch.default_model = Some("anthropic/claude-sonnet-5".into());
        patch.max_turns = Some(10);
        write_settings(&s, &patch).unwrap();

        let settings = read_settings(&s).unwrap();
        let a = settings.providers.iter().find(|p| p.key == "anthropic").unwrap();
        assert!(a.has_key);
        assert_eq!(a.key_hint, "…1234");
        assert_eq!(settings.default_model, "anthropic/claude-sonnet-5");
        assert_eq!(settings.max_turns, 10);

        let mut patch = SettingsPatch::default();
        patch.provider_keys.insert("anthropic".into(), None);
        write_settings(&s, &patch).unwrap();
        assert!(!read_settings(&s).unwrap().providers.iter().find(|p| p.key == "anthropic").unwrap().has_key);
    }

    #[test]
    fn apply_registers_only_keyed_providers() {
        let s = Store::open_in_memory().unwrap();
        s.setting_set(&setting_key("openai"), "sk-test").unwrap();
        let active = apply(&s).unwrap();
        assert_eq!(active, vec!["openai"]);
        let reg = ailoy::lang_model::get_lm_providers();
        let def = reg.get("default").unwrap();
        assert!(def.get("openai/gpt-5").is_some());
        assert!(def.get("anthropic/claude-opus-5").is_none() || std::env::var("ANTHROPIC_API_KEY").is_ok());
    }
}
```

- [ ] **Step 2: 구현** — `src/providers.rs`

```rust
//! Settings, and the one side effect they have: which model providers ailoy can call.

use std::collections::BTreeMap;

use ailoy::lang_model::{BedrockRegion, LangModelProvider, get_lm_providers_mut};

use crate::{
    error::{EngineError, Result},
    store::Store,
    types::{ProviderSetting, Settings, SettingsPatch},
};

pub struct ProviderDef {
    pub key: &'static str,
    pub label: &'static str,
    pub ailoy_prefix: &'static str,
    pub pattern: &'static str,
}

pub const PROVIDERS: &[ProviderDef] = &[
    ProviderDef { key: "anthropic", label: "Anthropic", ailoy_prefix: "anthropic", pattern: "anthropic/*" },
    ProviderDef { key: "openai", label: "OpenAI", ailoy_prefix: "openai", pattern: "openai/*" },
    ProviderDef { key: "google", label: "Google Gemini", ailoy_prefix: "google", pattern: "google/*" },
    ProviderDef { key: "xai", label: "xAI", ailoy_prefix: "x-ai", pattern: "x-ai/*" },
    ProviderDef { key: "deepseek", label: "DeepSeek", ailoy_prefix: "deepseek", pattern: "deepseek/*" },
    ProviderDef { key: "moonshotai", label: "Moonshot Kimi", ailoy_prefix: "moonshotai", pattern: "moonshotai/*" },
    ProviderDef { key: "bedrock", label: "Amazon Bedrock", ailoy_prefix: "bedrock", pattern: "bedrock/*" },
];

pub const BEDROCK_REGION_KEY: &str = "provider.bedrock.region";
pub const DEFAULT_MODEL: &str = "anthropic/claude-opus-5";
pub const DEFAULT_MAX_TOKENS: u64 = 32_000;
pub const DEFAULT_MAX_TURNS: u32 = 50;

pub fn setting_key(provider_key: &str) -> String {
    format!("provider.{provider_key}.api_key")
}

pub fn key_hint(key: &str) -> String {
    let tail: String = key.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("…{tail}")
}

fn provider(key: &str) -> Option<&'static ProviderDef> {
    PROVIDERS.iter().find(|p| p.key == key)
}

/// Register every provider that has a key, drop every one that does not, in ailoy's
/// process-wide `"default"` registry. Returns the keys now active.
pub fn apply(store: &Store) -> Result<Vec<&'static str>> {
    let mut active = Vec::new();
    let region = store.setting_get(BEDROCK_REGION_KEY)?.unwrap_or_else(|| "us-east-1".to_string());
    let mut registry = get_lm_providers_mut();
    let default = registry.entry("default".to_string()).or_insert_with(LangModelProvider::new);
    for def in PROVIDERS {
        let key = store.setting_get(&setting_key(def.key))?.filter(|k| !k.trim().is_empty());
        match key {
            None => default.remove(def.pattern),
            Some(k) => {
                let elem = match def.key {
                    "anthropic" => LangModelProvider::anthropic(k),
                    "openai" => LangModelProvider::openai(k),
                    "google" => LangModelProvider::gemini(k),
                    "xai" => LangModelProvider::grok(k),
                    "deepseek" => LangModelProvider::deepseek(k),
                    "moonshotai" => LangModelProvider::kimi(k),
                    "bedrock" => {
                        let region: BedrockRegion = region
                            .parse()
                            .map_err(|_| EngineError::Invalid(format!("unsupported Bedrock region {region:?}")))?;
                        LangModelProvider::bedrock(region, k)
                    }
                    _ => unreachable!("PROVIDERS is the closed list above"),
                };
                default.insert(def.pattern.to_string(), elem);
                active.push(def.key);
            }
        }
    }
    Ok(active)
}

pub fn read_settings(store: &Store) -> Result<Settings> {
    let mut providers = Vec::with_capacity(PROVIDERS.len());
    for def in PROVIDERS {
        let key = store.setting_get(&setting_key(def.key))?.filter(|k| !k.trim().is_empty());
        providers.push(ProviderSetting {
            key: def.key.into(),
            label: def.label.into(),
            has_key: key.is_some(),
            key_hint: key.as_deref().map(key_hint).unwrap_or_default(),
            region: if def.key == "bedrock" { store.setting_get(BEDROCK_REGION_KEY)? } else { None },
        });
    }
    Ok(Settings {
        providers,
        default_model: store.setting_get("default_model")?.unwrap_or_else(|| DEFAULT_MODEL.into()),
        max_tokens: store.setting_get("max_tokens")?.and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MAX_TOKENS),
        max_turns: store.setting_get("max_turns")?.and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MAX_TURNS),
        catalog_refresh: store.setting_get("catalog_refresh")?.map(|v| v == "true").unwrap_or(true),
    })
}

pub fn write_settings(store: &Store, patch: &SettingsPatch) -> Result<()> {
    for (key, value) in &patch.provider_keys {
        if provider(key).is_none() {
            return Err(EngineError::Invalid(format!("unknown provider {key}")));
        }
        match value.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            Some(v) => store.setting_set(&setting_key(key), v)?,
            None => store.setting_delete(&setting_key(key))?,
        }
    }
    if let Some(r) = &patch.bedrock_region { store.setting_set(BEDROCK_REGION_KEY, r.trim())?; }
    if let Some(m) = &patch.default_model { store.setting_set("default_model", m.trim())?; }
    if let Some(t) = patch.max_tokens { store.setting_set("max_tokens", &t.to_string())?; }
    if let Some(t) = patch.max_turns { store.setting_set("max_turns", &t.to_string())?; }
    if let Some(c) = patch.catalog_refresh { store.setting_set("catalog_refresh", if c { "true" } else { "false" })?; }
    let _ = BTreeMap::<String, String>::new(); // keeps the import meaningful if unused elsewhere
    Ok(())
}
```

(`BTreeMap` 더미 줄은 실제로 불필요하면 `use` 와 함께 삭제.) `src/lib.rs`에 `pub mod providers;`.

- [ ] **Step 3: 확인·커밋**

```bash
cargo test -p ailoy-desktop-core providers 2>&1 | grep -E 'test result|FAILED|panicked'
git add apps/desktop/core && git commit -m "feat(desktop-core): settings and provider registration into ailoy"
```

---

### Task B5: 시스템 프리앰블

**Files:**
- Create: `src/prompt.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `pub struct PromptInput<'a> { pub workfs_path: &'a Path, pub mounts: &'a [MountInfo], pub today: &'a str, pub os: &'a str, pub extra: Option<&'a str> }`, `pub fn build(input: &PromptInput) -> String`

- [ ] **Step 1: 테스트**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MountInfo, MountKind, MountStatus};

    #[test]
    fn preamble_names_workfs_mounts_and_readonly() {
        let mounts = vec![
            MountInfo { id: "r".into(), path: "/".into(), kind: MountKind::Root, label: "Workspace".into(), detail: "".into(), writable: true, status: MountStatus::Ok },
            MountInfo { id: "n".into(), path: "/notion".into(), kind: MountKind::Notion, label: "notion".into(), detail: "Notion".into(), writable: false, status: MountStatus::Ok },
        ];
        let s = build(&PromptInput { workfs_path: Path::new("/tmp/ws"), mounts: &mounts, today: "2026-09-11", os: "macos", extra: Some("Answer in Korean.") });
        assert!(s.contains("/tmp/ws"));
        assert!(s.contains("/notion"));
        assert!(s.contains("read-only"));
        assert!(s.contains("page.json"));
        assert!(s.contains("2026-09-11"));
        assert!(s.ends_with("Answer in Korean."));
    }
}
```

- [ ] **Step 2: 구현**

```rust
//! The system preamble. ailoy sends only what `instruction` says, so this is where the
//! agent learns where it stands and what is mounted there.

use std::path::Path;

use crate::types::{MountInfo, MountKind};

pub struct PromptInput<'a> {
    pub workfs_path: &'a Path,
    pub mounts: &'a [MountInfo],
    pub today: &'a str,
    pub os: &'a str,
    pub extra: Option<&'a str>,
}

pub fn build(input: &PromptInput) -> String {
    let mut s = String::new();
    s.push_str("You are Ailoy, a desktop assistant that works inside the user's workspace: a directory tree the user assembled from local folders and connected services. You read, write and run commands there with your tools, and you explain what you did in plain language.\n\n");
    s.push_str(&format!("Today is {}. The host OS is {}.\n\n", input.today, input.os));
    s.push_str(&format!(
        "# Workspace\n\nThe workspace root is `{}`. It is also the shell's working directory, so relative paths resolve inside it. Stay inside the workspace unless the user explicitly asks about another path.\n\n",
        input.workfs_path.display()
    ));
    s.push_str("## Mounts\n\n");
    for m in input.mounts {
        let access = if m.writable { "read-write" } else { "read-only" };
        let hint = match m.kind {
            MountKind::Root => "the workspace's own files",
            MountKind::Local => "a folder on this computer",
            MountKind::Notion => "a Notion workspace; each page is a directory whose `page.json` holds the page as JSON; databases are directories of pages",
            MountKind::S3 => "an S3 bucket; keys appear as files and directories",
        };
        let path = if m.path.is_empty() { "/" } else { m.path.as_str() };
        s.push_str(&format!("- `{path}` — {} ({access}): {hint}\n", m.label));
    }
    s.push_str("\nA read-only mount rejects writes; do not retry them — tell the user.\n\n");
    s.push_str("# Tools\n\n`shell` runs `sh -c` in the workspace (output over 30k characters is middle-truncated and flagged `truncated`; a command past its timeout is killed and reported `timed_out`). Prefer `read`, `write`, `edit`, `glob`, `grep` for files, and `shell` for everything else. Run independent tool calls in parallel when it saves time.\n");
    if let Some(extra) = input.extra.map(str::trim).filter(|e| !e.is_empty()) {
        s.push_str("\n# Additional instructions\n\n");
        s.push_str(extra);
    }
    s
}
```

- [ ] **Step 3: 확인·커밋**

```bash
cargo test -p ailoy-desktop-core prompt 2>&1 | grep -E 'test result|FAILED'
git add apps/desktop/core && git commit -m "feat(desktop-core): system preamble with workspace and mount table"
```

---

### Task B6: 워크스페이스 — SharedFs, fsops, 커넥터

**Files:**
- Create: `src/workspace/mod.rs`, `src/workspace/shared.rs`, `src/workspace/fsops.rs`, `src/workspace/connectors.rs`
- Modify: `src/lib.rs` (`pub mod workspace;`)

**Interfaces:**
- Produces:
  ```rust
  pub struct SharedFs(Arc<tokio::sync::RwLock<WorkFs>>);  impl FileSystem + Clone
  // fsops (모두 &dyn FileSystem 위 free fn, 경로는 "/"-rooted 문자열)
  pub async fn list(fs: &dyn FileSystem, path: &str) -> Result<Vec<Entry>>
  pub async fn read(fs: &dyn FileSystem, path: &str) -> Result<FileContent>
  pub async fn write(fs: &dyn FileSystem, path: &str, text: &str) -> Result<()>
  pub async fn mkdir(fs: &dyn FileSystem, path: &str) -> Result<()>
  pub async fn delete(fs: &dyn FileSystem, path: &str) -> Result<()>
  pub async fn rename(fs: &dyn FileSystem, from: &str, to: &str) -> Result<()>
  pub async fn import(fs: &dyn FileSystem, dest: &str, sources: Vec<PathBuf>) -> Result<ImportReport>
  pub fn join(dir: &str, name: &str) -> String
  // connectors
  pub fn normalize_mount_path(path: &str) -> Result<String>     // "/notion"
  pub fn describe(config: &MountConfig) -> (MountKind, String /*detail*/, bool /*writable*/)
  pub async fn build_and_probe(config: &MountConfig) -> Result<Box<dyn FileSystem>>   // 생성 + 연결 확인(15초 상한)
  ```

- [ ] **Step 1: `shared.rs`** — cortex-gui `shared.rs`를 그대로 옮긴다(모듈 doc 유지). 11개 메서드 모두 `let fs = self.0.read().await; fs.<method>(..).await` 로 전달. 시그니처는 cortex `FileSystem` 트레이트와 동일(`stat, list, read_at, create, mkdir, unlink, rmdir, write_at, truncate, rename, flush`).

```rust
use std::{io, path::Path, sync::Arc};

use cortex::{
    BoxFuture,
    fs::{Dirent, FileSystem, Stat, WorkFs},
};
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct SharedFs(pub Arc<RwLock<WorkFs>>);

impl FileSystem for SharedFs {
    fn stat<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move { self.0.read().await.stat(path).await })
    }
    fn list<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Vec<Dirent>>> {
        Box::pin(async move { self.0.read().await.list(path).await })
    }
    fn read_at<'a>(&'a self, path: &'a Path, buf: &'a mut [u8], offset: u64) -> BoxFuture<'a, io::Result<usize>> {
        Box::pin(async move { self.0.read().await.read_at(path, buf, offset).await })
    }
    fn create<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move { self.0.read().await.create(path).await })
    }
    fn mkdir<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move { self.0.read().await.mkdir(path).await })
    }
    fn unlink<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move { self.0.read().await.unlink(path).await })
    }
    fn rmdir<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move { self.0.read().await.rmdir(path).await })
    }
    fn write_at<'a>(&'a self, path: &'a Path, buf: &'a [u8], offset: u64) -> BoxFuture<'a, io::Result<usize>> {
        Box::pin(async move { self.0.read().await.write_at(path, buf, offset).await })
    }
    fn truncate<'a>(&'a self, path: &'a Path, size: u64) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move { self.0.read().await.truncate(path, size).await })
    }
    fn rename<'a>(&'a self, from: &'a Path, to: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move { self.0.read().await.rename(from, to).await })
    }
    fn flush<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move { self.0.read().await.flush(path).await })
    }
}
```

- [ ] **Step 2: `fsops.rs` 테스트** (`InMemFs` 루트의 `WorkFs` 위에서)

```rust
#[cfg(test)]
mod tests {
    use cortex::fs::{InMemFs, WorkFs};

    use super::*;

    fn ws() -> WorkFs {
        WorkFs::new().try_with_mount("", InMemFs::new()).unwrap()
    }

    #[tokio::test]
    async fn write_read_list_delete() {
        let fs = ws();
        mkdir(&fs, "/docs/notes").await.unwrap();
        write(&fs, "/docs/notes/a.md", "hello").await.unwrap();
        write(&fs, "/docs/notes/a.md", "hi").await.unwrap(); // shorter rewrite truncates
        let got = read(&fs, "/docs/notes/a.md").await.unwrap();
        assert_eq!(got.text.as_deref(), Some("hi"));
        assert!(!got.truncated);
        let entries = list(&fs, "/docs").await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, "dir");
        assert_eq!(entries[0].path, "/docs/notes");
        rename(&fs, "/docs/notes/a.md", "/docs/notes/b.md").await.unwrap();
        delete(&fs, "/docs/notes/b.md").await.unwrap();
        assert!(list(&fs, "/docs/notes").await.unwrap().is_empty());
        assert!(matches!(read(&fs, "/docs").await, Err(EngineError::Invalid(_))));
    }

    #[tokio::test]
    async fn import_copies_files_and_reports_skips() {
        let fs = ws();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("x.txt"), b"xx").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/y.txt"), b"y").unwrap();
        let report = import(&fs, "/", vec![dir.path().join("x.txt"), dir.path().join("sub"), dir.path().join("missing")]).await.unwrap();
        assert_eq!(report.files, 2);
        assert_eq!(report.bytes, 3);
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(read(&fs, "/sub/y.txt").await.unwrap().text.as_deref(), Some("y"));
    }
}
```

- [ ] **Step 3: `fsops.rs` 구현** — cortex-gui `fsops.rs`를 Tauri 의존 없이 옮긴다. 함수 시그니처를 Interfaces대로 바꾸고(`State` 제거, `&dyn FileSystem` 첫 인자), `Error::msg(..)` → `EngineError::Invalid(..)`, `epoch_ms` 는 로컬 헬퍼로. 반환 타입은 `crate::types::{Entry, FileContent, ImportReport}`. `READ_CAP = 1 << 20`, `IMPORT_CAP = 64 << 20`. `write_file`/`read_all`/`mkdir_p`/`as_text`/`kind_str`/`join` 은 그대로(`pub(crate)`). `fs_touch`는 `touch(fs, path)` 로 유지.

```rust
fn epoch_ms(t: std::time::SystemTime) -> Option<u64> {
    t.duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_millis() as u64)
}
```

`io::Error` → `EngineError` 변환은 `From` 이 있으므로 `?` 로 충분.

- [ ] **Step 4: `connectors.rs`**

```rust
//! Connectors: a `MountConfig` becomes a `FileSystem`, after one request proves it can answer.

use std::{path::Path, time::Duration};

use cortex::fs::{FileSystem, NotionConfig, NotionFs, PassthroughFs, S3Config, S3Fs};

use crate::{
    error::{EngineError, Result},
    types::{MountConfig, MountKind},
};

const PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// `/`-rooted, no trailing slash, never the root itself.
pub fn normalize_mount_path(path: &str) -> Result<String> {
    let cleaned = path.trim().trim_matches('/').trim();
    if cleaned.is_empty() {
        return Err(EngineError::Invalid("연결할 경로를 입력해 주세요 (예: /notion)".into()));
    }
    if cleaned.contains("..") {
        return Err(EngineError::Invalid("경로에 '..' 을 쓸 수 없습니다".into()));
    }
    Ok(format!("/{cleaned}"))
}

pub fn describe(config: &MountConfig) -> (MountKind, String, bool) {
    match config {
        MountConfig::Root => (MountKind::Root, "워크스페이스 파일".into(), true),
        MountConfig::Local { host_root } => (MountKind::Local, host_root.display().to_string(), true),
        MountConfig::Notion { .. } => (MountKind::Notion, "Notion workspace · 읽기 전용".into(), false),
        MountConfig::S3 { bucket, region, endpoint, .. } => (
            MountKind::S3,
            match endpoint { Some(e) => format!("s3://{bucket} · {e}"), None => format!("s3://{bucket} · {region}") },
            false,
        ),
    }
}

pub async fn build_and_probe(config: &MountConfig) -> Result<Box<dyn FileSystem>> {
    match config {
        MountConfig::Root => Err(EngineError::Invalid("루트는 커넥터로 추가할 수 없습니다".into())),
        MountConfig::Local { host_root } => {
            let meta = tokio::fs::metadata(host_root).await
                .map_err(|e| EngineError::Invalid(format!("{}: {e}", host_root.display())))?;
            if !meta.is_dir() {
                return Err(EngineError::Invalid("디렉터리를 선택해 주세요".into()));
            }
            Ok(Box::new(PassthroughFs::new(host_root.clone())))
        }
        MountConfig::Notion { api_key } => {
            let api_key = api_key.trim().to_string();
            if api_key.is_empty() {
                return Err(EngineError::Invalid("Notion 통합 토큰을 입력해 주세요".into()));
            }
            let store = NotionFs::new(&NotionConfig { api_key })?;
            tokio::time::timeout(PROBE_TIMEOUT, store.list(Path::new("")))
                .await
                .map_err(|_| EngineError::Invalid("Notion 응답이 없습니다 (15초)".into()))?
                .map_err(|e| EngineError::Invalid(format!("Notion에 연결하지 못했습니다: {e}")))?;
            Ok(Box::new(store))
        }
        MountConfig::S3 { bucket, region, access_key_id, secret_access_key, endpoint, key_prefix } => {
            let cfg = S3Config {
                bucket: bucket.trim().to_string(),
                region: region.trim().to_string(),
                access_key_id: access_key_id.trim().to_string(),
                secret_access_key: secret_access_key.clone(),
                endpoint: endpoint.clone().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()),
                key_prefix: key_prefix.clone().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()),
            };
            if cfg.bucket.is_empty() {
                return Err(EngineError::Invalid("버킷 이름을 입력해 주세요".into()));
            }
            let store = S3Fs::new(&cfg)?;
            tokio::time::timeout(PROBE_TIMEOUT, store.check_reachable())
                .await
                .map_err(|_| EngineError::Invalid("S3 응답이 없습니다 (15초)".into()))?
                .map_err(|e| EngineError::Invalid(format!("버킷에 연결하지 못했습니다: {e}")))?;
            Ok(Box::new(store))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mount_paths_are_normalized() {
        assert_eq!(normalize_mount_path(" notion/ ").unwrap(), "/notion");
        assert_eq!(normalize_mount_path("/a/b").unwrap(), "/a/b");
        assert!(normalize_mount_path("/").is_err());
        assert!(normalize_mount_path("/../x").is_err());
    }

    #[tokio::test]
    async fn local_connector_needs_a_directory() {
        let f = tempfile::NamedTempFile::new().unwrap();
        assert!(build_and_probe(&MountConfig::Local { host_root: f.path().to_path_buf() }).await.is_err());
        let d = tempfile::tempdir().unwrap();
        assert!(build_and_probe(&MountConfig::Local { host_root: d.path().to_path_buf() }).await.is_ok());
    }
}
```

`S3Fs::check_reachable` 시그니처가 `async fn check_reachable(&self) -> io::Result<()>` 인지 `../cortex/cortex/src/fs/filesystem/impl/s3.rs` 에서 확인하고 오류 타입 변환을 맞춘다.

- [ ] **Step 5: `workspace/mod.rs`(모듈 선언만 이 Task에서)**

```rust
pub mod connectors;
pub mod fsops;
pub mod shared;
pub mod manager;   // Task B7

pub use manager::*;
pub use shared::SharedFs;
```

`manager.rs` 는 다음 Task에서 만들므로, 이 Task의 컴파일을 위해 빈 파일 `src/workspace/manager.rs` 를 두고 시작한다.

- [ ] **Step 6: 확인·커밋**

```bash
cargo test -p ailoy-desktop-core workspace 2>&1 | grep -E 'test result|FAILED|panicked'
git add apps/desktop/core && git commit -m "feat(desktop-core): workspace SharedFs, fsops (ported from cortex-gui) and connectors"
```

---

### Task B7: `WorkspaceManager` — 마운트 수명, stale 정리, 커넥터 부착

**Files:**
- Create: `src/workspace/manager.rs`, `src/workspace/mount.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct WorkspaceMount(PathBuf);  impl cortex::fs::Mount   // mountpoint() = 경로
  pub struct WorkspaceManager { .. }
  impl WorkspaceManager {
    pub async fn start(files_root: PathBuf, mountpoint: PathBuf, mount: bool) -> WorkspaceManager;   // 실패 시에도 Degraded 로 시작
    pub fn info(&self) -> WorkspaceInfo;
    pub fn console_mount(&self) -> WorkspaceMount;   // Mounted → mountpoint, Degraded → files_root
    pub fn fs(&self) -> SharedFs;
    pub async fn attach(&self, info: MountInfo, store: Box<dyn FileSystem>) -> Result<()>;   // WorkFs::mount + 목록 갱신 (DB는 Engine 몫)
    pub async fn detach(&self, path: &str) -> Result<()>;
    pub async fn mounts(&self) -> Vec<MountInfo>;
    pub async fn set_mount_status(&self, path: &str, status: MountStatus);
    pub async fn shutdown(&self);   // 마운트 drop (spawn_blocking)
  }
  pub fn is_mounted(path: &Path) -> bool;          // `mount` 출력에 " on <path> (" 포함
  pub fn force_unmount(path: &Path) -> std::io::Result<()>;   // umount → diskutil unmount force
  ```

- [ ] **Step 1: `mount.rs`**

```rust
use std::path::{Path, PathBuf};

use cortex::fs::Mount;

/// Where the console stands: the FUSE-T mount point when the workspace is mounted, the
/// plain files directory when it is not. `Mount` asks for nothing but the path — a
/// directory answers a console the same way a mount point does.
pub struct WorkspaceMount(pub PathBuf);

impl Mount for WorkspaceMount {
    fn mountpoint(&self) -> &Path {
        &self.0
    }
}
```

- [ ] **Step 2: 테스트** (`manager.rs` 하단; FUSE 없이 `mount: false` 경로만 단위 테스트)

```rust
#[cfg(test)]
mod tests {
    use cortex::fs::InMemFs;

    use super::*;
    use crate::{types::{MountKind, MountStatus}, workspace::fsops};

    #[tokio::test]
    async fn degraded_manager_serves_files_root_and_attaches_stores() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        let mp = dir.path().join("workspace");
        let ws = WorkspaceManager::start(files.clone(), mp, false).await;
        assert!(matches!(ws.info().status, WorkspaceStatus::Degraded { .. }));
        assert_eq!(ws.console_mount().0, files);
        assert_eq!(ws.mounts().await.len(), 1, "the root row");

        // Root is the real directory: a write shows up on disk.
        fsops::write(&ws.fs(), "/hello.txt", "hi").await.unwrap();
        assert_eq!(std::fs::read_to_string(files.join("hello.txt")).unwrap(), "hi");

        let info = MountInfo { id: "m".into(), path: "/mem".into(), kind: MountKind::Local, label: "mem".into(), detail: "".into(), writable: true, status: MountStatus::Ok };
        ws.attach(info.clone(), Box::new(InMemFs::new())).await.unwrap();
        assert!(ws.attach(info, Box::new(InMemFs::new())).await.is_err(), "duplicate path");
        fsops::write(&ws.fs(), "/mem/a.txt", "x").await.unwrap();
        assert_eq!(fsops::list(&ws.fs(), "/mem").await.unwrap().len(), 1);
        ws.detach("/mem").await.unwrap();
        assert_eq!(ws.mounts().await.len(), 1);
        ws.shutdown().await;
    }
}
```

- [ ] **Step 3: 구현** — `manager.rs`

```rust
//! The workspace: one `WorkFs`, mounted for the life of the engine.

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex as StdMutex},
};

use cortex::fs::{FileSystem, FuseTMount, PassthroughFs, WorkFs};
use tokio::sync::RwLock;

use crate::{
    error::{EngineError, Result},
    types::{MountInfo, MountKind, MountStatus, WorkspaceInfo, WorkspaceStatus},
    workspace::{mount::WorkspaceMount, shared::SharedFs},
};

pub struct WorkspaceManager {
    fs: Arc<RwLock<WorkFs>>,
    files_root: PathBuf,
    mountpoint: PathBuf,
    /// Held for the life of the engine; dropping it unmounts. Behind a std mutex because
    /// `FuseTMount` is dropped on a blocking thread at shutdown.
    mount: StdMutex<Option<FuseTMount>>,
    status: RwLock<WorkspaceStatus>,
    mounts: RwLock<Vec<MountInfo>>,
}

impl WorkspaceManager {
    pub async fn start(files_root: PathBuf, mountpoint: PathBuf, mount: bool) -> WorkspaceManager {
        let status = match prepare_dirs(&files_root, &mountpoint) {
            Ok(()) => WorkspaceStatus::Mounted,
            Err(e) => WorkspaceStatus::Degraded { reason: e.to_string() },
        };
        let fs = Arc::new(RwLock::new(
            WorkFs::new()
                .try_with_mount("", PassthroughFs::new(files_root.clone()))
                .expect("an empty path is a valid mount key"),
        ));
        let root = MountInfo {
            id: "root".into(),
            path: "/".into(),
            kind: MountKind::Root,
            label: "Workspace".into(),
            detail: files_root.display().to_string(),
            writable: true,
            status: MountStatus::Ok,
        };
        let manager = WorkspaceManager {
            fs,
            files_root,
            mountpoint,
            mount: StdMutex::new(None),
            status: RwLock::new(if mount { status } else { WorkspaceStatus::Degraded { reason: "mounting disabled".into() } }),
            mounts: RwLock::new(vec![root]),
        };
        if mount && matches!(*manager.status.read().await, WorkspaceStatus::Mounted) {
            manager.mount_fuse().await;
        }
        manager
    }

    async fn mount_fuse(&self) {
        let shared = SharedFs(self.fs.clone());
        let at = self.mountpoint.clone();
        // Mounting blocks on a helper handshake, so it runs off the runtime.
        let result = tokio::task::spawn_blocking(move || FuseTMount::try_new(shared, &at)).await;
        match result {
            Ok(Ok(m)) => {
                *self.mount.lock().expect("mount mutex") = Some(m);
                tracing::info!("workspace mounted at {}", self.mountpoint.display());
            }
            Ok(Err(e)) => {
                tracing::warn!("workspace mount failed: {e}");
                *self.status.write().await = WorkspaceStatus::Degraded { reason: format!("FUSE-T mount failed: {e}") };
            }
            Err(e) => {
                *self.status.write().await = WorkspaceStatus::Degraded { reason: format!("mount task panicked: {e}") };
            }
        }
    }

    pub fn info(&self) -> WorkspaceInfo {
        WorkspaceInfo {
            mountpoint: self.mountpoint.clone(),
            files_root: self.files_root.clone(),
            status: self.status.try_read().map(|s| s.clone()).unwrap_or(WorkspaceStatus::Degraded { reason: "busy".into() }),
        }
    }

    pub fn console_mount(&self) -> WorkspaceMount {
        match self.info().status {
            WorkspaceStatus::Mounted => WorkspaceMount(self.mountpoint.clone()),
            WorkspaceStatus::Degraded { .. } => WorkspaceMount(self.files_root.clone()),
        }
    }

    pub fn fs(&self) -> SharedFs {
        SharedFs(self.fs.clone())
    }

    /// Record then mount, in that order: the list is what refuses a duplicate, and the tree
    /// has no undo that leaves the first store where it was.
    pub async fn attach(&self, info: MountInfo, store: Box<dyn FileSystem>) -> Result<()> {
        {
            let mut mounts = self.mounts.write().await;
            if mounts.iter().any(|m| m.path == info.path) {
                return Err(EngineError::Invalid(format!("{} 에는 이미 다른 저장소가 연결되어 있습니다", info.path)));
            }
            mounts.push(info.clone());
            mounts.sort_by(|a, b| a.path.cmp(&b.path));
        }
        if let Err(e) = self.fs.write().await.mount(Path::new(&info.path), store) {
            self.mounts.write().await.retain(|m| m.path != info.path);
            return Err(EngineError::Workspace(e.to_string()));
        }
        Ok(())
    }

    pub async fn detach(&self, path: &str) -> Result<()> {
        if path == "/" || path.is_empty() {
            return Err(EngineError::Invalid("루트는 분리할 수 없습니다".into()));
        }
        let _ = self.fs.write().await.unmount(Path::new(path));
        self.mounts.write().await.retain(|m| m.path != path);
        Ok(())
    }

    pub async fn mounts(&self) -> Vec<MountInfo> {
        self.mounts.read().await.clone()
    }

    /// Remember a connector that failed to restore, so the sidebar can show it with its error.
    pub async fn remember_failed(&self, info: MountInfo) {
        let mut mounts = self.mounts.write().await;
        mounts.retain(|m| m.path != info.path);
        mounts.push(info);
        mounts.sort_by(|a, b| a.path.cmp(&b.path));
    }

    pub async fn shutdown(&self) {
        let taken = self.mount.lock().expect("mount mutex").take();
        if let Some(m) = taken {
            let _ = tokio::task::spawn_blocking(move || drop(m)).await;
        }
    }
}

/// Create the two directories, and make sure the mount point is free: a mount left over
/// from a crash is unmounted, and anything else inside it is a refusal, not a deletion.
fn prepare_dirs(files_root: &Path, mountpoint: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(files_root)?;
    std::fs::create_dir_all(mountpoint)?;
    if is_mounted(mountpoint) {
        tracing::warn!("stale mount at {}, unmounting", mountpoint.display());
        force_unmount(mountpoint)?;
    }
    if std::fs::read_dir(mountpoint)?.next().is_some() {
        return Err(std::io::Error::other(format!("mount point {} is not empty", mountpoint.display())));
    }
    Ok(())
}

pub fn is_mounted(path: &Path) -> bool {
    let Ok(out) = Command::new("mount").output() else { return false };
    let needle = format!(" on {} (", path.display());
    String::from_utf8_lossy(&out.stdout).contains(&needle)
}

pub fn force_unmount(path: &Path) -> std::io::Result<()> {
    let _ = Command::new("umount").arg(path).status();
    if is_mounted(path) {
        let _ = Command::new("diskutil").args(["unmount", "force"]).arg(path).status();
    }
    if is_mounted(path) {
        return Err(std::io::Error::other(format!("could not unmount {}", path.display())));
    }
    Ok(())
}
```

`workspace/mod.rs` 에 `pub mod mount; pub use mount::WorkspaceMount;` 추가.

- [ ] **Step 4: FUSE-T 통합 테스트(`#[ignore]`)** — `apps/desktop/core/tests/live_workspace.rs`

```rust
//! Needs FUSE-T installed. Run: `cargo test -p ailoy-desktop-core --test live_workspace -- --ignored`

use ailoy_desktop_core::workspace::{WorkspaceManager, fsops, is_mounted};

#[tokio::test]
#[ignore]
async fn a_mounted_workspace_is_visible_to_the_kernel() {
    let dir = tempfile::tempdir().unwrap();
    let files = dir.path().join("files");
    let mp = dir.path().join("workspace");
    let ws = WorkspaceManager::start(files.clone(), mp.clone(), true).await;
    assert!(matches!(ws.info().status, ailoy_desktop_core::WorkspaceStatus::Mounted), "{:?}", ws.info());
    assert!(is_mounted(&mp));

    fsops::write(&ws.fs(), "/via-workfs.txt", "kernel sees me").await.unwrap();
    assert_eq!(std::fs::read_to_string(mp.join("via-workfs.txt")).unwrap(), "kernel sees me");
    assert_eq!(std::fs::read_to_string(files.join("via-workfs.txt")).unwrap(), "kernel sees me");

    ws.shutdown().await;
    assert!(!is_mounted(&mp));
}
```

- [ ] **Step 5: 확인·커밋**

```bash
cargo test -p ailoy-desktop-core workspace 2>&1 | grep -E 'test result|FAILED|panicked'
cargo test -p ailoy-desktop-core --test live_workspace -- --ignored 2>&1 | grep -E 'test result|FAILED|panicked'   # FUSE-T 있는 머신
git add apps/desktop/core && git commit -m "feat(desktop-core): WorkspaceManager with app-lifetime FUSE-T mount and stale cleanup"
```

---

### Task B8: 콘솔 팩토리

**Files:**
- Create: `src/console.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces:
  ```rust
  pub const CONSOLE_BIN_NAME: &str = "cortex-local-console";
  pub fn resolve_console_bin(explicit: Option<&Path>) -> Result<PathBuf>;   // explicit → $AILOY_CORTEX_BIN_DIR/<name> → cwd 상위 탐색 `cortex/target/{debug,release}/<name>` → PATH(which)
  pub struct ConsoleFactory { bin: PathBuf }
  impl ConsoleFactory { pub fn new(bin: PathBuf) -> Self; pub fn bin(&self) -> &Path; pub async fn spawn(&self, mount: WorkspaceMount) -> Result<cortex::console::Console> }
  ```

- [ ] **Step 1: 테스트**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_path_wins_when_it_exists() {
        let f = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(resolve_console_bin(Some(f.path())).unwrap(), f.path());
        assert!(matches!(resolve_console_bin(Some(Path::new("/nonexistent/bin"))), Err(EngineError::ConsoleUnavailable(_))));
    }

    #[test]
    fn env_dir_is_honoured() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join(CONSOLE_BIN_NAME);
        std::fs::write(&bin, b"").unwrap();
        // SAFETY: tests in this module are the only readers of this variable and run serially
        // under `--test-threads=1` in CI; locally a race only makes the assertion fail.
        unsafe { std::env::set_var("AILOY_CORTEX_BIN_DIR", dir.path()) };
        assert_eq!(resolve_console_bin(None).unwrap(), bin);
        unsafe { std::env::remove_var("AILOY_CORTEX_BIN_DIR") };
    }
}
```

- [ ] **Step 2: 구현**

```rust
//! Starting a `cortex-local-console` for one run.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

use cortex::console::{Console, stdio::StdioClient};
use tokio::process::Command;

use crate::{
    error::{EngineError, Result},
    workspace::WorkspaceMount,
};

pub const CONSOLE_BIN_NAME: &str = "cortex-local-console";

/// Where the console server binary is. In a bundle it sits beside the app binary (the Tauri
/// layer passes that path explicitly); in development it is the sibling checkout's build.
pub fn resolve_console_bin(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return if p.is_file() { Ok(p.to_path_buf()) } else { Err(EngineError::ConsoleUnavailable(format!("{} does not exist", p.display()))) };
    }
    if let Ok(dir) = std::env::var("AILOY_CORTEX_BIN_DIR") {
        let p = PathBuf::from(dir).join(CONSOLE_BIN_NAME);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            for profile in ["debug", "release"] {
                let p = ancestor.join("cortex").join("target").join(profile).join(CONSOLE_BIN_NAME);
                if p.is_file() {
                    return Ok(p);
                }
            }
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            let p = dir.join(CONSOLE_BIN_NAME);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    Err(EngineError::ConsoleUnavailable(format!(
        "{CONSOLE_BIN_NAME} not found; set AILOY_CORTEX_BIN_DIR or build it with `cargo build -p cortex-local-console` in ../cortex"
    )))
}

pub struct ConsoleFactory {
    bin: PathBuf,
}

impl ConsoleFactory {
    pub fn new(bin: PathBuf) -> Self {
        Self { bin }
    }

    pub fn bin(&self) -> &Path {
        &self.bin
    }

    /// One console standing in `mount`. Its `PATH` starts with the binary's own directory so
    /// sidecars beside it (`mem`, later) are commands the agent can name.
    pub async fn spawn(&self, mount: WorkspaceMount) -> Result<Console> {
        let mut cmd = Command::new(&self.bin);
        cmd.stderr(Stdio::inherit());
        if let Some(dir) = self.bin.parent() {
            let mut path = dir.as_os_str().to_os_string();
            if let Ok(existing) = std::env::var("PATH") {
                path.push(":");
                path.push(existing);
            }
            cmd.env("PATH", path);
        }
        let client = StdioClient::new(cmd).map_err(|e| EngineError::ConsoleUnavailable(e.to_string()))?;
        Console::builder()
            .client(client)
            .mount(mount)
            .build()
            .await
            .map_err(|e| EngineError::ConsoleUnavailable(e.to_string()))
    }
}
```

`src/lib.rs`에 `pub mod console;`.

- [ ] **Step 3: 라이브 테스트(`#[ignore]`)** — `tests/live_console.rs`

```rust
//! Needs a built `cortex-local-console` (AILOY_CORTEX_BIN_DIR or ../cortex/target/debug).
//! Run: `cargo test -p ailoy-desktop-core --test live_console -- --ignored`

use ailoy_desktop_core::{console::{ConsoleFactory, resolve_console_bin}, workspace::WorkspaceMount};

#[tokio::test]
#[ignore]
async fn a_console_stands_in_the_mount_it_was_given() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), b"here").unwrap();
    let factory = ConsoleFactory::new(resolve_console_bin(None).unwrap());
    let mut console = factory.spawn(WorkspaceMount(dir.path().to_path_buf())).await.unwrap();
    let out = console.exec(["cat", "marker.txt"], Some(5_000)).await.unwrap();
    assert_eq!(out.stdout, b"here");
}
```

- [ ] **Step 4: 확인·커밋**

```bash
cargo test -p ailoy-desktop-core console 2>&1 | grep -E 'test result|FAILED|panicked'
AILOY_CORTEX_BIN_DIR=../cortex/target/debug cargo test -p ailoy-desktop-core --test live_console -- --ignored 2>&1 | grep -E 'test result|FAILED'
git add apps/desktop/core && git commit -m "feat(desktop-core): console factory with binary discovery"
```

---

### Task B9: 델타 재조립기와 이벤트, 사용량 계산

**Files:**
- Create: `src/assembler.rs`, `src/events.rs`, `src/usage.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces:
  ```rust
  pub enum AssembledItem { Text(String), Thinking(String), Completed(Box<MessageOutput>) }
  pub struct MessageAssembler; impl { pub fn new() -> Self; pub fn push(&mut self, delta: MessageDeltaOutput) -> std::result::Result<Vec<AssembledItem>, String>; pub fn finish(&mut self) -> std::result::Result<Option<MessageOutput>, String>; pub fn partial_text(&self) -> String }
  #[serde(tag = "type", rename_all = "snake_case")]
  pub enum RunEvent {
    Started { run_id: String },
    TextDelta { text: String },
    ThinkingDelta { text: String },
    ToolCallStarted { id: String, name: String, arguments: Value },
    Message { seq: i64, depth: u8, source_agent: Option<String>, message: Message, usage: Option<TokenUsage> },
    Usage { usage: Option<TokenUsage>, rate_limit: Option<RateLimitInfo>, context_used: Option<u64>, context_limit: Option<u64> },
    AwaitingApproval { id: String, name: String, arguments: Value },
    Done, Cancelled,
    Error { kind: String, message: String },
  }
  pub fn context_used(u: &TokenUsage) -> u64;                                  // input + cache_read + cache_creation
  pub fn estimate_cost_usd(usages: &[TokenUsage], cost: &ModelCost) -> Option<f64>;
  pub fn session_usage(usages: &[TokenUsage], model: Option<&CatalogModel>) -> SessionUsage;
  ```

- [ ] **Step 1: `assembler.rs`** — agent-k `agent_stream.rs`를 이식하되 (a) `AgentStreamItem::Delta` → `AssembledItem::Text`, (b) `delta.delta.thinking` 조각을 `AssembledItem::Thinking` 으로 추가 방출(같은 최상위·비-Tool 조건), (c) `partial_text()` — 현재 `acc` 의 텍스트 파트 연결. agent-k 테스트 7개를 모두 옮기고(`Delta`→`Text`), thinking 테스트를 추가한다:

```rust
    #[test]
    fn thinking_fragments_are_emitted_separately() {
        let mut a = MessageAssembler::new();
        let mut d = delta(Some(Role::Assistant), "", false);
        d.delta.thinking = Some("hmm".into());
        let items = a.push(d).unwrap();
        assert!(matches!(items.as_slice(), [AssembledItem::Thinking(t)] if t == "hmm"));
        assert_eq!(a.partial_text(), "");
        let items = a.push(delta(None, "answer", true)).unwrap();
        assert!(matches!(&items[0], AssembledItem::Text(t) if t == "answer"));
        assert!(matches!(&items[1], AssembledItem::Completed(_)));
    }
```

`push` 의 1단계에 텍스트 조각 수집 뒤:

```rust
            if let Some(th) = delta.delta.thinking.as_deref().filter(|t| !t.is_empty()) {
                items.push(AssembledItem::Thinking(th.to_string()));
            }
```

`partial_text`:

```rust
    pub fn partial_text(&self) -> String {
        self.acc.delta.contents.iter().filter_map(|p| match p { PartDelta::Text { text } => Some(text.as_str()), _ => None }).collect()
    }
```

- [ ] **Step 2: `events.rs`** — Interfaces 그대로 `#[derive(Clone, Debug, Serialize, Deserialize)]`, `#[serde(tag = "type", rename_all = "snake_case")]`. `Value` 는 `ailoy::datatype::Value`.

- [ ] **Step 3: `usage.rs` 테스트와 구현**

```rust
//! Token arithmetic the UI shows: context in use, totals, cost.

use ailoy::message::TokenUsage;

use crate::{catalog::CatalogModel, types::{ModelCost, SessionUsage}};

/// The input the *next* call will carry, approximated by the last call's whole input:
/// Anthropic's `input_tokens` excludes what was read from cache, so all three are summed.
pub fn context_used(u: &TokenUsage) -> u64 {
    u.input_tokens + u.cache_read_input_tokens.unwrap_or(0) + u.cache_creation_input_tokens.unwrap_or(0)
}

pub fn estimate_cost_usd(usages: &[TokenUsage], cost: &ModelCost) -> Option<f64> {
    let (input, output) = (cost.input?, cost.output?);
    let mut total = 0.0;
    for u in usages {
        total += u.input_tokens as f64 * input;
        total += u.output_tokens as f64 * output;
        total += u.cache_read_input_tokens.unwrap_or(0) as f64 * cost.cache_read.unwrap_or(input);
        total += u.cache_creation_input_tokens.unwrap_or(0) as f64 * cost.cache_write.unwrap_or(input);
    }
    Some(total / 1_000_000.0)
}

pub fn session_usage(usages: &[TokenUsage], model: Option<&CatalogModel>) -> SessionUsage {
    SessionUsage {
        input_tokens: usages.iter().map(|u| u.input_tokens).sum(),
        output_tokens: usages.iter().map(|u| u.output_tokens).sum(),
        cache_read_tokens: usages.iter().map(|u| u.cache_read_input_tokens.unwrap_or(0)).sum(),
        cache_write_tokens: usages.iter().map(|u| u.cache_creation_input_tokens.unwrap_or(0)).sum(),
        estimated_cost_usd: model.and_then(|m| m.cost.as_ref()).and_then(|c| estimate_cost_usd(usages, c)),
        context_used: usages.last().map(context_used),
        context_limit: model.and_then(|m| m.context),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(i: u64, o: u64, cr: Option<u64>, cw: Option<u64>) -> TokenUsage {
        TokenUsage { input_tokens: i, output_tokens: o, cache_read_input_tokens: cr, cache_creation_input_tokens: cw }
    }

    #[test]
    fn context_used_sums_cached_input() {
        assert_eq!(context_used(&u(50, 10, Some(200_000), Some(1_000))), 201_050);
    }

    #[test]
    fn cost_uses_cache_prices_when_present() {
        let cost = ModelCost { input: Some(5.0), output: Some(25.0), cache_read: Some(0.5), cache_write: Some(6.25) };
        let usd = estimate_cost_usd(&[u(1_000_000, 1_000_000, Some(1_000_000), Some(1_000_000))], &cost).unwrap();
        assert!((usd - (5.0 + 25.0 + 0.5 + 6.25)).abs() < 1e-9);
        assert!(estimate_cost_usd(&[u(1, 1, None, None)], &ModelCost { input: None, output: Some(1.0), cache_read: None, cache_write: None }).is_none());
    }

    #[test]
    fn session_usage_reports_last_context_and_limit() {
        let model = CatalogModel { id: "m".into(), name: "m".into(), reasoning: true, tool_call: true, context: Some(1_000_000), output: None, cost: None };
        let s = session_usage(&[u(10, 5, None, None), u(30, 5, Some(70), None)], Some(&model));
        assert_eq!(s.input_tokens, 40);
        assert_eq!(s.cache_read_tokens, 70);
        assert_eq!(s.context_used, Some(100));
        assert_eq!(s.context_limit, Some(1_000_000));
        assert!(s.estimated_cost_usd.is_none());
    }
}
```

`src/lib.rs`에 `pub mod assembler; pub mod events; pub mod usage; pub use events::RunEvent;`.

- [ ] **Step 4: 확인·커밋**

```bash
cargo test -p ailoy-desktop-core assembler usage 2>&1 | grep -E 'test result|FAILED|panicked'
git add apps/desktop/core && git commit -m "feat(desktop-core): message assembler (ported from agent-k), RunEvent, usage math"
```

---

### Task B10: `RunManager` — actor, 이벤트, 취소

**Files:**
- Create: `src/run.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `Store`, `ConsoleFactory`, `WorkspaceManager`, `Catalog`, `providers::read_settings`, `prompt::build`, `MessageAssembler`, ailoy `AgentBuilder`, `RunControl`, `AgentError`
- Produces:
  ```rust
  pub struct RunDeps { pub store: Arc<Store>, pub console: Arc<ConsoleFactory>, pub workspace: Arc<WorkspaceManager>, pub catalog: Arc<Catalog> }
  pub struct RunHandle { pub run_id: String, pub events: broadcast::Receiver<RunEvent> }
  pub struct RunManager { .. }
  impl RunManager {
    pub fn new(deps: RunDeps) -> Self;
    pub async fn start(&self, session_id: &str, parts: Vec<Part>) -> Result<RunHandle>;
    pub async fn attach(&self, session_id: &str) -> Option<(RunHandle, String /*partial text*/)>;
    pub async fn cancel(&self, session_id: &str) -> bool;
    pub async fn is_running(&self, session_id: &str) -> bool;
    pub async fn cancel_all(&self);
  }
  ```

- [ ] **Step 1: 구현**

```rust
//! One actor per active run: assemble the agent, drive its stream, persist and broadcast.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex as StdMutex},
};

use ailoy::{
    agent::{AgentBuilder, AgentError, RunControl},
    message::{Message, Part, Role},
};
use futures::StreamExt as _;
use tokio::sync::{Mutex, broadcast};
use tokio_util::sync::CancellationToken;

use crate::{
    assembler::{AssembledItem, MessageAssembler},
    catalog::Catalog,
    console::ConsoleFactory,
    error::{EngineError, Result},
    events::RunEvent,
    prompt,
    providers,
    store::{NewMessage, Store},
    usage,
    workspace::WorkspaceManager,
};

pub struct RunDeps {
    pub store: Arc<Store>,
    pub console: Arc<ConsoleFactory>,
    pub workspace: Arc<WorkspaceManager>,
    pub catalog: Arc<Catalog>,
}

pub struct RunHandle {
    pub run_id: String,
    pub events: broadcast::Receiver<RunEvent>,
}

struct ActiveRun {
    run_id: String,
    cancel: CancellationToken,
    events: broadcast::Sender<RunEvent>,
    partial: Arc<StdMutex<String>>,
}

pub struct RunManager {
    deps: RunDeps,
    runs: Mutex<HashMap<String, ActiveRun>>,
}

const EVENT_BUFFER: usize = 4096;

impl RunManager {
    pub fn new(deps: RunDeps) -> Self {
        Self { deps, runs: Mutex::new(HashMap::new()) }
    }

    pub async fn is_running(&self, session_id: &str) -> bool {
        self.runs.lock().await.contains_key(session_id)
    }

    pub async fn attach(&self, session_id: &str) -> Option<(RunHandle, String)> {
        let runs = self.runs.lock().await;
        let run = runs.get(session_id)?;
        let partial = run.partial.lock().expect("partial mutex").clone();
        Some((RunHandle { run_id: run.run_id.clone(), events: run.events.subscribe() }, partial))
    }

    pub async fn cancel(&self, session_id: &str) -> bool {
        match self.runs.lock().await.get(session_id) {
            Some(run) => { run.cancel.cancel(); true }
            None => false,
        }
    }

    pub async fn cancel_all(&self) {
        for run in self.runs.lock().await.values() {
            run.cancel.cancel();
        }
    }

    pub async fn start(&self, session_id: &str, parts: Vec<Part>) -> Result<RunHandle> {
        let session = self.deps.store.session_get(session_id)?;
        let mut runs = self.runs.lock().await;
        if runs.contains_key(session_id) {
            return Err(EngineError::AlreadyRunning);
        }

        // The user's message is persisted before anything can fail, so a refresh shows it.
        let user_msg = Message::new(Role::User).with_contents(parts);
        let seq = self.deps.store.message_append(session_id, NewMessage { depth: 0, source_agent: None, message: &user_msg, usage: None })?;
        self.deps.store.session_touch(session_id)?;

        let run_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = broadcast::channel(EVENT_BUFFER);
        let cancel = CancellationToken::new();
        let partial = Arc::new(StdMutex::new(String::new()));
        runs.insert(session_id.to_string(), ActiveRun { run_id: run_id.clone(), cancel: cancel.clone(), events: tx.clone(), partial: partial.clone() });
        drop(runs);

        let _ = tx.send(RunEvent::Started { run_id: run_id.clone() });
        let _ = tx.send(RunEvent::Message { seq, depth: 0, source_agent: None, message: user_msg.clone(), usage: None });

        let deps = RunDeps { store: self.deps.store.clone(), console: self.deps.console.clone(), workspace: self.deps.workspace.clone(), catalog: self.deps.catalog.clone() };
        let sid = session_id.to_string();
        let model = session.model.clone();
        let runs_ref: *const Mutex<HashMap<String, ActiveRun>> = &self.runs;
        // The map outlives every run because `RunManager` lives in the `Engine`, which is
        // `Arc`ed for the app's lifetime; `remove_run` below goes through a clone-safe path.
        let _ = runs_ref;
        let finished = self.finisher(sid.clone());

        tokio::spawn(async move {
            let outcome = drive(deps, &sid, &model, user_msg, tx.clone(), cancel, partial).await;
            match outcome {
                Ok(()) => { let _ = tx.send(RunEvent::Done); }
                Err(RunEnd::Cancelled) => { let _ = tx.send(RunEvent::Cancelled); }
                Err(RunEnd::Failed { kind, message }) => { let _ = tx.send(RunEvent::Error { kind, message }); }
            }
            finished.await;
        });

        Ok(RunHandle { run_id, events: rx })
    }

    /// A future that removes `session_id` from the active map when awaited. Built while
    /// `self` is borrowed; awaited from the detached task through an `Arc` handle.
    fn finisher(&self, session_id: String) -> impl std::future::Future<Output = ()> + Send + 'static {
        let runs = self.runs_handle();
        async move {
            runs.lock().await.remove(&session_id);
        }
    }

    fn runs_handle(&self) -> Arc<Mutex<HashMap<String, ActiveRun>>> {
        // `runs` is stored behind an Arc so detached tasks can clean up after themselves.
        self.runs_arc.clone()
    }
}
```

위 스케치의 `runs_ref`/`runs_arc` 혼란을 피하기 위해 **필드를 `runs: Arc<Mutex<HashMap<String, ActiveRun>>>` 로 정의**하고, `runs_handle()` 은 `self.runs.clone()`, `finisher` 는 그 clone 을 캡처한다. `runs_ref` 관련 두 줄은 삭제한다. 즉 최종 구조체:

```rust
pub struct RunManager {
    deps: RunDeps,
    runs: Arc<Mutex<HashMap<String, ActiveRun>>>,
}
```

이어서 actor 본체:

```rust
enum RunEnd {
    Cancelled,
    Failed { kind: String, message: String },
}

async fn drive(
    deps: RunDeps,
    session_id: &str,
    model: &str,
    user_msg: Message,
    tx: broadcast::Sender<RunEvent>,
    cancel: CancellationToken,
    partial: Arc<StdMutex<String>>,
) -> std::result::Result<(), RunEnd> {
    let fail = |kind: &str, e: String| RunEnd::Failed { kind: kind.into(), message: e };

    let settings = providers::read_settings(&deps.store).map_err(|e| fail("storage", e.to_string()))?;
    let history = deps.store.message_history(session_id).map_err(|e| fail("storage", e.to_string()))?;
    // The user message was appended by `start`; the agent pushes its own copy of the query,
    // so the replayed history stops before it.
    let history: Vec<Message> = history.into_iter().take_while(|m| !(m.role == Role::User && m.contents == user_msg.contents && m.id == user_msg.id)).collect();

    let mounts = deps.workspace.mounts().await;
    let ws_mount = deps.workspace.console_mount();
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let preamble = prompt::build(&prompt::PromptInput {
        workfs_path: &ws_mount.0,
        mounts: &mounts,
        today: &today,
        os: std::env::consts::OS,
        extra: deps.store.setting_get("extra_instruction").ok().flatten().as_deref(),
    });

    let console = deps.console.spawn(ws_mount).await.map_err(|e| fail("console_unavailable", e.to_string()))?;

    let mut agent = AgentBuilder::new(model)
        .instruction(preamble)
        .system_tools()
        .web_search_tool(vec![])
        .web_fetch_tool()
        .max_tokens(settings.max_tokens)
        .history(history)
        .console(console)
        .build()
        .map_err(|e| fail("model", e.to_string()))?;

    let ctl = RunControl { cancel, max_turns: Some(settings.max_turns), ..Default::default() };
    let catalog_model = deps.catalog.lookup(model);
    let context_limit = catalog_model.as_ref().and_then(|m| m.context);

    let mut assembler = MessageAssembler::new();
    let mut end: Option<RunEnd> = None;
    {
        let mut stream = agent.run_stream_controlled(user_msg, ctl);
        while let Some(item) = stream.next().await {
            let delta = match item {
                Ok(d) => d,
                Err(AgentError::Cancelled) => { end = Some(RunEnd::Cancelled); break; }
                Err(AgentError::MaxTurns { turns }) => { end = Some(fail("max_turns", format!("turn limit reached after {turns} model calls"))); break; }
                Err(AgentError::Model(m)) => { end = Some(fail("model", m.to_string())); break; }
                Err(AgentError::Console(e)) => { end = Some(fail("console_unavailable", e.to_string())); break; }
                Err(e) => { end = Some(fail("tool", e.to_string())); break; }
            };
            if delta.usage.is_some() || delta.rate_limit.is_some() {
                let _ = tx.send(RunEvent::Usage {
                    context_used: delta.usage.as_ref().map(usage::context_used),
                    context_limit,
                    usage: delta.usage.clone(),
                    rate_limit: delta.rate_limit.clone(),
                });
            }
            let items = match assembler.push(delta) {
                Ok(items) => items,
                Err(e) => { end = Some(fail("stream", e)); break; }
            };
            for item in items {
                match item {
                    AssembledItem::Text(t) => {
                        partial.lock().expect("partial mutex").push_str(&t);
                        let _ = tx.send(RunEvent::TextDelta { text: t });
                    }
                    AssembledItem::Thinking(t) => { let _ = tx.send(RunEvent::ThinkingDelta { text: t }); }
                    AssembledItem::Completed(out) => {
                        partial.lock().expect("partial mutex").clear();
                        if out.message.role == Role::Assistant && let Some(calls) = &out.message.tool_calls {
                            for c in calls {
                                if let Some((id, name, args)) = c.as_function() {
                                    let _ = tx.send(RunEvent::ToolCallStarted { id: id.into(), name: name.into(), arguments: args.clone() });
                                }
                            }
                        }
                        persist(&deps.store, session_id, &tx, *out);
                    }
                }
            }
        }
    }
    // A stream that ended mid-message (cancel during the model phase) still has text worth
    // keeping — the agent committed it to its history, so mirror that here.
    if let Ok(Some(out)) = assembler.finish() {
        persist(&deps.store, session_id, &tx, out);
    }
    let _ = deps.store.session_touch(session_id);
    match end {
        None => Ok(()),
        Some(e) => Err(e),
    }
}

fn persist(store: &Store, session_id: &str, tx: &broadcast::Sender<RunEvent>, out: ailoy::message::MessageOutput) {
    let depth = out.depth.unwrap_or(0);
    match store.message_append(session_id, NewMessage { depth, source_agent: out.source_agent.as_deref(), message: &out.message, usage: out.usage.as_ref() }) {
        Ok(seq) => {
            let _ = tx.send(RunEvent::Message { seq, depth, source_agent: out.source_agent, message: out.message, usage: out.usage });
        }
        Err(e) => tracing::error!("persisting message for {session_id}: {e}"),
    }
}
```

주의: `history` 필터는 방금 추가한 user 메시지를 제외하기 위한 것이다. 더 단순하고 안전한 방법은 `message_history` 결과의 **마지막 원소**(방금 넣은 user 메시지)를 `pop()` 하는 것이다 — `start` 가 append 직후 `drive` 를 호출하므로 마지막이 그 메시지임이 보장된다. `take_while` 대신 `history.pop();` 으로 구현한다.

`src/lib.rs`에 `pub mod run;`.

- [ ] **Step 2: 오프라인 테스트** — `run.rs` `mod tests`. Plan A의 가짜 SSE 서버는 ailoy 크레이트 내부(`cfg(test)`)여서 여기서 못 쓴다. 같은 최소 버전을 이 테스트 모듈에 둔다(텍스트 응답 하나만 서빙):

```rust
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ailoy::{agent::{AgentProvider, get_agent_providers_mut}, lang_model::{LangModelProvider, get_lm_providers_mut}, message::Part, tool::{ToolProvider, get_tool_providers_mut}};
    use axum::{Router, body::Body, response::Response, routing::post};

    use super::*;
    use crate::{catalog::{Catalog, CatalogData}, console::ConsoleFactory, store::Store, workspace::WorkspaceManager};

    async fn fake_model_server(sse: &'static str) -> std::net::SocketAddr {
        let app = Router::new().route("/", post(move || async move {
            Response::builder().status(200).header("content-type", "text/event-stream").body(Body::from(sse)).unwrap()
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        addr
    }

    /// The engine builds agents against the `"default"` bundle; point its lang-model
    /// registry at the fake server under a `fake/*` pattern.
    fn point_default_at(addr: std::net::SocketAddr) {
        let mut lmps = get_lm_providers_mut();
        let def = lmps.entry("default".into()).or_insert_with(LangModelProvider::new);
        def.insert("fake/*".into(), LangModelProvider::chat_completion(&format!("http://{addr}/"), None).unwrap());
        drop(lmps);
        get_tool_providers_mut().entry("default".into()).or_insert_with(ToolProvider::new);
        get_agent_providers_mut().entry("default".into()).or_insert_with(|| AgentProvider::new("default", "default"));
    }

    async fn deps(dir: &std::path::Path) -> RunDeps {
        let store = Arc::new(Store::open_in_memory().unwrap());
        let workspace = Arc::new(WorkspaceManager::start(dir.join("files"), dir.join("ws"), false).await);
        // `true` is a shell builtin-ish binary that exits 0 immediately and ignores stdin;
        // the console is never used by a text-only run, so it just has to exist.
        let console = Arc::new(ConsoleFactory::new(std::path::PathBuf::from("/usr/bin/true")));
        RunDeps { store, console, workspace, catalog: Arc::new(Catalog::from_data(CatalogData::default())) }
    }

    #[tokio::test]
    async fn a_text_run_persists_user_and_assistant_and_ends_done() {
        const SSE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
                           data: [DONE]\n\n";
        let addr = fake_model_server(SSE).await;
        point_default_at(addr);
        let dir = tempfile::tempdir().unwrap();
        let d = deps(dir.path()).await;
        let store = d.store.clone();
        store.session_create("s1", "t", "fake/m").unwrap();
        let mgr = RunManager::new(d);

        let mut handle = mgr.start("s1", vec![Part::text("hi")]).await.unwrap();
        assert!(mgr.is_running("s1").await);
        assert!(matches!(mgr.start("s1", vec![Part::text("again")]).await, Err(EngineError::AlreadyRunning)));

        let mut text = String::new();
        let mut done = false;
        while let Ok(ev) = handle.events.recv().await {
            match ev {
                RunEvent::TextDelta { text: t } => text.push_str(&t),
                RunEvent::Done => { done = true; break; }
                RunEvent::Error { kind, message } => panic!("{kind}: {message}"),
                _ => {}
            }
        }
        assert!(done);
        assert_eq!(text, "Hello");
        let msgs = store.message_list("s1").unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].message.contents[0].as_text(), Some("Hello"));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!mgr.is_running("s1").await);
    }
}
```

주의: 콘솔 스폰은 `Console::builder().build()` 가 `init` 핸드셰이크를 기다리므로 `/usr/bin/true` 로는 실패한다(`ConsoleUnavailable`). 이 테스트가 그 이유로 실패하면, `drive` 의 콘솔 스폰을 **첫 툴 배치 직전이 아니라 시작 시** 하도록 두는 대신 다음 규칙으로 바꾼다: `ConsoleFactory::spawn` 실패는 `Error{kind:"console_unavailable"}` 로 끝내되, 테스트에서는 `AILOY_CORTEX_BIN_DIR` 가 없으면 `#[ignore]` 처리 대신 **`ConsoleFactory` 에 `Optional` 모드를 둔다**: `ConsoleFactory::disabled()` 는 `spawn` 시 `Err(ConsoleUnavailable)` 을 즉시 반환하고, `drive` 는 콘솔이 없으면 `AgentBuilder::console` 을 생략한다(pure 툴만 동작; 콘솔 툴 호출은 ailoy 가 "needs a console" 오류로 답한다). 테스트는 `ConsoleFactory::disabled()` 를 쓴다. 이 변형을 채택하면 `console.rs` 에 다음을 추가한다:

```rust
    /// A factory that never spawns: runs proceed without a console (tools that need one fail
    /// saying so). For tests and for a machine with no console binary.
    pub fn disabled() -> Self { Self { bin: PathBuf::new() } }
    pub fn is_disabled(&self) -> bool { self.bin.as_os_str().is_empty() }
```

그리고 `drive` 에서:

```rust
    let console = if deps.console.is_disabled() { None } else { Some(deps.console.spawn(ws_mount).await.map_err(|e| fail("console_unavailable", e.to_string()))?) };
    let mut builder = AgentBuilder::new(model)...;
    if let Some(c) = console { builder = builder.console(c); }
    let mut agent = builder.build()...;
```

- [ ] **Step 3: 확인·커밋**

```bash
cargo test -p ailoy-desktop-core run 2>&1 | grep -E 'test result|FAILED|panicked'
git add apps/desktop/core && git commit -m "feat(desktop-core): RunManager actor with events, persistence and cancellation"
```

---

### Task B11: `Engine` 파사드

**Files:**
- Create: `src/engine.rs`
- Modify: `src/lib.rs` (`pub mod engine; pub use engine::Engine;`)

**Interfaces:**
- Produces(스펙 §6.7):
  ```rust
  pub struct Engine { .. }
  impl Engine {
    pub async fn start(cfg: EngineConfig) -> Result<Arc<Engine>>;
    pub async fn shutdown(&self);
    pub async fn session_list(&self) -> Result<Vec<SessionSummary>>;
    pub async fn session_create(&self, model: Option<String>) -> Result<SessionSummary>;
    pub async fn session_rename(&self, id: &str, title: &str) -> Result<()>;
    pub async fn session_set_model(&self, id: &str, model: &str) -> Result<()>;
    pub async fn session_delete(&self, id: &str) -> Result<()>;
    pub async fn message_list(&self, session_id: &str) -> Result<Vec<StoredMessage>>;
    pub async fn run_start(&self, session_id: &str, parts: Vec<Part>) -> Result<RunHandle>;
    pub async fn run_attach(&self, session_id: &str) -> Option<(RunHandle, String)>;
    pub async fn run_cancel(&self, session_id: &str) -> bool;
    pub async fn fs_list(&self, path: &str) -> Result<Vec<Entry>>;  fs_read, fs_write, fs_mkdir, fs_delete, fs_rename, fs_import (fsops 위임)
    pub async fn mount_list(&self) -> Vec<MountInfo>;
    pub async fn mount_add(&self, req: MountRequest) -> Result<MountInfo>;
    pub async fn mount_remove(&self, path: &str) -> Result<()>;
    pub fn workspace_info(&self) -> WorkspaceInfo;
    pub async fn settings_get(&self) -> Result<Settings>;
    pub async fn settings_set(&self, patch: SettingsPatch) -> Result<Settings>;
    pub fn models_list(&self) -> Result<Vec<ModelInfo>>;
    pub async fn session_usage(&self, session_id: &str) -> Result<SessionUsage>;
  }
  ```

- [ ] **Step 1: 테스트** (`engine.rs` `mod tests`; 마운트 비활성, 콘솔 disabled)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    async fn engine() -> (tempfile::TempDir, Arc<Engine>) {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = EngineConfig::new(dir.path());
        cfg.mount_workspace = false;
        cfg.catalog_refresh = false;
        cfg.console_bin = Some(PathBuf::new()); // disabled console
        let e = Engine::start(cfg).await.unwrap();
        (dir, e)
    }

    #[tokio::test]
    async fn sessions_settings_and_mounts_round_trip_through_the_engine() {
        let (_dir, e) = engine().await;
        let s = e.session_create(None).await.unwrap();
        assert_eq!(s.model, providers::DEFAULT_MODEL);
        assert_eq!(e.session_list().await.unwrap().len(), 1);
        e.session_rename(&s.id, "renamed").await.unwrap();
        assert_eq!(e.session_list().await.unwrap()[0].title, "renamed");

        let mut patch = SettingsPatch::default();
        patch.provider_keys.insert("anthropic".into(), Some("sk-ant-test-1234".into()));
        let settings = e.settings_set(patch).await.unwrap();
        assert!(settings.providers.iter().any(|p| p.key == "anthropic" && p.has_key));
        let models = e.models_list().unwrap();
        assert!(models.iter().any(|m| m.id.starts_with("anthropic/") && m.available));
        assert!(models.iter().filter(|m| m.id.starts_with("openai/")).all(|m| !m.available));

        let local = tempfile::tempdir().unwrap();
        std::fs::write(local.path().join("f.txt"), b"hey").unwrap();
        let info = e.mount_add(MountRequest { path: "docs".into(), label: None, config: MountConfig::Local { host_root: local.path().to_path_buf() } }).await.unwrap();
        assert_eq!(info.path, "/docs");
        assert_eq!(info.label, "docs");
        assert_eq!(e.fs_read("/docs/f.txt").await.unwrap().text.as_deref(), Some("hey"));
        assert_eq!(e.mount_list().await.len(), 2);
        e.mount_remove("/docs").await.unwrap();
        assert_eq!(e.mount_list().await.len(), 1);
        e.shutdown().await;
    }

    #[tokio::test]
    async fn mounts_are_restored_on_restart() {
        let dir = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        let mk = || { let mut c = EngineConfig::new(dir.path()); c.mount_workspace = false; c.catalog_refresh = false; c.console_bin = Some(PathBuf::new()); c };
        {
            let e = Engine::start(mk()).await.unwrap();
            e.mount_add(MountRequest { path: "/data".into(), label: Some("데이터".into()), config: MountConfig::Local { host_root: local.path().to_path_buf() } }).await.unwrap();
            e.shutdown().await;
        }
        let e = Engine::start(mk()).await.unwrap();
        let mounts = e.mount_list().await;
        assert!(mounts.iter().any(|m| m.path == "/data" && m.label == "데이터" && matches!(m.status, MountStatus::Ok)));
        e.shutdown().await;
    }
}
```

- [ ] **Step 2: 구현**

```rust
//! The engine: everything the window can ask for, behind one `Arc`.

use std::{path::PathBuf, sync::Arc};

use ailoy::message::Part;

use crate::{
    catalog::{Catalog, split_model_id},
    config::EngineConfig,
    console::{ConsoleFactory, resolve_console_bin},
    error::{EngineError, Result},
    providers,
    run::{RunDeps, RunHandle, RunManager},
    store::{MountRow, Store},
    types::*,
    workspace::{WorkspaceManager, connectors, fsops},
};

pub struct Engine {
    cfg: EngineConfig,
    store: Arc<Store>,
    workspace: Arc<WorkspaceManager>,
    catalog: Arc<Catalog>,
    runs: RunManager,
}

impl Engine {
    pub async fn start(cfg: EngineConfig) -> Result<Arc<Engine>> {
        std::fs::create_dir_all(&cfg.data_dir)?;
        let store = Arc::new(Store::open(&cfg.db_path())?);
        let workspace = Arc::new(WorkspaceManager::start(cfg.files_root(), cfg.mountpoint(), cfg.mount_workspace).await);

        // Connectors come back from the database; one that cannot be rebuilt shows its error
        // in the list instead of taking the workspace down.
        for row in store.mount_list()? {
            let (kind, detail, writable) = connectors::describe(&row.config);
            let mut info = MountInfo { id: row.id.clone(), path: row.path.clone(), kind, label: row.label.clone(), detail, writable, status: MountStatus::Ok };
            match connectors::build_and_probe(&row.config).await {
                Ok(fs) => {
                    if let Err(e) = workspace.attach(info.clone(), fs).await {
                        info.status = MountStatus::Error { message: e.to_string() };
                        workspace.remember_failed(info).await;
                    }
                }
                Err(e) => {
                    info.status = MountStatus::Error { message: e.to_string() };
                    workspace.remember_failed(info).await;
                }
            }
        }

        let cache = cfg.cache_dir().join("models.json");
        let catalog = Arc::new(Catalog::load(Some(&cache)));
        let refresh = cfg.catalog_refresh && store.setting_get("catalog_refresh")?.map(|v| v != "false").unwrap_or(true);
        if refresh {
            let catalog = catalog.clone();
            tokio::spawn(async move {
                if let Err(e) = catalog.refresh(&cache).await {
                    tracing::warn!("catalog refresh failed: {e}");
                }
            });
        }

        providers::apply(&store)?;

        let console = Arc::new(match cfg.console_bin.as_deref() {
            Some(p) if p.as_os_str().is_empty() => ConsoleFactory::disabled(),
            explicit => match resolve_console_bin(explicit) {
                Ok(bin) => ConsoleFactory::new(bin),
                Err(e) => { tracing::warn!("{e}"); ConsoleFactory::disabled() }
            },
        });

        let runs = RunManager::new(RunDeps { store: store.clone(), console, workspace: workspace.clone(), catalog: catalog.clone() });
        Ok(Arc::new(Engine { cfg, store, workspace, catalog, runs }))
    }

    pub async fn shutdown(&self) {
        self.runs.cancel_all().await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        self.workspace.shutdown().await;
    }

    // ── sessions ────────────────────────────────────────────────────────────

    pub async fn session_list(&self) -> Result<Vec<SessionSummary>> {
        let mut out = Vec::new();
        for r in self.store.session_list()? {
            let running = self.runs.is_running(&r.id).await;
            out.push(SessionSummary { id: r.id, title: r.title, model: r.model, created_at: r.created_at, updated_at: r.updated_at, running });
        }
        Ok(out)
    }

    pub async fn session_create(&self, model: Option<String>) -> Result<SessionSummary> {
        let model = match model { Some(m) if !m.trim().is_empty() => m, _ => providers::read_settings(&self.store)?.default_model };
        let id = uuid::Uuid::new_v4().to_string();
        let r = self.store.session_create(&id, "새 대화", &model)?;
        Ok(SessionSummary { id: r.id, title: r.title, model: r.model, created_at: r.created_at, updated_at: r.updated_at, running: false })
    }

    pub async fn session_rename(&self, id: &str, title: &str) -> Result<()> {
        let title = title.trim();
        if title.is_empty() { return Err(EngineError::Invalid("제목을 입력해 주세요".into())); }
        self.store.session_rename(id, title)
    }

    pub async fn session_set_model(&self, id: &str, model: &str) -> Result<()> {
        if split_model_id(model).is_none() { return Err(EngineError::Invalid(format!("모델 ID 형식은 provider/model 입니다: {model}"))); }
        self.store.session_set_model(id, model)
    }

    pub async fn session_delete(&self, id: &str) -> Result<()> {
        self.runs.cancel(id).await;
        self.store.session_delete(id)
    }

    pub async fn message_list(&self, session_id: &str) -> Result<Vec<StoredMessage>> {
        self.store.session_get(session_id)?;
        self.store.message_list(session_id)
    }

    // ── runs ────────────────────────────────────────────────────────────────

    pub async fn run_start(&self, session_id: &str, parts: Vec<Part>) -> Result<RunHandle> {
        if parts.is_empty() { return Err(EngineError::Invalid("메시지가 비어 있습니다".into())); }
        self.runs.start(session_id, parts).await
    }

    pub async fn run_attach(&self, session_id: &str) -> Option<(RunHandle, String)> {
        self.runs.attach(session_id).await
    }

    pub async fn run_cancel(&self, session_id: &str) -> bool {
        self.runs.cancel(session_id).await
    }

    // ── workspace ───────────────────────────────────────────────────────────

    pub fn workspace_info(&self) -> WorkspaceInfo { self.workspace.info() }
    pub async fn fs_list(&self, path: &str) -> Result<Vec<Entry>> { fsops::list(&self.workspace.fs(), path).await }
    pub async fn fs_read(&self, path: &str) -> Result<FileContent> { fsops::read(&self.workspace.fs(), path).await }
    pub async fn fs_write(&self, path: &str, text: &str) -> Result<()> { fsops::write(&self.workspace.fs(), path, text).await }
    pub async fn fs_mkdir(&self, path: &str) -> Result<()> { fsops::mkdir(&self.workspace.fs(), path).await }
    pub async fn fs_delete(&self, path: &str) -> Result<()> { fsops::delete(&self.workspace.fs(), path).await }
    pub async fn fs_rename(&self, from: &str, to: &str) -> Result<()> { fsops::rename(&self.workspace.fs(), from, to).await }
    pub async fn fs_import(&self, dest: &str, sources: Vec<PathBuf>) -> Result<ImportReport> { fsops::import(&self.workspace.fs(), dest, sources).await }

    pub async fn mount_list(&self) -> Vec<MountInfo> { self.workspace.mounts().await }

    pub async fn mount_add(&self, req: MountRequest) -> Result<MountInfo> {
        let path = connectors::normalize_mount_path(&req.path)?;
        let (kind, detail, writable) = connectors::describe(&req.config);
        let label = req.label.map(|l| l.trim().to_string()).filter(|l| !l.is_empty())
            .unwrap_or_else(|| path.rsplit('/').next().unwrap_or(&path).to_string());
        let fs = connectors::build_and_probe(&req.config).await?;
        let info = MountInfo { id: uuid::Uuid::new_v4().to_string(), path: path.clone(), kind: kind.clone(), label: label.clone(), detail, writable, status: MountStatus::Ok };
        self.workspace.attach(info.clone(), fs).await?;
        let row = MountRow { id: info.id.clone(), path, kind, label, config: req.config, writable, created_at: now_ms() };
        if let Err(e) = self.store.mount_insert(&row) {
            let _ = self.workspace.detach(&row.path).await;
            return Err(e);
        }
        Ok(info)
    }

    pub async fn mount_remove(&self, path: &str) -> Result<()> {
        self.workspace.detach(path).await?;
        self.store.mount_delete(path)
    }

    // ── settings & catalog ──────────────────────────────────────────────────

    pub async fn settings_get(&self) -> Result<Settings> { providers::read_settings(&self.store) }

    pub async fn settings_set(&self, patch: SettingsPatch) -> Result<Settings> {
        providers::write_settings(&self.store, &patch)?;
        providers::apply(&self.store)?;
        providers::read_settings(&self.store)
    }

    pub fn models_list(&self) -> Result<Vec<ModelInfo>> {
        let settings = providers::read_settings(&self.store)?;
        let mut out = Vec::new();
        for def in providers::PROVIDERS {
            let available = settings.providers.iter().any(|p| p.key == def.key && p.has_key);
            for m in self.catalog.models_for(def.ailoy_prefix) {
                out.push(ModelInfo {
                    id: format!("{}/{}", def.ailoy_prefix, m.id),
                    provider: def.label.into(),
                    name: m.name,
                    context: m.context,
                    output: m.output,
                    cost: m.cost,
                    reasoning: m.reasoning,
                    tool_call: m.tool_call,
                    available,
                });
            }
        }
        out.sort_by(|a, b| b.available.cmp(&a.available).then(a.id.cmp(&b.id)));
        Ok(out)
    }

    pub async fn session_usage(&self, session_id: &str) -> Result<SessionUsage> {
        let session = self.store.session_get(session_id)?;
        let usages = self.store.message_usages(session_id)?;
        Ok(crate::usage::session_usage(&usages, self.catalog.lookup(&session.model).as_ref()))
    }

    pub fn config(&self) -> &EngineConfig { &self.cfg }
}
```

- [ ] **Step 3: 확인·커밋**

```bash
cargo test -p ailoy-desktop-core 2>&1 | grep -E 'test result|FAILED|panicked'
git add apps/desktop/core && git commit -m "feat(desktop-core): Engine facade over store, workspace, runs, settings, catalog"
```

---

### Task B12: 라이브 E2E(콘솔 + 툴콜)와 정리

**Files:**
- Create: `tests/live_run.rs`

- [ ] **Step 1: 라이브 run 테스트(`#[ignore]`)** — 가짜 모델이 `shell` 툴로 `echo` 를 호출하고, 결과가 DB에 기록되는지 확인. 콘솔 바이너리가 필요하다.

```rust
//! A run that calls the shell tool through a real `cortex-local-console`.
//! Run: `AILOY_CORTEX_BIN_DIR=../cortex/target/debug cargo test -p ailoy-desktop-core --test live_run -- --ignored`

use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

use ailoy::{agent::{AgentProvider, get_agent_providers_mut}, lang_model::{LangModelProvider, get_lm_providers_mut}, message::{Part, Role}, tool::{ToolProvider, get_tool_providers_mut}};
use ailoy_desktop_core::{Engine, EngineConfig, MountConfig, MountRequest, RunEvent};
use axum::{Router, body::Body, response::Response, routing::post};

const TOOL_CALL: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"{\\\"cmd\\\":\\\"cat hello.txt\\\"}\"}}]}}]}\n\n\
                         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                         data: [DONE]\n\n";
const ANSWER: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"It says hi.\"},\"finish_reason\":\"stop\"}]}\n\n\
                      data: [DONE]\n\n";

#[tokio::test]
#[ignore]
async fn a_run_reads_a_workspace_file_through_the_shell_tool() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let app = Router::new().route("/", post(move || {
        let c = c.clone();
        async move {
            let body = if c.fetch_add(1, Ordering::SeqCst) == 0 { TOOL_CALL } else { ANSWER };
            Response::builder().status(200).header("content-type", "text/event-stream").body(Body::from(body)).unwrap()
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    {
        let mut lmps = get_lm_providers_mut();
        lmps.entry("default".into()).or_insert_with(LangModelProvider::new)
            .insert("fake/*".into(), LangModelProvider::chat_completion(&format!("http://{addr}/"), None).unwrap());
        drop(lmps);
        get_tool_providers_mut().entry("default".into()).or_insert_with(ToolProvider::new);
        get_agent_providers_mut().entry("default".into()).or_insert_with(|| AgentProvider::new("default", "default"));
    }

    let dir = tempfile::tempdir().unwrap();
    let mut cfg = EngineConfig::new(dir.path());
    cfg.mount_workspace = false;   // the console stands in files/ directly
    cfg.catalog_refresh = false;
    let engine = Engine::start(cfg).await.unwrap();
    engine.fs_write("/hello.txt", "hi from the workspace").await.unwrap();

    let s = engine.session_create(Some("fake/m".into())).await.unwrap();
    let mut h = engine.run_start(&s.id, vec![Part::text("what does hello.txt say?")]).await.unwrap();
    let mut tool_started = false;
    loop {
        match h.events.recv().await.unwrap() {
            RunEvent::ToolCallStarted { name, .. } => { assert_eq!(name, "shell"); tool_started = true; }
            RunEvent::Done => break,
            RunEvent::Error { kind, message } => panic!("{kind}: {message}"),
            _ => {}
        }
    }
    assert!(tool_started);
    let msgs = engine.message_list(&s.id).await.unwrap();
    let tool = msgs.iter().find(|m| m.message.role == Role::Tool).expect("a tool result was persisted");
    let v = tool.message.contents[0].as_value().unwrap();
    assert!(v.pointer("/stdout").and_then(|s| s.as_str()).unwrap().contains("hi from the workspace"), "{v:?}");
    assert_eq!(msgs.last().unwrap().message.contents[0].as_text(), Some("It says hi."));
    engine.shutdown().await;
}
```

`axum` 을 `[dev-dependencies]` 에 추가한다(`axum = "0.8"`).

- [ ] **Step 2: 실행**

```bash
AILOY_CORTEX_BIN_DIR=../cortex/target/debug cargo test -p ailoy-desktop-core --test live_run -- --ignored 2>&1 | grep -E 'test result|FAILED|panicked'
```

- [ ] **Step 3: 전체 정리·커밋**

```bash
cargo fmt --all && cargo clippy -p ailoy-desktop-core --all-targets 2>&1 | grep -E '^(warning|error)' | head
cargo test -p ailoy-desktop-core 2>&1 | grep -E 'test result|FAILED'
git add -A && git commit -m "test(desktop-core): live run through cortex-local-console; fmt and clippy"
```

---

## Self-Review 체크리스트 (작성자용)

- 스펙 §6.1 모듈표 → B1(config), B2(store), B3(catalog), B4(providers), B5(prompt), B6/B7(workspace), B8(console), B9(assembler/events/usage), B10(run), B11(engine) ✓
- §6.2 스키마 → B2 SQL과 일치(`workspace_id` 기본값 'default') ✓
- §6.3 run 수명·이벤트 → B10 (`Started`→user `Message`→델타/툴콜/메시지/사용량→`Done|Cancelled|Error{kind}`) ✓; `AwaitingApproval` 은 정의만(v1 미발생) ✓
- §6.4 프리앰블 → B5 ✓ (`extra_instruction` 설정 키로 사용자 지시문)
- §6.5 카탈로그 → B3 (`gen-catalog` 바이너리가 스크립트 대신 Rust 필터 재사용) ✓
- §6.6 사용량 계산 → B9 ✓
- §6.7 공개 API → B11 (`session_set_model` 추가) ✓
- §10 오류 처리: Degraded 워크스페이스(B7), 커넥터 복원 실패 배지(B11), 콘솔 없음(B8/B10 `console_unavailable`), 턴 상한 `max_turns` kind(B10) ✓
- 타입 일치: `MountInfo{id,path,kind,label,detail,writable,status}` · `MountConfig::{Root,Local{host_root},Notion{api_key},S3{..}}` · `RunEvent` 변형명 · `EngineError` 변형명 · `ConsoleFactory::{new,disabled,is_disabled,spawn}` · `WorkspaceManager::{start,info,console_mount,fs,attach,detach,mounts,remember_failed,shutdown}` — B6~B12 전체 동일 ✓
