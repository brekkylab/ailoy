# Ailoy Desktop — 설계서 (v1)

- 작성일: 2026-09-11
- 상태: 리뷰 대기
- 범위: ailoy를 개발자 라이브러리에서 Claude/ChatGPT형 데스크톱 앱(Tauri)으로 전환하는 v1(minimal) 설계
- 관련 저장소: `brekkylab/ailoy`(이 저장소), `brekkylab/cortex`(형제 체크아웃 `../cortex`, 당분간 path 의존)

---

## 1. 목표와 범위

### 1.1 목표

세션 기반 대화형 UI를 가진 데스크톱 앱을 만든다. 에이전트는 **cortex workspace**(WorkFs를 FUSE-T로 마운트한 디렉터리) 안에서 `cortex-local-console`을 통해 셸 명령과 파일 I/O를 수행하고, 사용자는 로컬 폴더·Notion·S3를 하나의 파일시스템 트리에 붙여 에이전트에게 보여줄 수 있다(Rambox/Ferdium식 "내 FS에 서비스 붙이기"). 에이전틱 루프는 Claude/ChatGPT 수준의 기본 완성도(취소, 턴 상한, 승인 훅, 실패 복구, 사용량 표시)를 갖춘다.

### 1.2 v1 포함

- 세션 CRUD와 영속(SQLite), 앱 재시작 후 대화 복원
- 스트리밍 대화(텍스트·thinking 델타), 툴콜 카드(이름·인자·결과·상태)
- 시스템 툴 `shell/read/write/edit/glob/grep` + `web_search/web_fetch`
- 실행 취소, 턴 상한(기본 50), 툴 승인 훅(v1 정책은 자동 승인)
- 워크스페이스: 앱 수명 동안 유지되는 FUSE-T 마운트, 루트는 영속 로컬 디렉터리, 커넥터 로컬 폴더/Notion/S3
- 파일 브라우저(트리·텍스트 미리보기), 마운트 관리 UI
- 토큰 사용량: 메시지·세션 누적, 컨텍스트 창 사용률, 추정 비용, Anthropic/OpenAI(xAI 조건부) rate-limit 잔여율
- 설정: 프로바이더 API 키, 기본 모델
- 대상 플랫폼: macOS(Apple Silicon 우선). FUSE-T 설치 필요.

### 1.3 v1 제외 (후속 로드맵은 §12)

승인 UI·정책 저장, `mem` 메모리 툴 연결, micro-VM 콘솔, GDrive 커넥터(OAuth), stdout 스트리밍, 컨텍스트 요약, MCP, Keychain, Windows/Linux, `Length` 이어쓰기, 세션 제목 자동 생성, 계정 잔액·월 사용량 조회.

---

## 2. 현황 요약과 재활용 판단

### 2.1 ailoy 브랜치

| 브랜치 | 요지 | 판단 |
|---|---|---|
| `develop` (`60716d12`) | 순수 Rust 라이브러리. API 모델 5종(Anthropic/OpenAI Responses/ChatCompletion/Gemini/Bedrock). 자체 `runenv`(Local, microsandbox), `skill`, `python_repl`. MCP `todo!()`. | 기반으로 쓰지 않음. 단 #448(Bedrock wire)은 가져옴. |
| `origin/mem-applied` (`8d3f0238`, 9/7) ⊃ `cortex-applied` | `runenv`·`skill`·`python_repl` 제거. `cortex::console::Console` 재수출(`src/console.rs`). 도구는 cortex `exec/read/write` 직접 호출. 툴 배치마다 `console.start/stop`. `src/memory`(`mem` 실행파일 기반) + `mem_search/mem_insert` 툴. `cortex = { path = "../cortex/cortex" }`. **cortex main과 컴파일 확인.** | **작업 기반.** |
| `origin/feat/krun-sandbox` (8/6) | agent-k의 기반. ailoy가 libkrun VM을 직접 띄움. | cortex가 VM 콘솔을 흡수했으므로 채택하지 않음. |
| `origin/feat/vfs-provider-mounts` 등 (6~7월) | `src/vfs`(S3/Notion/GDrive + in-guest FUSE). | cortex `fs`로 대체됨. 참고만. |

`mem-applied`는 #448 이전에서 갈라져 `bedrock.rs`가 없다. 작업 브랜치는 `mem-applied`에서 따고 `develop`을 머지한다(#448은 그대로, #443은 삭제된 sandbox 코드 → 삭제로 해소, `web_fetch.rs` 변경은 유지).

### 2.2 cortex (`../cortex` main `dfabb34`)

- `console`: stdio 위 BSON JSON-RPC. 메서드 `init/exec/read/write/commit/start/stop/quit`. 서버는 `cortex-local-console`(호스트), `cortex-uvm-console`(micro-VM). **제약**: stdout 스트리밍 없음(설계상), 디렉터리 listing 없음, 콘솔당 요청 1개(`&mut self`), `timeout_ms`는 받지만 두 서버 모두 **미구현**, 64 MiB 프레임 상한(`ExecResp::truncated`).
- `fs`: `FileSystem` 트레이트(필수 `stat/list/read_at`, 나머지 기본 read-only), `WorkFs` 최장접두 마운트 테이블(자체가 `FileSystem`), 백엔드 `InMemFs/PassthroughFs/S3Fs/NotionFs/GdriveFs`, 호스트 마운트 `FuseMount`(macFUSE)/`FuseTMount`(FUSE-T, kext 불필요). `Mount` 트레이트: `mountpoint()`, drop 시 언마운트.
- `rootfs`/이미지 레이어, `mem`/`index` 실행파일(SQLite+vec0).
- **`origin/cortex-gui`** (jhlee525, 9/7): Tauri 2 + React PoC. WorkFs 파일 브라우저, 로컬/Notion/S3 커넥터, `SharedFs`(마운트에 넘겨도 WorkFs를 잃지 않는 핸들), FUSE-T 임시 마운트로 `mem init`. 세션 저장 없음. README: "에이전트 실행 — ailoy가 붙는 자리".

### 2.3 agent-k (`../agent-k`)

ailoy(`feat/krun-sandbox`) + cortex 위의 서버형 풀스택(axum + SQLite + SSE). 프론트는 구버전 backend에 묶여 재사용 불가. **이식 가치가 높은 조각**: `backend/src/agent_stream.rs`(델타→메시지 재조립 `MessageAssembler`, 테스트 포함), `state/session.rs`의 중단 처리(drain 모드, `completed_naturally`, 미완 tool_call stub), `SessionMessage{depth, source_agent, message}` 저장 형태와 "depth 0만 모델에 재주입" 규칙, `lib/toolCallFormat.ts`(툴콜 렌더/복사 단일 파서). 툴 승인은 없었다.

### 2.4 재활용 결정 요약

| 출처 | 그대로 사용 | 이식(코드 복사·수정) | 아이디어만 |
|---|---|---|---|
| ailoy `mem-applied` | 코어 전체(메시지 모델, LM 와이어 5종, 툴, 에이전트 루프, memory) | — | — |
| cortex main | `Console`, `WorkFs`, 백엔드 FS, `FuseTMount` | — | — |
| cortex-gui | — | `SharedFs`, `fsops`(list/read/write/import), `mounts`(커넥터 사전 검증), 파일 트리 UI 로직 | 3열 레이아웃 |
| agent-k | — | `MessageAssembler`, 중단 처리, `SessionMessage` 형태, 툴콜 포맷터 | `.ref` 지식 인덱싱(후속) |

---

## 3. 결정 사항 (Q&A 기록)

| 항목 | 결정 |
|---|---|
| 코어 기반 브랜치 | `mem-applied` (+ develop #448 머지) |
| v1 콘솔 백엔드 | `cortex-local-console` (호스트 실행). micro-VM은 후속 토글 |
| v1 범위 | 채팅 + 로컬 폴더 + 외부 커넥터(Notion/S3) |
| 툴 승인 | 훅과 이벤트는 지금, 정책은 자동 승인 |
| 코드 위치 | ailoy 저장소 `apps/desktop/`. 브랜치는 `mem-applied`에서. `mem-applied→develop` 병합은 브랜치 소유자 |
| cortex 수정 | 필요 시 cortex에 변경 브랜치 생성(path 의존으로 참조) |
| 워크스페이스 모델 | 앱에 워크스페이스 1개, 세션 N개(스키마에 `workspace_id` 유지) |
| UI 스택 | React 19 + Vite + TS + Tailwind v4 + shadcn/ui, 채팅 UI 직접 구현 |
| 런타임 배치 | 헤드리스 세션 엔진 크레이트 + Tauri 임베드 |
| 마운트 수명 | 앱 수명 동안 FUSE-T 마운트 유지 |
| 사용량 표시 | 계층 1(전 벤더 컨텍스트 사용률·누적·비용) + 계층 2(rate-limit 헤더) |
| 모델 메타데이터 | models.dev(`https://models.dev/api.json`) 스냅샷 내장 + 런타임 갱신 |

관례적 기본값(질문 없이 결정): SQLite는 `rusqlite`(bundled), API 키는 앱 데이터 디렉터리의 설정 DB에 저장(Keychain은 후속), 기본 모델 `anthropic/claude-opus-5`, `max_tokens` 기본 32,000, 턴 상한 50.

---

## 4. 전체 아키텍처

### 4.1 저장소 배치

```
ailoy/                         cargo workspace 루트 (패키지 `ailoy`는 그대로 라이브러리)
├─ Cargo.toml                  members = ["./", "apps/desktop/core"], exclude += ["apps/desktop/src-tauri"]
├─ src/                        ailoy 코어 (§5)
├─ apps/desktop/
│  ├─ package.json, vite.config.ts, tailwind, src/   React 프론트 (§8)
│  ├─ core/                    crate `ailoy-desktop-core` — 헤드리스 세션 엔진 (§6)
│  ├─ src-tauri/               crate `ailoy-desktop` — 자체 [workspace] (§7)
│  └─ scripts/                 models.dev 스냅샷 생성 등
└─ ../cortex/cortex            path 의존
```

`src-tauri`를 루트 workspace의 member로 두지 않는 이유는 cortex-gui와 같다: Tauri가 끌어오는 webview 의존 수백 개를 루트의 `cargo test`에 부담시키지 않는다. `core`는 가벼우므로(rusqlite, tokio, cortex, ailoy) member로 둔다.

### 4.2 프로세스와 데이터 흐름

```
┌──────────────── Tauri 앱 프로세스 ────────────────┐      stdio(BSON JSON-RPC)   ┌──────────────────────┐
│ WebView(React) ⇄ invoke/Channel ⇄ src-tauri       │ ───────────────────────────▶ │ cortex-local-console │ (run당 1개)
│                    └── ailoy-desktop-core          │                              │  cwd = workfs 마운트  │
│                          ├─ Agent(ailoy) ─ LLM API │                              └──────────┬───────────┘
│                          ├─ WorkFs ── FuseTMount ──┼── <appdata>/workspace ◀── 커널 FUSE ────┘
│                          └─ SQLite(<appdata>/db)   │
└───────────────────────────────────────────────────┘
```

- 파일 트리는 두 경로로 본다. 에이전트 셸은 FUSE-T 마운트 경로를 workfs로 받고, UI 파일 브라우저는 같은 `WorkFs`를 in-process `FileSystem::list/read`로 읽는다(FUSE 왕복 없음).
- 콘솔은 **run 하나에 하나**. cortex 콘솔은 요청을 하나씩만 받으므로 세션 간 동시 실행이 자연스럽고, run 종료 시 `quit`으로 정리된다. 한 세션에 run은 동시에 하나만.
- LLM 호출은 ailoy 코어가 앱 프로세스에서 직접 수행한다(키는 코어 프로세스에만 존재).

### 4.3 앱 데이터 디렉터리

`~/Library/Application Support/com.brekkylab.ailoy/`
- `ailoy.sqlite` — 세션·메시지·마운트·설정
- `files/` — 워크스페이스 루트(`PassthroughFs`)
- `workspace/` — FUSE-T 마운트포인트(비어 있어야 함)
- `cache/models.json` — models.dev 갱신 캐시

---

## 5. ailoy 코어 변경

원칙: 기존 `run`/`run_stream`과 `anyhow` 기반 공개 API는 유지하고, 제어 가능한 진입점과 타입 있는 에러를 **추가**한다. 라이브러리 소비자 누구나 같은 완성도를 얻도록 루프 정합성(취소·복구)은 코어에 둔다.

### 5.1 `RunControl`과 `run_stream_controlled`

```rust
// src/agent/control.rs (신규)
pub struct RunControl {
    pub cancel: tokio_util::sync::CancellationToken,
    pub max_turns: Option<u32>,            // 모델 호출 횟수 상한. None = 무제한(기존 동작)
    pub tool_gate: Arc<dyn ToolGate>,      // 기본 AllowAll
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

- 기존 `run_stream`은 `RunControl::default()`(취소 없음, 상한 없음, AllowAll)로 위임하고 `AgentError`를 `anyhow`로 감싼다.
- `MessageDeltaOutput` 아이템 타입은 바꾸지 않는다. "승인 대기" 같은 UI 이벤트는 게이트 구현체(엔진) 쪽에서 낸다.

### 5.2 루프 정합성 규칙

- **턴 상한**: 모델 호출 직전(직전 턴의 툴 결과가 모두 커밋된 지점)에 검사. 초과 시 `Err(AgentError::MaxTurns { turns })`. 이 시점의 history는 항상 재전송 가능하다.
- **취소 — 모델 응답 중**: `select!`로 스트림 `next()`와 `cancel.cancelled()`를 경합. 취소되면 누적된 부분 assistant 메시지를 커밋한다(텍스트·thinking은 그대로, 미완성 tool_call 조각은 버림, `finish_reason = Stop`). 텍스트도 tool_call도 없으면 커밋하지 않고 대기 중인 user 메시지를 pop(기존 롤백 규칙).
- **취소 — 툴 실행 중**: 툴 스트림을 drop(진행 중인 future 중단). 결과가 커밋되지 않은 tool_call마다 `Role::Tool` stub `"[Interrupted: cancelled before this tool call completed]"`을 같은 `id`로 history에 넣는다.
- **승인 거부**: `Deny{reason}`인 호출은 실행하지 않고 `Role::Tool` 결과 `{"error":"denied by user: <reason>","phase":"policy"}`로 기록한다. 같은 배치의 허용된 호출은 정상 실행.
- **모델 호출 실패(2턴 이후)**: 기존에는 assistant `tool_calls` 뒤에 결과가 없는 상태로 남을 수 있었다. 실패 시점에 미완 tool_call이 있으면 위와 같은 stub을 넣어 history를 닫은 뒤 에러를 반환한다.
- 어떤 경로로 종료되든 **history에 짝 없는 `tool_use`가 남지 않는다**(Anthropic 400 방지). 이는 테스트로 고정한다.

### 5.3 `AgentError`

```rust
pub enum AgentError {
    Cancelled,
    MaxTurns { turns: u32 },
    Model(ModelError),                 // status: Option<u16>, retryable: bool, provider_message: String
    Tool { name: String, source: anyhow::Error },
    Console(anyhow::Error),            // 콘솔 없음/기동 실패
    Other(anyhow::Error),
}
```

`LangModel` 계층에도 `ModelError`를 도입하고 HTTP 상태·본문·재시도 여부를 담는다. 기존 `anyhow` 경로는 `From<AgentError> for anyhow::Error`로 호환.

### 5.4 재시도와 클라이언트

- `send_with_retry`: 429 외에 **5xx와 transport 오류**도 지수 백오프(최대 3회, 10초 상한) 대상으로 포함. 4xx(429 제외)는 즉시 실패. 영구 quota 오류 분류(`is_permanent_quota_error`)는 유지.
- `reqwest::Client`를 `LangModel`에 캐시해 매 호출 TLS 핸드셰이크를 없앤다.

### 5.5 사용량과 rate-limit 정보

- `TokenUsage`는 5개 와이어 모두에서 채워진다(Gemini·OpenAI Responses는 캐시 필드 `None`; OpenAI Responses의 `input_tokens_details.cached_tokens`, Gemini `cachedContentTokenCount` 파싱을 추가한다).
- 신규 타입과 필드:

```rust
pub struct RateLimitWindow { pub limit: Option<u64>, pub remaining: Option<u64>, pub reset_at: Option<SystemTime> }
pub struct RateLimitInfo {
    pub requests: Option<RateLimitWindow>,
    pub tokens: Option<RateLimitWindow>,        // 통합 토큰(있는 벤더만)
    pub input_tokens: Option<RateLimitWindow>,
    pub output_tokens: Option<RateLimitWindow>,
}
// MessageOutput / MessageDeltaOutput 에 `pub rate_limit: Option<RateLimitInfo>` (serde skip_if_none) 추가.
// run_stream: 응답 헤더를 첫 델타에 실어 보낸다. run: MessageOutput에 실린다.
```

| 와이어 스키마 | 헤더 매핑 |
|---|---|
| Anthropic | `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}` (reset은 RFC 3339) |
| OpenAI Responses / ChatCompletion | `x-ratelimit-{limit,remaining,reset}-{requests,tokens}` (reset은 `1m2s` 형식 duration → 현재 시각에 더함). xAI·DeepSeek·Moonshot는 같은 파서를 시도하고 헤더가 없으면 `None` |
| Gemini, Bedrock | 헤더 없음 → `None` |

### 5.6 기반 브랜치 정리

`feat/desktop` 브랜치 = `origin/mem-applied` + `develop` 머지(#448 유지, #443의 `sandbox.rs` 변경은 파일 삭제로 해소). 이후 `develop`으로의 병합은 `mem-applied` 소유자와 조율한다.

---

## 6. 세션 엔진 `ailoy-desktop-core`

Tauri 의존이 없는 라이브러리 크레이트. 모든 공개 타입은 `serde` 직렬화 가능하며 그대로 IPC 페이로드가 된다.

### 6.1 모듈

| 모듈 | 책임 |
|---|---|
| `engine` | `Engine::start(config) -> Engine`: DB 열기·마이그레이션 → stale 마운트 정리 → WorkFs 조립·마운트 → 커넥터 복원 → 프로바이더 등록 → 모델 카탈로그 로드. `Engine::shutdown()`: 모든 run 취소 → 콘솔 quit → 마운트 drop. |
| `workspace` | `Arc<RwLock<WorkFs>>` 소유. `SharedFs`(cortex-gui 이식)를 `FuseTMount::try_new`에 넘겨 앱 수명 동안 유지. 루트 `""` = `PassthroughFs(<appdata>/files)`. 커넥터 `mount_add/remove`는 DB 갱신 + `WorkFs::mount/unmount`. 파일 브라우저용 `list/read/write/mkdir/delete/rename/import`(cortex-gui `fsops` 이식). `WorkspaceMount: cortex::fs::Mount` 구현체로 마운트포인트를 콘솔에 전달. |
| `console` | 사이드카 경로 해석 → `tokio::process::Command` → `Console::builder().client(StdioClient::new(cmd)).mount(WorkspaceMount).build()`. 환경 변수 `PATH`에 사이드카 디렉터리를 prepend(후속 `mem` 대비). run당 1개, run 종료 시 drop(`quit`). 스폰 실패는 `EngineError::ConsoleUnavailable`. |
| `store` | `rusqlite`(bundled, WAL). 스키마는 §6.2. `Message`는 ailoy serde JSON을 `{version, depth, source_agent, message}`로 감싸 저장. |
| `catalog` | models.dev 스냅샷(§6.5). 모델 목록, 컨텍스트 창, 단가, 기능 플래그 제공. |
| `providers` | 설정의 API 키를 ailoy 전역 `"default"` `LangModelProvider`에 등록/갱신(`get_lm_providers_mut`). 키 변경 즉시 반영. |
| `prompt` | 시스템 프리앰블 생성(§6.4). |
| `run` | 세션당 최대 1개의 actor 태스크(§6.3). |
| `usage` | `TokenUsage`·`RateLimitInfo`·카탈로그로 컨텍스트 사용률, 세션 누적, 추정 비용, 분당 잔여율 계산(§6.6). |
| `events` | `RunEvent`(§6.3), `EngineEvent`(마운트 변경·오류). |

### 6.2 데이터 모델 (SQLite)

```sql
CREATE TABLE workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at INTEGER NOT NULL);
-- v1은 'default' 한 행만 생성

CREATE TABLE mounts (
  id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  path TEXT NOT NULL,                 -- WorkFs 내 경로. 루트 행은 '' (표시는 '/')
  kind TEXT NOT NULL,                 -- 'root' | 'local' | 'notion' | 's3'
  label TEXT NOT NULL,
  config TEXT NOT NULL,               -- JSON. 자격 증명 포함(v1은 평문, 파일 권한 0600)
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
  depth INTEGER NOT NULL DEFAULT 0,   -- 0 = 최상위, ≥1 = 서브에이전트 내부
  source_agent TEXT,
  role TEXT NOT NULL,                 -- 조회 편의용 복제
  content TEXT NOT NULL,              -- {"version":1,"message":<ailoy Message JSON>}
  usage TEXT,                         -- TokenUsage JSON (assistant 메시지만)
  created_at INTEGER NOT NULL,
  PRIMARY KEY(session_id, seq)
);

CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- provider.<name>.api_key, default_model, catalog.last_refreshed 등
```

- 모델에 재주입하는 history는 `depth = 0`인 행만(agent-k 규칙). 서브에이전트 내부 메시지는 UI 표시용으로만 저장.
- `messages.usage`는 세션 누적 계산과 컨텍스트 사용률의 근거.
- 마이그레이션은 `PRAGMA user_version` 기반의 순차 SQL 파일(`core/migrations/NNNN_*.sql`).

### 6.3 실행(run) 수명과 이벤트

```
run_start(session, parts)
 ├─ 세션에 활성 run 있으면 EngineError::AlreadyRunning
 ├─ user 메시지 즉시 DB 기록(seq N) → RunEvent::Message
 ├─ 콘솔 스폰 → Agent 조립:
 │     AgentBuilder::new(model).instruction(preamble).system_tools()
 │        .web_search_tool(vec![]).web_fetch_tool()
 │        .history(depth0 메시지).console(console).build()
 │     spec.max_tokens = settings 또는 32_000
 ├─ tokio::spawn(actor):
 │     stream = agent.run_stream_controlled(user_msg, RunControl{cancel, max_turns: 50, tool_gate: AllowAll})
 │     MessageAssembler(agent-k 이식)로 델타→(TextDelta|ThinkingDelta|Completed) 분류
 │     Completed 메시지는 즉시 DB 기록(seq++) → RunEvent::Message
 │     usage/rate_limit이 실린 델타 → RunEvent::Usage
 │     종료: Ok → Done, Err(Cancelled) → Cancelled, Err(MaxTurns) → Error{kind:"max_turns"}, 그 외 → Error
 └─ 콘솔 drop(quit), 세션 updated_at 갱신
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
    AwaitingApproval { id: String, name: String, arguments: Value },   // v1에서는 발생하지 않음
    Done, Cancelled,
    Error { kind: String, message: String },
}
```

- 이벤트는 run별 `tokio::sync::broadcast`로 내보낸다. 창이 새로 고쳐져도 `run_attach`로 재구독하며, 이때 현재까지의 부분 텍스트를 한 번에 `TextDelta`로 보낸다.
- `run_cancel`은 `CancellationToken::cancel()` 후 actor 종료를 기다리지 않고 반환. 최종 상태는 이벤트로.
- 부분 assistant 텍스트는 코어가 커밋하므로(§5.2) 취소 후에도 DB에 남는다.

### 6.4 시스템 프리앰블

ailoy는 `instruction` 외에 아무것도 넣지 않으므로 엔진이 조립한다. 구성: (1) 정체성과 역할, (2) 날짜·OS, (3) 작업 디렉터리 = workfs 경로와 "모든 경로는 이 안"이라는 규칙, (4) 마운트 테이블: 경로·종류·읽기 전용 여부·각 커넥터 사용 안내(Notion은 `page.json` 렌더, S3는 객체 키 규칙), (5) 툴 사용 지침(`shell`은 `sh -c`, 결과 30k 문자 절단·`truncated` 플래그 의미), (6) 사용자 설정 추가 지시문(선택). 마운트가 바뀌면 다음 run부터 반영된다(세션 중간 변경은 새 시스템 메시지로 교체하지 않고 다음 run의 첫 메시지에 반영).

### 6.5 모델 카탈로그 (models.dev)

- 출처: `https://models.dev/api.json` (MIT, TOML 소스 + PR, 213 프로바이더·7,677 모델, 4.5 MB). 스키마: `provider.models[id] = { name, family, limit: {context, output}, cost: {input, output, cache_read, cache_write} (USD/1M), reasoning, tool_call, structured_output, modalities, release_date, ... }`.
- **빌드 시**: `apps/desktop/scripts/gen-catalog` 스크립트가 api.json을 받아 7개 프로바이더(`anthropic, openai, google, amazon-bedrock, xai, deepseek, moonshotai`)의 chat 모델만 남긴 스냅샷 `core/assets/models.json`(수십 KB)을 생성해 커밋. `include_str!`로 내장.
- **런타임**: 시작 시 24시간 캐시 기준으로 백그라운드 갱신 시도(`<appdata>/cache/models.json`). 실패는 조용히 내장 스냅샷으로 폴백. 설정에서 갱신 끄기 가능.
- **ID 매핑**: ailoy `provider/model` → models.dev `provider.models[model]`. 접두 매핑 `anthropic→anthropic, openai→openai, google→google, x-ai→xai, deepseek→deepseek, moonshotai→moonshotai, bedrock→amazon-bedrock`(Bedrock은 모델 ID 자체가 `anthropic.claude-...` 형태로 일치). 카탈로그에 없는 모델은 컨텍스트·비용 표시를 생략하고 "알 수 없음"으로.
- 모델 선택 UI 목록은 "키가 등록된 프로바이더 × 카탈로그의 `tool_call: true` 모델"이며, 사용자가 임의 ID를 직접 입력할 수도 있다.

### 6.6 사용량 계산

- 컨텍스트 사용률: 세션의 마지막 assistant 메시지 `usage`에서 `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`(각 `None`은 0) ÷ 카탈로그 `limit.context`. 다음 호출의 입력 크기에 대한 근사치로 표시(Anthropic `input_tokens`는 마지막 캐시 브레이크포인트 이후 토큰만이므로 세 항을 합쳐야 총 입력이 된다).
- 세션 누적: 모든 assistant `usage` 합. 비용 = Σ(input·cost.input + output·cost.output + cache_read·cost.cache_read + cache_write·cost.cache_write) / 1e6. 카탈로그에 단가가 없으면 비용 생략.
- 분당 잔여율: `RateLimitInfo`의 각 창에서 `remaining / limit`. `reset_at`까지 카운트다운. 헤더가 없는 프로바이더는 표시하지 않는다.

### 6.7 공개 API (엔진)

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
  pub fn workspace_info(&self) -> WorkspaceInfo;   // 마운트포인트, 상태
  // settings & catalog
  pub async fn settings_get(&self) -> Settings /*키는 마스킹*/; pub async fn settings_set(&self, patch: SettingsPatch);
  pub fn models_list(&self) -> Vec<ModelInfo>;
  pub async fn session_usage(&self, session_id) -> SessionUsage;
}
```

---

## 7. Tauri 계층 `ailoy-desktop`

- `tauri::Builder` setup에서 `Engine::start` → `app.manage(engine)`. 종료(`tauri::RunEvent::ExitRequested`, 엔진의 `RunEvent`와는 별개 타입)에서 `engine.shutdown()`을 `spawn_blocking`으로 기다린 뒤 종료(FUSE 언마운트는 스레드 join이 필요).
- 명령(모두 `async`, 엔진 호출을 감싸는 얇은 래퍼): `session_list/create/rename/delete`, `message_list`, `run_start(session_id, parts, on_event: Channel<RunEvent>) -> run_id`, `run_attach(session_id, on_event)`, `run_cancel(session_id)`, `fs_*`, `mount_list/add_local/add_notion/add_s3/remove`, `workspace_info`, `settings_get/set`, `models_list`, `session_usage`.
- 이벤트: run 이벤트는 `tauri::ipc::Channel<RunEvent>`(타입·순서 보장, 스트리밍용). 전역 변화(`workspace_changed`, `mount_error`, `catalog_refreshed`)만 `AppHandle::emit`.
- 사이드카: `tauri.conf.json` `bundle.externalBin: ["binaries/cortex-local-console"]`, capability `shell:allow-execute`(sidecar). cortex `StdioClient`가 `tokio::process::Command`를 요구하므로 플러그인 대신 경로만 해석한다: 번들에서는 실행 파일 옆(`current_exe().parent()`), dev에서는 `AILOY_CORTEX_BIN_DIR` → `../../../cortex/target/{debug,release}` 폴백. 빌드 스크립트가 `../cortex`에서 `cargo build -p cortex-local-console`을 돌려 `binaries/`에 target-triple 접미사로 복사한다.
- 보안: CSP `default-src 'self'`, 원격 콘텐츠 없음. 자격 증명은 프론트에 마스킹만 내려간다. `fs_read`는 텍스트 상한(1 MiB)과 바이너리 감지.
- 로그: `tracing` → 앱 데이터 `logs/`. 프론트에서 "로그 폴더 열기".

---

## 8. 프론트엔드

- 스택: React 19, Vite, TypeScript, Tailwind v4, shadcn/ui(Radix), TanStack Query(목록·설정), Zustand(스트림/런 상태), react-markdown + remark-gfm + shiki, lucide-react. `src/api.ts`가 `invoke`의 유일한 통로, `src/events.ts`가 `Channel`을 스토어에 연결(cortex-gui 패턴).
- 레이아웃(3열): 좌 세션 목록(+새 대화, 설정 버튼, 각 세션 제목·모델·최근 시각), 중앙 스레드(user/assistant 말풍선, 접히는 thinking, 툴콜 카드: 이름·인자 요약·결과·상태 running/done/error·소요시간, 컴포저: 모델 선택·전송·중지, 컴포저 위 컨텍스트 사용률 게이지와 세션 토큰·비용 요약, 프로바이더 배지에 분당 잔여율·리셋 카운트다운), 우 워크스페이스 패널(파일 트리, 마운트 목록·종류 배지·읽기전용 표시, "+ 연결" 다이얼로그 로컬/Notion/S3 — 연결 전 검증 요청 1회, 텍스트 미리보기·간단 편집).
- 설정 다이얼로그: 프로바이더별 API 키(마스킹, 저장 시 즉시 등록), 기본 모델, `max_tokens`, 턴 상한, 카탈로그 갱신 토글.
- 스트림 리듀서: `RunEvent`→ 현재 assistant 버블에 `TextDelta`/`ThinkingDelta` 누적, `ToolCallStarted`로 카드 생성, `Message`로 낙관적 상태를 저장된 메시지로 교체(tool 결과는 카드에 부착), `Usage`로 게이지 갱신, `Done/Cancelled/Error`로 종료 표시. 창 새로고침 시 `message_list` + `run_attach`.
- 툴콜 렌더: agent-k `toolCallFormat.ts` 이식(렌더와 복사가 같은 파서).
- 문자열은 `src/strings.ts` 한 파일(한국어 기본)로 모아 후속 i18n 대비.

---

## 9. cortex 변경

| 항목 | v1 처리 |
|---|---|
| `timeout_ms` 미구현 | **cortex 브랜치 `feat/exec-timeout`**: `cortex-local-console`의 `execute`에서 `tokio::time::timeout` 후 자식 kill, `Error::TIMED_OUT` 응답. ailoy `shell` 툴은 인자 `timeout_secs`를 `timeout_ms`로 전달(기본 600초). 앱은 `../cortex`가 이 브랜치인 상태를 전제로 하며 README에 명시. |
| stdout 스트리밍 없음 | v1은 "실행 중 + 경과 시간" 표시로 대체. 요구사항(server→client `exec.output` 알림, 또는 chunked 응답)을 cortex 이슈로 기록. |
| 콘솔 listing 없음 | in-process `FileSystem::list` 사용. 변경 불필요. |
| `Drop for Console`이 런타임 필요 | 엔진이 run 종료 시 명시적으로 drop(런타임 안), 종료 시 `spawn_blocking`. |

`../cortex`는 path 의존이므로 체크아웃된 브랜치가 빌드에 그대로 반영된다. 개발 중 `feat/exec-timeout`을 체크아웃하고, 병합 후 main으로 되돌린다.

---

## 10. 오류 처리

| 상황 | 동작 |
|---|---|
| FUSE-T 미설치 / 마운트 실패 | 엔진 시작은 성공하되 `WorkspaceInfo.status = Degraded{reason}`. run 시작 시 FUSE 없이 `<appdata>/files` 디렉터리 자체를 `Mount` 구현체(마운트포인트만 반환)로 콘솔에 넘기고 UI에 경고 배너("커넥터는 에이전트에게 보이지 않음"). |
| stale 마운트(비정상 종료) | 시작 시 마운트포인트가 마운트 상태이거나 비어 있지 않으면 `umount`(실패 시 `diskutil unmount force`) 후 재시도. 그래도 실패면 Degraded. |
| 콘솔 스폰 실패 | `Error{kind:"console_unavailable"}` 이벤트, 설정의 사이드카 경로 안내. |
| 커넥터 검증 실패(키·버킷 오류) | `mount_add`가 400류 오류 메시지를 그대로 반환, 마운트하지 않음. 시작 시 복원 실패는 `mount_error` 전역 이벤트 + 목록에 오류 배지. |
| 모델 오류 | `ModelError{status, retryable}`을 `Error{kind:"model", message}`로. 401은 설정 다이얼로그 열기 제안, 429 spend cap(`enforced_spend_limit_reached`)은 별도 문구. |
| 턴 상한 | `Error{kind:"max_turns"}` + "계속" 버튼(새 user 메시지 "continue" 전송). |
| 취소 | 부분 텍스트 유지, 미완 툴콜 카드는 "중단됨". |
| DB 오류 | 치명. 시작 실패 다이얼로그(파일 경로 표시). |

---

## 11. 테스트와 검증

- **ailoy 코어**: dev-deps의 `axum`으로 OpenAI 호환 가짜 서버를 띄워 (a) 취소 시점별(모델 중/툴 중) history 최종 형태, (b) 턴 상한, (c) `ToolGate` 거부, (d) 5xx 재시도, (e) rate-limit 헤더 파싱(Anthropic/OpenAI 형식)을 검증. 기존 라이브 테스트는 유지.
- **엔진**: in-memory SQLite로 store/마이그레이션, 이식한 `MessageAssembler` 테스트, 카탈로그 매핑·사용량 계산 단위 테스트. 실제 `cortex-local-console`이 필요한 run 테스트와 FUSE-T 마운트 테스트는 `#[ignore]`(`AILOY_CORTEX_BIN_DIR` 설정 시 실행).
- **앱**: `cargo check`(src-tauri), `tsc --noEmit`, 스트림 리듀서·툴콜 포맷터 vitest.
- **수동 E2E 체크리스트**(실제 키): 세션 생성 → 메시지 → `shell`로 workfs `ls` → 로컬 폴더 연결 후 그 안 파일 `cat` → Notion 연결 후 `page.json` 읽기 → 실행 중 취소 → 앱 재시작 후 대화·마운트 복원 → 컨텍스트 게이지·비용·잔여율 표시 확인.

---

## 12. 후속 로드맵 (우선순위 순)

1. 승인 UI와 정책 저장(`ToolGate` 구현체 교체, `AwaitingApproval` 이벤트 활성화, 세션/워크스페이스별 허용 규칙)
2. `mem` 메모리 툴 연결(사이드카 `mem` + `mem init`, cortex-gui의 `+ memory` UI)
3. micro-VM 콘솔 토글(`cortex-uvm-console`, 이미지·네트워크 정책 설정)
4. GDrive 커넥터(OAuth 데스크톱 플로우)
5. stdout 스트리밍(cortex 프로토콜 확장)
6. 컨텍스트 요약/컴팩션(Anthropic 서버측 compaction 포함)
7. MCP 클라이언트(ailoy `ToolProviderElem::MCP` 구현)
8. 계정 잔액·월 사용량: DeepSeek/Moonshot 잔액 API, Anthropic/OpenAI Admin 키 옵트인
9. Keychain 저장, 다중 워크스페이스, Windows/Linux, `Length` 이어쓰기, 세션 제목 자동 생성

---

## 13. 리스크와 오픈 이슈

- **FUSE-T 설치 의존**: 최종 사용자 배포 시 설치 안내가 필요. macFUSE 대안(`fuse` 피처)은 kext 승인 부담.
- **stale 마운트**: 비정상 종료 뒤 `<appdata>/workspace`가 남는 경우의 정리 루틴이 실패하면 Degraded 모드로만 동작.
- **path 의존**: `../cortex`의 체크아웃 브랜치가 빌드를 좌우. 병합 전까지 README에 요구 브랜치를 명시. 장기적으로 git rev 핀 또는 crates.io 배포로 전환.
- **팀 조율**: `mem-applied`·`cortex-gui`는 jhlee525의 진행 중 브랜치. 이식한 코드의 출처를 커밋 메시지에 남기고, `mem-applied→develop` 병합 시점을 합의한다.
- **models.dev 가용성**: 외부 서비스 중단 시 내장 스냅샷으로 폴백하므로 기능은 유지되나 신모델 정보가 늦어질 수 있다.
- **cortex 프로토콜 제약**: 콘솔당 요청 1개·스트리밍 없음은 v1에서 수용. 병렬 툴콜은 콘솔 기준으로 직렬 실행된다(모델은 병렬로 요청하지만 실행은 순차).
- **Anthropic thinking 표시**: 최신 모델은 기본 `omitted`이므로 thinking 델타가 비어 올 수 있다. 표시 옵션(`display: summarized`)은 ailoy 마샬 옵션 추가가 필요해 후속으로 둔다.
