# Remote run 작업 플랜

작업하는 동안 관리하는 문서입니다. 커밋하지 않습니다.

## 확정된 전제

- Spec은 이름만 담고, 구현(모델·도구·키)은 서버의 provider가 준비합니다. 서버에 없는 이름은 거절됩니다.
- ailoy 서버는 virtx 서버와 별개이고, virtx 클라이언트로 sandbox를 다룹니다. virtx 서버 코드를 상속하지 않습니다.
- 턴 실패 시 history 동작은 그대로 둡니다(첫 메시지 전 실패만 query를 뺌). 서버는 메시지가 history에 들어갈 때마다 state를 저장합니다.
- sandbox snapshot은 0.4.0 범위 밖입니다.
- 폴더는 virtx처럼 root에 crate별로 둡니다.

## 개념과 타입

| 개념 | 지금 | 바뀌는 점 |
| --- | --- | --- |
| `AgentSpec` | 이름만 담은 serde 타입 | `use_sandbox: bool` → `sandbox: Option<SandboxCreateArgs>` |
| `AgentProvider` | 없음 (프로세스 전역 registry) | 모델·도구 registry, virtx 클라이언트, 저장 위치를 담는 값. 전역 `set_*_provider`는 기본 인스턴스에 등록 |
| `AgentState` | history + sandbox 핸들 + browser | 저장 형태는 `id`, history, `context_tokens`, sandbox id. 핸들은 실행 중에만 provider가 붙임 |
| Run | 없음 (스트림만) | 한 번의 턴 실행. id, 상태, 이벤트 로그, 요약 |

진입점은 virtx의 `Client::spawn`/`connect`처럼 하나입니다.

```rust
let provider = AgentProvider::local();              // 이 프로세스의 registry
let provider = AgentProvider::spawn("ailoy-stdio")?;
let provider = AgentProvider::connect(url, auth)?;

provider.deploy("image-editor", spec).await?;
let run = provider.run("image-editor", query, Some(state_id));  // 이벤트 스트림
```

기존 `Agent`/`AgentBuilder`의 run loop는 ailoy-runner로 가고, crate `ailoy`의 공개 API는 클라이언트 위에서 유지합니다(Local의 형태는 미정 2번).

## 폴더 구조

root는 패키지 없는 workspace가 됩니다.

```
ailoy-core/           virtx-core 자리. 타입과 프로토콜. 만들어짐: message, datatype, AgentSpec,
                      ToolDesc, LangModelOptions.
                      예정: state 데이터, run 이벤트, 프로토콜의 args/outputs,
                      wire의 client 끝(feature `client`)과 server 끝(feature `server`)
ailoy-runner/         virtx-uvm 자리. 지금의 src/ 대부분: run loop, 모델, 도구(내장 도구의 desc 포함),
                      브라우저, AgentProvider, 그것을 서빙하는 router. 두 서버와 바인딩이 함께 의존
ailoy-rs/             virtx-rs 자리. crate `ailoy`: ailoy-core의 client 위의 클라이언트 API.
                      ailoy-core 타입을 re-export
ailoy-server-stdio/   ailoy-runner의 router를 stdio로 서빙하는 `ailoy-stdio` 바이너리
ailoy-server-tcp/     같은 것을 TCP로, 인증 포함
ailoy-py/             지금의 bindings/python
ailoy-node/           지금의 bindings/node
examples/ docs/ scripts/ skills/   그대로
```

- 이동은 한 번에 합니다.
- 함께 고칠 곳: root `Cargo.toml`의 `[[example]]`, `include`, `.github/workflows/packages.yml`, `scripts/packages/`, `.githooks/`, `AGENTS.md`·`CONTRIBUTING.md`·docs의 경로.
- router는 ailoy-runner의 라이브러리에 둡니다. 바인딩에서 `serve`를 불러야 함수 도구를 쓰는 팀이 Remote로 배포할 수 있습니다.

## Transport

별도 저장소의 crate `http-over-x`(github.com/ailoy-co/http-over-x)를 씁니다. 장기적으로 virtx-core도 이것으로 옮깁니다.

- 주는 것: `Client`(spawn·connect·over, 재시도, 인증), JSON 요청과 거절(`Error::Status`), SSE 로그의 양 끝(`sse::Log`, `sse::log`), `serve_stdio`·`serve_tcp`·`serve_tls`, `from_body`
- 없는 것: WebDAV, port forward (virtx 전용)
- 의존 방식: 개발 중에는 git `rev`, 릴리스 때는 crates.io에 먼저 publish

## 프로토콜

run = job + SSE 로그 (virtx의 exec와 같은 모델).

| 엔드포인트 | 역할 |
| --- | --- |
| `GET /tools`, `GET /models` | 이 provider가 가진 도구(`ToolDesc`)와 모델 org |
| `GET /agents/`, `PUT·GET·DELETE /agents/{name}` | 목록, `{spec, model_key_provider}` 배포, 조회, 제거. 배포 시 spec의 이름과 `model_key_provider`를 provider에 대해 검증 |
| `POST /agents/{name}/runs` | `{query, state?}`로 실행 시작. state가 없으면 새 id 발급 |
| `GET /runs/{id}`, `GET /runs/{id}/events` | 요약(`running` 포함)과 이벤트 스트림. 마지막 항목은 `end`. `Last-Event-ID`로 이어 읽기 |
| `POST /runs/{id}/cancel` | 중단 |
| `GET·DELETE /states/{id}` | history와 sandbox id 조회, 제거(sandbox도 함께) |
| `GET /agents/{name}/runs` | 실행 요약 목록. 지표는 이것을 모아 계산 |

## 저장

Provider의 디렉터리에 파일로 둡니다. 조회 API는 파일 구조를 드러내지 않습니다.

```
agents/<name>/spec.json
states/<id>/state.json            # 메시지마다 갱신
runs/<id>/summary.json            # agent, state, 시작·종료, 결과, 토큰
runs/<id>/events.jsonl            # 모델·도구 호출의 시작·종료, 오류
```

## 미정

1. **호출자의 파일 접근**: ailoy 서버가 띄운 virtx-stdio에는 호출자가 닿을 수 없습니다. 안: `/states/{id}/files/...`를 virtx 파일 API로 넘기는 중계. 대안: 메시지에 실어 보내는 것만 허용.
2. **Local의 형태**: crate `ailoy`는 클라이언트라 in-process 실행을 직접 갖지 않음. 초안: `ailoy`의 `local` feature가 같은 프로세스에서 ailoy-runner를 띄우고 `Client::over`(메모리 파이프)로 연결(함수 도구를 그대로 씀). 대안: virtx처럼 항상 `ailoy-stdio`를 띄움(호출자 프로세스의 함수 도구는 못 씀).
3. **spec 배포 방식**: 초안은 이름으로 배포 후 실행. 대안: 요청마다 spec 전송.
4. **remote의 browser**: `Browser::user()`는 호출자 쪽 브라우저라 remote에서 거절. isolated/headless만 허용할지.
5. **호출자의 모델 키**: 배포의 `model_key_provider`(`server` 기본, `client`)가 누가 키를 준비하는지 정함. 모델을 부르는 쪽은 항상 서버. `client`일 때 호출자가 run 요청에 키를 싣는 방식(헤더 안)과, 서버가 `client`를 받지 않을 때의 거절이 미정. 키는 spec·state·로그에 남기지 않음.
