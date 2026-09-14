# Ailoy Desktop

macOS 데스크톱 앱. Tauri 2 셸(`src-tauri`) + React 웹뷰(`src`)이고, 실제 일은
`apps/desktop/core`의 엔진이 한다. 엔진 쪽 설계와 테스트 규칙은
[`docs/desktop-development.md`](../../docs/desktop-development.md)에 있다.

## 준비물

- **macOS.** v1은 macOS 전용이다. 워크스페이스 마운트가 FUSE-T에 묶여 있다.
- **FUSE-T** — `brew install --cask fuse-t`. 없으면 앱은 뜨지만 워크스페이스가
  `degraded`가 되고 커넥터가 에이전트에게 보이지 않는다.
- **Rust ≥ 1.95** (워크스페이스 `rust-version`), **Node ≥ 22**.
- **`../cortex` 체크아웃.** `cortex`를 path 의존으로 쓰기 때문에 이 저장소의 형제
  디렉터리에 있어야 하고, 브랜치는 `feat/exec-timeout`, 커밋은 `3dd05ef0` 이상이어야
  한다. 그 앞 커밋에서는 shell 툴의 `timeout_secs`가 무시된다.

## 실행

```sh
cd apps/desktop
npm install
npm run tauri:dev
```

`tauri dev`를 직접 부르지 말 것. 사이드카(`cortex-local-console`)를 먼저 빌드해
`src-tauri/binaries/`에 넣어야 하는데 그 일을 `npm run tauri:dev`가 한다
(`scripts/build-sidecar.sh`). `../cortex`가 다른 곳에 있으면 `CORTEX_DIR`로 알려준다.

`.app` 번들은 `npm run tauri:build` — 결과는
`src-tauri/target/release/bundle/macos/Ailoy.app`.

첫 실행에는 키가 없다. **설정**에서 쓰려는 프로바이더의 API 키를 입력해야 모델 목록이
`available`이 되고 대화를 시작할 수 있다. 키는 엔진이 보관하고 웹뷰로 돌려주지 않는다.

## 데이터

전부 `~/Library/Application Support/com.brekkylab.ailoy/` 아래에 있다.

| 경로            | 내용                                                      |
| --------------- | --------------------------------------------------------- |
| `ailoy.sqlite`  | 대화·메시지·설정·마운트. 키도 여기에 있다 (파일 권한 0600) |
| `files/`        | 워크스페이스의 실제 파일                                   |
| `workspace/`    | FUSE-T 마운트포인트. 에이전트가 보는 경로                  |
| `cache/`        | 모델 카탈로그 캐시                                         |
| `logs/`         | `ailoy.log.<날짜>` (일 단위 롤링)                          |
| `engine.lock`   | 한 데이터 디렉터리에 한 인스턴스만 허용하는 잠금           |

지우면 처음 상태로 돌아간다.

## 로그

기본 레벨은 `info`, `RUST_LOG`로 바꾼다 (`RUST_LOG=ailoy_desktop_core=debug npm run
tauri:dev`). 파일은 위의 `logs/ailoy.log.<날짜>`이고, 개발 중에는 stderr로도 나온다.
앱에서는 **설정 → 로그 폴더 열기**가 그 디렉터리를 Finder로 연다.

## 테스트

```sh
cd apps/desktop && npm test          # 웹뷰 (vitest)
cargo test -p ailoy-desktop-core     # 엔진
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml   # Tauri 커맨드
```

저장소 루트에서 `cargo test`를 인자 없이 돌리지 말 것. 루트 크레이트에는 `.env`의 키로
실제 API를 호출하는 테스트가 있다.

실물이 필요한 테스트 넷은 `#[ignore]`다 — FUSE-T가 필요한 `live_workspace`, 빌드된
콘솔이 필요한 `live_console`과 `live_run`(둘).

```sh
cargo build --manifest-path ../cortex/Cargo.toml -p cortex-local-console
AILOY_CORTEX_BIN_DIR=$PWD/../cortex/target/debug \
  cargo test -p ailoy-desktop-core --test live_run -- --ignored
```

## 알려진 제한

- **macOS 전용.**
- **승인 UI가 없다.** 툴은 곧바로 실행된다. 엔진은 `awaiting_approval` 이벤트를 알지만
  v1의 웹뷰는 그것을 물어보지 않는다.
- **툴 실행 중 stdout 스트리밍이 없다.** 툴카드는 호출과 최종 결과를 보여줄 뿐,
  진행 중인 출력을 흘리지 않는다.
- **서명하지 않은 번들.** 다른 기계에서는 Gatekeeper를 우회해서 열어야 한다.
- **데이터 디렉터리당 한 인스턴스.** 두 번째 실행은 잠금을 얻지 못하고 오류 대화상자와
  함께 종료한다.
- **Notion·S3 커넥터는 읽기 전용.** 로컬 폴더 연결만 쓰기를 지원한다.
