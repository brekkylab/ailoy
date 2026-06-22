# ailoy 외부 Provider 연동 기술 검토 리포트 (S3 / Notion / Google Drive)

> 작성일: 2026-06-22
> 최종 방향: **NFS 방식 (호스트 userspace NFS 서버 + 게스트 커널 NFS 클라이언트)**

## 1. 배경 / 목표

mirage + agno + Claude PoC로 외부 provider(S3, Notion, GDrive)에 대한 read/write가
에이전트에서 동작함을 검증했다. 이를 ailoy로 가져오기 위한 기술 방향을 모색하며, 핵심 제약은:

- ailoy 본체에 **`src/vfs` 모듈**로 구현 (신규 크레이트 X), mirage provider 로직을 **구조적 패리티**로 이식.
- 에이전트의 **custom tool 최소화** (가능하면 0개).
- 자격증명은 **호스트에만** 유지 (샌드박스로 유출 금지).
- **ailoy 릴리스에 플랫폼별 게스트 바이너리 같은 추가 아티팩트를 늘리지 않을 것.**
- 범위: S3(read/write 완전), Notion(read + page-create/block-append), GDrive(read + GDocs append).

## 2. 핵심 사실관계 (검증으로 확정)

### 2.1 mirage에는 두 가지 노출 모드가 있다
- **in-process VFS**: `Workspace.execute("ls /s3")` 를 in-process로 파싱·디스패치 → boto3/HTTP 직접 호출.
  agno `MirageToolkit` 이 이 모드를 사용. **FUSE 불필요** → 우리 PoC가 macOS에서 무설치로 동작한 이유.
- **FUSE 모드**: 실제 파일시스템으로 마운트. macOS에선 `macfuse` 설치 필요 (`python/mirage/fuse/README.md`).

### 2.2 ailoy 샌드박스는 진짜 Linux microVM + 진짜 셸
- microsandbox(libkrun) 기반 microVM. 에이전트는 `shell` tool(`runenv.exec_shell` → `sh -c`)로 **실제 coreutils** 실행.
- `grep`/`glob`/`shell` tool은 모두 `exec_shell` 로 **실제 바이너리를 셸 아웃**(`src/tool/impl/builtins/grep.rs` 가 `rg`/`grep` 직접 호출) → Rust 레이어에서 가로챌 수 없음.
- `read`/`write` tool만 `runenv.read/write` 직접 호출 → override 가능.
- **결론**: 진짜 셸이 `/s3` 를 보게 하려면 그 경로가 **게스트 실제 파일시스템에 존재**해야 함.

### 2.3 검증 매트릭스 (실측)

| 항목 | 결과 | 비고 |
|---|---|---|
| in-process VFS + agno tool (S3/Notion/GDrive) | ✅ read/write 전부 | macOS 무설치. Notion page-create, GDocs append 포함 |
| 호스트 mirage FUSE 마운트 (macFUSE) | ✅ 동작 | M4 Pro: Recovery에서 Reduced Security + kext 승인 필요 |
| FUSE 마운트 위에서 agno(ShellTools)로 read/write | ✅ S3 write+readback, 3 provider read | 순수 shell 명령만으로 동작 (custom tool 0) |
| **방식 A**: 호스트 FUSE → microsandbox `Bind` 로 게스트 주입 | ❌ **실패** | `mount mnt_...: Operation not permitted (os error 1)` — libkrun 공유가 macFUSE 마운트를 통과 못함 |
| 대조: 일반 호스트 디렉토리 `Bind` | ✅ read/write 양방향 | bind 자체는 정상 |
| **방식 B**: 게스트 내부 FUSE (`/dev/fuse`) | ✅ **완전 동작** | 커널 6.12 FUSE 내장, root, `fuse3`+`bindfs` 마운트/read/write/round-trip 성공 |

> msb는 ailoy 의존성(microsandbox 0.5.6)에 맞춰 **0.5.6으로 핀 설치**한 상태에서 검증.

## 3. 노출 방식 옵션 전체 비교

진짜 셸 투명성(에이전트가 기존 `shell`/`grep`/`read` tool로 `/s3` 사용)을 얻으려면 가상 경로가
게스트 실제 FS에 있어야 한다. 이를 만족하는 방식과 그 외 폴백:

| 옵션 | 투명 셸 | 게스트측 요구 | 호스트측 요구 | 자격증명 | 릴리스 아티팩트 | Notion | 검증 |
|---|---|---|---|---|---|---|---|
| **A. 호스트 FUSE + bind** | ✅ 0 tool | 없음 | 호스트 FUSE 마운트 | 호스트 ✅ | 없음 | 가능 | ❌ **불가**(2.3) |
| **B1. 게스트 내부 FUSE (자급식)** | ✅ 0 tool | 정적 FUSE 바이너리 | 없음 | **게스트(유출)** | 게스트 바이너리 | 가능 | ✅ 전제 검증됨 |
| **B2. 게스트 FUSE + 호스트 RPC** | ✅ 0 tool | 정적 FUSE 바이너리 | `src/vfs` 데몬 | 호스트 ✅ | 게스트 바이너리 | 가능 | 전제 검증됨 |
| **C. NFS (호스트 서버 + 게스트 커널 mount)** | ✅ 0 tool | stock `mount`(bash) | NFS 서버 어댑터 | 호스트 ✅ | **없음** | 가능 | 스파이크 예정 |
| D. rclone serve nfs + 게스트 mount | ✅ 0 tool | stock `mount` | rclone 설정 | 호스트 ✅ | 없음 | ❌ 백엔드 없음 | — |
| E. in-process `vfs` tool | ❌ 1 tool | 없음 | tool(=lib 일부) | 호스트 ✅ | 가능 | ✅ (agno로 검증) |

### 방식별 핵심 메모
- **A 폐기**: libkrun의 호스트 디렉토리 공유가 macFUSE(userspace) 마운트를 통과하지 못해 게스트 마운트가 실패. macOS에서 구조적으로 불가.
- **B1/B2**: 게스트 내부 FUSE는 실증됐으나, FUSE 엔진은 `/dev/fuse` 를 말하는 **정적 바이너리(C-free Rust + root `mount(2)`)** 가 게스트에 있어야 함 → ailoy 릴리스에 linux aarch64/x86_64 바이너리 추가 부담. B1은 자격증명 유출까지.
- **D**: S3/GDrive는 무료지만 **Notion 백엔드 부재**로 탈락.
- **E**: 가장 단순(인프라 0)하나 투명 셸 아님(tool 1개). 폴백.

## 4. 성능 검토 (NFS vs FUSE)

read/write의 지배적 비용은 **provider API 왕복(~50–300ms; 측정치 S3 stat~100ms, read~95–280ms)**.
로컬 전송 오버헤드(FUSE ~10–50µs / NFS-virtio ~100µs–1ms)는 그 위의 **0.01~1% 잔차**.

- cloud-backed I/O에선 FUSE/NFS 체감 차이 **사실상 0** (둘 다 provider-bound).
- 메타데이터 폭주(`ls -la`, `grep -r`)에선 **캐싱 전략**이 체감을 좌우. NFS 커널 클라이언트의 attr 캐시/readahead/write-back이 오히려 유리할 수 있음(단 staleness 주의).
- **자격증명 호스트 유지 전제(B2/NFS)** 에선 둘 다 게스트↔호스트 hop 1회로 전송 비용 동급. NFS는 chatter가 많은 대신 커널 캐싱 성숙.
- 결론: **성능은 결정 기준이 아님.** `actimeo`, `rsize/wsize`(NFS) / attr timeout(FUSE) 같은 캐싱 튜닝이 실제 체감을 좌우.

## 5. 최종 결정: NFS 방식 (옵션 C)

```
┌─ Host (ailoy 단일 빌드) ──────────────┐         ┌─ Guest microVM ─────────┐
│  src/vfs ─ S3/Notion/GDrive (creds)   │  NFS    │  mount -t nfs 127.0.0.1 │
│  └ userspace NFS 서버 (포워드 포트)   │ ◄─────► │   → /s3 /notion /gdrive │
└───────────────────────────────────────┘  (TCP)  │  기존 shell tool 그대로 │
                                                    └─────────────────────────┘
```

### 선정 이유
- **모든 제약 동시 충족**: custom tool 0개 / 자격증명 호스트 유지 / **게스트 바이너리·릴리스 아티팩트 0** / 임의 base image(커널 NFS + stock `mount`, bash 런처) / Notion 지원(서버를 우리가 구현).
- 커스텀 코드가 **호스트 ailoy 단일 빌드** 안에 집중. 게스트는 stock `mount -t nfs` 뿐.
- 성능은 provider-bound라 FUSE 대비 손해 없음.

### 구현 개요
- Rust userspace **NFSv3 서버**(`nfsserve` 류 크레이트) 트레이트를 `src/vfs` 의 `Resource` 위에 노출. read/write/readdir/lookup/getattr 등이 거의 1:1 매핑.
- 게스트↔호스트 채널: 검증된 microsandbox `-p` 포트포워딩으로 NFS 포트 연결.
- 부팅 훅(`Sandbox::new` 의 `inner.shell(...)` 자리)에서 게스트 `mount -t nfs` 자동 실행.
- 도메인 쓰기(Notion page-create 등)는 `/<mount>/.cmd/<op>` 컨트롤 경로 write → 서버가 해당 core 함수로 라우팅.

## 6. mirage에서 참고할 부분 (NFS 경로 기준)

NFS·FUSE 모두 "파일시스템 op 서버"라 참고 범위가 동일하며, mirage의 가장 큰 덩어리는 스킵된다.

### 참고가치 높음 — `src/vfs` 로 구조적 패리티 이식
- `resource/base.py` — Resource 인터페이스(read/write/readdir/stat/glob/fingerprint).
- `core/s3/*`, `core/notion/{pages,blocks,pathing,normalize}.py`, `core/gdocs/write.py`, `core/google/config.py`
  — provider I/O + 경로 매핑 노하우(S3 key↔경로, Notion `<title>__<id>` 트리·`page.json`, `.gdoc.json` 렌더링).
- **`fuse/fs.py` (MirageFS)** — 파일시스템 콜백 → resource op **변환 + errno 매핑**. NFS 서버 어댑터의 **가장 직접적 레퍼런스**. op 단위로 정독 권장.
- `cache/` (file cache, dir index, TTL, post-write invalidation) + `fingerprint()` — readdir 시 매번 S3 re-list 방지.
- 각 resource의 `PROMPT`/`WRITE_PROMPT` — system prompt mount 섹션.

### 참고 불필요 — NFS의 이득
- `commands/builtin/**` (제너릭 + 백엔드 래퍼, ~7만 줄): 게스트의 **실제 coreutils** 가 처리.
- `shell/`(tree-sitter 파서), `workspace/node/command_dispatch.py`: in-process tool 경로 전용.

### 매핑 차이 — 도메인 쓰기
- API 호출 로직(`core/notion/pages.py`·`blocks.py`, `core/gdocs/write.py`)은 그대로 재사용.
- 호출 표면만 셸 명령 → `/.cmd/` 컨트롤 파일 write로 변경.

## 7. 다음 단계

1. **스파이크**: 게스트에서 stock `mount -t nfs` 로 호스트 userspace NFS 서버(`nfsserve` 더미) 마운트 →
   `ls`/`cat`/`tee` round-trip 확인 (포트포워딩 경유). 이게 통과하면 "0-tool 투명 + 바이너리 0" 확정.
2. `src/vfs` 코어 + `Resource` + **S3 어댑터** 이식 → NFS 서버로 S3 노출 E2E.
3. Notion + GDrive 어댑터 + `/.cmd/` 도메인 연산.
4. ailoy 통합: 부팅 훅 자동 마운트 + `.vfs(VfsConfig)` 빌더 + system prompt mount 섹션 + 캐싱 튜닝(`actimeo`/`rsize`).
5. 폴백 비상구: NFS 서버 구현이 과하면 **옵션 E(in-process `vfs` tool, 1개)** 로 후퇴 가능 (이미 검증됨).

## 부록 A. 검증 산출물
- `mirage/examples/python/agents/agno/agno_multi_poc.py` — in-process VFS, 3 provider read/write (Claude).
- `mirage/examples/python/agents/agno/agno_fuse_poc.py` — 호스트 FUSE 마운트 + ShellTools(순수 shell)로 read/write.
- `mirage/examples/python/agents/agno/{notion_only,gdrive_only}.py` — provider별 격리 검증.

## 부록 B. macOS(M4 Pro) macFUSE 활성화 메모
호스트 FUSE 로컬 검증용. 게스트 NFS 방식에는 **불필요**(게스트 커널 NFS만 사용).
1. Recovery: 전원 길게 → Options → Startup Security Utility → **Reduced Security** + **"Allow user management of kernel extensions"**.
2. System Settings → Privacy & Security → 'Benjamin Fleischer' 소프트웨어 **Allow**.
3. 재부팅 → `kmutil showloaded | grep -i fuse` 로 로드 확인.
