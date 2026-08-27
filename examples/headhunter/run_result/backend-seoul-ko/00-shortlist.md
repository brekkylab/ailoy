## Picks
1. urn:li:person:k6qb3nwd — 채원 노 — Halcyon Systems에서 Rust로 이벤트 수집 게이트웨이를 직접 재작성하고 오프셋 관리·중복 제거 계층을 설계한 경험이 JD의 1번(수집·전달 계층 재작성)·2번(순서·1회 반영) 과업과 그대로 겹친다.
2. urn:li:person:w3gm8rbq — 지훈 류 — "이벤트가 순서대로, 한 번만 처리되도록"이라는 본인 요약 문구가 JD의 요구사항을 거의 그대로 반복하며, Kafka 토픽 설계와 PostgreSQL 저장 구조 재편 경험도 있다.
3. urn:li:person:n7kv4jsb — 도윤 반 — PostgreSQL에 대한 idempotent write, Kafka consumer group rebalancing, replay tooling을 현직에서 직접 소유하고 있어 우대사항(멱등성·재시도, Kafka)을 실무 근거로 갖춘 후보.

## 진행 방법

**게이트(필수 조건만)**: `headhunting search --city Seoul --skill Rust --min-years 4` — 서울, Rust 스킬, 백엔드 4년 이상만 조건으로 걸었다(우대사항인 Kafka/멱등성/대용량은 이 단계에 넣지 않음, 넣으면 그 어휘를 안 쓴 적합자를 놓친다). `distribution skill rust`로 Rust 표기 변형(rust-lang, Rust Lang, Rust (Programming Language), Async Rust)까지 다 포함되는지 먼저 확인했고, `--skill Rust`가 이 5개 표기 중 정확히 "Rust"만 잡는다는 점을 first-line에서 확인한 뒤 검색을 실행했다(52명 보유, 실제로 매칭된 건 이 스킬명 표기 그대로인 사람들).

45명이 나왔고, 그 안에서 `--mentions kafka`(우대사항)로 랭킹을 좁혔다. 이후 `--mentions 분산`, `--mentions distributed`, `--mentions idempotent`도 확인해 한국어/영어 표기 모두 어휘가 없는지 점검했다(`멱등성`, `idempotency`, `대용량`, `traffic`은 스킬 축에 없어 0건 — 어휘가 다르게 쓰였을 가능성을 `read`에서 본문으로 직접 확인).

45명 중 후보 다수를 `read`로 여러 명 동시에 읽어 직책 서술을 직접 확인했다(headline/skill만 보고 판단하지 않음).

## 비교 근거

- 채원 노(k6qb3nwd): tenure 10.0y = naive_years 10.0y (중복 없음). open to work, contact inmail. `wants`가 "Backend Engineer · Hybrid in Seoul · Full-time"으로 JD의 서울 근무와 정확히 일치. 직무 서술에 "오프셋 관리와 중복 제거 계층을 직접 설계해 재처리 상황에서도 중복이 생기지 않게 했습니다" — JD 우대사항(멱등성·재시도)의 직접 증거.
- 지훈 류(w3gm8rbq): tenure 9.7y = naive_years 9.7y. open, contact inmail. 근무지 On-site Seoul(Larkfield Networks). Kafka 토픽 설계, 컨슈머 재처리 로직, PostgreSQL 저장 구조 재편 — JD의 세 과업(수집·전달, 순서/1회 처리, PostgreSQL 재설계) 중 세 개 모두에 근거 문장이 있다.
- 도윤 반(n7kv4jsb): tenure 9.7y = naive_years 9.7y. open, contact inmail. `wants`에 "Platform Engineer · On-site in Seoul · Full-time"이 있어 서울 상근 의사 확인. 직무 서술에 "idempotent writes into PostgreSQL"과 "replay tooling"이 명시되어 우대사항과 직접 대응. 프로필 언어가 `en`이라 메일은 영어로 작성했다(이름이 한국어라도 규칙상 프로필 언어를 따름).

<!-- rejected -->

- Casey Dunmore (urn:li:person:bqsuo43r): `years`10.0 vs 이 사람은 `years` 14.0, `naive_years` 17.3 — 두 직책(Halcyon Platform과 Finlogic Platform)의 재직 기간이 겹쳐 순진하게 더하면 부풀려진다. 게다가 `not open to work`, contact가 없어(unreachable) 접촉 불가.
- Jordan Merrick (urn:li:person:h4tb9wzr): headline과 mentions kafka 검색에는 걸리지만, skill은 `Rust(7)` — endorsement가 낮고 두 직책 서술 모두 Java 서비스에 대한 내용뿐, Rust를 실제로 썼다는 문장이 어디에도 없다. headline-bait/skills-without-evidence에 해당해 "Rust 프로덕션 경험" 필수 조건을 사실상 충족하지 못한다.
- 예은 연 (urn:li:person:b2pq7x72): Rust·분산 시스템·PostgreSQL 파티셔닝까지 경력상 가장 화려해 보이지만(15.0y), `contacts`가 없어 연락할 방법이 없다(trap: no contact).
- 예준 설 (urn:li:person:ebbl1eok): 스킬에 Async Rust가 있지만 직무 서술을 보면 전부 사내 자동화 도구·Kubernetes/AWS 인프라 운영이고, "프로덕션 서비스"를 Rust로 만든 게 아니라 반복 작업 자동화 스크립트다. 게다가 contact 없음.
- 소율 함 (urn:li:person:oobyp8gk): `not open to work`이고 `last_updated_at`이 2024-01-06으로 오래된 프로필. headline은 "Python and Java"이고 Rust는 "일부 지연에 민감한 구간"에만 쓴 보조 언어라 주 사용 언어가 아니다.
- 민준 심 (urn:li:person:y5cf9pxn), 수아 강 (urn:li:person:gyu50i73), 현우 위 (urn:li:person:1hriea8p): 각각 Rust/분산 시스템 관련 서술이 있지만 `contacts`가 없어 연락 불가. 특히 수아 강과 현우 위는 Rust가 "일부 모듈만 옮기는 실험/이관" 단계로 프로덕션 소유 경험이라 보기 어렵다.
- 서연 강 (urn:li:person:k8vq3mrt): Rust 프로덕션 경험(재고 서비스, Kafka 연동, PostgreSQL 파티셔닝)이 튼튼하지만 `wants`가 두 건 모두 Remote로 명시되어 있어 "서울 성동구" 상근을 요구하는 이 자리와 근무 형태가 맞지 않을 위험이 있다. 근거리 대체 후보로만 남긴다.
