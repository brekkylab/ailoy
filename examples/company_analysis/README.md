# Company Analysis Example

기업정보 데이터 레이크를 파일시스템으로 들고, 유저의 쿼리를 조사·분석해 리포트로 답하는 에이전트 예제.

```
질문 (자연어)
    │
    ▼
[ company analysis agent ] ──shell / grep / glob / read──▶ data/   (읽기 전용)
    │                       └─python_repl (duckdb, pandas)─┘
    │
    └──write──▶ artifacts/<slug>/report.md
                artifacts/<slug>/evidence.md
                artifacts/<slug>/queries/*.sql|*.py
```

---

## 1. 시나리오

국내·해외 기업정보를 모아둔 데이터 레이크가 로컬 디스크에 있다. 사용자는 자연어로 질문을 던지고, 에이전트는 데이터를 탐색해 근거와 함께 답한다.

대표 질문 4종 (§7의 프리셋에 대응):

1. **기업 분석** — "Acme Materials의 최근 3년 실적과 사업 구조를 분석해줘."
2. **공급망 리스크** — "우리 배터리 라인의 2차 협력사 중 제재·지정학 리스크가 있는 곳은?"
3. **기업 조사** — "Nova Chem Ltd.와 거래하기 전에 알아야 할 것 (실소유주, 제재, 소송, 관계사)."
4. **자동화** — "watchlist 30개사에 대해 지난주 변경사항을 요약해줘."

에이전트는 `data/`에 대해 **읽기 전용**이다. 쓰기는 `artifacts/`(산출물)와 `workspace/`(중간 계산)에만 허용한다.

---

## 2. 범위

**포함**

- 국내·해외 기업정보를 흉내 낸 합성 데이터 레이크 (`data/`)
- 자유 형식 쿼리 실행 환경: `shell` + `read`/`grep`/`glob` + `python_repl`(duckdb/pandas)
- 데이터 지도(`data/CATALOG.md`) 기반의 탐색 → 쿼리 → 검증 → 리포트 루프
- 근거 추적: 리포트의 모든 수치가 파일·쿼리로 역추적 가능

**미포함 (후속)**

- 실제 DART / SEC EDGAR / GLEIF / OFAC API 연동 및 수집 파이프라인
- 실시간 갱신, 증분 적재, 데이터 품질 모니터링
- 임베딩 기반 문서 검색 (현재 `ailoy` 크레이트에 임베딩 API 없음 — 공시 원문은 grep/키워드로 접근)
- 웹 UI, 리포트 배포, 알림 전송

---

## 3. 데이터: 파일시스템 데이터 레이크

### 3.1 설계 원칙

- **텍스트 우선** — jsonl / csv / md 만 사용한다. grep과 duckdb 양쪽에서 바로 읽히고, diff와 리뷰가 가능하다. parquet은 쓰지 않는다.
- **한 파일 = 한 테이블** (또는 한 문서). 파일명만 보고 내용이 짐작되어야 한다.
- **모든 레코드에 `source`와 `as_of`** — 어디서 온 값이고 언제 기준인지 없는 행은 만들지 않는다.
- **조인 키는 명시적으로** — 회사 식별자는 §3.3의 `company_id` 하나로 통일하고, 원천 식별자(사업자번호, CIK, LEI 등)는 별도 매핑 테이블에 둔다.

### 3.2 디렉터리 구조

```
data/
  CATALOG.md              # 데이터 지도. 에이전트가 가장 먼저 읽는 파일
  registry/
    companies.jsonl       # 기업 마스터
    identifiers.csv       # company_id ↔ 원천 식별자 매핑
    aliases.csv           # 상호 표기 변형 (한글/영문/약칭/구 상호)
    ownership.csv         # 지분 관계 (모회사-자회사, 지분율)
  financials/
    kr/<company_id>/fs_<year>.csv     # 요약 재무제표 (K-IFRS)
    us/<company_id>/fs_<year>.csv     # 요약 재무제표 (US GAAP)
  filings/
    kr/<company_id>/<date>-<type>.md  # 공시 원문 (요약본)
    us/<company_id>/<date>-<form>.md  # 10-K / 8-K 등 발췌
  supplychain/
    edges.csv             # 거래 관계 (buyer ← supplier)
    sites.csv             # 생산·물류 거점
  trade/
    customs_kr.csv        # 수출입 신고 요약 (HS 코드 기준)
    bol_us.csv            # 선하증권 기반 수입 기록
  risk/
    sanctions.csv         # 제재 대상 목록 (OFAC SDN / EU 통합 리스트 형식)
    litigation.jsonl      # 소송·분쟁 이력
    incidents.jsonl       # 사고·리콜·환경/노동 이슈
    credit.csv            # 신용등급 / 부실 징후 지표
  news/
    <YYYY>/<MM>/<date>-<slug>.md      # 기사 (제목/매체/일자/본문 요약)
  reference/
    hs_codes.csv, ksic.csv, naics.csv, countries.csv, fx_rates.csv
  watchlist.csv           # 자동화 시나리오용 모니터링 대상
```

규모 기준: 기업 200~300개(국내 60%, 해외 40%), 공급망 엣지 800~1,200개, 뉴스 150건, 공시 80건. **3~4단계 공급망 추적이 가능해야 하고**, 제재 대상이 2차·3차 협력사에 숨어 있는 케이스를 최소 두 건 심는다.

### 3.3 핵심 스키마

`registry/companies.jsonl`

```jsonc
{
  "company_id": "kr-acme-materials",     // 전 데이터셋 공통 조인 키
  "legal_name": "주식회사 에이크메머티리얼즈",
  "legal_name_en": "Acme Materials Co., Ltd.",
  "country": "KR",
  "status": "ACTIVE",                    // ACTIVE | DISSOLVED | MERGED
  "incorporated_on": "2004-06-11",
  "industry": { "ksic": "20119", "naics": "325180" },
  "listed": { "market": "KOSDAQ", "ticker": "123456" },  // 비상장이면 null
  "employees": 412,
  "hq": { "country": "KR", "region": "충청북도", "city": "청주시" },
  "website": "https://example.com",
  "source": "dart-mock",
  "as_of": "2026-06-30"
}
```

`registry/identifiers.csv` — `company_id, scheme, value, source, as_of`
`scheme` ∈ `brn`(사업자등록번호) | `corp_no`(법인등록번호) | `cik`(SEC) | `lei`(GLEIF) | `duns` | `ticker`

`supplychain/edges.csv`

```
buyer_id,supplier_id,tier,hs_code,item,share_pct,since,source,as_of,confidence
kr-acme-materials,cn-hanjiang-chem,1,290369,전해액 첨가제,34.0,2021-03,customs+bol,2026-06-30,0.82
```

- `tier`는 **기준 기업으로부터의 거리**가 아니라 **이 엣지의 관측 근거 수준**을 뜻하지 않도록 주의한다. 여기서는 "buyer 기준 직접 거래 = 1"로 고정하고, n차 추적은 엣지를 재귀 조인해서 계산한다.
- `confidence`는 0~1. 관세 신고와 선하증권 양쪽에서 관측되면 높고, 뉴스 언급 1건뿐이면 낮다. **리포트는 이 값을 반드시 함께 표기한다.**

`risk/sanctions.csv` — `list_name, entity_name, aliases, country, program, listed_on, source` (OFAC SDN 형식 차용). **`company_id`가 미리 매핑되어 있지 않다.** 이름 매칭은 에이전트의 일이다(§5.4).

### 3.4 `data/CATALOG.md`

에이전트가 데이터를 헤매지 않도록 하는 진입점. 사람이 손으로 관리한다.

- 각 파일의 경로, 한 줄 설명, 행 수, 컬럼 목록과 타입
- 조인 키와 조인 예시 (SQL 3~4개)
- 알려진 한계: 커버리지 구멍, 통화 단위, 회계기준 차이, 갱신 시점
- **자주 하는 실수** 항목 (예: `edges.csv`의 `share_pct`는 buyer 기준 조달 비중이지 supplier 기준 매출 비중이 아님)

---

## 4. 입력: 자유 형식 질문

CLI 인자 또는 파일로 받는다.

```sh
--task "Acme Materials의 중국 의존도를 2차 협력사까지 따져줘"
--task-file ./tasks/supply-chain.md
--preset supply-chain-risk --company "Acme Materials"
```

프리셋은 §7의 4종에 대응하며, 내부적으로는 잘 쓰인 task 문장 + 산출물 템플릿 지정일 뿐이다. 프리셋 없이 임의의 질문을 던져도 동작해야 한다.

---

## 5. 에이전트

### 5.1 환경 = 파일시스템 + 쿼리 런타임

전용 툴을 만들지 않고 **built-in 툴**을 쓴다. 질문의 형태를 미리 정할 수 없기 때문에, 스키마를 고정한 툴은 오히려 표현력을 깎는다.

| 툴 | 용도 |
| --- | --- |
| `read` | `CATALOG.md`, 공시·뉴스 원문, csv 헤더 확인 |
| `glob` | 파일 존재 여부와 커버리지 확인 (`filings/us/*/2026-*-8-K.md`) |
| `grep` | 원문 문서 검색, 상호·키워드 스캔 |
| `shell` | 빠른 집계 (`wc -l`, `cut`, `sort | uniq -c`), 파일 크기 확인 |
| `python_repl` | 본 계산. duckdb로 csv/jsonl에 SQL을 직접 실행, pandas로 후처리 |

`python_repl`은 `pip_install`을 지원하므로 첫 호출에서 `duckdb`를 설치한다. duckdb는 `read_csv_auto` / `read_json_auto`로 파일을 그대로 테이블처럼 다루므로, 별도 적재 단계가 필요 없다.

```python
import duckdb
con = duckdb.connect()
con.sql("""
  SELECT s.legal_name, e.item, e.share_pct, e.confidence
  FROM read_csv_auto('data/supplychain/edges.csv') e
  JOIN read_json_auto('data/registry/companies.jsonl') s ON s.company_id = e.supplier_id
  WHERE e.buyer_id = 'kr-acme-materials'
  ORDER BY e.share_pct DESC
""").show()
```

### 5.2 쓰기 경계

- `data/` — 읽기 전용. 수정·삭제 금지를 인스트럭션에 명시하고, 실행 전후로 데이터 디렉터리 해시를 비교해 위반을 검출한다.
- `workspace/<run-slug>/` — 중간 산출물(추출한 csv, 그래프 계산 결과). 실행마다 비운다.
- `artifacts/<run-slug>/` — 최종 산출물.

### 5.3 루프

1. `data/CATALOG.md`를 읽어 무엇이 있는지 파악한다.
2. 질문을 답변 가능한 하위 질문으로 분해하고, 각각에 필요한 파일을 지목한다.
3. 대상 기업을 식별한다 (§5.4).
4. 하위 질문별로 쿼리를 작성·실행한다. 실패하면 스키마를 다시 확인하고 고친다.
5. 핵심 수치는 **다른 경로로 한 번 더 검증한다** (예: 공급망 비중을 `edges.csv`와 `trade/customs_kr.csv` 양쪽에서).
6. 리포트, 근거 목록, 실행한 쿼리를 `artifacts/`에 기록한다.

### 5.4 기업 식별(entity resolution)

이 데이터셋의 가장 큰 함정. 별도 요구사항으로 둔다.

- 입력은 사람이 부르는 이름("에이크메", "Acme Materials", "ACME MATERIALS CO LTD")이고, 데이터의 키는 `company_id`다. `registry/aliases.csv`로 먼저 해소한다.
- 법인격 접미사(`(주)`, `주식회사`, `Co., Ltd.`, `Inc.`, `GmbH`)와 대소문자·공백은 정규화 후 비교한다.
- **동명이인 케이스를 반드시 사람에게 되묻는다.** 후보가 둘 이상이면 임의로 고르지 않고, 국가·업종·설립연도를 제시해 확인을 요청하거나 리포트 상단에 모호성을 명시한다.
- 제재 리스트 매칭은 이름 유사도 기반이므로 **후보 제시**까지만 한다. "제재 대상이다"라고 단정하지 않고, `가능성 있는 일치 — 확인 필요`로 표기하고 근거 행을 인용한다. 오탐의 비용이 큰 영역이다.

### 5.5 분석 품질 요구사항

- **모든 수치에 출처.** 파일 경로 + 필터 조건, 또는 실행한 쿼리 파일명을 함께 적는다.
- **데이터에 없으면 "없다"고 쓴다.** 추정할 경우 추정임을 명시하고 계산 과정을 남긴다. 일반 상식으로 기업 정보를 채워 넣지 않는다.
- **기준일을 항상 표기.** 서로 다른 `as_of`의 값을 한 표에 섞을 때는 열을 나눈다.
- **통화·회계기준을 섞지 않는다.** KRW/USD 환산은 `reference/fx_rates.csv`의 명시된 기준일 환율로만 하고, K-IFRS와 US GAAP 수치를 직접 비교하지 않는다.
- **`confidence`가 낮은 엣지에 기반한 결론은 그 사실을 함께 쓴다.**
- 리스크 점수를 낼 경우 **가중치와 계산식을 리포트에 노출한다.** 블랙박스 점수는 금지.

---

## 6. 출력: `artifacts/`

```
artifacts/
  2026-08-16-acme-supply-chain-risk/
    report.md          # 사람이 읽는 최종 리포트
    evidence.md        # 주장 ↔ 근거(파일/쿼리) 대응표
    findings.json      # 기계가 읽는 구조화 결과 (자동화 시나리오용)
    queries/
      01-direct-suppliers.sql
      02-tier2-expansion.py
      03-sanctions-match.py
```

`report.md` 공통 골격:

- **요약** — 질문에 대한 3~5줄 답변. 여기서 결론이 끝나야 한다.
- **핵심 발견** — 항목별로 사실 + 근거 + 신뢰도
- **분석 본문** — 프리셋별 구성 (§7)
- **데이터 한계** — 커버리지 구멍, 오래된 기준일, 낮은 confidence, 미해소 동명이인
- **다음 단계** — 사람이 확인해야 할 것, 추가로 필요한 데이터

`findings.json`은 자동화 시나리오에서 실행 간 비교(diff)를 하기 위한 것이므로 스키마를 고정한다: `run_id`, `task`, `entities[]`, `findings[] {severity, statement, evidence[], confidence}`, `data_gaps[]`.

---

## 7. 프리셋 시나리오

| 프리셋 | 질문 | 리포트 본문 구성 |
| --- | --- | --- |
| `company-profile` | 기업 분석 | 개요·지배구조 / 재무 추이 3년 / 사업·제품 / 주요 거래처 / 최근 공시·뉴스 / 강점·약점 |
| `supply-chain-risk` | 공급망 리스크 | n차 협력사 전개 / 집중도(단일 공급처·국가 편중) / 제재·지정학 노출 / 대체 공급처 후보 / 시나리오별 영향 |
| `due-diligence` | 기업 조사 | 실체 확인 / 지분·실소유 구조 / 제재·소송·사고 이력 / 재무 건전성 / 관계사 네트워크 / 레드 플래그 |
| `watchlist-monitor` | 자동화 | watchlist 대상별 지난 실행 이후 변화 / 신규 리스크 / 변화 없음 항목은 한 줄로 |

`watchlist-monitor`는 이전 실행의 `findings.json`을 입력으로 받아 **변화만** 보고한다. 매번 전체를 다시 쓰지 않는 것이 요구사항이다.

---

## 8. 실행

```sh
export ANTHROPIC_API_KEY=...   # 또는 OPENAI_API_KEY 등

# 자유 질문
cargo run -p company_analysis -- \
  --task "Acme Materials의 중국 의존도를 2차 협력사까지 분석해줘" \
  --data ./data --out ./artifacts

# 프리셋
cargo run -p company_analysis -- \
  --preset supply-chain-risk --company "Acme Materials"

# 자동화 (이전 결과와 비교)
cargo run -p company_analysis -- \
  --preset watchlist-monitor --since ./artifacts/2026-08-09-watchlist/findings.json
```

- 모델은 `--model`로 지정, 기본값 `anthropic/claude-sonnet-4-6`.
- 진행 상황은 `cortex` 콘솔로 스트리밍한다.
- 종료 시 생성 파일 목록과 실행한 쿼리 수를 출력한다.

---

## 9. 디렉터리 구조

```
examples/company_analysis/
  README.md
  Cargo.toml
  src/
    main.rs        # CLI, 에이전트 조립, 실행
    preset.rs      # 프리셋 → task 문장 + 리포트 템플릿
    prompt.rs      # 시스템 인스트럭션 (데이터 규칙, 근거 규칙, 쓰기 경계)
    guard.rs       # data/ 무결성 검사, 산출물 경로 제한
  data/            # 합성 데이터 레이크 (커밋)
  tasks/           # 예시 질문 모음
  workspace/       # gitignore
  artifacts/       # gitignore (샘플 1건만 커밋)
```

루트 `Cargo.toml`의 `[workspace] members`에 `examples/company_analysis`를 추가한다.

---

## 10. 완료 기준

- [ ] API 키만 있으면 `cargo run -p company_analysis`가 동작한다
- [ ] 프리셋 4종이 각각 `report.md` + `evidence.md` + `findings.json`을 생성한다
- [ ] 리포트의 모든 수치가 `evidence.md`를 통해 파일·쿼리로 역추적된다 (수동 검증)
- [ ] 데이터에 없는 기업·거래·제재를 지어내지 않는다
- [ ] 2차·3차 협력사에 심어둔 제재 리스크 케이스를 실제로 찾아낸다
- [ ] 동명이인 케이스에서 임의로 하나를 고르지 않는다
- [ ] 실행 후 `data/`가 변경되지 않았음이 검증된다
- [ ] `watchlist-monitor`가 변화 없는 항목을 반복 서술하지 않는다

---

## 11. 열린 질문

1. **duckdb 의존을 전제할 것인가.** `python_repl`의 `pip_install`에 의존하면 오프라인 환경에서 깨진다. 순수 python(csv+dict)만으로도 돌아가게 할지, duckdb를 필수로 둘지.
2. **공시·뉴스 원문 검색을 grep으로 버틸 수 있는가.** 문서 수가 늘면 키워드 검색의 재현율이 떨어진다. 임베딩이 없는 지금, 문서에 프론트매터 태그(기업/주제/일자)를 붙여 구조화 검색으로 대신할지.
3. **서브에이전트 분리 여부.** 공급망 n차 전개처럼 컨텍스트를 많이 먹는 작업을 서브에이전트로 뺄지. 기업당 조사 서브에이전트가 자연스럽지만 예제의 가독성은 떨어진다.
4. **리스크 점수 산식을 코드로 고정할지, 에이전트가 매번 설계할지.** 고정하면 재현 가능하지만 자유 형식 질문의 취지와 어긋난다. 프리셋에서만 고정하는 절충안이 유력하다.
5. **합성 데이터의 현실성 수준.** 실제 DART/EDGAR 스키마를 얼마나 충실히 흉내 낼지. 너무 충실하면 예제가 무거워지고, 너무 단순하면 free-form 쿼리의 어려움이 사라진다.
6. **실존 기업명 사용 금지 범위.** 제재 리스트는 형식만 차용하고 대상은 전부 가공해야 한다. 국가명·HS 코드는 실제 값을 써도 되는지.
