## Picks
1. urn:li:person:b3tq9wmk — 하은 성 — rewrote settlement batch processing in Rust (daily→hourly) with PostgreSQL partitioning/read replicas at Nordwind Works, the exact scope of this posting's 주요 업무
2. urn:li:person:k6qb3nwd — 채원 노 — built the Rust event-ingestion gateway at Halcyon Systems with hand-designed offset management and dedup for safe reprocessing, matching the posting's idempotency/offset requirement
3. urn:li:person:n7kv4jsb — 도윤 반 — owns Rust Kafka-consumer services doing idempotent PostgreSQL writes and replay tooling at Larkfield Dynamics, wants an on-site Seoul role starting immediately

## Method

**Must-haves gated** (`search --city Seoul --min-years 4 --skill Rust ...`): production Rust experience, 4+ years backend, Seoul (posting requires 3 days/week onsite, so I treated Seoul as the necessary proxy — `distribution city` confirmed Seoul is the pool's dominant city and matches "서울 오피스"). I did **not** gate on Kafka, payments, or idempotency — those are 우대 사항 (nice-to-haves) — to avoid dropping a qualified person who never used those words on a searchable surface.

`distribution skill rust` showed the pool spells the skill five ways (Rust, rust-lang, Rust (Programming Language), Rust Lang, Async Rust) plus two Korean spellings not listed there (러스트) and Tokio — I ran `search --skill` for every spelling and merged the 49 unique ids into one candidate set.

Within that set of 49, I ranked with `--mentions` for the domain vocabulary from 우대 사항: `Kafka`, `분산 시스템`/`distributed`, `정산`/`payment`/`결제`, `재시도`/`retry`, `장애`/`failover`. This is ranking only, not a filter — nobody was dropped from the 49 for lacking these words.

I then `read` the top hits together to check tenure math (years vs naive_years), `open_to_work`, `wants` arrangement, and whether the description text (not just the headline) backs the skill.

## Why these three over other high scorers

- **하은 성** is the only person in the set whose position description literally names 정산 배치 재작성 + PostgreSQL 파티셔닝 + 읽기 복제, i.e. this posting's 주요 업무 verbatim. 10.4y tenure with `years == naive_years` (sequential positions, no overlap inflation). Open to work, inmail contact.
- **채원 노** built exactly the "이벤트 수집 계층의 재처리 설계 — 멱등성과 오프셋 관리" the posting asks for, at 8만 TPS scale, and explicitly wants Hybrid-in-Seoul work — matching the 주 3일 출근 arrangement. 10.0y tenure, no overlap.
- **도윤 반**'s current role is idempotent Kafka-consumer writes into PostgreSQL plus on-call replay tooling — the reprocessing/idempotency 우대 사항 again, from description text not just headline. Wants On-site-in-Seoul starting immediately. 9.7y tenure, no overlap.

<!-- rejected -->

- **urn:li:person:bqsuo43r (Casey Dunmore)** — `years`=14.0 vs `naive_years`=17.3 (3.3y of concurrent-employment double counting: a still-open Finlogic Platform role runs the entire 2012–present span while a second FULL_TIME Halcyon Platform job and a current part-time Foldgate Platform gig overlap it) — a naive-sum ranking would have put him near the top. His only Rust position is a legacy Kafka service he "continues maintaining" while his live day-to-day work is Python scripting part-time. `not open to work`, and no contact row exists. Rejected on staleness/overlap and unreachability.
- **urn:li:person:r2vt6nsw (Sage Radcliffe)** — strong Rust/distributed-systems Staff engineer (8.5y, no overlap), but `wants` is explicitly Remote-only starting 2027-01, which conflicts with the posting's 주 3일 출근 requirement. A search on "distributed systems" alone would rank him highly; the mismatch only shows up in `wants`.
- **urn:li:person:d8kf3prw (Rowan Voss)** — excellent Rust/replication/failover background, but profile is `not open to work`. Fails a practical must-have even though the skill match is strong.
- **urn:li:person:b2pq7x72 (예은 연)** — 15y tenure, PostgreSQL partitioning + Rust distributed work, but `not open to work` and profile last updated 2024-12-31 (stale relative to others updated in 2026), so I discounted it further even before the open-to-work gate.
- **urn:li:person:t4qm9wbd (Morgan Thorne)** — headline says "Rust, payments" and description covers payment authorization/reconciliation exactly on-domain, but `years`=5.0 vs `naive_years`=8.0 (Backend Consultant contract 2019–2022 overlaps a concurrent FULL_TIME role 2018–2023 by 3 years), and there is **no contact row** — unreachable. A payments-keyword search would have ranked him near the top; both the tenure and reachability facts only surface on `read`.
- **urn:li:person:8rs80uy8 (하은 하)** — settlement-batch and payment-verification Rust work in the description, on-domain, but `not open to work`.
- **urn:li:person:w3gm8rbq (지훈 류)** — close 4th choice: Kafka + exactly-once event processing in Rust at Larkfield Networks, 9.7y no-overlap tenure, inmail contact, open to work. Held out of the top 3 because his profile carries no `wants` row (no stated arrangement, unlike the picks who explicitly want Seoul hybrid/on-site) and his description stays at the microservice-coordination level without the settlement/payments-specific evidence the top 3 have.
- **urn:li:person:a4rn7qvt (Alex Vance)** and **urn:li:person:u5cz9jhr (Tatum Ellery)** — both strong Rust/distributed-systems engineers (8.9y and 8.7y respectively, no overlap), but neither's description touches settlement, payments, or Kafka streaming; Ellery additionally wants Remote-only, again a mismatch with onsite requirement. Kept as backups, not top 3.
