# Evidence

Every path below is written from the mount root down (no leading `./`), per
instructions. Registry: live API mirrored as a filesystem; each read is one request.

## GLEIF

1. **Claim**: Searching GLEIF's `entity.legalName` field for "Samsung Electronics"
   returns 47 records, none of which is the Korean parent — it returns national sales
   subsidiaries (e.g. SAMSUNG ELECTRONICS AMERICA, INC., SAMSUNG ELECTRONICS (UK)
   LIMITED, SAMSUNG ELECTRONICS GMBH, …) plus one unrelated leveraged-ETF fund named
   "ProShares Ultra Samsung Electronics" (LEI `529900JUHLXQE4F70208`, category `FUND`)
   and two CSOP Hong Kong leveraged/inverse products.
   Source: `gleif/search/entity.legalName/Samsung Electronics/pages/page-001.json`

2. **Claim**: The Korean parent company's legal name is recorded in GLEIF in Korean
   script — 삼성전자(주) — with "SAMSUNG ELECTRONICS CO., LTD" carried only as an
   `ALTERNATIVE_LANGUAGE_LEGAL_NAME` under `otherNames`; found via GLEIF's `fulltext`
   filter, LEI `9884007ER46L6N7EI764`.
   Source: `gleif/search/fulltext/Samsung Electronics Co Ltd/pages/page-001.json`

3. **Claim**: Full LEI record for `9884007ER46L6N7EI764`:
   - Legal name (ko): 삼성전자(주); English alt name: SAMSUNG ELECTRONICS CO., LTD
   - Legal/HQ address: 경기도 수원시 영통구 삼성로 129 (매탄동), 삼성전자 — Suwon,
     Gyeonggi-do, KR, postal code 16677
   - Registered as (Korean business registration no.): `124-81-00998`, validated at
     registration authority id `RA000657`
   - Jurisdiction: `KR`; legal form id `5RCH`; entity status `ACTIVE`
   - Entity creation date: `1969-01-13`
   - LEI registration status: `ISSUED`; initial registration `2017-09-27`; last update
     `2025-10-31`; next renewal `2026-10-31`; managing LOU `9884008RRMX1X5HV6625`
   - BIC: `SECTKRSEXXX`
   - S&P Global company id (`spglobal`): `91868`
   Source: `gleif/by-lei/9884007ER46L6N7EI764/record/record.json`

4. **Claim**: GLEIF carries no reportable corporate parent for this LEI — both direct
   and ultimate parent fields resolve to a reporting exception, category
   `DIRECT_ACCOUNTING_CONSOLIDATION_PARENT` / `ULTIMATE_ACCOUNTING_CONSOLIDATION_PARENT`,
   reason `NO_KNOWN_PERSON`.
   Sources:
   `gleif/by-lei/9884007ER46L6N7EI764/direct-parent-reporting-exception/direct-parent-reporting-exception.json`
   `gleif/by-lei/9884007ER46L6N7EI764/ultimate-parent-reporting-exception/ultimate-parent-reporting-exception.json`

5. **Claim**: This LEI has 15 associated ISINs, e.g. `US7960505086`, `USY74718AQ37`,
   `US7960508700`, `US7960504097`, `US7960503008`, `US7960508056`, `US7960506076`,
   `US796050AE22`, `US7960502018`, `US7960508882`, `US7960508627`, `US796050AA00`,
   `US7960507066`, `US0019071044`, `US796050AB82`.
   Source: `gleif/by-lei/9884007ER46L6N7EI764/isins/isins.json`

6. **Claim**: This LEI has one recorded branch — SAMSUNG ELECTRONICS COMPANY LIMITED,
   LEI `335800A16O7QD1TDQO62`, category `BRANCH`, jurisdiction `KR` (branch physically
   registered in Mumbai, India, registeredAs `F04529`), status `ACTIVE`.
   Source: `gleif/by-lei/9884007ER46L6N7EI764/branches/branches.json`

7. **Claim**: Direct-children relationship page (6 of total) includes SAMSUNG MALAYSIA
   ELECTRONICS (SME) SDN. BHD. (MY), SAMSUNG R&D INSTITUTE INDIA – BANGALORE PRIVATE
   LIMITED (IN), 삼성디스플레이(주) / SAMSUNG DISPLAY Co., Ltd. (KR), SAMSUNG ELETRONICA
   DA AMAZONIA LTDA (BR), 哈曼（中国）投资有限公司 / HARMAN INTERNATIONAL (CHINA)
   HOLDINGS CO., LTD. (CN), HARMAN INTERNATIONAL INDUSTRIES, INCORPORATED (US-DE).
   Source: `gleif/by-lei/9884007ER46L6N7EI764/direct-children/direct-children.json`

8. **Claim**: Ultimate-children total per GLEIF pagination metadata is 22 (page of 15
   read), spanning Samsung Display, Samsung Malaysia Electronics, Samsung R&D
   Institute India–Bangalore, Samsung Display Noida, Samsung Eletronica da Amazonia
   (Brazil), Samsung Oak Holdings Inc. (US-DE), Harman Industries Holding Mauritius,
   and multiple Harman entities across Hungary, Romania, Netherlands, Denmark, Russia,
   Mexico, Germany, US.
   Source: `gleif/by-lei/9884007ER46L6N7EI764/ultimate-children/ultimate-children.json`

## SEC EDGAR

9. **Claim**: EDGAR full-text search for entityName "Samsung Electronics" returns 213
   total hits (elasticsearch `hits.total.value`); most-recent hit is a joint SC 13G/A
   filed by SunEdison Semiconductor Ltd (CIK 0001585854) and SAMSUNG ELECTRONICS CO LTD
   /FI (CIK 0000879316), filed 2015-01-20.
   Source: `edgar/search/entityName/Samsung Electronics/pages/page-001.json`

10. **Claim**: EDGAR filer CIK `0000879316` is registered under the name "SAMSUNG
    ELECTRONICS CO LTD /FI"; `entityType` = `other`; `ein` = `953170778`; `lei` field
    is `null`; `stateOfIncorporation` = `M5` ("Korea, Republic of"); no tickers, no
    exchanges, no former names; mailing/business address "250 2 KA TAEPYUNG RO CHUNG
    KU, SEOUL", zip 100742, Korea.
    Source: `edgar/by-cik/0000879316/submissions/submissions.json`

11. **Claim**: `filings.recent` for CIK 0000879316 lists 251 filings, dated
    1995-03-06 to 2015-01-20, form-type counts: SUPPL 196, SC 14D1/A 18, ARS 9,
    SC 13E3/A 9, SC 13D/A 6, SC 13G/A 2, SC 13G 2, SC 14D1 2, SC 14D9/A 2, Form 4 1,
    Form 3 1, SC 13D 1, SC 13E3 1, SC 14D9 1. No 10-K/10-Q/20-F present.
    Source: `edgar/by-cik/0000879316/submissions/submissions.json`

12. **Claim**: The XBRL company-facts endpoint for CIK 0000879316 returns an S3
    "NoSuchKey" error rather than data — no structured XBRL financial facts are on
    file for this filer.
    Source: `edgar/by-cik/0000879316/facts/facts.json`

## Cross-registry note

13. **Claim**: No shared identifier links the GLEIF LEI record
    (`9884007ER46L6N7EI764`) to the EDGAR CIK record (`0000879316`); EDGAR's `lei`
    field for this CIK is `null` (see item 10), and GLEIF's record carries no SEC CIK
    field at all. The association made in this report between the two is a **name and
    jurisdiction match** (Korean "Samsung Electronics" entity in both), consistent
    with the top-level catalog's warning that no reliable key bridges GLEIF and
    EDGAR.
    Sources: `CATALOG.md`; `gleif/by-lei/9884007ER46L6N7EI764/record/record.json`;
    `edgar/by-cik/0000879316/submissions/submissions.json`
