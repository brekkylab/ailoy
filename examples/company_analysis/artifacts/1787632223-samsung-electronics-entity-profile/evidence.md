# Evidence log

All paths below are relative to the mount root (the
`ailoy-company-analysis` prefix is omitted, per instructions). Each path was
read live during this session.

## GLEIF

1. **Path:** `gleif/search/entity.legalName/Samsung Electronics Co., Ltd/entity.legalAddress.country/KR/pages/page-001.json`
   **Claim:** An exact legal-name search for "Samsung Electronics Co., Ltd"
   filtered to Korean addresses returns 50 records, none of which is Samsung
   Electronics itself — because GLEIF stores its legal name in Korean script.
   **Detail:** `meta.pagination.total = 50`; scanned all 50 `entity.legalName.name`
   values, none contain "samsung electronics" (case-insensitive).

2. **Path:** `gleif/search/fulltext/Samsung Electronics/pages/page-001.json`
   **Claim:** Full-text search for "Samsung Electronics" returns 81 records
   (one page, under the 200/page cap). Record `id`/`lei` =
   `9884007ER46L6N7EI764` is Samsung Electronics Co., Ltd.
   **Detail:** record's `entity.legalName.name` = "삼성전자(주)" (language `ko`);
   `entity.otherNames[0]` = `{"name":"SAMSUNG ELECTRONICS CO., LTD","language":"en","type":"ALTERNATIVE_LANGUAGE_LEGAL_NAME"}`;
   `entity.legalAddress` = Suwon, Gyeonggi-do (KR-41), KR 16677;
   `entity.registeredAs` = "124-81-00998"; `entity.jurisdiction` = "KR";
   `entity.legalForm.id` = "5RCH"; `entity.status` = "ACTIVE";
   `entity.creationDate` = "1969-01-13T00:00:00Z";
   `registration.status` = "ISSUED"; `registration.initialRegistrationDate` =
   "2017-09-27T00:00:00Z"; `registration.lastUpdateDate` = "2025-10-31T09:40:29Z";
   `registration.managingLou` = "9884008RRMX1X5HV6625";
   `registration.validatedAt.id` = "RA000657"; `registration.validatedAs` =
   "124-81-00998"; `bic` = ["SECTKRSEXXX"]; `spglobal` = ["91868"];
   `conformityFlag` = "CONFORMING".

3. **Path:** `gleif/search/ownedBy/9884007ER46L6N7EI764/pages/page-001.json`
   **Claim:** GLEIF records 22 entities directly/ultimately owned by this LEI.
   **Detail:** `meta.pagination.total = 22`. Names include: Samsung Display
   Noida Private Limited (IN), Samsung OAK Holdings, Inc. (US), Samsung
   Malaysia Electronics (SME) Sdn. Bhd. (MY), Samsung R&D Institute India –
   Bangalore Private Limited (IN), 삼성디스플레이(주) (KR), Samsung Eletronica
   da Amazonia Ltda (BR), and 12 Harman-branded subsidiaries across HU, NL, DE,
   GB, DK, MU, US, AT, RU, CN (×3), MX, RO.

4. **Path:** `gleif/search/owns/9884007ER46L6N7EI764/pages/page-001.json`
   **Claim:** GLEIF records 0 entities that own this LEI (empty parent
   relationship set at the `owns` search endpoint).
   **Detail:** `meta.pagination.total = 0`, `data = []`.

5. **Path:** `gleif/by-lei/9884007ER46L6N7EI764/direct-parent-reporting-exception/direct-parent-reporting-exception.json`
   **Claim:** No direct parent is reported; reason given is `NO_KNOWN_PERSON`.
   **Detail:** `data.attributes.category` = "DIRECT_ACCOUNTING_CONSOLIDATION_PARENT",
   `data.attributes.reason` = "NO_KNOWN_PERSON".

6. **Path:** `gleif/by-lei/9884007ER46L6N7EI764/ultimate-parent-reporting-exception/ultimate-parent-reporting-exception.json`
   **Claim:** No ultimate parent is reported; same reason.
   **Detail:** `data.attributes.category` = "ULTIMATE_ACCOUNTING_CONSOLIDATION_PARENT",
   `data.attributes.reason` = "NO_KNOWN_PERSON".

7. **Path:** `gleif/by-lei/9884007ER46L6N7EI764/isins/isins.json`
   **Claim:** 15 ISINs are linked to this LEI.
   **Detail:** `meta.pagination.total = 15`; ISINs: US7960503008, US7960504097,
   US7960506076, US796050AA00, US7960508056, US7960507066, US7960508882,
   US7960502018, US796050AE22, US796050AB82, US0019071044, US7960508700,
   US7960505086, USY74718AQ37, US7960508627.

8. **Path (checked, negative result):** `gleif/search/entity.legalName/Samsung Electronics/entity.legalAddress.country/KR/pages/page-001.json`
   **Claim:** An exact legal-name search for the bare string "Samsung
   Electronics" (no "Co., Ltd") filtered to KR returns 0 records — GLEIF's
   `entity.legalName` filter requires an exact match, and the Korean legal
   name doesn't literally read "Samsung Electronics" in Latin script.
   **Detail:** `meta.pagination.total = 0`, `data = []`.

## SEC EDGAR

9. **Path:** `edgar/search/entityName/Samsung Electronics/pages/page-001.json`
   **Claim:** Full-text filing search for "Samsung Electronics" returns 213
   filing hits, spanning only 4 distinct CIKs: 0000879316 (Samsung
   Electronics itself), 0000917273 (Rambus Inc.), 0001137789 (Seagate
   Technology plc), 0001585854 (SunEdison Semiconductor Ltd).
   **Detail:** `hits.total.value = 213`; ciks collected from all
   `hits.hits[]._source.ciks`.

10. **Path:** `edgar/search/entityName/Samsung Electronics Co Ltd/pages/page-001.json`
    **Claim:** A near-identical query without punctuation returns the same
    213-hit total, confirming this is a full-text search over filing
    documents rather than an exact company-name index.
    **Detail:** `hits.total.value = 213`.

11. **Path:** `edgar/by-cik/0000879316/submissions/submissions.json`
    **Claim:** CIK 0000879316 is registered under the name "SAMSUNG
    ELECTRONICS CO LTD /FI"; `entityType` = "other"; EIN = "953170778"; `lei`
    field = null; `stateOfIncorporation` = "M5" (Korea, Republic of);
    `fiscalYearEnd` = "1231"; mailing/business address = "250 2 KA TAEPYUNG
    RO CHUNG KU, SEOUL, KOREA, 100742"; `tickers` = []; `exchanges` = [];
    `description`/`website`/`investorWebsite` = "" (empty).
    **Detail:** direct field values as listed, read from the JSON object's
    top level (excluding `filings`).

12. **Path:** `edgar/by-cik/0000879316/submissions/submissions.json`
    (`filings.recent`, zipped by index)
    **Claim:** 251 filings on record for this CIK, dated 1995-03-06 to
    2015-01-20. Form-type counts: SUPPL 196, SC 14D1/A 18, ARS 9, SC 13E3/A
    9, SC 13D/A 6, SC 13G/A 2, SC 13G 2, SC 14D1 2, SC 14D9/A 2, Form 4 1,
    Form 3 1, SC 13D 1, SC 13E3 1, SC 14D9 1.
    **Detail:** computed via `collections.Counter` over
    `filings.recent.form` (251 entries), min/max over
    `filings.recent.filingDate`.

13. **Path:** `edgar/by-cik/0000879316/submissions/submissions.json`
    (`filings.recent`, individual rows)
    **Claim:** 1995 filings (SC 14D1, SC 14D1/A ×8, SC 14D9, SC 14D9/A ×2)
    and 1997 filings (SC 14D1, SC 13E3, and 9 rounds of amendments to each)
    relate to a tender offer / going-private transaction; several 1997
    `primaryDocDescription` values explicitly read "AST/SAMSUNG MERGER" or
    "SAMSUNG / AST MERGER".
    **Detail:** e.g. accession `0001017062-97-000688`,
    `primaryDocDescription` = "SCHEDULE 13E3-3 RE SAMSUNG / AST MERGER";
    accession `0001017062-97-000832`, `primaryDocDescription` = "AMEND. #2
    TO SCH13E-3 - AST/SAMSUNG MERGER".

14. **Path:** `edgar/by-cik/0000879316/submissions/submissions.json`
    (`filings.recent`, individual rows)
    **Claim:** Samsung Electronics is named as a co-filer/subject on SC 13D
    and SC 13D/A filings for Seagate Technology plc (2011–2013), an SC 13G/A
    for Rambus Inc. (2012), SC 13G/SC 13G-A for SunEdison Semiconductor Ltd
    (2014–2015), and Forms 3/4 tied to the Seagate position (2012–2013).
    **Detail:** e.g. accession `0001193125-13-406331`,
    `display_names` = ["Seagate Technology plc (STX) (CIK 0001137789)",
    "SAMSUNG ELECTRONICS CO LTD /FI (CIK 0000879316)"]; accession
    `0001193125-15-014566`, `display_names` = ["SunEdison Semiconductor Ltd
    (CIK 0001585854)", "SAMSUNG ELECTRONICS CO LTD /FI (CIK 0000879316)"].

15. **Path:** `edgar/by-cik/0000879316/facts/facts.json`
    **Claim:** No XBRL company-facts document exists for this CIK; the
    endpoint returns an S3 "NoSuchKey" error instead.
    **Detail:** raw response body: `<?xml version="1.0" encoding="UTF-8"?>
    <Error><Code>NoSuchKey</Code><Message>The specified key does not
    exist.</Message><Key>api/xbrl/companyfacts/CIK0000879316.json</Key>...`.

## Cross-registry note

No path in either registry cites the other's identifier: GLEIF's record for
LEI `9884007ER46L6N7EI764` has no CIK/EIN field, and EDGAR's `lei` field for
CIK `0000879316` is `null` (evidence #11). The association between the two
records made in this report rests on matching the company name ("Samsung
Electronics") and Korea as the home jurisdiction in both records — a name
match, per the catalog's guidance, not a confirmed shared-entity fact.
