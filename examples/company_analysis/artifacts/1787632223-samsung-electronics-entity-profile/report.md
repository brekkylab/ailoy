# Samsung Electronics — Entity Profile

## Summary

- **Samsung Electronics Co., Ltd.** (Korean legal name 삼성전자(주)) is a South Korean
  company headquartered in Suwon, Gyeonggi-do, holding an **active** Legal Entity
  Identifier (LEI) `9884007ER46L6N7EI764` issued under GLEIF, validated against the
  Korean business registration number `124-81-00998` (registration authority
  `RA000657`).
- GLEIF reports **no parent** for Samsung Electronics — both the direct- and
  ultimate-parent relationships carry the exception reason `NO_KNOWN_PERSON`, meaning
  no controlling entity was reported to the LEI system. GLEIF does list 22 entities
  Samsung Electronics itself owns (e.g. Samsung Display, Samsung Malaysia Electronics,
  and the Harman International group).
- In SEC EDGAR, Samsung Electronics is registered as filer **CIK 0000879316**,
  "SAMSUNG ELECTRONICS CO LTD /FI", a foreign private issuer incorporated in the
  Republic of Korea (state code `M5`), EIN `953170778`. Its EDGAR filing history is
  narrow and dated: 251 filings from 1995 to 2015, dominated by beneficial-ownership
  and tender-offer schedules (SC 13D/G, SC 14D1/9, SC 13E3) and paper annual-report
  supplements — no 10-K/20-F-style disclosures and nothing filed after January 2015.
- No XBRL company-facts are on file for this CIK (the API returns a "NoSuchKey"
  error), and no ticker/exchange listing is recorded — consistent with Samsung
  Electronics not being an SEC reporting company in the ordinary sense, but a foreign
  entity that has occasionally been a subject or filer in US ownership/tender-offer
  filings.
- The two registries share no common identifier: GLEIF's LEI and EDGAR's CIK were
  joined here only by matching the company name and its Korean HQ address/description
  across records — see "Identifiers" below.

## 1. The entity

Samsung Electronics is a Korean electronics manufacturer. The GLEIF record's legal
name is stored in Korean script — **삼성전자(주)** — with the English form
"SAMSUNG ELECTRONICS CO., LTD" recorded as an `ALTERNATIVE_LANGUAGE_LEGAL_NAME`.
`gleif/search/fulltext/Samsung Electronics/pages/page-001.json`, record with
`lei = 9884007ER46L6N7EI764`.

Its legal and headquarters address, per GLEIF, is:

> 경기도 수원시 영통구 삼성로 129 (매탄동), 삼성전자 — Suwon, Gyeonggi-do (KR-41), KR 16677
> (English form: 129, Samsung-ro, Yeongtong-gu, Suwon-si, Gyeonggi-do)

`gleif/search/fulltext/Samsung Electronics/pages/page-001.json` (same record,
`entity.legalAddress` / `entity.otherAddresses`).

GLEIF's `entity.creationDate` for this LEI record is `1969-01-13`, i.e. the entity's
founding date as reported to GLEIF. Same source.

In SEC EDGAR, the filer of record under this name is CIK `0000879316`, name on file
"SAMSUNG ELECTRONICS CO LTD /FI" (the "/FI" suffix is EDGAR's own disambiguation
suffix, not part of the legal name), `entityType: "other"`, incorporated/organized in
`M5` = "Korea, Republic of." `edgar/by-cik/0000879316/submissions/submissions.json`.

A full-text search of EDGAR filings for "Samsung Electronics" returns 213 filing hits
across only four distinct CIKs — Samsung Electronics itself (0000879316), and three
unrelated US filers (Rambus Inc., Seagate Technology plc, SunEdison Semiconductor
Ltd) whose filings merely *mention* Samsung Electronics (e.g. as a 5%+ beneficial
owner in SC 13D/G schedules). `edgar/search/entityName/Samsung Electronics/pages/page-001.json`.

## 2. Identifiers and where each comes from

| Identifier | Value | Source | Notes |
|---|---|---|---|
| LEI (GLEIF) | `9884007ER46L6N7EI764` | `gleif/search/fulltext/Samsung Electronics/pages/page-001.json` | Status `ISSUED`, `conformityFlag: CONFORMING`. Managing LOU: `9884008RRMX1X5HV6625`. |
| Local registration number (validated by GLEIF) | `124-81-00998` | same record, `entity.registeredAs` / `registration.validatedAs` | Validated at registration authority `RA000657` (Korea), i.e. a Korean business registration number, not the EDGAR CIK or EIN. |
| BIC | `SECTKRSEXXX` | same record, `attributes.bic` | Bank identifier code associated with the entity in GLEIF's data. |
| ISINs (15 listed) | e.g. `US7960503008`, `US7960504097`, `US0019071044`, `USY74718AQ37`, … | `gleif/by-lei/9884007ER46L6N7EI764/isins/isins.json` | Securities linked to this LEI; several `US7960...` and `US796050...` ISINs plus one non-matching prefix (`US0019071044`, `USY74718AQ37`) that GLEIF nonetheless associates with this LEI. |
| SEC CIK | `0000879316` | `edgar/by-cik/0000879316/submissions/submissions.json` | Filer name on record: "SAMSUNG ELECTRONICS CO LTD /FI". |
| EIN (US tax ID, as recorded by EDGAR) | `953170778` | same submissions.json | EDGAR's `ein` field; per the registry-crossing rule this is *not* a bridge to GLEIF, which records a local registration number instead. |
| GLEIF `lei` field inside EDGAR | `null` | same submissions.json | Confirms the catalog's warning that EDGAR's `lei` field is essentially never populated for this filer. |

**No shared key exists between the two records.** They were matched here on **name +
jurisdiction** (Korean electronics company named "Samsung Electronics," headquartered
in / incorporated under Korea in both records) — this is a **name match, treated as a
candidate joined and cross-checked on jurisdiction, not a guaranteed same-legal-entity
link** the way the catalogs describe. Nothing in either record cites the other's
identifier directly.

## 3. Registration and status

**GLEIF (LEI record):**
- Status: `ACTIVE` (entity status) / `ISSUED` (LEI registration status)
- Initial LEI registration: 2017-09-27; last update 2025-10-31; next renewal
  2026-10-31
- Jurisdiction: `KR`; legal form code `5RCH` (a GLEIF-internal legal-form code; the
  record does not spell out its label, and no local legal-form lookup table is
  available in this mount)
- Direct-parent / ultimate-parent: both report `NO_KNOWN_PERSON` — GLEIF was told
  there is no reportable controlling entity above Samsung Electronics.
- Direct/ultimate subsidiaries reported to GLEIF (`ownedBy` relationship, 22 total):
  includes Samsung Display Noida (India), Samsung OAK Holdings Inc. (US), Samsung
  Malaysia Electronics, Samsung R&D Institute India–Bangalore, 삼성디스플레이(주)
  (Samsung Display Co., Ltd, Korea), Samsung Eletrônica da Amazônia (Brazil), and the
  Harman International group of subsidiaries (US, DE, NL, HU, DK, AT, MU, RU, CN, MX,
  RO). `gleif/search/ownedBy/9884007ER46L6N7EI764/pages/page-001.json`.

**SEC EDGAR (filer record):**
- `entityType: "other"`; no ticker, no exchange listed
  (`tickers: []`, `exchanges: []`)
- State/country of incorporation: `M5` = Korea, Republic of
- Fiscal year end: 12/31
- Filing history spans **1995-03-06 to 2015-01-20** (251 filings on record); nothing
  filed since January 2015 under this CIK.
- Form mix (251 filings): 196 `SUPPL` (auto-generated paper-document placeholders),
  18 `SC 14D1/A`, 9 `ARS` (annual report to shareholders, paper), 9 `SC 13E3/A`, 6
  `SC 13D/A`, 2 each of `SC 13G/A`, `SC 13G`, `SC 14D1`, `SC 14D9/A`, and 1 each of
  `SC 13D`, `SC 13E3`, `SC 14D9`, Form `3`, Form `4`.
  `edgar/by-cik/0000879316/submissions/submissions.json`.
- Company-facts (XBRL) endpoint returns an S3 "NoSuchKey" error rather than a facts
  document — no structured financial facts are on file for this CIK.
  `edgar/by-cik/0000879316/facts/facts.json`.

## 4. What the filings say

The EDGAR filing history under CIK 0000879316 is not a normal US-issuer disclosure
record (no 10-K/10-Q, no XBRL facts). It consists almost entirely of:

- **Tender-offer / going-private paperwork from 1995 and 1997**: Schedule 14D-1 and
  14D-9 filings (and their amendments) plus Schedule 13E-3 filings, associated with a
  1995 tender offer and a 1997 going-private transaction involving **AST Research**
  (the 1997 filings are explicitly described as "AST/SAMSUNG MERGER" /
  "SAMSUNG/AST MERGER" in `primaryDocDescription`). `edgar/by-cik/0000879316/submissions/submissions.json`, filings dated 1995-03-06 through 1997-08-15.
- **Beneficial-ownership schedules where Samsung Electronics is the reporting
  person or subject** on other companies' securities: Schedule 13D/13D-A on Seagate
  Technology plc (2011–2013) and Rambus Inc. (2012), Schedule 13G/13G-A on SunEdison
  Semiconductor Ltd (2014–2015), and individual Forms 3/4 tied to the Seagate
  position (2012–2013). `edgar/search/entityName/Samsung Electronics/pages/page-001.json`.
- **Paper annual reports (ARS) and paper "SUPPL" filings from 2002–2010**, i.e.
  Samsung Electronics periodically submitted its annual report to shareholders (as a
  foreign private issuer furnishing information under Rule 12g3-2(b)) as
  auto-generated paper documents rather than electronic disclosures.
  `edgar/by-cik/0000879316/submissions/submissions.json`.

No 20-F, 10-K, or other periodic US disclosure form appears in the filing history, and
no filings are on record after **2015-01-20**. EDGAR's `description` and `website`
fields for this CIK are both empty strings — the registry does not carry a
self-description or website for this filer.

## Data limits

- **No structured financial disclosures**: EDGAR carries no XBRL company facts for
  this CIK (`facts.json` — S3 "NoSuchKey" error) and no 10-K/20-F on record — this
  registry cannot answer questions about Samsung Electronics' revenue, headcount, or
  financial statements, and the task instructions note this is out of scope for both
  registries generally (non-US financial statements aren't covered).
- **No shared identifier confirms the two records are the same legal entity beyond
  name + Korean jurisdiction.** The join between GLEIF's LEI `9884007ER46L6N7EI764`
  and EDGAR's CIK `0000879316` is a **possible match — needs confirmation**, checked
  here only against: (a) both naming "Samsung Electronics," and (b) both citing Korea
  as the home jurisdiction. Neither record cites the other's identifier.
- **GLEIF's `entity.legalForm.id` (`5RCH`) and `registeredAt.id` (`RA000657`) are
  opaque codes** in the record; no legal-form or registration-authority lookup table
  was available in this mount to translate them into plain text, so they are reported
  as-is.
- **EDGAR's `lei` field for this filer is `null`**, matching the catalog's warning
  that this field is almost never populated — it could not be used to confirm the
  cross-registry match.
- **EDGAR's filing history stops in January 2015.** Whether Samsung Electronics has
  had any EDGAR-reportable events since then (e.g. further Schedule 13D/G activity)
  is not something this record can speak to; only what is on file for CIK 0000879316
  was read.
- The GLEIF full-text search for "Samsung Electronics" returned 81 records on a
  single page (under the API's 200-per-page cap) and was not paginated further, but a
  broader legal-name search (without "Electronics") could return additional
  Samsung-affiliated entities not surfaced here; the search here was scoped to
  entities whose name contains "Samsung Electronics" specifically.

## Next steps

- If a full ownership/subsidiary map is needed, walk GLEIF's
  `direct-children`/`ultimate-children` relationship endpoints beyond the 22 entities
  already surfaced under `ownedBy`, and decode legal-form code `5RCH` against GLEIF's
  full ISO 20275 legal-form-code reference (not available in this mount).
  Ownership structure and control cannot be confirmed further within this registry.
- If current (post-2015) SEC activity is in scope, re-run the EDGAR full-text search
  narrowed by a recent `startdt` to check whether Samsung Electronics has appeared as
  a subject of any new Schedule 13D/G filings by other CIKs.
- For financial statements, corporate registry filings, or news about Samsung
  Electronics, a different data source is required — neither GLEIF nor EDGAR carries
  this, per the catalog's own statement of scope.
