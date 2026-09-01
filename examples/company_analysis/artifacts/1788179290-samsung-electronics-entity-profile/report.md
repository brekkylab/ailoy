# Samsung Electronics — Entity Profile

## Summary

- The entity is **삼성전자(주)** / **SAMSUNG ELECTRONICS CO., LTD**, a South Korean
  general-category legal entity registered in Suwon, Gyeonggi-do, Korea, holding LEI
  **9884007ER46L6N7EI764** (GLEIF, active, issued 2017-09-27, last updated 2025-10-31).
- Its Korean business registration number is **124-81-00998**, validated by the Korean
  local operating unit RA000657; GLEIF records no parent LEI for it — both the direct
  and ultimate parent fields carry a `NO_KNOWN_PERSON` reporting exception, i.e.
  Samsung Electronics is not itself consolidated under another LEI-registered parent
  in GLEIF's data.
- It also appears in **SEC EDGAR** under **CIK 0000879316** as *"SAMSUNG ELECTRONICS CO
  LTD /FI"* (foreign filer), which has never filed a 10-K or held a US ticker/exchange
  listing; its 251 listed filings are almost all Schedule 13D/13G, tender-offer
  (SC 14D1/14D9), going-private (SC 13E3), and paper "SUPPL"/annual-report (ARS)
  filings tied to its stakes in, or tender offers for, other US issuers (e.g. Seagate,
  Rambus, SunEdison Semiconductor), spanning 1995–2015.
- Samsung Electronics operates a vast network of legally distinct national
  subsidiaries and one Korean branch abroad (in India); GLEIF alone returns 47 records
  whose legal name contains "Samsung Electronics" plus further Korean-script entities
  found only by full-text search — a name collision hazard the registries warn about
  explicitly.
- EDGAR carries no XBRL company facts for CIK 0000879316 (S3 `NoSuchKey` on the facts
  endpoint) and no `lei` value in `submissions.json` — this filer entry predates and is
  unlinked to the GLEIF LEI record above; the two are matched here **by name and Korean
  jurisdiction only**, not by a shared identifier.

## 1. The entity

Samsung Electronics is South Korea's flagship electronics manufacturer. The two
registries in scope here — GLEIF and SEC EDGAR — each hold one primary record that
plausibly refers to the parent company, plus dozens of subsidiary/affiliate records
that share the "Samsung Electronics" name:

- **GLEIF**: the Korean parent's legal name is recorded in Korean script,
  **삼성전자(주)**, with the English form **"SAMSUNG ELECTRONICS CO., LTD"** carried as
  an `ALTERNATIVE_LANGUAGE_LEGAL_NAME` (`otherNames`), not as the primary `legalName`
  field — which is why a search on `entity.legalName=Samsung Electronics` (English)
  does **not** return this record; it only surfaces subsidiaries whose legal names are
  registered in English (e.g. SAMSUNG ELECTRONICS AMERICA, INC.), plus one unrelated
  fund. The parent was found instead via GLEIF's `fulltext` filter.
  (`gleif/search/entity.legalName/Samsung Electronics/pages/page-001.json`;
  `gleif/search/fulltext/Samsung Electronics Co Ltd/pages/page-001.json`)

- **SEC EDGAR**: CIK 0000879316 is filed under the name **"SAMSUNG ELECTRONICS CO LTD
  /FI"** (the `/FI` suffix marks it as a foreign issuer in EDGAR's own naming
  convention). `entityType` is `"other"` and it has never listed a ticker or exchange
  in EDGAR. (`edgar/by-cik/0000879316/submissions/submissions.json`)

## 2. Identifiers and where each comes from

| Identifier | Value | Source | Notes |
|---|---|---|---|
| LEI | `9884007ER46L6N7EI764` | GLEIF record | Status ACTIVE / registration status ISSUED |
| Korean business registration no. | `124-81-00998` | GLEIF `entity.registeredAs`, validated at registration authority id `RA000657` | This is the Korean corporate registry number, not a US EIN |
| BIC (SWIFT) | `SECTKRSEXXX` | GLEIF record | |
| S&P Global company id | `91868` | GLEIF record (`spglobal` field) | Third-party cross-reference GLEIF carries, not itself an official ID |
| ISINs (15 securities) | e.g. `US7960505086`, `US0019071044`, `US796050AA00`, … | GLEIF `isins` relationship | These are US-format ISINs for Samsung ADR/debt-type instruments associated with the LEI; full list of 15 in evidence.md |
| SEC CIK | `0000879316` | EDGAR `submissions.json` | Filer name "SAMSUNG ELECTRONICS CO LTD /FI" |
| EDGAR-recorded EIN | `95-3170778` | EDGAR `submissions.json` (`ein` field) | Per the cross-registry catalog, EDGAR's `ein` field for a foreign filer is not necessarily an actual US EIN issued to the entity's home jurisdiction — treat as EDGAR bookkeeping, not a confirmed US tax ID |
| EDGAR `lei` field | `null` | EDGAR `submissions.json` | Confirms the catalog's warning: EDGAR's own LEI field is empty for this filer, so the CIK↔LEI link above is **our own name/jurisdiction match**, not a link either registry asserts |

## 3. Registration and status

- **GLEIF (LEI 9884007ER46L6N7EI764)**: Category `GENERAL`, legal form code `5RCH`,
  jurisdiction `KR`, status `ACTIVE`, entity creation date `1969-01-13`. Legal and
  headquarters address: 129 Samsung-ro, Yeongtong-gu, Suwon-si, Gyeonggi-do, Korea,
  postal code 16677. LEI registration status `ISSUED`, initial registration
  2017-09-27, last updated 2025-10-31, next renewal 2026-10-31, managed by LOU
  `9884008RRMX1X5HV6625`. (`gleif/by-lei/9884007ER46L6N7EI764/record/record.json`)
- **Parent reporting**: both `direct-parent-reporting-exception` and
  `ultimate-parent-reporting-exception` are populated with category
  `DIRECT_ACCOUNTING_CONSOLIDATION_PARENT` / `ULTIMATE_ACCOUNTING_CONSOLIDATION_PARENT`
  and reason `NO_KNOWN_PERSON` — GLEIF's data model treats Samsung Electronics as not
  having a reportable corporate parent (consistent with its being the ultimate,
  widely-held listed parent of the Samsung Electronics group, not a subsidiary of
  another LEI-bearing entity).
  (`gleif/by-lei/9884007ER46L6N7EI764/direct-parent-reporting-exception/direct-parent-reporting-exception.json`,
  `gleif/by-lei/9884007ER46L6N7EI764/ultimate-parent-reporting-exception/ultimate-parent-reporting-exception.json`)
- **Branch**: GLEIF records one branch of this LEI — **SAMSUNG ELECTRONICS COMPANY
  LIMITED**, LEI `335800A16O7QD1TDQO62`, category `BRANCH`, registered in Mumbai,
  India (registeredAs `F04529`), status `ACTIVE`.
  (`gleif/by-lei/9884007ER46L6N7EI764/branches/branches.json`)
- **Direct/ultimate children**: GLEIF lists 6 direct-child relationships (page shown)
  and 22 ultimate-child relationships in total, including 삼성디스플레이(주) / SAMSUNG
  DISPLAY Co., Ltd. (Korea), Samsung Malaysia Electronics, Samsung R&D Institute India
  – Bangalore, Samsung Eletronica da Amazonia (Brazil), and the Harman International
  group of entities (US, China, Hungary, Romania, Netherlands, Denmark, Russia,
  Mexico, Germany) — consistent with Samsung's 2017 acquisition of Harman
  International.
  (`gleif/by-lei/9884007ER46L6N7EI764/direct-children/direct-children.json`,
  `gleif/by-lei/9884007ER46L6N7EI764/ultimate-children/ultimate-children.json`)
- **EDGAR (CIK 0000879316)**: `stateOfIncorporation` = `M5` ("Korea, Republic of"),
  `entityType` = `other`, no tickers, no exchanges, no former names on file. Mailing
  and business address on file: 250 2-ka Taepyung-ro, Chung-ku, Seoul, Korea
  (zip 100742) — an older-format Seoul central-Seoul address, distinct from the
  Suwon address GLEIF carries for the same company, consistent with EDGAR not having
  refreshed this filer's address since its filings ended in 2015.
  (`edgar/by-cik/0000879316/submissions/submissions.json`)

## 4. What the filings say

- EDGAR's full-text search for "Samsung Electronics" returns **213 hits**, capped to
  documents where the name appears; the CIK 0000879316 filing history (251 filings in
  `submissions.json`, `filings.recent`) is dominated by:
  - **196 "SUPPL"** filings — auto-generated paper submissions dated 2002–2010, mostly
    routine supplemental filings tied to a paper file number (082-03109);
  - **9 "ARS"** (Annual Report to Shareholders, paper) filings;
  - **18 SC 14D1/A + 2 SC 14D1** and **2 SC 14D9/A + 1 SC 14D9** — tender-offer related
    filings;
  - **9 SC 13E3/A + 1 SC 13E3** — going-private transaction filings;
  - **6 SC 13D/A + 1 SC 13D** and **2 SC 13G/A + 2 SC 13G** — beneficial-ownership
    disclosures, most recently in the Seagate Technology plc, Rambus Inc., and
    SunEdison Semiconductor Ltd filings (2011–2015), where Samsung Electronics
    co-files as the reporting person alongside the issuer;
  - **1 Form 3** and **1 Form 4** — insider ownership reports (2012, 2013), again
    tied to the Seagate Technology plc holding.
  - Filing dates in `submissions.json` run **1995-03-06 to 2015-01-20**; no filings
    are recorded after January 2015.
    (`edgar/by-cik/0000879316/submissions/submissions.json`)
- No 10-K, 10-Q, 20-F, or other annual/periodic US-GAAP or IFRS financial-statement
  filing is present for this CIK, and the XBRL company-facts endpoint returns an S3
  "NoSuchKey" error rather than data — EDGAR carries **no structured financial facts**
  for Samsung Electronics. This is consistent with Samsung Electronics being a
  Korea-listed issuer that reports financials to Korea's own regulator (DART/KRX), not
  as a US SEC periodic filer; the disclosures on EDGAR here are limited to
  ownership/tender-offer events involving Samsung as an investor in US issuers.
  (`edgar/by-cik/0000879316/facts/facts.json`)
- GLEIF's ISIN relationship for LEI `9884007ER46L6N7EI764` lists 15 US-format ISINs
  (e.g. `US7960505086`, `US7960503008`, `US0019071044`); GLEIF does not describe what
  each ISIN represents (equity, ADR, or debt instrument) — that detail is outside what
  either registry discloses here.
  (`gleif/by-lei/9884007ER46L6N7EI764/isins/isins.json`)

## Data limits

- **No confirmed cross-registry key.** GLEIF's LEI and EDGAR's CIK are joined here only
  by legal name ("Samsung Electronics") and jurisdiction (Korea/`M5`); EDGAR's own
  `lei` field for CIK 0000879316 is `null`, so this is a **possible match — needs
  confirmation** against a further source (e.g. matching the Suwon vs. Seoul addresses,
  or a DART filing) if a firm one-to-one link is required.
- **No US financial statements.** Neither registry carries Samsung Electronics' income
  statement, balance sheet, revenue, or employee count; EDGAR has no XBRL facts for
  this CIK, and GLEIF does not carry financial statements at all (per the top-level
  catalog). Any question about Samsung's revenue, profit, or headcount cannot be
  answered from these two registries.
- **Name collision risk realized.** A plain GLEIF `entity.legalName` search for
  "Samsung Electronics" returns 47 records, including **"ProShares Ultra Samsung
  Electronics"**, a US leveraged ETF fund (LEI `529900JUHLXQE4F70208`, category
  `FUND`) that tracks Samsung Electronics stock but is not Samsung Electronics itself,
  and two Hong Kong CSOP leveraged/inverse products with the same characteristic. These
  are excluded from the profile above but are worth flagging if the same search is
  repeated.
- **EDGAR coverage stops in 2015.** The CIK 0000879316 filing history contains nothing
  after 2015-01-20; whether Samsung Electronics has any more recent EDGAR-registered
  activity under a different CIK was not separately searched beyond the 213-hit
  full-text query (capped at the first page of results), so a residual gap is possible.
- **GLEIF group structure is partial by design.** The "ultimate-children" total (22)
  and "direct-children" total (6, on the page read) reflect only relationships GLEIF
  itself has corroborated; Samsung Electronics' full corporate structure (it has
  dozens of subsidiaries worldwide, as the 47-record legalName search alone shows) is
  not exhaustively modeled as parent/child links in GLEIF — most of the 47
  "Samsung Electronics …" national entities found above are not listed under
  direct/ultimate-children, meaning GLEIF's ownership graph for this LEI is
  incomplete relative to the actual corporate group.

## Next steps

- If a firm CIK↔LEI link is required, cross-check the EDGAR business address
  (Seoul, Taepyung-ro) against a dated Samsung Electronics corporate filing (e.g. a
  DART/KRX disclosure) to confirm both records describe the same legal entity at
  different points in time.
- To answer questions about Samsung Electronics' financial performance, headcount, or
  business description, consult Korea's DART system or Samsung's own investor-relations
  disclosures — outside the scope of GLEIF and EDGAR.
- To map the full corporate group, treat the GLEIF `ultimate-children` list (22 total)
  as a floor, not a ceiling — the fact that most of the 47 "Samsung Electronics"-named
  national sales subsidiaries found by legalName search do not appear in that list
  indicates GLEIF's parent/child linkage is not exhaustively populated for this LEI.
