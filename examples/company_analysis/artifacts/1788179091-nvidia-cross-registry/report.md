# NVIDIA — GLEIF vs. EDGAR

## Summary

NVIDIA appears in both registries as a clean, high-confidence match: SEC EDGAR's
**NVIDIA CORP** (CIK 0001045810) and GLEIF's **NVIDIA CORPORATION**
(LEI `549300S4KLFTLO7GSQ80`). Both describe the same Delaware corporation
headquartered at 2788 San Tomas Expressway, Santa Clara, CA — the address, state of
incorporation, and legal-name stem all line up. They disagree only where the two
registries simply track different things: EDGAR carries an EIN, ticker, SIC code and
filing history that GLEIF doesn't record, and GLEIF carries a registered-agent address,
legal-entity ID and a corporate family tree (parent/subsidiaries) that EDGAR doesn't
carry. GLEIF's own `lei` field is null in EDGAR's record, so the registries could not be
joined by ID — the match rests on name plus address/state corroboration.

## The two records

**EDGAR** — `edgar/by-cik/0001045810/submissions/submissions.json`
- Name: `NVIDIA CORP`, tickers `["NVDA"]`, exchange `Nasdaq`
- CIK: `0001045810`; EIN: `943177549`; `lei` field in this record: `null`
- SIC: `3674` — "Semiconductors & Related Devices"
- State of incorporation: `DE`
- Business/mailing address: 2788 San Tomas Expressway, Santa Clara, CA 95051
- Fiscal year end: `0131`; filer category: `Large accelerated filer`
- Former name: `NVIDIA CORP/CA` (1998-05-07 to 2002-06-04)
- 300+ recent filings on record (10-K, 10-Q, 8-K, Forms 3/4/5, 13F-HR, SCHEDULE 13G, etc.)

**GLEIF** — `gleif/by-lei/549300S4KLFTLO7GSQ80/record/record.json` (identical to the hit
returned by `gleif/search/entity.legalName/NVIDIA/pages/page-001.json`)
- Legal name: `NVIDIA CORPORATION`
- LEI: `549300S4KLFTLO7GSQ80`
- Legal address (registered agent): C/O Corporation Service Company, 251 Little Falls
  Drive, Wilmington, DE 19808
- Headquarters address: 2788 San Tomas Expressway, Santa Clara, CA 95051
- Jurisdiction: `US-DE`; legal form code `XTIQ`; category `GENERAL`; status `ACTIVE`
- Registered-as (Delaware entity number): `2862596`; `ocid`: `us_de/2862596`
- Entity creation date: `1998-02-24`
- LEI initial registration: `2013-04-09`; last update: `2026-01-07`; next renewal:
  `2027-02-06`
- BIC: `NVDAUS6SXXX`; S&P Global company ID: `32307`
- Three direct subsidiaries recorded: `NVIDIA INTERNATIONAL, INC.` (US-DE,
  LEI `549300EK80J3WR5TSA69`), `NVIDIA GRAPHICS PRIVATE LIMITED` (India), and
  `MELLANOX TECHNOLOGIES, LTD.` (Israel)

## How they were matched, and how sure

The two registries share no common identifier — EDGAR's `lei` field for CIK 0001045810
is `null`, so the join is by name plus corroborating facts, per the cross-registry
catalog's guidance. Confidence: **high**.

Checks performed:
1. **Name**: `NVIDIA CORP` (EDGAR) vs. `NVIDIA CORPORATION` (GLEIF) — same stem, only the
   legal-form suffix differs (this is exactly the kind of suffix variance the catalog
   warns about, so it alone would not be conclusive).
2. **Address**: EDGAR's business/mailing address (2788 San Tomas Expressway, Santa
   Clara, CA 95051) is byte-identical to GLEIF's `headquartersAddress` for
   `549300S4KLFTLO7GSQ80`. This is the strongest corroborating signal.
3. **State of incorporation**: EDGAR says `DE`; GLEIF's `jurisdiction` is `US-DE` and
   `legalAddress` is a Delaware registered-agent address (Corporation Service Company,
   Wilmington) — consistent with a Delaware-incorporated company.
4. **Candidate elimination**: the same GLEIF name search returned 20 records, including
   `NVIDIA INTERNATIONAL, INC.` (a subsidiary, also GLEIF-listed with US-DE jurisdiction
   and the same HQ address, but a different legal entity and LEI) and several ETFs/funds
   named after NVIDIA (`Defiance Nvidia Ventures ETF`, `NINEPOINT NVIDIA HIGHSHARES ETF`,
   category `FUND`) that are not the manufacturer at all — confirming the catalog's
   warning that "a search for NVIDIA returns a fund before it returns the manufacturer."
   `NVIDIA CORPORATION` was chosen over these because its category is `GENERAL` (an
   operating company, not a fund) and its name/address match EDGAR's registrant exactly.

Given the exact address match plus consistent jurisdiction and the name being the
company's evident legal-form variant (CORP vs. CORPORATION), this is treated as the same
legal entity, not merely a plausible candidate — but it remains a **name-based match**,
not an ID-based one, since no shared key exists.

## Where they agree

- **Same company, same HQ**: 2788 San Tomas Expressway, Santa Clara, CA 95051 —
  identical in both (`edgar/by-cik/0001045810/submissions/submissions.json`
  `addresses.business`; `gleif` record `entity.headquartersAddress`).
- **Same jurisdiction of incorporation**: Delaware (`stateOfIncorporation":"DE"` in
  EDGAR; `jurisdiction":"US-DE"` in GLEIF).
- **Same legal-name stem**: "NVIDIA" is the operative word in both; GLEIF's fuller
  "CORPORATION" vs. EDGAR's abbreviated "CORP" is a suffix/style difference, not a
  disagreement about the entity.
- **Both show it as an active, currently operating entity**: EDGAR files continuously
  through 2026; GLEIF status is `ACTIVE` with an unexpired LEI (`expiration.date: null`,
  next renewal `2027-02-06`).

## Where they differ

These are differences of *coverage*, not disagreements about fact — each registry
records what its regulator asks for:

| Field | EDGAR | GLEIF | Note |
|---|---|---|---|
| Primary identifier | CIK `0001045810` | LEI `549300S4KLFTLO7GSQ80` | No shared key; EDGAR's own `lei` field is `null` |
| Legal name | `NVIDIA CORP` | `NVIDIA CORPORATION` | Suffix/style difference only |
| EIN | `943177549` | not carried | GLEIF records the state filing number instead (see below) |
| Registered/filing number | not carried | Delaware entity no. `2862596` (`registeredAs`) | EDGAR does not carry a state filing number |
| Registered address used | single business/mailing address (Santa Clara) | separate legal address (registered agent, Wilmington DE) vs. headquarters address (Santa Clara) | GLEIF distinguishes legal-domicile address from HQ; EDGAR does not |
| Ticker / exchange | `NVDA` / Nasdaq | not carried | GLEIF has no securities-market field of this kind (ISIN linkage exists but wasn't queried here) |
| SIC / industry code | `3674` Semiconductors & Related Devices | not carried | Industry classification is an EDGAR-only field |
| Corporate family (subsidiaries) | not carried | 3 direct children listed (NVIDIA International Inc., NVIDIA Graphics Private Limited, Mellanox Technologies Ltd.) | GLEIF's relationship data has no EDGAR counterpart in this record |
| Former name | `NVIDIA CORP/CA` (1998–2002) | not shown as a name history field (creation date `1998-02-24` only) | EDGAR tracks name-change history explicitly |
| Filing/disclosure history | extensive (10-K, 10-Q, 8-K, Forms 3/4/5, 13F-HR, etc.) | none — GLEIF is not a disclosure registry | Out of scope for GLEIF by design |

## Data limits

- **No shared key.** EDGAR's `lei` field for this CIK is `null`, so the match is by
  name and address, corroborated but not identifier-based.
- **GLEIF name search returns non-company entities.** The same query surfaced ETFs and
  funds named after NVIDIA; these were excluded on category (`FUND`) grounds, not
  examined further, since they were clearly not candidates for the operating company.
- **Subsidiary detail is one-sided.** GLEIF's parent/child relationships (e.g., Mellanox
  Technologies, NVIDIA International Inc., NVIDIA Graphics Private Limited) have no
  cross-check performed against EDGAR subsidiary disclosures (e.g., Exhibit 21) in this
  run — that would need a separate document pull.
- **Financial statements, sanctions, litigation, news**: neither registry carries these
  (per `CATALOG.md`), so no comparison on those dimensions is possible here.
- **EDGAR's full-text search returns filings, not just the company.** Many hits in
  `edgar/search/entityName/NVIDIA/pages/page-001.json` are insiders (Form 4/3 filers)
  and other companies' filings that merely mention NVIDIA (e.g., Nebius Group N.V.
  Schedule 13G); only rows carrying CIK `0001045810` as the registrant were used.

## Next steps

- If a harder link is needed, check GLEIF's `isins` relationship for
  `549300S4KLFTLO7GSQ80` against NVDA's known ISIN/CUSIP to add a second, independent
  corroboration beyond address.
- Pull EDGAR's Exhibit 21 (list of subsidiaries) from a recent 10-K and compare it
  against GLEIF's direct/ultimate-children list to see how many subsidiary entities
  appear in both registries.
- If ownership structure matters, walk GLEIF's `ultimate-parent`/`ultimate-children`
  relationships for `549300S4KLFTLO7GSQ80` (not fetched in this run — its own record
  shows no `direct-parent` link, consistent with NVIDIA Corporation being the top of its
  GLEIF-recorded family).
