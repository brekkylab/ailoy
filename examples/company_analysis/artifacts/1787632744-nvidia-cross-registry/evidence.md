# Evidence

All paths below are relative to the mount root (the `.../ailoy-company-analysis`
prefix is omitted per instructions).

## GLEIF

**Path:** `gleif/search/entity.legalName/NVIDIA/pages/page-001.json`
Retrieved by descending `gleif/search/entity.legalName/NVIDIA/pages` and reading
`page-001.json` (the only paid request in this branch). `meta.pagination.total`
= 19.

Claims sourced from `data[0]` (the `NVIDIA CORPORATION` record, LEI
`549300S4KLFTLO7GSQ80`):
- LEI `549300S4KLFTLO7GSQ80`
- Legal name "NVIDIA CORPORATION"
- Category `GENERAL`, status `ACTIVE`, legal form id `XTIQ`
- Jurisdiction `US-DE`; `registeredAs` "2862596"; `registeredAt.id` "RA000602"
- Legal address: C/O Corporation Service Company, 251 Little Falls Drive,
  Wilmington, US-DE, 19808
- Headquarters address: 2788 San Tomas Expressway, Santa Clara, US-CA, 95051
- `creationDate` 1998-02-24T00:00:00Z
- Event: `CHANGE_HQ_ADDRESS`, effective 2024-02-05
- Registration: `initialRegistrationDate` 2013-04-09T17:39:00Z, `lastUpdateDate`
  2026-01-07T01:29:18Z, `nextRenewalDate` 2027-02-06T17:45:00Z,
  `corroborationLevel` "FULLY_CORROBORATED"
- `bic`: `NVDAUS6SXXX`; `spglobal`: `["32307"]`

Claims sourced from `data[1]` (`NVIDIA INTERNATIONAL, INC.`, LEI
`549300EK80J3WR5TSA69`) — used only to show what was ruled out, not as the
matched record:
- Legal name "NVIDIA INTERNATIONAL, INC."
- Same headquarters address as `data[0]` (2788 San Tomas Expressway, Santa
  Clara, US-CA, 95051) and same jurisdiction `US-DE`
- Different `registeredAs`: "6004882"; `creationDate` 2021-06-28

Claims sourced from `data[2..18]` (funds/ETFs/subsidiaries) — used to show
what the "NVIDIA" name match returns besides the parent company:
- "Defiance Nvidia Ventures ETF", LEI `529900YQV0X0R0T0AZ11`, category `FUND`
- "NINEPOINT NVIDIA HIGHSHARES ETF", LEI `894500UB1PEXN75HFT11`, category
  `FUND`, jurisdiction `CA` (Canada)
- "NVIDIA GRAPHICS PRIVATE LIMITED", LEI `335800TGZ6N1BWAN7Q35`, jurisdiction
  `IN`, category `GENERAL`
- "NVIDIA SINGAPORE PTE LTD", LEI `549300PVTLDCMDI67P44`, jurisdiction `SG`,
  category `GENERAL`
- "Nvidia Land Development, LLC", LEI `54930030GWBYMACGJH60`, jurisdiction
  `US-DE`, category `GENERAL`
- Remaining hits are further fund/ETF products (Kurv, Tuttle, CHINAAMC,
  Harvest, Purpose, PurePlay, CSOP, T-Rex, ASA), all category `FUND`.

## EDGAR

**Path:** `edgar/search/entityName/NVIDIA/pages/page-001.json`
Retrieved by descending `edgar/search/entityName/NVIDIA/pages` and reading
`page-001.json`. `hits.total.value` = 2415 (capped display: the search backend
caps at 10,000; this number is below the cap, so it is exact per the catalog's
rule).

Claim sourced: top hit's `_source.display_names` includes
`"NVIDIA CORP  (NVDA)  (CIK 0001045810)"`, and `_source.ciks` includes
`"0001045810"` — this is the CIK used for the by-CIK lookup below. Other CIKs
appearing in this result page (insiders, Vanguard entities, CoreWeave, Nebius
Group) are filers who filed *about* NVIDIA (e.g. Form 4 insider filings,
13F holdings, Schedule 13G), not NVIDIA itself.

**Path:** `edgar/by-cik/0001045810/submissions/submissions.json`
Retrieved directly since the CIK was confirmed via the search above.

Claims sourced:
- `cik`: `"0001045810"`
- `name`: `"NVIDIA CORP"`
- `tickers`: `["NVDA"]`; `exchanges`: `["Nasdaq"]`
- `ein`: `"943177549"`
- `lei`: `null` — confirms the catalog's statement that this field is
  "almost always null"
- `stateOfIncorporation`: `"DE"`
- `sic`: `"3674"`, `sicDescription`: `"Semiconductors & Related Devices"`
- `category`: `"Large accelerated filer"`; `fiscalYearEnd`: `"0131"`
- `formerNames`: `[{"name": "NVIDIA CORP/CA", "from": "1998-05-07T00:00:00.000Z",
  "to": "2002-06-04T00:00:00.000Z"}]`
- `addresses.mailing` and `addresses.business`: both
  `2788 SAN TOMAS EXPRESSWAY, SANTA CLARA, CA, 95051`

## Cross-registry comparison basis

- Address match (GLEIF `entity.headquartersAddress` vs. EDGAR
  `addresses.business`/`addresses.mailing`): both "2788 San Tomas Expressway,
  Santa Clara, CA 95051" — used as the primary corroborating fact for the
  match, per `gleif/search/.../page-001.json` and
  `edgar/by-cik/0001045810/submissions/submissions.json`.
- Jurisdiction match: GLEIF `entity.jurisdiction` = `US-DE`; EDGAR
  `stateOfIncorporation` = `DE` — same fact, different field names.

## Catalog statements relied on directly

- `CATALOG.md` (root): "GLEIF's identifier is the LEI and EDGAR's is the CIK,
  and neither registry carries the other's... EIN does not bridge either" —
  basis for treating this as a name match rather than a key join.
- `edgar/CATALOG.md`: "`ciks` is a list of zero-padded strings —
  `["0001045810"]` — and that is the one to use" — followed when extracting
  the CIK from the search hit.
- `gleif/CATALOG.md`: "Descending is free; opening `pages` is not" — followed
  by naming the filter (`entity.legalName/NVIDIA`) before opening `pages`.
