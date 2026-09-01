# Evidence

Each line: claim — source path (from mount root, no leading `.`).

## Identification

- EDGAR registrant for "NVIDIA" is CIK `0001045810`, name `NVIDIA CORP`, ticker `NVDA` —
  edgar/search/entityName/NVIDIA/pages/page-001.json (hit `_source.ciks` /
  `display_names`: `"NVIDIA CORP  (NVDA)  (CIK 0001045810)"`)
- Confirmed registrant record — edgar/by-cik/0001045810/submissions/submissions.json
  (`"cik":"0001045810"`, `"name":"NVIDIA CORP"`, `"tickers":["NVDA"]`,
  `"exchanges":["Nasdaq"]`)
- GLEIF name search for "NVIDIA" returns 20 records, 1 page —
  gleif/search/entity.legalName/NVIDIA/pages/_README.md
  ("This query has 1 page(s), 20 record(s).")
- GLEIF record selected as the operating company: `NVIDIA CORPORATION`,
  LEI `549300S4KLFTLO7GSQ80`, `category":"GENERAL"` —
  gleif/search/entity.legalName/NVIDIA/pages/page-001.json (first record in `data`)
- Same record, addressed directly — gleif/by-lei/549300S4KLFTLO7GSQ80/record/record.json
  (same content as the search hit)

## Candidates excluded from the GLEIF match

- `NVIDIA INTERNATIONAL, INC.`, LEI `549300EK80J3WR5TSA69`, US-DE, same HQ address as
  NVIDIA CORPORATION but a distinct legal entity (subsidiary, per
  direct-children listing below) — gleif/search/entity.legalName/NVIDIA/pages/page-001.json
- `Defiance Nvidia Ventures ETF`, LEI `529900YQV0X0R0T0AZ11`, `category":"FUND"` — a fund
  tracking NVIDIA, not the company — gleif/search/entity.legalName/NVIDIA/pages/page-001.json
- `NINEPOINT NVIDIA HIGHSHARES ETF`, LEI `894500UB1PEXN75HFT11`, another fund named after
  NVIDIA — gleif/search/entity.legalName/NVIDIA/pages/page-001.json

## Where they agree

- Headquarters address 2788 San Tomas Expressway, Santa Clara, CA 95051 in both:
  - edgar/by-cik/0001045810/submissions/submissions.json
    (`"addresses":{"mailing":{"street1":"2788 SAN TOMAS EXPRESSWAY","city":"SANTA CLARA","stateOrCountry":"CA","zipCode":"95051"}...`, `"business":{...same...}`)
  - gleif/by-lei/549300S4KLFTLO7GSQ80/record/record.json
    (`"headquartersAddress":{"addressLines":["2788 SAN TOMAS EXPRESSWAY"],"city":"SANTA CLARA","region":"US-CA","country":"US","postalCode":"95051"}`)
- Delaware incorporation/jurisdiction in both:
  - edgar/by-cik/0001045810/submissions/submissions.json (`"stateOfIncorporation":"DE"`)
  - gleif/by-lei/549300S4KLFTLO7GSQ80/record/record.json (`"jurisdiction":"US-DE"`)
- Active/operating status in both:
  - edgar/by-cik/0001045810/submissions/submissions.json (`"entityType":"operating"`,
    `"category":"Large accelerated filer"`, continuing filings through 2026)
  - gleif/by-lei/549300S4KLFTLO7GSQ80/record/record.json (`"status":"ACTIVE"`,
    `"expiration":{"date":null,"reason":null}`, `"registration":{"status":"ISSUED",...,"nextRenewalDate":"2027-02-06T17:45:00Z"}`)

## Where they differ (coverage, not conflict)

- EDGAR carries EIN `943177549`; GLEIF carries no EIN field (per
  ./CATALOG.md: "EIN does not bridge either, because GLEIF records ... a state filing
  number ... rather than the EIN") —
  edgar/by-cik/0001045810/submissions/submissions.json (`"ein":"943177549"`);
  GLEIF's own record has no EIN key at all —
  gleif/by-lei/549300S4KLFTLO7GSQ80/record/record.json
- GLEIF carries a Delaware registered-agent (`legalAddress`) distinct from HQ, and a
  state registration number (`registeredAs":"2862596"`, `ocid":"us_de/2862596"`) that
  EDGAR does not carry —
  gleif/by-lei/549300S4KLFTLO7GSQ80/record/record.json
- EDGAR's `lei` field for this CIK is null, confirming no ID bridge exists —
  edgar/by-cik/0001045810/submissions/submissions.json (`"lei":null`)
- EDGAR carries SIC code `3674` ("Semiconductors & Related Devices"); GLEIF record has
  no industry-classification field —
  edgar/by-cik/0001045810/submissions/submissions.json (`"sic":"3674"`,
  `"sicDescription":"Semiconductors & Related Devices"`)
- EDGAR carries former name `NVIDIA CORP/CA` (1998-05-07 to 2002-06-04); GLEIF has no
  name-history array populated for this record (only `creationDate":"1998-02-24"`) —
  edgar/by-cik/0001045810/submissions/submissions.json (`"formerNames":[{"name":"NVIDIA CORP/CA","from":"1998-05-07T04:00:00.000Z","to":"2002-06-04T04:00:00.000Z"}]`);
  gleif/by-lei/549300S4KLFTLO7GSQ80/record/record.json (`"otherNames":[]`)
- GLEIF lists 3 direct subsidiaries; EDGAR's submissions.json carries no
  subsidiary/relationship data —
  gleif/by-lei/549300S4KLFTLO7GSQ80/direct-children/direct-children.json
  (`NVIDIA GRAPHICS PRIVATE LIMITED`, LEI `335800TGZ6N1BWAN7Q35`, India;
  `MELLANOX TECHNOLOGIES, LTD.`, LEI `5299007E0TX53VXVFJ97`, Israel;
  `NVIDIA INTERNATIONAL, INC.`, LEI `549300EK80J3WR5TSA69`, US-DE)
- EDGAR carries an extensive filing history (10-K/10-Q/8-K/Forms 3,4,5/13F-HR/Schedule
  13G, etc.); GLEIF is not a disclosure registry and carries none of this —
  edgar/by-cik/0001045810/submissions/submissions.json (`filings.recent.accessionNumber`
  array, hundreds of entries spanning 2024–2026)

## Registry scope notes

- Neither registry covers financial statements narrative, sanctions, or litigation —
  ./CATALOG.md ("What neither covers")
- GLEIF search fields available are limited to a fixed list (no city/date filters) —
  gleif/CATALOG.md
- EDGAR full-text search results include filings that merely *mention* NVIDIA (e.g.
  insider Form 4 filers, or Nebius Group N.V.'s Schedule 13G naming NVIDIA as the other
  filer) — only rows where CIK `0001045810` is itself the registrant were treated as
  NVIDIA's own record —
  edgar/search/entityName/NVIDIA/pages/page-001.json (e.g. hit
  `"display_names":["Nebius Group N.V.  (NBIS)  (CIK 0001513845)","NVIDIA CORP  (CIK 0001045810)"]`)
