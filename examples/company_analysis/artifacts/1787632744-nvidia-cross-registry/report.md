# NVIDIA — GLEIF vs. EDGAR

## Summary

NVIDIA's operating company is identified in GLEIF as **NVIDIA CORPORATION**, LEI
`549300S4KLFTLO7GSQ80`, and in EDGAR as **NVIDIA CORP**, CIK `0001045810` / ticker
`NVDA`. The two records agree on headquarters address (2788 San Tomas Expressway,
Santa Clara, CA), state of incorporation (Delaware), and active/operating status.
They diverge in what they cover — GLEIF carries a Delaware file number and LEI
registration history, EDGAR carries EIN, SIC code, tickers, exchange and filing
history — because the two registries record different things about the same
company. The match is **high confidence**: no shared key exists (see Data limits),
but name, jurisdiction and exact street address all line up on one candidate out
of 19 GLEIF hits for "NVIDIA".

## 1. The two records

**GLEIF** (`gleif/search/entity.legalName/NVIDIA/pages/page-001.json`, `data[0]`):
- LEI: `549300S4KLFTLO7GSQ80`
- Legal name: `NVIDIA CORPORATION`
- Legal form: `XTIQ`, category `GENERAL`, status `ACTIVE`
- Jurisdiction: `US-DE`; registered as `2862596` with Delaware registrar `RA000602`
- Legal address: c/o Corporation Service Company, 251 Little Falls Drive,
  Wilmington, US-DE 19808
- Headquarters address: 2788 San Tomas Expressway, Santa Clara, US-CA 95051
- Entity creation date: 1998-02-24; last HQ-address change event: 2024-02-05
- LEI registration: initial 2013-04-09, last update 2026-01-07, next renewal
  2027-02-06, fully corroborated
- BIC: `NVDAUS6SXXX`; associated S&P Global ID `32307`

**EDGAR** (`edgar/by-cik/0001045810/submissions/submissions.json`):
- CIK: `0001045810`
- Name: `NVIDIA CORP` (former name `NVIDIA CORP/CA`, 1998-05-07 to 2002-06-04)
- Ticker: `NVDA` on Nasdaq
- EIN: `943177549`
- State of incorporation: `DE`
- SIC: `3674` — Semiconductors & Related Devices
- Filer category: Large accelerated filer; fiscal year end 01/31
- Mailing and business address (identical): 2788 San Tomas Expressway,
  Santa Clara, CA 95051
- `lei` field in this record: `null`

## 2. How they were matched, and how sure

There is no shared identifier — GLEIF exposes LEIs, EDGAR exposes CIKs, and
EDGAR's own `lei` field for NVIDIA is empty (confirmed above), matching the
catalog's warning that this bridge does not work in practice.

The match was made by:
1. Searching GLEIF for legal name "NVIDIA" — 19 records returned, most of them
   funds/ETFs tracking NVIDIA (e.g. "Defiance Nvidia Ventures ETF",
   "T-Rex 2X Long NVIDIA Daily Target ETF") or foreign subsidiaries
   ("NVIDIA SINGAPORE PTE LTD", "NVIDIA GRAPHICS PRIVATE LIMITED").
2. Selecting the one record categorized `GENERAL` (not `FUND`), jurisdiction
   `US-DE`, legal name exactly "NVIDIA CORPORATION" — the parent entity, not a
   subsidiary or a tracking fund.
3. Searching EDGAR full text index for "NVIDIA" — top hit is CIK `0001045810`,
   `NVIDIA CORP (NVDA)`.
4. Cross-checking the two candidates on data outside the name: **both give the
   identical headquarters street address** (2788 San Tomas Expressway, Santa
   Clara, CA 95051) and **both give Delaware as the jurisdiction of
   incorporation**.

Given an exact address match plus an exact jurisdiction match, on top of the
name match, this is a **high-confidence match** — as high as this task
structure permits, since it is still ultimately a name-plus-attributes
correlation rather than a shared key. It is reported as such, not as fact.

Two entities were explicitly *not* picked despite matching on name:
- `NVIDIA INTERNATIONAL, INC.` (LEI `549300EK80J3WR5TSA69`) — same Santa Clara
  HQ address and same Delaware jurisdiction, but a distinct legal entity
  (different LEI, different Delaware file number `6004882` vs `2862596`,
  created 2021 vs 1998). This looks like a subsidiary of NVIDIA CORPORATION,
  not the SEC registrant.
- Several fund/ETF names containing "NVIDIA" (Defiance, Kurv, Tuttle, Ninepoint,
  Harvest, CSOP, T-Rex, ASA, PurePlay) — these are investment products that
  hold or track NVIDIA stock, evidenced by category `FUND` and fund-manager
  addresses unrelated to Santa Clara. They are not NVIDIA itself.

## 3. Where they agree

| Fact | GLEIF | EDGAR |
|---|---|---|
| Headquarters street address | 2788 San Tomas Expressway, Santa Clara, CA 95051 | 2788 San Tomas Expressway, Santa Clara, CA 95051 |
| State/jurisdiction of incorporation | US-DE (Delaware) | DE |
| Entity is active/operating | status `ACTIVE` | filer status implies operating (Large accelerated filer, current filings) |
| Legal name (core) | "NVIDIA CORPORATION" | "NVIDIA CORP" |

Sources: `gleif/search/entity.legalName/NVIDIA/pages/page-001.json` (record with
LEI `549300S4KLFTLO7GSQ80`); `edgar/by-cik/0001045810/submissions/submissions.json`.

## 4. Where they disagree / do not overlap

These are not contradictions — each registry carries fields the other does not:

- **Legal-form spelling**: GLEIF's legal name is "NVIDIA CORPORATION"; EDGAR's
  is "NVIDIA CORP" — a suffix difference typical of EDGAR's abbreviated
  registrant names, not a discrepancy about the underlying entity.
- **Registered address vs. mailing address**: GLEIF's *legal* address (used for
  service of process) is c/o Corporation Service Company, 251 Little Falls
  Drive, Wilmington, DE 19808 — a registered-agent address, different from its
  own *headquarters* address field, which matches EDGAR's business/mailing
  address. EDGAR does not distinguish a separate "legal" address at all.
- **Identifiers each side carries that the other lacks**: GLEIF has no EIN, SIC
  code, ticker, or exchange; EDGAR has no LEI (field is `null` for this CIK,
  confirmed directly), no Delaware file number, no LEI-registration dates.
- **Former name**: EDGAR records a former name, "NVIDIA CORP/CA" (1998–2002).
  GLEIF's record shows no `otherNames` entries for the current record — this
  is not evidence the name change didn't happen, only that GLEIF's snapshot
  does not carry legal-name history the way EDGAR does.
- **Creation/registration dates differ in what they measure**: GLEIF's
  `creationDate` (1998-02-24) is entity formation; EDGAR carries no formation
  date at all, only a former-name window starting 1998-05-07 — close but not
  the same event, and not comparable without more care.

## Data limits

- No shared identifier bridges the two registries; the match above is a name
  match corroborated by address and jurisdiction, per the catalog's warning
  that "a search for NVIDIA returns a fund before it returns the manufacturer."
  Confirmed directly: EDGAR's own `lei` field for CIK 0001045810 is `null`.
- GLEIF's "NVIDIA" search returned 19 records; most were fund/ETF products
  tracking NVIDIA stock or NVIDIA subsidiaries, not NVIDIA CORPORATION itself.
  Only the one flagged `GENERAL`/`US-DE` with the matching HQ address was used.
- `NVIDIA INTERNATIONAL, INC.` (LEI `549300EK80J3WR5TSA69`) shares the same HQ
  address and Delaware jurisdiction as NVIDIA CORPORATION but is a separate LEI
  record — presumed subsidiary, not confirmed against a parent/child
  relationship record (GLEIF exposes `direct-parent`/`ultimate-parent`
  relationship links for this LEI that were not queried in this run).
- Neither registry contains financial statements, sanctions, litigation, or
  news — as the top-level catalog states outright, that class of question has
  no answer in this data.
- EDGAR's full-text search (2,415 hits for "NVIDIA", capped display at one
  page of ~24 shown here) surfaces filings that *mention* NVIDIA — insiders,
  index funds, and counterparties like CoreWeave and Nebius filing about
  NVIDIA — not just NVIDIA's own filings; only rows with CIK `0001045810` are
  NVIDIA the registrant.

## Next steps

- Query `gleif/by-lei/549300S4KLFTLO7GSQ80/direct-parent` and
  `.../ultimate-children` (or the `direct-children` relationship link already
  present in the GLEIF record) to confirm `NVIDIA INTERNATIONAL, INC.` and the
  Singapore/India subsidiaries as children of NVIDIA CORPORATION, rather than
  inferring it from address overlap alone.
- If a financial or ownership comparison is wanted, EDGAR's `facts`/`concept`
  endpoints under `by-cik/0001045810/` carry XBRL financials that GLEIF does
  not — a natural follow-up query, not run here since it wasn't asked for.
