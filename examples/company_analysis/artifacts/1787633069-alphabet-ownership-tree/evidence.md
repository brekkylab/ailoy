# Evidence

All paths below are relative to the mount root and omit the
`/var/folders/.../ailoy-company-analysis` prefix, per instructions.

## Identifying Alphabet Inc. in GLEIF

- **Claim:** Alphabet Inc.'s LEI is `5493006MHB84DD0ZWV18`, jurisdiction US-DE,
  registered as 5786925, HQ 1600 Amphitheatre Parkway, Mountain View, CA, status ACTIVE.
  - `gleif/search/entity.legalName/Alphabet Inc/pages/page-001.json` — record with
    `id: "5493006MHB84DD0ZWV18"`, `legalName.name: "ALPHABET INC."`,
    `legalAddress.country: "US"`, `jurisdiction: "US-DE"`,
    `headquartersAddress: {city: "Mountain View", region: "US-CA"}`.
- **Claim:** the same search also returns four unrelated "Alphabet …" entities that do
  not belong to this group: Alphabet Energy Inc (India, Bilaspur), Alphabet Minerals
  Inc (India, Bilaspur), Alphabet Holding Company, Inc. (Delaware, registered as
  4846758, status LAPSED — a different registration number from Alphabet Inc.'s
  5786925), and Alphabet Millcreek Self Storage Inc. (Ontario, Canada).
  - Same file, records with ids `984500443D78642F8280`, `335800TESL8Z1SFPDI66`,
    `5493000J3OND4HN47659`, `549300HKFWRE70KP4M82`.

## Alphabet Inc.'s own parent (or lack of one)

- **Claim:** GLEIF records no direct parent for Alphabet Inc.; the reason given is
  "no known person."
  - `gleif/by-lei/5493006MHB84DD0ZWV18/direct-parent-reporting-exception/direct-parent-reporting-exception.json`
    — `category: "DIRECT_ACCOUNTING_CONSOLIDATION_PARENT"`, `reason: "NO_KNOWN_PERSON"`.
- **Claim:** same for the ultimate parent.
  - `gleif/by-lei/5493006MHB84DD0ZWV18/ultimate-parent-reporting-exception/ultimate-parent-reporting-exception.json`
    — `category: "ULTIMATE_ACCOUNTING_CONSOLIDATION_PARENT"`, `reason: "NO_KNOWN_PERSON"`.
- **Claim:** no entity in GLEIF's index is recorded as owning Alphabet Inc.
  - `gleif/search/owns/5493006MHB84DD0ZWV18/pages/page-001.json` — `meta.pagination.total: 0`.

## Alphabet Inc.'s disclosed subsidiaries (50 total)

- **Claim:** Alphabet Inc. is recorded as the ultimate owner of 50 entities.
  - `gleif/search/ownedBy/5493006MHB84DD0ZWV18/pages/page-001.json` — `meta.pagination.total: 50`,
    all 50 records returned on a single page (perPage 200).
  - Cross-checked against `gleif/by-lei/5493006MHB84DD0ZWV18/ultimate-children/ultimate-children.json`
    — `meta.pagination.total: 50` (page 1 shows 15 of them; all 15 are a subset of the
    ownedBy list of 50, confirmed programmatically).
- **Claim:** Alphabet Inc. has 33 *direct* children (a strict subset of the 50 above);
  the rest sit one or two tiers further down, under Google LLC, Google International
  LLC, Wiz Inc., or Google Cloud EMEA Limited.
  - `gleif/by-lei/5493006MHB84DD0ZWV18/direct-children/direct-children.json` —
    `meta.pagination.total: 33`.
  - `gleif/by-lei/5493006MHB84DD0ZWV18/direct-child-relationships/direct-child-relationships.json`
    — same total (33), giving the relationship type `IS_DIRECTLY_CONSOLIDATED_BY` for
    each, e.g. Firebase, Inc. (`9845003D44CCA145CC76`) → Alphabet Inc., relationship
    period starting 2014-10-17.

### Entity-by-entity parent (all 50), read from each subsidiary's own record

Format: subsidiary — direct parent as disclosed on the subsidiary's own GLEIF record
(`gleif/by-lei/<LEI>/direct-parent/direct-parent.json`), or, where that file does not
exist, the reporting exception plus the ultimate parent that is still on file.

| Subsidiary (LEI) | Direct parent (source) |
|---|---|
| GOOGLE ARGENTINA S.R.L. (`98450066EDFC0Y6C2A54`) | ALPHABET INC. — `by-lei/98450066EDFC0Y6C2A54/direct-parent/direct-parent.json` |
| Google Austria GmbH (`984500D8D6D8C77L5B32`) | ALPHABET INC. — `by-lei/984500D8D6D8C77L5B32/direct-parent/direct-parent.json` |
| GOOGLE BELGIUM (`213800WO2QK7HUL8R680`) | ALPHABET INC. — `by-lei/213800WO2QK7HUL8R680/direct-parent/direct-parent.json` |
| GOOGLE BRASIL INTERNET LTDA. (`984500B5170E75C99056`) | ALPHABET INC. — `by-lei/984500B5170E75C99056/direct-parent/direct-parent.json` |
| GOOGLE CLOUD CANADA CORPORATION (`984500074CCB812A5D80`) | ALPHABET INC. — `by-lei/984500074CCB812A5D80/direct-parent/direct-parent.json` |
| Google Switzerland GmbH (`984500B1C1A7S4898E77`) | ALPHABET INC. — `by-lei/984500B1C1A7S4898E77/direct-parent/direct-parent.json` |
| GOOGLE CHILE LIMITADA (`984500C04A37BB9B5215`) | ALPHABET INC. — `by-lei/984500C04A37BB9B5215/direct-parent/direct-parent.json` |
| Google Germany GmbH (`529900H4SBHKV7XQXK22`) | ALPHABET INC. — `by-lei/529900H4SBHKV7XQXK22/direct-parent/direct-parent.json` |
| GLUI Engineering Oy (`74370096HBTUWJRFIT29`) | withheld, reason `NO_LEI`; ultimate parent on file = ALPHABET INC. — `by-lei/74370096HBTUWJRFIT29/direct-parent-reporting-exception/…json`, `.../ultimate-parent/ultimate-parent.json` |
| Tuike Finland Oy (`743700HGB1CRPBRA1K29`) | ALPHABET INC. — `by-lei/743700HGB1CRPBRA1K29/direct-parent/direct-parent.json` |
| GOOGLE CLOUD FRANCE (`9845003T366A4EB2EE58`) | ALPHABET INC. — `by-lei/9845003T366A4EB2EE58/direct-parent/direct-parent.json` |
| GOOGLE FRANCE (`9845004B3AADDD5FB669`) | ALPHABET INC. — `by-lei/9845004B3AADDD5FB669/direct-parent/direct-parent.json` |
| GOOGLE PAYMENT LIMITED (`6488U5HN5HSK59195V82`) | ALPHABET INC. — `by-lei/6488U5HN5HSK59195V82/direct-parent/direct-parent.json` |
| GOOGLE UK LIMITED (`64886BIOSLNV78184624`) | ALPHABET INC. — `by-lei/64886BIOSLNV78184624/direct-parent/direct-parent.json` |
| WIZ CLOUD LIMITED (`6488R1WLKN077T974P96`) | WIZ, INC. — `by-lei/6488R1WLKN077T974P96/direct-parent/direct-parent.json` |
| GOOGLE PAYMENT IRELAND LIMITED (`9845003AA4F14A013C44`) | ALPHABET INC. — `by-lei/9845003AA4F14A013C44/direct-parent/direct-parent.json` |
| GOOGLE CLOUD EMEA LIMITED (`98450052CF14CFEB6435`) | ALPHABET INC. — `by-lei/98450052CF14CFEB6435/direct-parent/direct-parent.json` |
| GOOGLE EUROPE, MIDDLE EAST AND AFRICA UNLIMITED COMPANY (`549300YY0W0QP4SZAG19`) | ALPHABET INC. — `by-lei/549300YY0W0QP4SZAG19/direct-parent/direct-parent.json` |
| GOOGLE IRELAND HOLDINGS UNLIMITED COMPANY (`1FN51LDHILXJ06W1QN20`) | ALPHABET INC. — `by-lei/1FN51LDHILXJ06W1QN20/direct-parent/direct-parent.json` |
| GOOGLE IRELAND LIMITED (`YYPPRNO5HB304LHFVG31`) | GOOGLE INTERNATIONAL LLC — `by-lei/YYPPRNO5HB304LHFVG31/direct-parent/direct-parent.json` |
| WAYMO IT SERVICES INDIA PRIVATE LIMITED (`335800LW63MGNBQTB908`) | withheld, reason `NO_LEI`; ultimate parent = ALPHABET INC. — `by-lei/335800LW63MGNBQTB908/direct-parent-reporting-exception/…json`, `.../ultimate-parent/ultimate-parent.json` |
| GOOGLE IT SERVICES INDIA PRIVATE LIMITED (`335800VPK652AEFF1O79`) | withheld, reason `NON_PUBLIC`; ultimate parent = ALPHABET INC. — same file pattern under this LEI |
| RAIDEN INFOTECH INDIA PRIVATE LIMITED (`335800FMIGZ24EJ2XW80`) | withheld, reason `NON_PUBLIC`; ultimate parent = ALPHABET INC. — same file pattern |
| SZS TECH PRIVATE LIMITED (`335800VQ73CQOLRRSK16`) | withheld, reason `NON_PUBLIC`; ultimate parent = ALPHABET INC. — same file pattern |
| MANDIANT CYBERSECURITY PRIVATE LIMITED (`335800APEHUAALHEZA12`) | withheld, reason `NON_PUBLIC`; ultimate parent = ALPHABET INC. — same file pattern |
| GOOGLE PAYMENT INDIA PRIVATE LIMITED (`335800SCE2XZHDWXCG42`) | withheld, reason `NON_PUBLIC`; ultimate parent = ALPHABET INC. — same file pattern |
| GOOGLE CLOUD INDIA PRIVATE LIMITED (`335800WBAET9Q971AT75`) | ALPHABET INC. — `by-lei/335800WBAET9Q971AT75/direct-parent/direct-parent.json` |
| GOOGLE INDIA DIGITAL SERVICES PRIVATE LIMITED (`3358004GEBDX73EU6859`) | ALPHABET INC. — `by-lei/3358004GEBDX73EU6859/direct-parent/direct-parent.json` |
| GOOGLE CAPITAL ADVISORS INDIA PRIVATE LIMITED (`335800DFHCGEFFZXRA34`) | ALPHABET INC. — `by-lei/335800DFHCGEFFZXRA34/direct-parent/direct-parent.json` |
| GOOGLE INFORMATION SERVICES INDIA PRIVATE LIMITED (`335800FCLGQFJQJGHP87`) | ALPHABET INC. — `by-lei/335800FCLGQFJQJGHP87/direct-parent/direct-parent.json` |
| GOOGLE CONNECT SERVICES INDIA PRIVATE LIMITED (`3358009UB5HS2WQF6Y18`) | ALPHABET INC. — `by-lei/3358009UB5HS2WQF6Y18/direct-parent/direct-parent.json` |
| GOOGLE INDIA PRIVATE LIMITED (`335800P2W6AY6E8BEP06`) | GOOGLE INTERNATIONAL LLC — `by-lei/335800P2W6AY6E8BEP06/direct-parent/direct-parent.json` |
| GOOGLE CLOUD ITALY S.R.L. (`529900H6FVQK3TP02Z71`) | ALPHABET INC. — `by-lei/529900H6FVQK3TP02Z71/direct-parent/direct-parent.json` |
| Google Payment Lithuania, UAB (`6488JJ7LW4MS85G57298`) | ALPHABET INC. — `by-lei/6488JJ7LW4MS85G57298/direct-parent/direct-parent.json` |
| GOOGLE CLOUD MEXICO S DE RL DE CV (`984500F71C16CDC30C27`) | ALPHABET INC. — `by-lei/984500F71C16CDC30C27/direct-parent/direct-parent.json` |
| Google Netherlands B.V. (`984500DO4RF63E8D6604`) | ALPHABET INC. — `by-lei/984500DO4RF63E8D6604/direct-parent/direct-parent.json` |
| GOOGLE POLAND SP. Z O.O. (`984500E14WF63DAD2C91`) | GOOGLE INTERNATIONAL LLC — `by-lei/984500E14WF63DAD2C91/direct-parent/direct-parent.json` |
| GOOGLE CLOUD POLAND SP. Z O.O. (`984500B8CC95C15F8Q65`) | GOOGLE CLOUD EMEA LIMITED — `by-lei/984500B8CC95C15F8Q65/direct-parent/direct-parent.json`; also confirmed from the parent side: `by-lei/98450052CF14CFEB6435/direct-children/direct-children.json` lists this as its only direct child |
| GOOGLE HOLDINGS PTE. LTD. (`549300AYV236IWGM7312`) | ALPHABET INC. — `by-lei/549300AYV236IWGM7312/direct-parent/direct-parent.json` |
| GOOGLE ASIA PACIFIC PTE. LTD. (`RXU43ANVI3MGLXA9UI89`) | ALPHABET INC. — `by-lei/RXU43ANVI3MGLXA9UI89/direct-parent/direct-parent.json` |
| ALPHABET CAPITAL US LLC (`549300KG6S007HTRLN18`) | withheld, reason `NON_PUBLIC`; ultimate parent = ALPHABET INC. — `by-lei/549300KG6S007HTRLN18/direct-parent-reporting-exception/…json`, `.../ultimate-parent/ultimate-parent.json` |
| CHARLESTON ROAD REGISTRY INC. (`984500639BT036CND679`) | GOOGLE LLC — `by-lei/984500639BT036CND679/direct-parent/direct-parent.json` |
| GOC INTERNATIONAL LLC (`984500DI9BZD41OF4781`) | ALPHABET INC. — `by-lei/984500DI9BZD41OF4781/direct-parent/direct-parent.json` (also in Alphabet's own `direct-child-relationships.json`, relationship period starting 2018-10-25) |
| Design, LLC (`9845000BD467666E3930`) | ALPHABET INC. — `by-lei/9845000BD467666E3930/direct-parent/direct-parent.json` (also in Alphabet's own `direct-child-relationships.json`, relationship period starting 2004-11-30, registration status LAPSED) |
| FIREBASE, INC. (`9845003D44CCA145CC76`) | ALPHABET INC. — `by-lei/9845003D44CCA145CC76/direct-parent/direct-parent.json` |
| WIZ, INC. (`6488C2T08TXT755U3P86`) | GOOGLE LLC — `by-lei/6488C2T08TXT755U3P86/direct-parent/direct-parent.json` |
| GOOGLE ENERGY LLC (`549300XRQBENEKTBS153`) | GOOGLE LLC — `by-lei/549300XRQBENEKTBS153/direct-parent/direct-parent.json` |
| GOOGLE INTERNATIONAL LLC (`549300Y7W34DK0WBPY51`) | GOOGLE LLC — `by-lei/549300Y7W34DK0WBPY51/direct-parent/direct-parent.json` |
| ALPHABET CAPITAL US II LLC (`549300SW82SFEORONS31`) | ALPHABET INC. — `by-lei/549300SW82SFEORONS31/direct-parent/direct-parent.json` (also in Alphabet's own `direct-child-relationships.json`, relationship period starting 2005-12-20) |
| GOOGLE LLC (`7ZW8QJWVPR4P1J1KQY45`) | ALPHABET INC. — `by-lei/7ZW8QJWVPR4P1J1KQY45/direct-parent/direct-parent.json`, relationship period starting 2002-10-22 per `by-lei/7ZW8QJWVPR4P1J1KQY45/direct-parent-relationship/direct-parent-relationship.json` |

## Confirming the tree bottoms out (no fifth tier)

- **Claim:** none of the 50 subsidiaries has any direct-children beyond the four
  already accounted for (Google LLC, Google International LLC, Google Cloud EMEA
  Limited, Wiz, Inc.).
  - Read `direct-children/direct-children.json` for all 50 LEIs (path pattern
    `gleif/by-lei/<LEI>/direct-children/direct-children.json`); only these four LEIs
    have that file present with non-empty `data`:
    - `gleif/by-lei/7ZW8QJWVPR4P1J1KQY45/direct-children/direct-children.json` (GOOGLE
      LLC) — 4 children: Charleston Road Registry Inc., Wiz Inc., Google Energy LLC,
      Google International LLC.
    - `gleif/by-lei/549300Y7W34DK0WBPY51/direct-children/direct-children.json` (GOOGLE
      INTERNATIONAL LLC) — 3 children: Google Poland, Google India Private Limited,
      Google Ireland Limited.
    - `gleif/by-lei/98450052CF14CFEB6435/direct-children/direct-children.json` (GOOGLE
      CLOUD EMEA LIMITED) — 1 child: Google Cloud Poland Sp. z o.o.
    - `gleif/by-lei/6488C2T08TXT755U3P86/direct-children/direct-children.json` (WIZ,
      INC.) — 1 child: Wiz Cloud Limited.
  - All other 46 LEIs: no `direct-children` file exists at that path (verified with a
    directory listing under each `gleif/by-lei/<LEI>/`), meaning GLEIF records zero
    further descendants for them.

## EDGAR side

- **Claim:** Alphabet Inc.'s SEC CIK is `0001652044`; this matches the GLEIF record on
  name and address (Mountain View, CA; EIN 61-1767919, Delaware incorporation).
  - `edgar/search/entityName/Alphabet Inc/pages/page-001.json` — hit with
    `display_names: ["Alphabet Inc.  (GOOG, GOOGL, GOOGM, GOOGN)  (CIK 0001652044)"]`.
  - `edgar/by-cik/0001652044/submissions/submissions.json` — `name: "Alphabet Inc."`,
    `stateOfIncorporation: "DE"`, `ein: "611767919"`, mailing/business address
    "1600 AMPHITHEATRE PARKWAY, MOUNTAIN VIEW, CA", `lei: null` (as the cross-registry
    catalog states is typical — EDGAR does not carry a usable LEI cross-reference).
- **Claim:** EDGAR's full-text search on "Alphabet Inc" surfaces routine issuer
  filings and insider Form 3/4/5s (e.g. Sergey Brin, John L. Hennessy, Frances Arnold)
  but no structured subsidiary/ownership graph was retrieved from this registry in
  this run.
  - `edgar/search/entityName/Alphabet Inc/pages/page-001.json` — `hits.total.value: 2760`;
    sample rows include forms `4`, `8-K`, `13F-HR`, `424B2`, `424B5`, `FWP`, `144`.
