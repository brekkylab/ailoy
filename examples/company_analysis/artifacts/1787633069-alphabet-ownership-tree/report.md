# Alphabet Inc. — Disclosed Ownership Tree

## Summary

Alphabet Inc. (LEI `5493006MHB84DD0ZWV18`, Delaware, HQ Mountain View CA) sits at the
top of the tree GLEIF discloses: no entity is recorded as owning it, and GLEIF
explicitly marks both its direct- and ultimate-parent fields as exceptions with reason
"no known person" — i.e., Alphabet has no controlling parent to disclose, consistent
with it being a widely-held public holding company. Below it, GLEIF's Level 2
("who owns whom") data lists exactly **50 entities** consolidated under Alphabet,
arranged in up to four tiers (Alphabet → Google LLC → Google International LLC/
Wiz, Inc. → their own subsidiaries), all bottoming out — no fifth tier appears
anywhere in the disclosed data. Several of the 50 have their *direct* parent withheld
(reason `NON_PUBLIC` or `NO_LEI`) even though their *ultimate* parent is still recorded
as Alphabet Inc., so the tree is complete at the ultimate-owner level but has small
gaps in the middle links. EDGAR, the other registry available, does not carry
ownership/subsidiary structure for Alphabet in the fields checked here — it confirms
Alphabet as CIK `0001652044`, a public filer with no parent of its own on that side
either, but contributes no additional subsidiary graph.

## The group as disclosed (GLEIF Level 2 data)

Alphabet Inc. is the ultimate parent of 50 entities, per
`gleif/search/ownedBy/5493006MHB84DD0ZWV18/pages/page-001.json` (total: 50) and cross-checked
against `gleif/by-lei/.../ultimate-children/ultimate-children.json` (total: 50, first
page's 15 records all appear in the ownedBy list). The chain has up to four tiers:

```
ALPHABET INC. (US-DE, LEI 5493006MHB84DD0ZWV18)
│  [parent: none disclosed — reporting exception "NO_KNOWN_PERSON"]
│
├── GOOGLE LLC (US-DE)                          — direct child of Alphabet
│     ├── CHARLESTON ROAD REGISTRY INC. (US)     — direct child of Google LLC
│     ├── WIZ, INC. (US)                         — direct child of Google LLC
│     │     └── WIZ CLOUD LIMITED (GB)           — direct child of Wiz, Inc.
│     ├── GOOGLE ENERGY LLC (US)                  — direct child of Google LLC
│     └── GOOGLE INTERNATIONAL LLC (US)           — direct child of Google LLC
│           ├── GOOGLE POLAND SP. Z O.O. (PL)
│           ├── GOOGLE INDIA PRIVATE LIMITED (IN)
│           └── GOOGLE IRELAND LIMITED (IE)
│
├── GOOGLE CLOUD EMEA LIMITED (IE)               — direct child of Alphabet
│     └── GOOGLE CLOUD POLAND SP. Z O.O. (PL)     — direct child of Google Cloud EMEA
│
└── 40 further entities recorded as direct children of Alphabet Inc. itself,
    spanning Argentina, Austria, Belgium, Brazil, Canada, Switzerland, Chile,
    Germany, Finland (2), France (2), UK (2, incl. Google Payment Limited),
    Ireland (3 more: Google Payment Ireland, Google Europe/Middle East/Africa
    Unlimited Co., Google Ireland Holdings Unlimited Co.), 10 "…India Private
    Limited" entities, Italy, Lithuania, Mexico, Netherlands, Singapore (2),
    and 3 US entities (Alphabet Capital US LLC, Alphabet Capital US II LLC,
    GOC International LLC, Design LLC, Firebase Inc.)
```

Full entity-by-entity parent listing is in `evidence.md`.

No entity among the 50 has any further direct-children beyond the four already shown
(Google LLC, Google International LLC, Google Cloud EMEA Limited, Wiz, Inc.) —
checked by reading `direct-children/direct-children.json` for all 50 LEIs; every other
one returns none. The tree, as GLEIF discloses it, therefore has a maximum depth of
four tiers below Alphabet and does not extend further.

## Direction of each relationship

GLEIF's `direct-parent` / `ultimate-parent` files on a subsidiary's own record point
**upward** (child → parent: "this entity `IS_DIRECTLY_CONSOLIDATED_BY` Alphabet Inc.").
The `search/ownedBy/<LEI>` and `search/owns/<LEI>` index, read from Alphabet's side,
gives the same relationships from the parent's side: `ownedBy` returned Alphabet's 50
children, and `owns` (which entity owns Alphabet) returned zero results — confirming
the direction and confirming no parent is disclosed above Alphabet.
(`gleif/CATALOG.md` warns the naming is counter-intuitive; both directions were
cross-checked against each other and agree.)

## Where the tree stops

- **Above Alphabet:** nothing. `direct-parent-reporting-exception` and
  `ultimate-parent-reporting-exception` for Alphabet both give
  `category: DIRECT/ULTIMATE_ACCOUNTING_CONSOLIDATION_PARENT`,
  `reason: NO_KNOWN_PERSON` — GLEIF's own statement that no natural or legal person is
  known to control Alphabet, i.e., its shareholding is dispersed enough (or LEI
  reporting rules judge it) that no consolidating parent is named.
- **Below the 50 disclosed subsidiaries:** nothing further appears. Every one of the 50
  was checked for its own `direct-children`; only four (Google LLC, Google
  International LLC, Google Cloud EMEA Limited, Wiz, Inc.) have any, and those
  children are already inside the 50 — i.e., the set is closed and the deepest chain is
  four links (Alphabet → Google LLC → Google International LLC → Google Poland /
  Google India / Google Ireland, or Alphabet → Google LLC → Wiz, Inc. → Wiz Cloud
  Limited).
- **Middle links sometimes withheld:** nine of the 50 (GLUI Engineering Oy, Waymo IT
  Services India, Google IT Services India, Raiden Infotech India, SZS Tech India,
  Mandiant Cybersecurity India, Google Payment India, Alphabet Capital US LLC) have
  their *direct*-parent relationship withheld under a reporting exception
  (`NO_LEI` — the immediate parent has no LEI of its own to name — or `NON_PUBLIC` —
  the immediate parent chose not to disclose). In every one of these cases the
  *ultimate*-parent field is still populated and still names Alphabet Inc., so the
  end of the chain is known even where a link in the middle is not shown.
- **EDGAR side:** Alphabet's own `submissions.json` carries no subsidiary list and an
  empty `lei` field (as the cross-registry catalog warns is typical). EDGAR's
  full-text search over Alphabet's filings surfaces insiders (Form 3/4/5 filers such
  as Sergey Brin, John L. Hennessy) and routine issuer filings (8-K, 424B, 13F-HR) but
  not a machine-readable ownership graph; EDGAR was not the source of the subsidiary
  tree above.

## What the shape does not tell you

- **Ownership percentage / voting control.** GLEIF's relationship type here is
  accounting consolidation (`IS_DIRECTLY_CONSOLIDATED_BY` /
  `IS_ULTIMATELY_CONSOLIDATED_BY`), not a stated percentage shareholding. A 100%-owned
  subsidiary and a majority-owned-but-consolidated one look identical in this data.
- **Completeness of Alphabet's actual corporate family.** GLEIF only shows entities
  that (a) hold an LEI and (b) have filed a relationship record with GLEIF. Alphabet's
  SEC 10-K Exhibit 21 (list of subsidiaries) would likely show more entities than the
  50 with LEIs found here, including ones with no LEI at all (e.g., many small or
  dormant subsidiaries) — that exhibit was not read in this run (EDGAR full-text
  search on entity name alone does not surface a specific exhibit without a further,
  narrower query).
- **Whether "Alphabet" name-collisions are related.** The GLEIF search for
  "Alphabet Inc" also returned "ALPHABET ENERGY INC" (India), "ALPHABET MINERALS INC"
  (India), "ALPHABET HOLDING COMPANY, INC." (Delaware, different registration number
  4846758 vs. Alphabet Inc.'s 5786925, LAPSED status), and
  "Alphabet Millcreek Self Storage Inc." (Canada) — none of these appear in Alphabet
  Inc.'s ownedBy/ultimate-children list, and their addresses/registration numbers do
  not match, so they are treated here as unrelated name collisions, not part of the
  tree.
- **Direction of dependency vs. legal risk/liability.** A consolidation relationship
  says an entity's financials roll up into Alphabet's; it says nothing about
  guarantees, intercompany debt, or litigation exposure between the entities.

## Data limits

- GLEIF's registry is name/LEI-indexed only; percentages of ownership and the legal
  basis for "control" are not part of the fields read here.
- Nine of the 50 disclosed subsidiaries have their immediate parent withheld
  (`NON_PUBLIC` or `NO_LEI`); only the ultimate-parent link to Alphabet is visible for
  those nine.
- EDGAR was checked but does not carry a machine-readable subsidiary/ownership graph
  for Alphabet in the paths available (`submissions.json`, full-text search); a
  narrower EDGAR full-text search restricted to Exhibit 21 filings could surface
  Alphabet's SEC-disclosed subsidiary list, which was not pulled in this run.
- The four "Alphabet *" name matches found in GLEIF search (Alphabet Energy Inc,
  Alphabet Minerals Inc, Alphabet Holding Company Inc., Alphabet Millcreek Self
  Storage Inc.) are **possible matches — needs confirmation** only in the sense that a
  bare name search surfaces them; on inspection (country, registration number,
  address) none of them corroborate as part of Alphabet Inc.'s group, and none appear
  in its ownedBy list, so they are excluded from the tree above.

## Next steps

- Pull Alphabet's most recent 10-K Exhibit 21 (list of subsidiaries) from EDGAR via a
  narrower full-text search (`forms=10-K`, `q=Exhibit 21` or similar) to compare
  against GLEIF's 50-entity list and see whether SEC disclosure names entities GLEIF's
  LEI graph does not (e.g., subsidiaries without an LEI).
- For the nine entities with a withheld direct parent, if the immediate parent is
  later assigned an LEI, re-check `direct-parent` to fill in the missing middle link.
- If ownership *percentages* are needed rather than consolidation relationships, this
  question requires a different source (e.g., the 10-K itself or a proxy statement),
  which neither registry mounted here provides in structured form.
