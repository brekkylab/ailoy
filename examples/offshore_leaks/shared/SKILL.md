---
name: offshore-leaks
description: Query ICIJ's Offshore Leaks database, the offshore companies, the people and firms behind them and the addresses in the Panama, Paradise, Pandora Papers, Bahamas Leaks and Offshore Leaks, with SQL. Use it to find a person or company in the leaks, follow who is connected to whom, or count and chart by jurisdiction, intermediary, year or leak.
---

# Offshore Leaks

The Offshore Leaks database is what the International Consortium of Investigative
Journalists (ICIJ) published from five leaks: the Offshore Leaks (2013), the Panama Papers
(2016), the Bahamas Leaks (2016), the Paradise Papers (2017) and the Pandora Papers (2021).
It is a graph of about 2 million nodes and 3.3 million relationships: offshore companies and
trusts, the people and companies that own or run them, the intermediaries who set them up,
and their addresses. It is at `/context/offshore_leaks.duckdb`, a DuckDB file opened
read-only.

## What it is not

**Being in the database is not wrongdoing.** In ICIJ's words: there are legitimate uses for
offshore companies and trusts, and it does not intend to suggest or imply that any person,
company or other entity in the database has broken the law or otherwise acted improperly.
Say so when you name a person, and report what the records show — a role, a date, a
jurisdiction — rather than what it might mean.

**A name is not a person.** Two officers named `KIM SOO IN` are two nodes, and nothing but
the name says they are one person; the leaks were never deduplicated against each other. When
you match someone the user named, say what the match rests on (the name, and a country, an
address or a date if they agree) and that it may be somebody else with the same name. The
`same_name_as`, `similar` and `probably_same_officer_as` relationships are ICIJ's own
guesses of the same kind, not identities.

**It is partial and old.** Each leak is what one or a few service providers held, up to the
year in `valid_until` (2015 for the Panama Papers). Absence from it says nothing, and a
company's status is its status then.

## Running it

Run `oldb.py` from this directory. There are three commands, and each prints one JSON object.

```sh
python3 oldb.py search 'mossack fonseca' --kind intermediary
python3 oldb.py node 11011863
python3 oldb.py sql "SELECT jurisdiction_description, count(*) AS n FROM entities GROUP BY 1 ORDER BY n DESC" --limit 20
```

* `search NAME` finds the nodes whose name holds every word of NAME, case aside, closest
  first, with how many relationships each has. `--kind` keeps one of `entity`, `officer`,
  `intermediary`, `address` or `other`, and `--limit` how many come back (20).
* `node ID` gives a node, every column of it, and its relationships in both directions, each
  with the id, kind and name of the node at the other end. `--limit` caps them (200), and
  `relationships_total` says how many there are.
* `sql QUERY` runs any DuckDB query and prints the first `--limit` rows (100). `--out PATH`
  writes every row to PATH as CSV, Parquet or JSON, by its extension. `truncated` says there
  were more rows than printed.

A query takes well under a second, even over all relationships. A query DuckDB refuses
comes back as `{"error": ...}` with exit code 1, with DuckDB's message, which names the
column or table it did not know.

For more than a table — a chart, a network drawing, several steps in pandas — use DuckDB
from Python yourself, with `duckdb.connect('/context/offshore_leaks.duckdb',
read_only=True)`. pandas, matplotlib and networkx are installed. Put what the user is to see
under `/artifacts`.

## The tables

Every node has a `node_id` (BIGINT), unique across all kinds, a `name`, `countries` and
`country_codes` (ISO 3166 alpha-3), `sourceID` (which leak, such as `Panama Papers` or
`Paradise Papers - Appleby`), `valid_until` and `note`. A node linked to several countries
has them joined with `;`, as `Andorra;Republic of Moldova`, so match them with `LIKE` or
split them with `string_split(countries, ';')`.

* `entities` (815,000) — the offshore companies, trusts and foundations. Also
  `original_name`, `former_name`, `jurisdiction` (a code) and `jurisdiction_description`
  (where it is registered, such as `British Virgin Islands`), `company_type`, `address`,
  `internal_id`, `incorporation_date`, `inactivation_date`, `struck_off_date`, `dorm_date`,
  `status` (such as `Active`, `Defaulted`, `Struck Off`), `service_provider` and `ibcRUC`.
  `countries` is where the entity's owners or address are, not its jurisdiction.
* `officers` (771,000) — people and companies with a role in an entity: directors,
  shareholders, beneficiaries, secretaries.
* `intermediaries` (27,000) — the law firms, banks and agents that asked the service
  provider for an entity. Also `status`, `internal_id` and `address`.
* `addresses` (402,000) — `address` is the text of it; `name` is mostly empty.
* `others` (3,000) — companies of the corporate registries that are none of the above. Also
  `type`, `incorporation_date`, `struck_off_date`, `closed_date` and the jurisdiction.
* `nodes` — a view of all five with `node_id`, `kind` (`entity`, `officer`,
  `intermediary`, `address`, `other`), `name` (the address, for an address), `countries`,
  `country_codes` and `sourceID`. Join a relationship's ends to it when their kind is not
  known.
* `relationships` (3.3 million) — `node_id_start`, `node_id_end`, `rel_type`, `link` (the
  role in words), `status`, `start_date`, `end_date` and `sourceID`. It is indexed on both
  ends. `sourceID` is empty on about 500,000 of them; take the leak from a node's instead.
* `meta` — `source`, `generated_on` (the day ICIJ exported it), and `license`.

Dates are DATEs. Some are NULL where the leak's text could not be read as a date, and a few
are typos, such as years 0199 or 2812; keep a range such as 1950 to 2025 when counting by
year.

## The relationships

A relationship points from the node that has the role to the node it has it in. Most are
of three kinds:

* `officer_of`, from an officer (sometimes an intermediary) to an entity. `link` is the role:
  `shareholder of`, `director of`, `secretary of`, `beneficiary of`, `beneficial owner of`,
  `nominee director of`, and more, spelled in several ways, so group by `lower(link)` or match
  with `ILIKE '%shareholder%'`.
* `intermediary_of`, from an intermediary to an entity it had set up or ran for a client.
* `registered_address`, from an entity, officer or intermediary to an address.

The rest say two nodes may be one: `same_name_as`, `same_as`, `same_company_as`,
`same_id_as`, `similar`, `similar_company_as`, `probably_same_officer_as`, `same_address_as`
and `same_intermediary_as`. `connected_to` and `underlying` are loose links from the
registries and the Offshore Leaks.

Who is behind an entity is its `officer_of` and `intermediary_of` relationships' starts;
what a person or firm is in is its relationships' ends. Two hops, officer to entity to
officer, is who shares a company with whom, and grows fast: an intermediary such as Mossack
Fonseca has thousands of entities, and a nominee director hundreds. Count before listing.

```sql
-- Who else is an officer of the entities an officer is in.
SELECT o2.node_id, o2.name, count(DISTINCT r1.node_id_end) AS shared_entities
FROM relationships r1
JOIN relationships r2 ON r2.node_id_end = r1.node_id_end AND r2.rel_type = 'officer_of'
JOIN officers o2 ON o2.node_id = r2.node_id_start
WHERE r1.node_id_start = 12000001 AND r1.rel_type = 'officer_of' AND r2.node_id_start <> 12000001
GROUP BY ALL ORDER BY shared_entities DESC
```

## Citing it

Each node has a page on ICIJ's site, `https://offshoreleaks.icij.org/nodes/<node_id>`. Give
it with each person or company you name, so the user can check the record there. The data is
ICIJ's, under the Open Database License, and its contents under CC BY-SA: a report or a chart
built on it says it is from the ICIJ Offshore Leaks Database.
