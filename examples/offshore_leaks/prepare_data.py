"""Download ICIJ's Offshore Leaks database and load it into one DuckDB file.

    uv run prepare_data.py [DATA_DIR] [DATABASE]

DATA_DIR is `data/` beside this file by default, and DATABASE `context/offshore_leaks.duckdb`.
The archive ICIJ publishes, `full-oldb.LATEST.zip`, goes into `DATA_DIR` and its CSVs into
`DATA_DIR/csv`; the database is built from them and is what the guest is given. It is built
again when ICIJ publishes a new archive, which its ETag tells.

* `OFFSHORE_LEAKS_URL` -- the archive to download, URL below by default.

The CSVs are the graph ICIJ's site shows: five files of nodes (entities, officers,
intermediaries, addresses and others) and one of the relationships between them. Loading them
takes three fixes of what DuckDB would guess:

* **Quotes.** DuckDB's sniffer finds no quoting in `relationships.csv`, whose first quoted
  field is 345,000 lines in, and splits `"owner, director and shareholder of"` at its comma.
  The quote and escape are named instead.
* **Dates.** They are text in three formats -- `08-MAR-2004` in most leaks, `2000-07-24` in
  the 2013 Offshore Leaks, and a few `Sep 25, 2012` -- and become DATEs. What is in none of
  them, such as `11/10/01`, whose day and month cannot be told apart, is NULL.
* **Ids.** `node_id` and the relationships' ends become BIGINTs, and the relationships are
  indexed on both ends, since most questions walk them from a node.

A `nodes` view puts the five kinds of node in one table, and `meta` says where and when the
data is from.
"""

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import duckdb

URL = "https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip"

CSV = "header = true, all_varchar = true, quote = '\"', escape = '\"'"
DATE_FORMATS = "['%d-%b-%Y', '%Y-%m-%d', '%b %d, %Y']"

# Table, file, and the columns that are dates.
NODES = [
    ("entities", "nodes-entities.csv", ["incorporation_date", "inactivation_date", "struck_off_date", "dorm_date"]),
    ("officers", "nodes-officers.csv", []),
    ("intermediaries", "nodes-intermediaries.csv", []),
    ("addresses", "nodes-addresses.csv", []),
    ("others", "nodes-others.csv", ["incorporation_date", "struck_off_date", "closed_date"]),
]


def etag(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
        return resp.headers.get("ETag", "").strip('"')


def built_from(database: Path) -> str | None:
    """The ETag of the archive `database` was built from, if it is complete."""
    try:
        with duckdb.connect(str(database), read_only=True) as con:
            return con.sql("SELECT value FROM meta WHERE key = 'etag'").fetchone()[0]
    except (duckdb.Error, TypeError):
        return None


def download(url: str, data: Path) -> Path:
    archive = data / "full-oldb.zip"
    print(f"offshore_leaks: downloading {url}", flush=True)
    part = archive.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, part.open("wb") as out:
        shutil.copyfileobj(resp, out)
    part.replace(archive)
    csv = data / "csv"
    shutil.rmtree(csv, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(csv)
    return csv


def select(file: Path, dates: list[str]) -> str:
    """A SELECT of `file` with its ids and `dates` typed."""
    replace = ["CAST(node_id AS BIGINT) AS node_id"]
    replace += [f"CAST(try_strptime({d}, {DATE_FORMATS}) AS DATE) AS {d}" for d in dates]
    return f"SELECT * REPLACE ({', '.join(replace)}) FROM read_csv('{file}', {CSV})"


def build(csv: Path, database: Path, url: str, tag: str):
    part = database.with_suffix(".part")
    part.unlink(missing_ok=True)
    with duckdb.connect(str(part)) as con:
        for table, file, dates in NODES:
            print(f"offshore_leaks: loading {table}", flush=True)
            con.execute(f"CREATE TABLE {table} AS {select(csv / file, dates)} ORDER BY node_id")
        print("offshore_leaks: loading relationships", flush=True)
        con.execute(f"""
            CREATE TABLE relationships AS
            SELECT * REPLACE (
                CAST(node_id_start AS BIGINT) AS node_id_start,
                CAST(node_id_end AS BIGINT) AS node_id_end,
                CAST(try_strptime(start_date, {DATE_FORMATS}) AS DATE) AS start_date,
                CAST(try_strptime(end_date, {DATE_FORMATS}) AS DATE) AS end_date)
            FROM read_csv('{csv / "relationships.csv"}', {CSV})
        """)
        con.execute("CREATE INDEX relationships_start ON relationships (node_id_start)")
        con.execute("CREATE INDEX relationships_end ON relationships (node_id_end)")
        # An address has a `name` only now and then; its `address` is what names it.
        con.execute("""
            CREATE VIEW nodes AS
                SELECT node_id, 'entity' AS kind, name, countries, country_codes, sourceID FROM entities
            UNION ALL
                SELECT node_id, 'officer', name, countries, country_codes, sourceID FROM officers
            UNION ALL
                SELECT node_id, 'intermediary', name, countries, country_codes, sourceID FROM intermediaries
            UNION ALL
                SELECT node_id, 'address', coalesce(address, name), countries, country_codes, sourceID FROM addresses
            UNION ALL
                SELECT node_id, 'other', name, countries, country_codes, sourceID FROM others
        """)
        generated = next((p.stem.removeprefix("GENERATED_ON_") for p in csv.glob("GENERATED_ON_*")), None)
        con.execute("CREATE TABLE meta (key VARCHAR, value VARCHAR)")
        con.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [("source", url), ("generated_on", generated), ("etag", tag),
             ("license", "Open Database License (ODbL) for the database, CC BY-SA for its contents; "
                         "source: ICIJ Offshore Leaks Database, https://offshoreleaks.icij.org")],
        )
        con.execute("CHECKPOINT")
    part.replace(database)


def main():
    here = Path(__file__).parent
    data = (Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data").resolve()
    database = (Path(sys.argv[2]) if len(sys.argv) > 2 else here / "context" / "offshore_leaks.duckdb").resolve()
    url = os.environ.get("OFFSHORE_LEAKS_URL", URL)

    try:
        tag = etag(url)
    except OSError as e:
        # Offline: what was built before is still the data.
        if built_from(database) is None:
            raise
        print(f"offshore_leaks: {e}; keeping {database.name}")
        return
    if built_from(database) == tag:
        print(f"offshore_leaks: {database.name} is up to date")
        return
    data.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    csv = download(url, data)
    build(csv, database, url, tag)
    print(f"offshore_leaks: wrote {database}")


if __name__ == "__main__":
    main()
