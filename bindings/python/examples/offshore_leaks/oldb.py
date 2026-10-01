"""Query ICIJ's Offshore Leaks database, the DuckDB file `prepare_data.py` built.

Run inside the guest as `python3 oldb.py COMMAND ...`, from the offshore-leaks skill mounted
at /skills/offshore-leaks. `OFFSHORE_LEAKS_DB` names another file than
/context/offshore_leaks.duckdb, to run it on the host.

* `sql QUERY [--limit N] [--out PATH]` -- run QUERY and print its first N rows (100 by
  default); with `--out`, write every row to PATH, as CSV, Parquet or JSON by its extension.
* `search NAME [--kind KIND] [--limit N]` -- the nodes whose name holds every word of NAME,
  closest first, each with how many relationships it has.
* `node ID [--limit N]` -- a node, every column of it, and its relationships in both
  directions with the name and kind of the node at the other end.

stdout is one JSON object. A query DuckDB refuses is `{"error": ...}` with exit code 1.

The file is opened read-only, as /context is: nothing can be written to the database, and a
table to keep goes to a file with `--out` instead.
"""

import argparse
import json
import os
import sys

import duckdb

DB = os.environ.get("OFFSHORE_LEAKS_DB", "/context/offshore_leaks.duckdb")

TABLES = {
    "entity": "entities",
    "officer": "officers",
    "intermediary": "intermediaries",
    "address": "addresses",
    "other": "others",
}


def rows(rel, limit: int) -> dict:
    """The first `limit` rows of `rel` as records, and whether there were more."""
    columns = rel.columns
    fetched = rel.limit(limit + 1).fetchall()
    return {
        "columns": columns,
        "rows": [dict(zip(columns, r)) for r in fetched[:limit]],
        "truncated": len(fetched) > limit,
    }


def sql(con, args) -> dict:
    rel = con.sql(args.query)
    if rel is None:
        return {"ok": True}
    out = rows(rel, args.limit)
    if args.out:
        fmt = {".csv": "CSV", ".parquet": "PARQUET", ".json": "JSON"}.get(os.path.splitext(args.out)[1].lower())
        if fmt is None:
            raise SystemExit("--out takes a .csv, .parquet or .json path")
        path = args.out.replace("'", "''")
        written = con.execute(f"COPY ({args.query.rstrip().rstrip(';')}) TO '{path}' (FORMAT {fmt})").fetchone()
        out["out"] = args.out
        out["out_rows"] = written[0]
    return out


def search(con, args) -> dict:
    words = args.name.split()
    if not words:
        raise SystemExit("search takes a name")
    where = " AND ".join(["name ILIKE ?"] * len(words))
    params = [f"%{w}%" for w in words]
    if args.kind:
        where += " AND kind = ?"
        params.append(args.kind)
    rel = con.sql(
        f"""
        WITH hits AS (SELECT * FROM nodes WHERE {where})
        SELECT hits.*,
            (SELECT count(*) FROM relationships WHERE node_id_start = hits.node_id)
            + (SELECT count(*) FROM relationships WHERE node_id_end = hits.node_id) AS relationships
        FROM hits
        ORDER BY jaro_winkler_similarity(upper(name), upper(?)) DESC, relationships DESC
        """,
        params=params + [args.name],
    )
    return rows(rel, args.limit)


def node(con, args) -> dict:
    found = con.execute("SELECT kind FROM nodes WHERE node_id = ?", [args.id]).fetchone()
    if found is None:
        return {"error": f"no node {args.id}"}
    kind = found[0]
    rel = con.sql(f"SELECT * FROM {TABLES[kind]} WHERE node_id = ?", params=[args.id])
    columns = rel.columns
    record = dict(zip(columns, rel.fetchone()))
    # `direction` is which end of the relationship this node is: `out` when it is the start,
    # as an officer is of the entity it is an officer of.
    edges = con.sql(
        """
        SELECT 'out' AS direction, r.rel_type, r.link, n.node_id, n.kind, n.name, n.countries,
            r.status, r.start_date, r.end_date, r.sourceID
        FROM relationships r JOIN nodes n ON n.node_id = r.node_id_end
        WHERE r.node_id_start = $id
        UNION ALL
        SELECT 'in', r.rel_type, r.link, n.node_id, n.kind, n.name, n.countries,
            r.status, r.start_date, r.end_date, r.sourceID
        FROM relationships r JOIN nodes n ON n.node_id = r.node_id_start
        WHERE r.node_id_end = $id
        ORDER BY 1 DESC, 2, 6
        """,
        params={"id": args.id},
    )
    listed = rows(edges, args.limit)
    return {
        "kind": kind,
        "node": record,
        "relationships": listed["rows"],
        "relationships_total": con.execute(
            "SELECT (SELECT count(*) FROM relationships WHERE node_id_start = $id)"
            " + (SELECT count(*) FROM relationships WHERE node_id_end = $id)",
            {"id": args.id},
        ).fetchone()[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sql")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out")
    p = sub.add_parser("search")
    p.add_argument("name")
    p.add_argument("--kind", choices=sorted(TABLES))
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("node")
    p.add_argument("id", type=int)
    p.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    con = duckdb.connect(DB, read_only=True)
    try:
        out = {"sql": sql, "search": search, "node": node}[args.command](con, args)
    except duckdb.Error as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)
    print(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
