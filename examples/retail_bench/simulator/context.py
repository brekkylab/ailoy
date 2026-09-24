"""The tree the agent reads, written fresh and cut to today.

RetailBench gives an agent nineteen tools and fifteen of them only look. Here
those fifteen are not tools: what they would have answered is written as files,
and the agent reads them the way it reads any other workspace — with its file
tools, in a console, against a read-only mount.

Two halves, because they come from two different places:

* **`market/`** — the dataset's own files, *as they are*, with the rows dated
  after today removed. Nothing is reshaped: `<UPC>_suppliers.json` has the keys
  RetailBench ships, metadata included. This is the half that makes the cut
  necessary — the files run to 1996 and the store is in 1991.
* **`store/`** — what the simulator knows and no file holds: the inventory, the
  funds, the orders in transit, the store's own sales and reviews so far. Each
  file is the `formatted` text of the `view_*` tool it replaces, so the wording
  the agent reads is the wording upstream's own agents read.

The two are rewritten on different clocks.
`store/` changes when the agent acts — an order placed has to show up in
`orders_in_transit.md` — and is small; `market/` is a function of the date alone,
so it is written once a day.

**What is where is not written into the tree.** It used to be, as a `README.md`
generated beside the files, and the agent was told to read that first. It is in
the system prompt now instead (`agent/system.md`): one description rather than
two, and the agent is not spending its first tool call reading a map. The cost is
that this file and that one can drift — if a path moves here, it moves there.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

# The tools whose answer is a record of one day, each written to its own file
# under `store/` (or `news/`) and never rewritten: the tool, the directory, and
# whether the day has to be split across categories to fit.
#
# It fits for five of the six. `exec_tools` truncates a `formatted` answer at
# fifty thousand characters, and one day of the whole store runs to twenty-four
# thousand at the widest (the news) — except the reviews, which reach sixty on
# the busiest days the dataset seeded and would lose the overflow silently. Per
# category they are three thousand, so the split is where the environment's own
# limit puts it and nowhere else.
RECORDS = (
    ("view_sku_sales_history", "store/sales", False),
    ("view_return_rates", "store/returns", False),
    ("view_returns", "store/return_records", False),
    ("view_sku_reviews", "store/reviews", True),
    ("view_sku_avg_ratings", "store/ratings", False),
    ("view_news_history", "news", False),
)

# `view_returns` and `view_news_history` take no SKU list; the rest do.
NO_SKUS = ("view_returns", "view_news_history")


def parse_date(value: Any) -> date | None:
    """A date as the dataset spells it — ISO, or Dominick's `MM/DD/YY`."""
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def rows_upto(rows: Iterable[dict[str, Any]], end: date) -> list[dict[str, Any]]:
    """The rows dated on or before `end`, in the order they were given.

    Only the future is cut. The dataset's own first row is where these files
    start, because a day the data records is a day that happened, and there is
    no reason for this tree to know it and not say so.

    A row whose date cannot be read is dropped rather than kept: it cannot be
    shown to be in the past, and the whole point of this file is that nothing in
    the future is shown.
    """
    kept = []
    for row in rows:
        when = parse_date(row.get("date"))
        if when is not None and when <= end:
            kept.append(row)
    return kept


class Market:
    """The dataset's daily files, loaded once and cut at today when written.

    Only the SKUs the config selected — the tree under `data_dir` carries every
    SKU Dominick's ever recorded — and only the days the run can reach. That
    last bound is the run's horizon and not a window on the past: the dataset
    runs to 1997 and the last day of a 180-day run is 1992, so rows after it
    cannot be written on any day of the run however far the agent looks. The
    supplier files are six megabytes apiece and cutting there is what keeps
    them in memory; nothing the agent could otherwise have seen is dropped.
    """

    def __init__(self, config: dict[str, Any], env: Any, max_days: int):
        self.config = config
        self.store_root = Path(config["data_dir"]) / str(config["store_id"])

        begin = parse_date(config.get("store_begin_time")) or date(1991, 9, 7)
        self.horizon = begin + timedelta(days=max_days)

        self.categories: dict[str, str] = {}
        for sku_id, sku in env.skus_id_map.items():
            self.categories[str(sku_id)] = str(getattr(sku, "category", "") or "unknown")

        self.sku_ids = [str(s) for s in env.skus_id_map]
        self.suppliers: dict[str, dict[str, Any]] = {}
        self.cost_prices: dict[str, list[dict[str, Any]]] = {}
        for sku_id in self.sku_ids:
            self.suppliers[sku_id] = self._load_suppliers(sku_id)
            self.cost_prices[sku_id] = self._load_cost_prices(sku_id)

        self.customers = self._load_customers()
        self.catalog = self._load_catalog()

    # ── loading ───────────────────────────────────────────────────

    def directory(self, sku_id: str) -> str:
        """The category, spelled the way a directory is."""
        return (self.categories.get(sku_id) or "unknown").replace(" ", "_")

    def _sku_dir(self, sku_id: str) -> Path | None:
        """Where a SKU's files are.

        The metadata spells a category with spaces and the directory with
        underscores, so both are tried before falling back to a walk.
        """
        category = self.categories.get(sku_id, "")
        for name in {category, category.replace(" ", "_"), category.replace("_", " ")}:
            if name and (self.store_root / name).is_dir():
                return self.store_root / name
        for path in self.store_root.glob(f"*/{sku_id}_suppliers.json"):
            return path.parent
        return None

    def _load_suppliers(self, sku_id: str) -> dict[str, Any]:
        directory = self._sku_dir(sku_id)
        path = directory / f"{sku_id}_suppliers.json" if directory else None
        if not path or not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        # The suppliers are the list-valued keys. The rest — `_corr_price_quality`
        # and three per-supplier tables, lead times among them — is the file's own
        # metadata, carried through as it is: it has no dates to cut by, and
        # dropping it would make this a file the dataset does not ship.
        return {
            str(key): rows_upto(value, self.horizon) if isinstance(value, list) else value
            for key, value in payload.items()
        }

    def _load_cost_prices(self, sku_id: str) -> list[dict[str, Any]]:
        directory = self._sku_dir(sku_id)
        path = directory / f"{sku_id}_daily.json" if directory else None
        if not path or not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return rows_upto(payload, self.horizon) if isinstance(payload, list) else []

    def _load_customers(self) -> list[dict[str, Any]]:
        root = Path(self.config["customer_data_path"])
        store = str(self.config["store_id"])
        for path in (root / f"store_{store}" / f"store_{store}_data.json", root / store / "data.json"):
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return rows_upto(payload, self.horizon) if isinstance(payload, list) else []
        return []

    def _load_catalog(self) -> list[dict[str, Any]]:
        """The item master, trimmed to the SKUs in play.

        Kept whole per entry — descriptions, shelf life, the price band the
        dataset records — because that is the static half an agent is meant to
        know without spending a day finding out.
        """
        path = self.store_root / "upc.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = list(payload.values()) if isinstance(payload, dict) else payload
        traded = set(self.sku_ids)
        return [e for e in entries if str(e.get("UPC")) in traded]

    # ── writing ───────────────────────────────────────────────────

    def write(self, root: Path, today: date) -> int:
        """The market half of the tree, as of `today`. Returns files written.

        The dated rows go out as CSV; what is not a table stays JSON. For the
        supplier files that means the quotes themselves in
        `<UPC>_suppliers.csv` — every supplier in one table, since `supplier_id`
        is already a column — and the four keys that are not rows in one
        `suppliers/meta.json` keyed by UPC. Three of those four only restate
        what the columns already say (`cycle_days`, `transport_days_min/max`);
        `_corr_price_quality` is the one fact that is nowhere else, and dropping
        it would lose something the dataset ships.
        """
        written = 0
        meta: dict[str, dict[str, Any]] = {}
        for sku_id in self.sku_ids:
            category = self.directory(sku_id)
            quotes, aside = [], {}
            for key, value in self.suppliers.get(sku_id, {}).items():
                if isinstance(value, list):
                    quotes.extend(rows_upto(value, today))
                else:
                    aside[key] = value
            if quotes:
                quotes.sort(key=lambda row: (str(row.get("date")), str(row.get("supplier_id"))))
                table(root / "suppliers" / category / f"{sku_id}_suppliers.csv", quotes)
                written += 1
            if aside:
                meta[sku_id] = aside
            costs = rows_upto(self.cost_prices.get(sku_id, []), today)
            if costs:
                table(root / "cost_prices" / category / f"{sku_id}_daily.csv", costs)
                written += 1
        if meta:
            dump(root / "suppliers" / "meta.json", meta)
            written += 1
        if self.customers:
            table(root / "customer_count.csv", rows_upto(self.customers, today))
            written += 1
        if self.catalog:
            table(root / "catalog.csv", [as_row(entry) for entry in self.catalog])
            written += 1
        return written


def as_row(entry: dict[str, Any]) -> dict[str, Any]:
    """One catalog entry, flat enough to be a row.

    `DELEVERY_TIME` is the only value in the item master that is not a scalar:
    a `[min, max]` pair of days, always two and always in order. Split, it is
    two columns named the way the supplier files already name their own pair,
    and the catalog is a table like everything else under `market/`.
    """
    row = dict(entry)
    window = row.pop("DELEVERY_TIME", None)
    if isinstance(window, (list, tuple)) and len(window) == 2:
        row["DELEVERY_TIME_MIN"], row["DELEVERY_TIME_MAX"] = window
    elif window is not None:
        row["DELEVERY_TIME_MIN"] = row["DELEVERY_TIME_MAX"] = window
    return row


def table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Dated rows as CSV, which is what rows are.

    The dataset ships these as JSON and the shape is a table: every row in a
    supplier file carries the same fifteen keys, and pretty-printed JSON spends
    four fifths of its bytes repeating them. A megabyte and a half a SKU becomes
    three hundred kilobytes, and `head`, `cut` and `awk` work on it — which is
    what the agent has, since it reads this tree with file tools and a shell.

    Columns are the union of the keys, in the order they were first seen, so a
    row that carries an extra field does not silently lose it.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def looked(env: Any, tool: str, **arguments: Any) -> str:
    """What a `view_*` tool would have said, as text.

    A tool that refuses — news with the news manager off, reviews before any
    exist — writes what it said rather than failing the morning.
    """
    try:
        result = env.exec_tools(tool, **arguments)
    except Exception as exc:
        return f"({tool} is unavailable: {type(exc).__name__}: {exc})"
    said = result.get("formatted")
    if said:
        return str(said)
    return json.dumps(result.get("result"), ensure_ascii=False, indent=2, default=str)


def answered(env: Any, tool: str, **arguments: Any) -> tuple[str, bool]:
    """[`looked`], with whether the tool found anything to say.

    The text alone cannot be asked: a tool with nothing to report still returns
    a heading — `## News 1991-06-06 ~ 1991-06-06 (0 items)` — and that heading
    is forty-two bytes that look like a file with news in it. The `result`
    payload is where the answer is empty or not, so it is what is asked.

    A refusal counts as something said, and is written: the agent should see
    that a tool was unavailable rather than find a day quietly missing.
    """
    try:
        result = env.exec_tools(tool, **arguments)
    except Exception as exc:
        return f"({tool} is unavailable: {type(exc).__name__}: {exc})", True
    said = result.get("formatted")
    text = str(said) if said else json.dumps(
        result.get("result"), ensure_ascii=False, indent=2, default=str
    )
    return text, has_rows(result.get("result"))


def has_rows(payload: Any) -> bool:
    """Whether a tool's `result` holds any row at all, however deeply."""
    if isinstance(payload, dict):
        return any(has_rows(value) for value in payload.values())
    if isinstance(payload, list):
        return any(has_rows(value) for value in payload)
    return payload is not None and payload != ""


def write(
    root: Path,
    *,
    env: Any,
    market: Market | None,
    day: int,
    max_days: int,
    begin: date | None,
) -> dict[str, Any]:
    """Write the tree for today. `market=None` writes only the half that moves."""
    today = env.current_date
    if isinstance(today, datetime):
        today = today.date()
    sku_ids = [str(s) for s in env.skus_id_map]
    now = today.isoformat()

    root.mkdir(parents=True, exist_ok=True)
    text(root / "today.md", today_page(env, day, max_days, today))

    # The snapshots: what is true at this moment and has no date of its own.
    # Rewritten on every action, because an order placed has to show up before
    # the agent looks again.
    store = root / "store"
    text(store / "funds.md", looked(env, "view_funds_and_date"))
    text(store / "inventory.md", looked(env, "view_inventory"))
    text(store / "orders_in_transit.md", looked(env, "view_current_orders"))
    text(store / "shelf_prices.md", looked(env, "view_sku_prices", sku_ids=sku_ids))
    text(root / "market" / "quotes_today.md", looked(env, "view_current_date_supplier_prices", sku_ids=sku_ids))
    text(root / "news" / "today.md", news_today(env))

    days = record_days(root, env, sku_ids, today, begin)
    files = market.write(root / "market", today) if market else 0
    return {
        "written": True,
        "day": day,
        "date": now,
        "record_files": days,
        "market_files": files,
        "market": market is not None,
    }


def record_days(root: Path, env: Any, sku_ids: list[str], today: date, begin: date | None) -> int:
    """One file to the day, for every day the store has a record of.

    **Written once and never rewritten.** A day that is over cannot change, so
    the file holding it is the same on day 180 as it was the morning after it
    closed — the agent can read back any day of the run, and the days before it
    that the dataset seeded, without the tree deciding for it how far back is
    far enough.

    Days strictly before today, and today not at all: sales, returns and
    reviews settle when `end_today` runs, so today has nothing to record until
    it is over, and a file written mid-day would say a day was empty that was
    not. The first write of a run therefore lays down `data_begin_time` up to
    the morning it opens — the seed the store came with — and every day-turn
    after it adds exactly one day to each directory.
    """
    if begin is None:
        return 0
    groups = group_by_category(env)
    written = 0
    for name, where, split in RECORDS:
        directory = root / where
        for category, ids in (groups.items() if split else ((None, sku_ids),)):
            here = directory / category if category else directory
            here.mkdir(parents=True, exist_ok=True)
            when = begin
            while when < today:
                path = here / f"{when.isoformat()}.md"
                if not path.exists():
                    spelled = when.isoformat()
                    arguments: dict[str, Any] = {"start_date": spelled, "end_date": spelled}
                    if name not in NO_SKUS:
                        arguments["sku_ids"] = ids
                    body, found = answered(env, name, **arguments)
                    # A day with nothing to report gets no file. The store's own
                    # news begins when it opens, so every one of the seeded days
                    # before that would otherwise be a heading saying `(0 items)`
                    # — ninety-three files that look like news and are not.
                    if found:
                        text(path, body)
                        written += 1
                when += timedelta(days=1)
    return written


def group_by_category(env: Any) -> dict[str, list[str]]:
    """The traded SKUs, grouped the way the store is laid out."""
    grouped: dict[str, list[str]] = {}
    for sku_id, sku in env.skus_id_map.items():
        name = (str(getattr(sku, "category", "") or "unknown")).replace(" ", "_")
        grouped.setdefault(name, []).append(str(sku_id))
    return grouped


def news_today(env: Any) -> str:
    """Today's headlines, each with the detail the agent would have opened.

    The detail is here rather than behind a second call because in this harness
    reading is not a tool call. It is the clearest place where this departs from
    upstream, where opening a story is a decision that costs something.
    """
    listing = looked(env, "view_today_news")
    parts = [listing]
    try:
        today = env.exec_tools("view_today_news").get("result") or []
    except Exception:
        return listing
    for item in today if isinstance(today, list) else []:
        news_id = item.get("id") or item.get("record_id")
        if news_id is None:
            continue
        parts.append(
            f"\n## {item.get('title', news_id)}\n\n"
            + looked(env, "view_news_detail", news_id=str(news_id))
        )
    return "\n".join(parts)


def today_page(env: Any, day: int, max_days: int, today: date) -> str:
    """The one page that is written rather than quoted: where the store stands."""
    stock = env.inventory.compute_net_worth(today)
    in_transit = sum(order.cost for order in env.order_manager.get_current_orders())
    capacity = getattr(env.inventory, "capacity", None)
    return f"""# Day {day} of {max_days} — {today.isoformat()}

| | |
| --- | --- |
| Funds | {env.funds:,.2f} |
| Stock, at cost | {stock:,.2f} |
| Paid for, in transit | {in_transit:,.2f} |
| **Net worth** | **{env.funds + stock + in_transit:,.2f}** |
| Rent, charged every night | {env.config.get("everyday_rent")} |
| Inventory capacity | {capacity if capacity is not None else "unbounded"} |
| SKUs you trade | {len(env.skus_id_map)} |

Net worth is funds plus stock at cost plus what is paid for and in transit — what the run is scored on.
Rent comes off every night whether or not anything sold, and it comes out of funds, not out of net worth.
"""
