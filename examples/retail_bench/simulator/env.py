"""The store: one run of RetailBench's environment, and what is scored.

This is the only module that touches `upstream/`. Everything above it — the
REST server, the CLI — goes through [`Session`], and [`Session`] is deliberately
narrow: three actions, any number of views, and the numbers the paper counts.

## Three actions, and only three

RetailBench declares nineteen tools. Fifteen of them only look, and in this
harness what they would have answered is written into the context directory as
files instead (see `context.py`). What is left is what *changes* something:

* `place_order` — spends funds, creates an order in transit
* `modify_sku_price` — sets what the store charges
* `end_today` — settles the day and moves the clock

They stay callable because they can be **refused** — an order for a supplier who
does not quote that SKU today, or one the store cannot afford, has to come back
as a refusal the agent can act on. A file cannot say no.

`add_note` is upstream's fourth state-changing tool and is *not* exposed: in this
harness the agent's memory is a file it writes itself, which `context.py` carries
into the next morning. Nothing here can be rejected, so nothing here needs a tool.
"""

from __future__ import annotations

import functools
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"

# `module/` imports its siblings bare (`from sku import SKU`), so the package
# directory goes on the path as well as the root it sits in.
for path in (UPSTREAM, UPSTREAM / "module"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import context  # noqa: E402

# The three the agent may call. Anything else reaching `act` is a bug in the
# caller, not a refusal for the agent to think about.
ACTIONS = ("place_order", "modify_sku_price", "end_today")

# Days with funds below zero, in a row, before the store is done. Upstream's
# runners' number, not ours.
BANKRUPT_AFTER = 5

# Config keys whose value is a path into the dataset. Rewritten under `--data`
# so that the dataset can live anywhere while the config keeps upstream's own
# relative spelling.
DATA_PATHS = (
    "data_dir",
    "customer_data_path",
    "init_sql_path",
    "review_model_path",
    "review_source_path",
    "news_source_path",
)


def quiet() -> None:
    """Put the simulator's logger on stderr before it puts itself on stdout.

    `util.logger.get_logger` installs a `StreamHandler(sys.stdout)` the first
    time it is called, and only when the logger has no handler yet — so giving
    it one here is what keeps the simulator's chatter out of whatever this
    process's stdout is being read as.
    """
    import logging

    from util.logger import logger as simulator

    if not simulator.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        simulator.addHandler(handler)
    simulator.propagate = False


def record_orders_once() -> None:
    """Keep a delivered order out of `supplier_orders` a second time.

    Upstream writes a supplier order to that table twice: once where it is
    placed (`RetailEnvironment.place_order`) and once more where it lands
    (`OrderManager.step`, given the record manager by the daily step). So every
    row doubles on the day its goods arrive, and a run's
    `order_records/records.db` reports more bought than was ever paid for —
    $49,248 against $28,193, five days into the run this was found in.

    Nothing the store runs on reads that table: funds, net worth and the
    agent's own `store/orders_in_transit.md` all come from the order manager
    holding the orders. It is the record of the run that is wrong, and anything
    summing `cost` or counting rows out of it afterwards that is wrong with it.

    The write at order time is the one to keep. It carries the arrival date the
    supplier quoted, which is what the second write overwrites with a clock
    that has already moved — and it is there for an order still in transit,
    which the second write is not. So the delivery-time write is what goes, by
    withholding from `step` the record manager it needs to make it.
    """
    from module.order_manager import OrderManager

    if getattr(OrderManager.step, "records_once", False):
        return

    upstream_step = OrderManager.step

    @functools.wraps(upstream_step)
    def step(self, current_date, record_manager=None):  # noqa: ARG001
        return upstream_step(self, current_date, record_manager=None)

    step.records_once = True
    OrderManager.step = step


# The environment the paper describes, which the published config does not.
# Section 3.1: "an initial budget of 30,000, a daily rent of 600, an inventory
# capacity of 15,000 units, and a shelf capacity of 40 products". The repo ships
# 50,000, 1,000 and 40,000 instead, and a run against those is not the run the
# paper's table reports: at a rent of 1,000 the best assortment this demand
# model allows needs a 37% margin to break even, against 22% at 600, so the sign
# of the result turns on the number rather than on the agent.
#
# The fourth, the shelf, is not here. Nothing upstream implements it, and
# holding the shelf to forty products is worth about a tenth of the volume —
# because demand is a category sum, dropping a SKU takes its own contribution
# out along with the crowding it caused. An agent free to carry forty can reach
# the same place by carrying forty; the cap only forecloses carrying all
# ninety-six, and which products to carry is the agent's decision to get wrong.
PAPER = {
    "initial_funds": 30000,
    "everyday_rent": 600,
    "inventory_capacity": 15000,
}

# A.3 again: "Category Effects: All categories have a uniform effect of -0.1".
# The repo ships -0.2 for all twenty, which is the difference between a demand
# model that reproduces this store's own recorded past and one that does not —
# at -0.1 it predicts 0.54 of the 5,309 units a day the seeded history holds,
# at -0.2 only 0.33. The coefficient subtracts a share of every same-category
# neighbour's pull from a product's own, so doubling it does not halve demand,
# it clamps the weaker half of the shelf to nothing.
CATEGORY_EFFECT = -0.1

def deliver_per_sku_once() -> None:
    """Land each SKU of an order on the day that SKU was quoted for.

    `place_order` draws a transport time inside the per-SKU loop — every line
    gets its own `shipping_days`, and the merchandise it builds is stamped with
    `begin_time` and `expired_time` from that draw. The `Order` it then makes is
    built once, after the loop, from whatever `delivery_days` the *last* line
    happened to leave behind, and `OrderManager.step` delivers the whole order
    on that one date.

    So a line quoted four days lands with one quoted nine, and a line quoted
    nine lands with one quoted four. The second case is the one that shows:
    goods arrive before their own `begin_time`, and `compute_net_worth` values
    stock at `buy_price * remaining / total_span` — with `current_date` before
    `begin_time`, `remaining` exceeds `total_span` and the ratio passes one. A
    day-two net worth of 29,006.15 against the 28,800 that two nights of rent
    leave is that, and nothing else: stock whose shelf-life clock has not
    started, marked above what was paid for it.

    Fixed where the mistake is rather than where it shows: an order now
    delivers the items that are due and keeps the rest, so `begin_time` is the
    day a thing actually arrives and the ratio cannot exceed one. What is still
    in transit keeps the cost of what is still in transit, and the order's date
    moves to the next line due, so `store/orders_in_transit.md` says a true
    thing on both counts.

    This replaces `step` outright, so it also takes on what [`record_orders_once`]
    was there for: no record manager is ever passed on, and a delivered order is
    written to `supplier_orders` once, where it was placed.
    """
    from module.order_manager import OrderManager

    if getattr(OrderManager.step, "delivers_per_sku", False):
        return

    def step(self, current_date, record_manager=None):  # noqa: ARG001
        delivered = []
        emptied = []
        for order_id, order in self.orders.items():
            due, held = [], []
            for merch in order.items:
                (due if merch.begin_time <= current_date else held).append(merch)
            if not due:
                continue
            delivered.extend(due)
            order.items = held
            if held:
                order.cost = sum(getattr(m, "buy_price", 0.0) for m in held)
                order.expected_delivery_date = min(m.begin_time for m in held)
            else:
                emptied.append(order_id)
        for order_id in emptied:
            self.orders.pop(order_id)
        return delivered

    step.delivers_per_sku = True
    # What `record_orders_once` guards against, guarded by this being the whole
    # of `step`: it can return early rather than wrap what it no longer needs to.
    step.records_once = True
    OrderManager.step = step


def build_config(kind: str, data: Path, run: Path) -> dict[str, Any]:
    """Upstream's own config for a dataset, with every path made absolute."""
    from util.default_config import (
        create_dynamic_hard_config,
        create_dynamic_middle_config,
        create_still_hard_config,
        create_still_middle_config,
    )

    makers = {
        "dynamic_hard": create_dynamic_hard_config,
        "dynamic_middle": create_dynamic_middle_config,
        "still_hard": create_still_hard_config,
        "still_middle": create_still_middle_config,
    }
    if kind not in makers:
        raise ValueError(f"unknown config {kind!r} (one of: {', '.join(makers)})")

    config = makers[kind]()
    # The paper's numbers, over the repo's. See [`PAPER`].
    config.update(PAPER)
    config["category_effects"] = {
        name: CATEGORY_EFFECT for name in config.get("category_effects", {})
    }
    for key in DATA_PATHS:
        if config.get(key):
            config[key] = str(data / config[key])
    # Where the simulator writes: its order-record database and its own logs,
    # both under the run, so a second run shares nothing with the first.
    config["order_record_dir"] = str(run / "order_records")
    config["log_dir"] = str(run / "sim_logs")
    config["debug"] = False

    # A partial fetch has to become a partial store. `sim fetch --skus` leaves a
    # manifest saying which SKUs it took; without this the environment builds
    # every SKU the parameter file names, and the ones whose supplier file was
    # never fetched have nobody quoting them.
    manifest = data / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if not recorded.get("full", True) and recorded.get("skus"):
            config["sku_ids"] = [str(s) for s in recorded["skus"]]
    return config


def missing_data(config: dict[str, Any]) -> list[str]:
    """The dataset files the config names and the disk does not have.

    Checked before the environment is built, because the simulator answers a
    missing `sku_model_parameter.json` with a warning, an empty SKU map, and a
    run that looks like it started.
    """
    wanted = [Path(config[key]) for key in DATA_PATHS if config.get(key)]
    store = Path(config["data_dir"]) / str(config["store_id"])
    wanted += [store / "sku_model_parameter.json", store / "upc.json"]
    return [str(p) for p in wanted if not p.exists()]


class Refused(Exception):
    """An action the store will not take, in words the agent can act on.

    Upstream refuses in two shapes — a `ValueError` for anything malformed or
    unknown, and a result carrying an `error` key when the funds are short —
    and [`Session.act`] turns both into this one.
    """


class Session:
    """One run: the environment, the clock, and the numbers it is scored on."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        max_days: int,
        context_dir: Path | None = None,
    ):
        quiet()
        deliver_per_sku_once()
        record_orders_once()
        from retail_environment import RetailEnvironment

        self.config = config
        self.max_days = max_days
        self.context_dir = context_dir

        self.env = RetailEnvironment(config)
        self.market = context.Market(config, self.env, max_days)

        self.day = 1
        self.days_completed = 0
        self.negative_days = 0
        self.terminated: str | None = None
        self.context_written: str | None = None

        # The paper's numbers, accumulated as the days go by: a run that ends in
        # bankruptcy still has to report the sales it made.
        self.total_sales = 0
        self.total_returns = 0
        self.total_expired = 0
        self.stockout_days = 0
        self.actions = 0
        self.refusals = 0

    # ── what the agent may do ─────────────────────────────────────

    def act(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """One action, applied — or refused with a reason.

        The context is rewritten on the way out, because an order that was
        placed has to be visible in `store/orders_in_transit.md` before the
        agent looks again. Only the cheap half after an ordinary action; the
        whole tree when the day turned, since that is when the market moves.
        """
        if name not in ACTIONS:
            raise Refused(f"{name!r} is not an action; the views are files under the context directory")
        if self.terminated:
            raise Refused(f"the run is over ({self.terminated})")

        try:
            result = self.env.exec_tools(name, **arguments)
        except (TypeError, ValueError) as exc:
            # Belt and braces: `exec_tools` swallows these itself and answers
            # with an `error` result, so this only fires if a future upstream
            # lets one through.
            self.refusals += 1
            raise Refused(unwrap(str(exc), name)) from exc

        payload = result.get("result")
        # Every refusal arrives this way. `exec_tools` catches its own
        # exceptions and hands back a result carrying `error` — so a bad
        # supplier, a malformed argument and insufficient funds all look alike
        # here, and none of them raised.
        if isinstance(payload, dict) and payload.get("error"):
            self.refusals += 1
            raise Refused(unwrap(str(payload["error"]), name))

        self.actions += 1
        reply = {
            "ok": True,
            "action": name,
            "formatted": result.get("formatted", ""),
            "result": jsonable(payload),
        }
        if name == "end_today":
            reply.update(self._day_ended(payload or {}))
            self.write_context()
        else:
            self.write_context(market=False)
        return reply

    def view(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """One of the fifteen read-only tools, for looking without an agent.

        Not what the agent uses — it reads files — but the same call the files
        are written from, which is what makes it worth exposing: it answers
        "what will be in `store/inventory.md` tomorrow" without running a day.
        """
        if name in ACTIONS:
            raise Refused(f"{name} changes the store; POST it as an action")
        result = self.env.exec_tools(name, **arguments)
        return {"ok": True, "view": name, "formatted": result.get("formatted", ""), "result": jsonable(result.get("result"))}

    # ── the clock ─────────────────────────────────────────────────

    def _day_ended(self, step: dict[str, Any]) -> dict[str, Any]:
        """Book the day `end_today` just closed, and say whether there is another.

        `end_today` is the only thing that moves the clock, so this is the one
        place a run can end: the store is out of days, or it has been in the red
        for [`BANKRUPT_AFTER`] of them.
        """
        self.days_completed = self.day
        self.day += 1

        sold = step.get("sales_by_sku") or {}
        returned = step.get("returns_by_sku") or {}
        expired = step.get("expired_discount_by_sku") or {}
        self.total_sales += sum(int(v) for v in sold.values())
        self.total_returns += sum(int(v) for v in returned.values())
        self.total_expired += sum(int(v) for v in expired.values())
        if step.get("insufficient_skus"):
            self.stockout_days += 1

        funds = step.get("funds", self.env.funds)
        self.negative_days = self.negative_days + 1 if funds < 0 else 0
        if self.negative_days >= BANKRUPT_AFTER:
            self.terminated = f"bankrupt: funds below zero for {self.negative_days} days running"
        elif self.days_completed >= self.max_days:
            self.terminated = f"horizon reached: {self.max_days} days"

        return {
            "day_over": True,
            "day": self.days_completed,
            "next_day": None if self.terminated else self.day,
            "terminated": self.terminated,
        }

    # ── what it is worth ──────────────────────────────────────────

    def worth(self) -> float:
        """Net worth the way the leaderboard counts it: stock in transit included.

        Upstream has two definitions and they differ by the orders already paid
        for and not yet delivered: `step` adds them, the `_log_tool_call` line
        does not. `benchmark_results.json` says its trajectories come from the
        second, and the numbers say otherwise — every one of the eight published
        runs moves by exactly the rent on days one and two, to the cent, and the
        oracle policy buys on day one. Only the first definition does that.
        Money spent on goods that have not arrived is still the store's money.
        """
        return self.worth_on_shelf() + sum(
            order.cost for order in self.env.order_manager.get_current_orders()
        )

    def worth_on_shelf(self) -> float:
        """The other definition: funds plus what is actually on the shelf.

        Reported beside the scored one because it is what a day's liquidity
        looks like — a store can be worth a great deal and still be unable to
        pay the rent.
        """
        return self.env.inventory.compute_net_worth(self.env.current_date) + self.env.funds

    def state(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "days_completed": self.days_completed,
            "max_days": self.max_days,
            "date": str(self.env.current_date),
            "funds": self.env.funds,
            "net_worth": self.worth(),
            "terminated": self.terminated,
            "skus": len(self.env.skus_id_map),
            "context_dir": str(self.context_dir) if self.context_dir else None,
            "context_written": self.context_written,
        }

    def metrics(self) -> dict[str, Any]:
        """What the leaderboard is made of, by the paper's definitions.

        `run_days` first, then `final_networth`, then `total_sales` — the
        survival-first selection rule — and the diagnostic ratios under them.
        """
        days = max(self.days_completed, 1)
        return {
            "run_days": self.days_completed,
            "final_networth": self.worth(),
            "final_networth_on_shelf": self.worth_on_shelf(),
            "final_funds": self.env.funds,
            "total_sales": self.total_sales,
            "return_ratio": self.total_returns / self.total_sales if self.total_sales else 0.0,
            "expired_ratio": self.total_expired / self.total_sales if self.total_sales else 0.0,
            "stockout_ratio": self.stockout_days / days,
            "actions": self.actions,
            "refusals": self.refusals,
            "terminated": self.terminated,
        }

    # ── stopping, and starting again ──────────────────────────────

    # What upstream's checkpoint does not carry, and this has to. Everything the
    # environment holds is in its own file; these are the counters the run is
    # scored by, which live up here.
    OURS = (
        "day",
        "days_completed",
        "negative_days",
        "terminated",
        "total_sales",
        "total_returns",
        "total_expired",
        "stockout_days",
        "actions",
        "refusals",
    )

    def save(self, directory: Path) -> dict[str, Any]:
        """Write the store where it stands, so a later run can pick it up.

        Two files: upstream's own checkpoint (funds, date, prices, inventory,
        orders, news, and a SQL dump of the whole record database) and ours
        (the counters above). Written to a directory beside the live one and
        swapped in, so a crash mid-save leaves the last good checkpoint rather
        than half of this one.

        What the agent remembers is not in here, and does not need to be: its
        notes are files in `artifacts/`, which is already in the run directory
        this is saved under.
        """
        directory = Path(directory)
        staging = directory.with_name(directory.name + ".writing")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        self.env.save_checkpoint(staging / "environment.json")
        (staging / "session.json").write_text(
            json.dumps(
                {
                    "saved": datetime.now().isoformat(timespec="seconds"),
                    "date": str(self.env.current_date),
                    "max_days": self.max_days,
                    **{name: getattr(self, name) for name in self.OURS},
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        if directory.exists():
            shutil.rmtree(directory)
        staging.rename(directory)
        return {"saved": str(directory), "day": self.day, "date": str(self.env.current_date)}

    @classmethod
    def restore(
        cls,
        directory: Path,
        *,
        max_days: int | None = None,
        context_dir: Path | None = None,
    ) -> "Session":
        """Pick a store up where [`Session.save`] left it.

        **A resumed run is not the same run as one that was never stopped.**
        Rebuilding the environment re-seeds `random`, so the demand, reviews and
        news sampled from here on are not the ones the uninterrupted run would
        have drawn. Everything the store *holds* is restored exactly; the
        stream it draws from is not. Resume is recovery, not continuation, and
        two runs that were resumed at different points are not comparable to
        each other on anything that depends on that stream.
        """
        quiet()
        deliver_per_sku_once()
        record_orders_once()
        from retail_environment import RetailEnvironment

        directory = Path(directory)
        saved = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        env = RetailEnvironment.recover_from_checkpoint(directory / "environment.json")

        # The environment applies initial ratings while it is built — before the
        # record database it would compute them from has been restored. Asked
        # again here, with the restored records in place, so a resumed store's
        # ratings are the ones its own history implies.
        try:
            env._apply_initial_ratings(env.review_manager if env.review_manager.enabled else None)
        except Exception as exc:  # a SKU with no ratings yet raises; it did at day one too
            print(f"[warn] initial ratings after restore: {type(exc).__name__}: {exc}")

        session = cls.__new__(cls)
        session.config = env.config
        session.max_days = max_days or int(saved.get("max_days", 180))
        session.context_dir = context_dir
        session.env = env
        session.market = context.Market(env.config, env, session.max_days)
        session.context_written = None
        for name in cls.OURS:
            setattr(session, name, saved.get(name))
        # A store restored at its horizon would refuse every action; asking for
        # more days is how a run is extended, so the ending is recomputed.
        if session.days_completed < session.max_days and session.terminated and "horizon" in str(session.terminated):
            session.terminated = None
        return session

    # ── what the agent reads ──────────────────────────────────────

    def write_context(self, market: bool = True) -> dict[str, Any]:
        """Write the context tree for today. See `context.py`.

        `market=False` skips the dataset half, which is a function of the date
        and not of anything an action did: rewriting seven megabytes of supplier
        quotes because a price changed would be work for no difference.
        """
        if self.context_dir is None:
            return {"written": False, "why": "no context directory was named"}
        written = context.write(
            self.context_dir,
            env=self.env,
            market=self.market if market else None,
            day=self.day,
            max_days=self.max_days,
            begin=context.parse_date(self.config.get("data_begin_time")),
        )
        self.context_written = datetime.now().isoformat(timespec="seconds")
        return written


def unwrap(message: str, tool: str) -> str:
    """`exec_tools` re-raises everything as `Error executing X: ValueError: …`.

    The agent is being told why the store said no, and the two words of Python
    in front of the reason are not part of that. Stripped once, here, so every
    refusal reads the same whether upstream wrapped it or not.
    """
    prefix = f"Error executing {tool}: "
    if message.startswith(prefix):
        message = message[len(prefix) :]
        _, _, rest = message.partition(": ")
        return rest or message
    return message


def jsonable(value: Any) -> Any:
    """The simulator hands back its own objects; the wire takes JSON.

    Dates become their ISO spelling and anything else without one becomes its
    `str`, which is what upstream's own logs do with the same values.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    for attribute in ("get_detail", "to_dict"):
        if hasattr(value, attribute):
            try:
                return jsonable(getattr(value, attribute)())
            except Exception:
                pass
    if hasattr(value, "sku_id"):
        return str(value.sku_id)
    return str(value)
