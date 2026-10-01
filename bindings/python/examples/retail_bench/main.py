"""Run RetailBench (https://github.com/linghuazhang01/RetailBench) against an agent: one
supermarket, one day at a time, for as many days as the store survives.

    uv run main.py --smoke            # eight SKUs, two days: the machinery
    uv run main.py --days 3           # three days of the real store
    uv run main.py                    # the benchmark: 96 SKUs, 180 days

The Python side of the Rust `retail_bench` example in `examples/retail_bench`, whose `main.rs`
has the long form. The store is RetailBench's own simulator, run as a local REST server by
that example's `simulator/sim`, which this one shares, upstream code and dataset cache
included. This folder is a uv project of its own (`pyproject.toml`), which `uv run` sets up
with ailoy from this checkout; run it from here.

Every morning the store writes what the agent may read into `runs/<slug>/context/`, mounted
read-only at `/context`. The agent reads it, calls `place_order` and `modify_sku_price` to
change something, and closes the day with `end_today`. Then the store writes tomorrow's tree
and the agent is asked again, with an empty history: only its notes in `/artifacts/notes/`
survive the night.

    runs/<slug>/
      metrics.json      run_days, final_networth, total_sales, and the ratios
      days.jsonl        one line per closed day: funds, net worth, sales
      tool_calls.jsonl  one line per action, in RetailBench's own field names
      days/NNN.json     the messages of one day's turn
      context/          the tree as it stood when the run ended
      artifacts/        what the agent wrote, notes/<date>.md among it

The store stops with the run; `--keep-store` leaves it up to be asked.

Environment:

* `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its
  provider's API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).

Read from `.env` as well.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Absolute, because a mount is named to the console server as a `file://` URL.
HERE = Path(__file__).resolve().parent

# This folder is a uv project of its own. Started from another environment, as with
# `uv run python examples/retail_bench/main.py` from `bindings/python`, run again in this one.
if Path(sys.prefix).resolve() != HERE / ".venv" and "AILOY_EXAMPLE_REEXEC" not in os.environ:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["AILOY_EXAMPLE_REEXEC"] = "1"  # once: not again if uv puts the environment elsewhere
    os.execvpe("uv", ["uv", "run", "--directory", str(HERE), "main.py", *sys.argv[1:]], env)

import ailoy  # noqa: E402
from ailoy.virtx import ConsoleClient, Recipe  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# The Rust example's simulator, shared rather than copied: it keeps the upstream code and the
# dataset it fetched beside itself.
SIM = HERE.parents[3] / "examples" / "retail_bench" / "simulator" / "sim"

SYSTEM = (HERE / "system.md").read_text(encoding="utf-8")
# Sent every morning.
USER = (HERE / "user.md").read_text(encoding="utf-8")
# Sent when a turn ends without closing the day.
NUDGE = (HERE / "nudge.md").read_text(encoding="utf-8")

# The tools the agent may call, `end_today` last because that is where it belongs in a day.
ACTIONS = ["place_order", "modify_sku_price", "end_today"]

# The name the tool and agent providers carrying the three actions are registered under.
PROVIDER = "retail_bench"

# How many times a day is asked before the harness closes it itself: a day left open would
# stop the clock, and the run would never end.
NUDGES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--days", type=int, help="the horizon (default: 180)")
    parser.add_argument(
        "--config",
        default="dynamic_hard",
        help="dynamic_hard | dynamic_middle | still_hard | still_middle",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="eight SKUs and two days: the machinery, not a result"
    )
    parser.add_argument(
        "--keep-store",
        action="store_true",
        help="leave the store's server running when the run ends",
    )
    args = parser.parse_args()
    if args.days is None:
        args.days = 2 if args.smoke else 180
    return args


async def main(args: argparse.Namespace) -> None:
    model = os.environ.get("AILOY_MODEL", "openai/gpt-6-astra")
    prepare(args.smoke)

    run_dir = HERE / "runs" / slug(model, args.config)
    context = run_dir / "context"
    artifacts = run_dir / "artifacts"
    for d in [context, artifacts / "notes", run_dir / "days"]:
        d.mkdir(parents=True, exist_ok=True)
    record = Record(run_dir)

    # ── the store
    store = Store.start(run_dir, context, args.config, args.days)

    # ── the agent
    day_over = threading.Event()
    tools = register_actions(store, record, day_over)
    console = await (
        ConsoleClient.builder()
        .image(Recipe("python:3.12-slim-trixie"))
        .mount_readonly(context, "/context")
        .mount(artifacts, "/artifacts")
        .network(False)
        .vcpus(2)
        .memory_mib(2048)
        .build()
    )

    print(f"  model  {model}")
    print(f"  store  {store.base}")
    print(f"  run    {run_dir}\n")

    # ── the days
    failure = None
    async with console:
        for day in range(1, args.days + 1):
            try:
                reason = await run_day(model, tools, console, store, record, day_over, day, args.days)
            # A failed day ends the run and not the process: the days before it are a
            # result, and late in a run that is hours of work.
            except Exception as e:
                print(f"\nday {day} failed: {e}")
                failure = f"day {day}: {e}"
                break
            if reason is not None:
                print(f"\nthe run ended on day {day}: {reason}")
                break

    try:
        metrics = store.get("/metrics")
    except Exception:
        metrics = None
    record.write(
        "metrics.json",
        {
            "model": model,
            "config": args.config,
            "horizon": args.days,
            "failed": failure,
            "metrics": metrics,
        },
    )
    print(f"\n{json.dumps(metrics, indent=2, ensure_ascii=False)}")
    print(f"\nwritten to {run_dir}")

    if args.keep_store:
        print(f"the store is still up at {store.base} — `sim stop --run {run_dir}` ends it")
    else:
        try:
            store.post("/stop", {})
        except Exception:
            pass


async def run_day(
    model: str,
    tools: list,
    console: ConsoleClient,
    store: "Store",
    record: "Record",
    day_over: threading.Event,
    day: int,
    max_days: int,
) -> str | None:
    """One day: write its tree, ask the agent until it closes the day, and write down what the
    day came to. Returns why the run ended, if this day ended it."""
    # Written before the turn, so the agent's first read is of a tree whose date matches the
    # prompt. Day one's was written when the store booted.
    if day > 1:
        store.post("/context", {"market": True})
    state = store.get("/state")

    def fill(template: str) -> str:
        return (
            template.replace("{{day}}", str(day))
            .replace("{{max_days}}", str(max_days))
            .replace("{{date}}", state.get("date") or "")
            .replace("{{funds}}", number(state.get("funds")))
            .replace("{{net_worth}}", number(state.get("net_worth")))
        )

    # An empty history every morning: the system message, and nothing the agent did yesterday
    # except what it wrote into its notes. `Agent.history` cannot be set, so it is a new agent
    # on the same console.
    agent = await (
        ailoy.AgentBuilder(model)
        .agent_provider(PROVIDER)
        .instruction(SYSTEM)
        .system_tools()
        .tools(tools)
        .console(console)
        .build()
    )
    day_over.clear()

    async with agent:
        # Whether the model answered at all today. A day where it never did is a model that
        # could not be reached, not one that decided to do nothing, and is not played as an
        # empty day.
        heard = False
        error = None
        for nudge in range(NUDGES):
            run = agent.run(fill(USER if nudge == 0 else NUDGE))
            try:
                async for output in run:
                    heard = True
                    show(day, output["message"])
                    # The day closes inside a tool call, so this is checked after every
                    # message: what the model would say next belongs to a day that has not
                    # been written yet.
                    if day_over.is_set():
                        break
            except Exception as e:
                error = str(e)
            finally:
                # Ends the turn where it stands, and lets go of the agent for `history`.
                await run.aclose()
            if day_over.is_set():
                break
        record.write(
            f"days/{day:03}.json",
            {"day": day, "state": state, "error": error, "messages": agent.history},
        )

    ended = None
    if not day_over.is_set():
        if not heard:
            raise RuntimeError(f"the model never answered: {error or ''}")
        print(f"{day:>3}   closed by the harness: the agent did not call end_today")
        call = store.act("end_today", {})
        record.tool_call("end_today", {"forced": True}, call, store, 0.0)
        ended = call.terminated

    metrics = store.get("/metrics")
    record.append(
        "days.jsonl",
        {
            "day": day,
            "date": state.get("date"),
            "funds": metrics.get("final_funds"),
            "net_worth": metrics.get("final_networth"),
            "total_sales": metrics.get("total_sales"),
            "refusals": metrics.get("refusals"),
        },
    )
    print(
        f"day {day:>3}  funds {number(metrics.get('final_funds')):>12}"
        f"  net worth {number(metrics.get('final_networth')):>12}"
        f"  sales {number(metrics.get('total_sales')):>8}"
    )
    if ended:
        return ended
    terminated = metrics.get("terminated")
    return terminated if isinstance(terminated, str) else None


def register_actions(store: "Store", record: "Record", day_over: threading.Event) -> list:
    """Register the three actions under `PROVIDER` and return their descriptions, whose
    schemas are the store's own rather than written here.

    A tool description in a spec is resolved by name against a tool provider, so a new one
    is added, with the built-in tools, and the three are registered into it."""
    declared = store.get("/actions").get("actions") or []

    ailoy.add_tool_provider(PROVIDER)
    descs = []
    for name in ACTIONS:
        # A name the store does not declare means this example and the simulator have drifted
        # apart, which is worth stopping for rather than running without the tool.
        spec = next((spec for spec in declared if spec.get("name") == name), None)
        if spec is None:
            raise RuntimeError(f"the store does not declare '{name}'")
        description = spec.get("description") or ""
        if name == "end_today":
            # Holds only in this harness, so the store's own description does not say it.
            description += (
                " This ends your turn: call it once, when you are done for the day, and expect "
                "no further instructions afterwards."
            )
        desc = {"name": name, "description": description, "parameters": spec["input_schema"]}
        descs.append(ailoy.register_tool(desc, action(name, store, record, day_over), provider=PROVIDER))

    ailoy.add_agent_provider(PROVIDER, tool_provider=PROVIDER)
    return descs


def action(name: str, store: "Store", record: "Record", day_over: threading.Event):
    """The tool function for one action. Synchronous: ailoy runs it off the event loop."""

    def call(**args: Any) -> str:
        started = time.monotonic()
        try:
            result = store.act(name, args)
        # Nothing the model can fix, but saying so beats a silent failure: the day never
        # closes, and the harness closes it.
        except Exception as e:
            return f"The store could not be reached: {e}"
        record.tool_call(name, args, result, store, time.monotonic() - started)
        if result.day_over:
            day_over.set()
        return said(result)

    return call


class Call:
    """What one action came back as. A refusal is an answer, not an error."""

    def __init__(self, said: str, ok: bool, day_over: bool, terminated: str | None, body: dict):
        self.said = said
        self.ok = ok
        self.day_over = day_over
        self.terminated = terminated
        self.body = body


def said(call: Call) -> str:
    """What the model is shown for one action: the store's own words, and a line saying so when
    the day just closed. Without it, a model that was just told yesterday's sales tends to keep
    going."""
    if not call.ok:
        return f"Refused: {call.said}"
    text = call.said
    if call.day_over:
        text += "\n\n---\nThe day is over. Stop here — do not call anything else."
        if call.terminated:
            text += f" The run has ended: {call.terminated}."
        else:
            text += " Tomorrow's files will be waiting when you are asked again."
    return text


def sim(*args: str, **kwargs: Any) -> subprocess.CompletedProcess:
    """`sim`, under this interpreter: it imports nothing outside the standard library, and
    Windows cannot execute its shebang."""
    return subprocess.run([sys.executable, str(SIM), *args], **kwargs)


def prepare(smoke: bool) -> None:
    """Set the store up if it is not: the upstream code, an interpreter for it, and the
    dataset."""
    if not smoke and sim("status", capture_output=True).returncode == 0:
        return
    status = sim("setup", "--no-serve", *(["--skus", "8"] if smoke else [])).returncode
    if status != 0:
        sys.exit(f"setting the store up: exit status {status}")


class Store:
    """The store, over HTTP. Each action tool holds it."""

    def __init__(self, base: str):
        self.base = base

    @classmethod
    def start(cls, run: Path, context: Path, config: str, days: int) -> "Store":
        """Start this run's server — booting loads the store's data, and `sim serve` waits
        for it."""
        status = sim(
            "serve", "--run", str(run), "--context", str(context),
            "--config", config, "--days", str(days),
        ).returncode
        if status != 0:
            sys.exit(f"`sim serve` failed: exit status {status}")
        port = (run / "sim.port").read_text().strip()
        store = cls(f"http://127.0.0.1:{port}")
        store.get("/state")
        return store

    def _request(self, method: str, path: str, payload: Any = None) -> tuple[int, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def get(self, path: str) -> Any:
        status, body = self._request("GET", path)
        if status // 100 != 2:
            raise RuntimeError(f"GET {path} failed ({status}): {body}")
        return body

    def post(self, path: str, payload: Any) -> Any:
        status, body = self._request("POST", path, payload)
        if status // 100 != 2:
            raise RuntimeError(f"POST {path} failed ({status}): {body}")
        return body

    def act(self, name: str, arguments: Any) -> Call:
        """One action. `409` is the store refusing a well-formed request, which the agent is
        meant to read and try differently; anything else that is not `200` is this side being
        wrong."""
        status, body = self._request("POST", f"/actions/{name}", arguments)
        if status not in (200, 409):
            raise RuntimeError(f"{name} failed ({status}): {body.get('error')}")
        ok = status == 200
        terminated = body.get("terminated")
        return Call(
            said=body.get("formatted" if ok else "error") or "",
            ok=ok,
            day_over=body.get("day_over") is True,
            terminated=terminated if isinstance(terminated, str) else None,
            body=body,
        )


def _bytes(value: object) -> str:
    """What JSON is written for an image's bytes in a day's messages."""
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class Record:
    """The run's directory, written to as the run goes, so a run stopped on day 90 has ninety
    days on disk."""

    def __init__(self, path: Path):
        self.path = path
        # Actions are appended from ailoy's threads.
        self.lock = threading.Lock()

    def write(self, name: str, value: Any) -> None:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=_bytes)
        (self.path / name).write_text(text, encoding="utf-8")

    def append(self, name: str, value: Any) -> None:
        line = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_bytes)
        try:
            with self.lock, open(self.path / name, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            print(f"could not append to {self.path / name}: {e}", file=sys.stderr)

    def tool_call(self, tool: str, args: Any, call: Call, store: Store, elapsed: float) -> None:
        """One action, in the field names RetailBench's own `analysis/` scripts read."""
        try:
            state = store.get("/state")
        except Exception:
            state = {}
        self.append(
            "tool_calls.jsonl",
            {
                "ts": int(time.time()),
                "tool": tool,
                "args": args,
                "ok": call.ok,
                "result": call.body.get("result"),
                "formatted": call.said,
                "funds": state.get("funds"),
                "net_worth": state.get("net_worth"),
                "current_date": state.get("current_date"),
                "day": state.get("day"),
                "elapsed_time": round(elapsed, 4),
            },
        )


def show(day: int, message: dict) -> None:
    """One line per thing the agent says or calls, so a long run can be watched."""
    for part in message["contents"]:
        if part["type"] == "text":
            lines = part["text"].strip().splitlines()
            if lines:
                print(f"{day:>3}   {truncate(lines[0], 140)}")
    for call in message.get("tool_calls") or []:
        function = call["function"]
        args = json.dumps(function["arguments"], ensure_ascii=False, separators=(",", ":"))
        print(f"{day:>3} → {function['name']}({truncate(args, 120)})")
    sys.stdout.flush()


def truncate(text: str, at: int) -> str:
    return text if len(text) <= at else text[:at] + "…"


def number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.2f}"
    return "-"


def slug(model: str, config: str) -> str:
    """A sortable name for this run: when it started, which model, which configuration."""
    for c in "/: ":
        model = model.replace(c, "-")
    return f"{int(time.time())}_{model}_{config}"


if __name__ == "__main__":
    # From the nearest `.env` up from this file.
    load_dotenv()
    asyncio.run(main(parse_args()))
