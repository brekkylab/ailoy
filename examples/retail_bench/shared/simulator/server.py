#!/usr/bin/env python3
"""The store, as a REST server.

    GET  /                      what is here, and where the store stands
    GET  /state                 day, date, funds, net worth, is it over
    GET  /metrics               the numbers the run is scored on
    GET  /actions               the three tools, with their schemas
    POST /actions/<name>        take an action — 200 taken, 409 refused
    GET  /views                 the fifteen read-only tools, by name
    GET  /views/<name>          what that tool says now (text, or ?format=json)
    GET  /context               where the context tree is, and when it was written
    POST /context               write it again (body: {"market": true})
    POST /stop                  shut the server down

One process per run, started by `sim serve` and outliving any one client. That
is the point of a server here rather than a pipe: booting the environment reads
about six hundred megabytes and takes four seconds, and an agent takes a few
dozen actions a day for a hundred and eighty days. The boot is paid once.

It also means the run can be *watched*. While an agent works, `curl .../state`
in another terminal answers, and so does `GET /views/view_inventory` — the same
call the context files are written from.

Bound to 127.0.0.1 on a port the kernel picks, written to `<run>/sim.port`.
Nothing here authenticates anything: it is a local process holding a simulated
supermarket, and it should never be reachable from anywhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as environment  # noqa: E402
from env import ACTIONS, Refused, Session  # noqa: E402

# One request at a time. The environment is a pile of mutable Python objects
# with no locking of its own, and upstream is explicit that a session takes one
# call at a time; the server is threaded only so that a slow client cannot wedge
# the others out.
LOCK = threading.Lock()

# Upstream tools that change something and are not exposed as actions. The
# agent's notes are files it writes itself, one to the day, in the artifacts
# tree the store never touches — so its note tools would be a second,
# invisible way to keep memory.
WRITERS = ("add_note", "view_notes")


class Store:
    """What the server holds: one session, and where its run lives."""

    def __init__(self, session: Session, run: Path):
        self.session = session
        self.run = run
        self.log = run / "actions.jsonl"
        self.checkpoint = run / "checkpoint"


class Handler(BaseHTTPRequestHandler):
    store: Store  # set on the class before the server starts

    # ── routing ───────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802  (http.server's spelling)
        route = urlparse(self.path)
        parts = [p for p in route.path.split("/") if p]
        query = parse_qs(route.query)
        session = self.store.session

        try:
            if not parts:
                return self.json(self.index())
            if parts == ["state"]:
                with LOCK:
                    return self.json(session.state())
            if parts == ["metrics"]:
                with LOCK:
                    return self.json(session.metrics())
            if parts == ["actions"]:
                with LOCK:
                    tools = session.env.get_tools()
                # `name`, `description`, `input_schema` — upstream also carries a
                # `parameters` copy of the schema, and printing both makes the
                # page twice as long and no clearer.
                return self.json(
                    {
                        "actions": [
                            {
                                "name": name,
                                "description": tools[name].get("description", ""),
                                "input_schema": tools[name].get("input_schema") or tools[name].get("parameters"),
                            }
                            for name in ACTIONS
                            if name in tools
                        ]
                    }
                )
            if parts == ["views"]:
                with LOCK:
                    names = [n for n in session.env.get_tools() if n not in ACTIONS and n not in WRITERS]
                    # `view_returns` is dispatched by `exec_tools` and missing
                    # from the tool declarations upstream ships. It is a view all
                    # the same, and the returns it lists are how a bad supplier
                    # is caught, so it is named here rather than lost.
                    names.append("view_returns")
                return self.json(
                    {
                        "views": sorted(names),
                        "note": "read-only. The agent does not call these; the context files are written from them.",
                        "example": f"/views/view_inventory  ·  /views/view_sku_prices?sku_ids={next(iter(session.env.skus_id_map), '')}",
                    }
                )
            if len(parts) == 2 and parts[0] == "views":
                return self.view(parts[1], query)
            if parts == ["context"]:
                return self.json(self.context_state())
            if parts == ["checkpoint"]:
                directory = self.store.checkpoint
                saved = directory / "session.json"
                return self.json(
                    {
                        "dir": str(directory),
                        "saved": json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else None,
                    }
                )
            return self.fail(404, f"no such path: /{'/'.join(parts)}")
        except Refused as refusal:
            return self.fail(409, str(refusal))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return self.fail(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        parts = [p for p in route.path.split("/") if p]

        try:
            body = self.body()
            if len(parts) == 2 and parts[0] == "actions":
                return self.act(parts[1], body)
            if parts == ["context"]:
                with LOCK:
                    written = self.store.session.write_context(market=bool(body.get("market", True)))
                return self.json(written)
            if parts == ["checkpoint"]:
                with LOCK:
                    return self.json(self.store.session.save(self.store.checkpoint))
            if parts == ["stop"]:
                self.json({"ok": True, "stopping": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return None
            return self.fail(404, f"no such path: /{'/'.join(parts)}")
        except Refused as refusal:
            return self.fail(409, str(refusal))
        except ValueError as exc:  # a body that is not JSON
            return self.fail(400, str(exc))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return self.fail(500, f"{type(exc).__name__}: {exc}")

    # ── the four that do something ────────────────────────────────

    def index(self) -> dict[str, Any]:
        with LOCK:
            state = self.store.session.state()
        return {
            "what": "RetailBench, one store, as a REST server",
            "state": state,
            "paths": {
                "GET /state": "day, date, funds, net worth, is it over",
                "GET /metrics": "run_days, final_networth, total_sales, and the ratios",
                "GET /actions": "the three tools the agent may call, with schemas",
                "POST /actions/place_order": "buy — 200 taken, 409 refused with a reason",
                "POST /actions/modify_sku_price": "set a shelf price",
                "POST /actions/end_today": "settle the day and move the clock",
                "GET /views": "the fifteen read-only tools",
                "GET /views/<name>": "what one of them says now (text/plain)",
                "GET /context": "where the agent's tree is, and when it was written",
                "POST /context": "write it again",
                "GET /checkpoint": "when the store was last saved, and where",
                "POST /checkpoint": "save it now (it saves itself after every day)",
                "POST /stop": "shut down",
            },
        }

    def act(self, name: str, arguments: dict[str, Any]) -> None:
        """One action, applied or refused — and recorded either way.

        A refusal is a `409`: the request was well-formed and the store said no.
        That distinction matters to the caller, which retries a 409 with
        different arguments and gives up on a 400.
        """
        with LOCK:
            session = self.store.session
            # Taken before the call, because `end_today` moves the clock: after
            # it returns, `session.day` is tomorrow and `days_completed` is the
            # day that just closed. Either way this is the day the action was
            # taken *on*.
            day = session.day
            try:
                result = session.act(name, arguments)
            except Refused as refusal:
                self.record({"action": name, "arguments": arguments, "ok": False, "day": day, "refused": str(refusal)})
                raise
            self.record({"action": name, "arguments": arguments, "ok": True, "day": day})
            # After the day turned, and only then: mid-day there is a half-made
            # day to come back to, and nothing here can express that. The save
            # is a directory swap, so the previous one stands until this one is
            # whole.
            if name == "end_today":
                try:
                    session.save(self.store.checkpoint)
                except Exception as exc:
                    print(f"[warn] could not checkpoint after day {day}: {exc}", file=sys.stderr)
        return self.json(result)

    def view(self, name: str, query: dict[str, list[str]]) -> None:
        """One read-only tool. Query strings become its arguments.

        `?sku_ids=a&sku_ids=b` and `?sku_ids=a,b` both arrive as a list, because
        a browser address bar writes the second and a script writes the first.
        """
        arguments: dict[str, Any] = {}
        for key, values in query.items():
            if key == "format":
                continue
            if key.endswith("_ids") or key == "ratings":
                flat: list[Any] = []
                for value in values:
                    flat += [v for v in value.split(",") if v]
                arguments[key] = [int(v) if key == "ratings" else v for v in flat]
            elif len(values) == 1:
                arguments[key] = values[0]
            else:
                arguments[key] = values

        with LOCK:
            session = self.store.session
            # The tools that take a SKU list want one, and a browser is not
            # going to type ninety-six ids to see the inventory.
            if "sku_ids" not in arguments and name in (
                "view_sku_prices",
                "view_sku_sales_history",
                "view_sku_reviews",
                "view_sku_avg_ratings",
                "view_return_rates",
                "view_current_date_supplier_prices",
            ):
                arguments["sku_ids"] = [str(s) for s in session.env.skus_id_map]
            result = session.view(name, arguments)

        if query.get("format", ["text"])[0] == "json":
            return self.json(result)
        return self.text(result["formatted"])

    def context_state(self) -> dict[str, Any]:
        session = self.store.session
        directory = session.context_dir
        files = sum(1 for _ in directory.rglob("*") if _.is_file()) if directory and directory.exists() else 0
        return {
            "dir": str(directory) if directory else None,
            "written": session.context_written,
            "files": files,
        }

    # ── plumbing ──────────────────────────────────────────────────

    def body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"the body is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("the body must be a JSON object")
        return parsed

    def record(self, entry: dict[str, Any]) -> None:
        """Every action and every refusal, in the order they happened.

        The run's audit trail: what was asked for, what the store said, on which
        day. A refused action is in here too — the refusals are half of what a
        run is worth looking at afterwards.
        """
        from datetime import datetime

        entry = {"at": datetime.now().isoformat(timespec="seconds"), **entry}
        with self.store.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def json(self, payload: Any, status: int = 200) -> None:
        self.send(status, "application/json", json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")

    def text(self, body: str, status: int = 200) -> None:
        self.send(status, "text/plain; charset=utf-8", body.rstrip() + "\n")

    def fail(self, status: int, message: str) -> None:
        self.json({"ok": False, "error": message}, status=status)

    def send(self, status: int, content_type: str, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args: Any) -> None:
        """One line per request, on stderr, with the status.

        `http.server` logs to stderr already; this only trims it to something a
        run's log can carry beside the simulator's own lines.
        """
        sys.stderr.write(f"[sim] {self.address_string()} {fmt % args}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="the dataset root: a mirror of RetailBench's own data/")
    parser.add_argument("--run", required=True, help="where this run's records, logs and port file go")
    parser.add_argument("--context", help="where to write the tree the agent reads")
    parser.add_argument("--config", default="dynamic_hard",
                        choices=("dynamic_hard", "dynamic_middle", "still_hard", "still_middle"))
    parser.add_argument("--days", type=int, default=180, help="the horizon: how many days the run may last")
    parser.add_argument("--port", type=int, default=0, help="0 lets the kernel pick, which is the default")
    parser.add_argument("--resume", metavar="DIR",
                        help="a checkpoint to start from instead of day one (default: <run>/checkpoint)")
    args = parser.parse_args()

    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    data = Path(args.data).resolve()

    # Everything this process prints goes to the log from here on, and the one
    # line the starter reads goes to the real stdout through `hello`. The
    # simulator prints as it builds — a line per SKU whose initial rating it
    # worked out — and without this that chatter is what `sim serve` tries to
    # parse as the port.
    hello = sys.stdout
    sys.stdout = sys.stderr
    environment.quiet()
    config = environment.build_config(args.config, data, run)
    absent = environment.missing_data(config)
    if absent:
        print(json.dumps({"ok": False, "error": "dataset files are missing; run `sim fetch`", "missing": absent}),
              file=hello, flush=True)
        return 1

    context_dir = Path(args.context).resolve() if args.context else None
    if args.resume:
        resuming = Path(args.resume).resolve()
        if resuming.is_dir() and not (resuming / "session.json").exists():
            resuming = resuming / "checkpoint"
        if not (resuming / "session.json").exists():
            print(json.dumps({"ok": False, "error": f"no checkpoint in {resuming}"}), file=hello, flush=True)
            return 1
        session = Session.restore(
            resuming,
            max_days=args.days,
            context_dir=context_dir,
        )
        print(f"[sim] resumed from {resuming} at day {session.day}", file=sys.stderr)
    else:
        session = Session(
            config,
            max_days=args.days,
            context_dir=context_dir,
        )
    if session.context_dir:
        session.write_context()

    Handler.store = Store(session, run)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    port = server.server_address[1]
    (run / "sim.port").write_text(f"{port}\n", encoding="utf-8")
    (run / "sim.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    # The one line on stdout, so that whoever started this can read the port
    # without watching for a file to appear.
    print(json.dumps({"ok": True, "url": f"http://127.0.0.1:{port}", "state": session.state()}, default=str),
          file=hello, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for leftover in (run / "sim.port", run / "sim.pid"):
            leftover.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    raise SystemExit(main())
