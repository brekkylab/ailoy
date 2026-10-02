#!/usr/bin/env python3
"""Play OpenTTD: look at the world, build, buy vehicles and let time pass.

    python3 ttd.py COMMAND [ARGS]    (python3 ttd.py -h lists the commands)

Each command goes to the game's bridge script over the admin port. The game stands paused
between commands, and runs only while one is carried out or while `wait` lets time pass.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from admin import SHOT_ADDRESSES, Admin, AdminError, connect  # noqa: E402

# `--json`: print results as they come, for a program to read.
AS_JSON = False


def tile(text):
    try:
        x, y = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"a tile is X,Y, not {text!r}")
    return [x, y]


def place(text):
    """`X,Y`, `town:ID`, `industry:ID` or `station:ID`."""
    if ":" in text:
        kind, _, id_ = text.partition(":")
        if kind not in ("town", "industry", "station") or not id_.isdigit():
            raise argparse.ArgumentTypeError(f"a place is X,Y, town:ID, industry:ID or station:ID, not {text!r}")
        return {kind: int(id_)}
    return {"tile": tile(text)}


def stop(text):
    """`STATION` or `STATION:flag+flag`, such as `12:full` or `7:unload+noload`."""
    station, _, flags = text.partition(":")
    if not station.isdigit():
        raise argparse.ArgumentTypeError(f"an order is STATION[:flag+flag], not {text!r}")
    return {"station": int(station), "flags": [f for f in flags.split("+") if f]}


def ids(text):
    return [int(v) for v in text.split(",")]


def amount(text):
    return text if text in ("max", "min") else int(text)


def call(admin, cmd, args=None, timeout=900):
    """Run `cmd` in the bridge with the game running, and pause it again."""
    admin.rcon("unpause")
    try:
        answer = admin.gamescript(
            {"id": random.randrange(1 << 30), "cmd": cmd, "args": args or {}}, timeout=timeout
        )
    finally:
        admin.rcon("pause")
    if not answer.get("ok"):
        raise AdminError(answer.get("error", "the bridge failed"))
    return answer.get("result", {})


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def show(result):
    """A result a line to each field, then each item of its list a line to each."""
    if AS_JSON:
        print(json.dumps(result, ensure_ascii=False))
        return
    items = result.pop("items", None)
    for key, value in result.items():
        print(f"{key}: {compact(value)}")
    for item in items or []:
        print(compact(item))


def show_map(result):
    if AS_JSON:
        show(result)
        return
    x0, y0 = result["top_left"]
    rows = result["items"]
    width = len(rows[0]) if rows else 0
    xs = [x0 + i for i in range(width)]
    pad = " " * 5
    print(pad + "".join(str(x // 100 % 10) if x >= 100 else " " for x in xs))
    print(pad + "".join(str(x // 10 % 10) for x in xs))
    print(pad + "".join(str(x % 10) for x in xs))
    for i, row in enumerate(rows):
        print(f"{y0 + i:>4} {row}")
    print(f"\nx runs left to right, y top to bottom. {result['legend']}")
    for key in ("towns", "industries", "our_stations"):
        if result[key]:
            print(f"{key.replace('_', ' ')}: " + "; ".join(f"{o['id']} {o['name']} at {o['tile'][0]},{o['tile'][1]}" for o in result[key]))


def screenshot(name):
    """Have the game's console take a screenshot, and return where it put it."""
    with connect(SHOT_ADDRESSES, 10) as conn:
        conn.settimeout(60)
        conn.sendall(f"{name}\n".encode())
        answer = conn.makefile("rb").readline().decode().strip()
    status, _, rest = answer.partition(" ")
    if status != "ok":
        raise AdminError(f"the screenshot failed: {rest or 'no answer'}")
    return rest


def main():
    p = argparse.ArgumentParser(prog="ttd.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="print results as JSON")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    sub.add_parser("status", help="the date, money, loan and what the company has")
    sub.add_parser("report", help="status, and what needs looking at: idle, losing or overfull")
    sub.add_parser("cargos", help="the cargos, and what each pays")
    s = sub.add_parser("towns", help="towns, the largest first or the nearest to --near")
    s.add_argument("--near", type=place)
    s.add_argument("--limit", type=int, default=15)
    s = sub.add_parser("industries", help="industries and what they make and take")
    s.add_argument("--cargo", help="only those producing this cargo (or accepting it, with --accepting)")
    s.add_argument("--accepting", action="store_true")
    s.add_argument("--near", type=place)
    s.add_argument("--limit", type=int, default=15)
    s = sub.add_parser("map", help="the tiles around a place, one character each")
    s.add_argument("at", type=place)
    s.add_argument("--radius", type=int, default=12, help="up to 25")
    s = sub.add_parser("tile", help="what is on a tile, and what a stop there would serve")
    s.add_argument("tile", type=tile)
    s = sub.add_parser("look", help="point the view at a place and take a screenshot")
    s.add_argument("at", type=place)

    s = sub.add_parser("station", help="build a bus or truck stop: at --tile, or the best place --near")
    s.add_argument("--cargo")
    s.add_argument("--near", type=place)
    s.add_argument("--mode", choices=["pickup", "dropoff", "both"], default="pickup")
    s.add_argument("--kind", choices=["bus", "truck"])
    s.add_argument("--search", type=int, default=8, help="how far from --near to look")
    s.add_argument("--tile", type=tile)
    s.add_argument("--front", type=tile, help="with --tile: the side the road comes in from")
    s = sub.add_parser("road", help="find and build a road between two places")
    s.add_argument("frm", type=place, metavar="FROM")
    s.add_argument("to", type=place)
    s = sub.add_parser("depot", help="build a road depot beside a road near a place")
    s.add_argument("near", type=place)
    sub.add_parser("airports", help="the airport types, and which can be built now")
    s = sub.add_parser("airport", help="build an airport: at --tile, or the best place --near")
    s.add_argument("--near", type=place)
    s.add_argument("--tile", type=tile)
    s.add_argument("--type", default="small")
    s.add_argument("--search", type=int, default=12)
    s.add_argument("--no-level", action="store_true", help="do not level land for it")
    s = sub.add_parser("demolish", help="clear a tile")
    s.add_argument("tile", type=tile)

    s = sub.add_parser("engines", help="the vehicles on sale")
    s.add_argument("--type", choices=["road", "air"], default="road")
    s.add_argument("--cargo")
    s = sub.add_parser("buy", help="buy vehicles at a depot or hangar, give them orders and start them")
    s.add_argument("--depot", type=tile, required=True)
    s.add_argument("--engine", type=int, required=True)
    s.add_argument("--cargo", help="refit to this cargo")
    s.add_argument("--count", type=int, default=1)
    s.add_argument("--orders", type=stop, nargs="+", metavar="STOP", help="STATION[:flag+flag] …")
    s.add_argument("--share-with", type=int, metavar="VEHICLE")
    s.add_argument("--no-start", action="store_true")
    s = sub.add_parser("orders", help="replace vehicles' orders")
    s.add_argument("vehicles", type=ids, help="ID[,ID…]")
    s.add_argument("orders", type=stop, nargs="*", metavar="STOP")
    s.add_argument("--share-with", type=int, metavar="VEHICLE")
    s = sub.add_parser("vehicles", help="the company's vehicles and how each is doing")
    s.add_argument("--station", type=int)
    s.add_argument("--ids", type=ids)
    s = sub.add_parser("vehicle", help="start, stop, send to depot, sell or clone vehicles")
    s.add_argument("action", choices=["start", "stop", "depot", "sell", "clone"])
    s.add_argument("ids", type=ids, help="ID[,ID…]")
    s.add_argument("--count", type=int, default=1, help="with clone: how many copies each")
    s.add_argument("--depot", type=tile, help="with clone: where to build them")
    sub.add_parser("stations", help="the company's stations, what waits at each and its ratings")
    s = sub.add_parser("loan", help="set the loan: an amount, max or min")
    s.add_argument("amount", type=amount)
    s = sub.add_parser("wait", help="let the game run for some days, then report")
    s.add_argument("days", type=int)
    s = sub.add_parser("save", help="save the game in the saves folder")
    s.add_argument("name")

    a = p.parse_args()
    global AS_JSON
    AS_JSON = a.json
    with Admin() as admin:
        c = a.cmd
        if c in ("status", "report", "cargos", "airports", "stations"):
            show(call(admin, c))
        elif c == "towns":
            show(call(admin, c, {"near": a.near, "limit": a.limit}))
        elif c == "industries":
            show(call(admin, c, {"cargo": a.cargo, "accepting": a.accepting, "near": a.near, "limit": a.limit}))
        elif c == "map":
            show_map(call(admin, c, {"at": a.at, "radius": a.radius}))
        elif c == "tile":
            show(call(admin, c, {"tile": a.tile}))
        elif c == "look":
            r = call(admin, c, a.at)
            time.sleep(1.5)
            name = f"{r['tile'][0]}_{r['tile'][1]}_{int(time.time())}"
            print(f"screenshot of {r['tile'][0]},{r['tile'][1]}: {screenshot(name)}")
        elif c == "station":
            if a.tile is None and (a.near is None or a.cargo is None):
                p.error("station needs --tile, or --near and --cargo")
            show(call(admin, "build_station", {
                "tile": a.tile, "front": a.front, "near": a.near, "cargo": a.cargo,
                "mode": a.mode, "kind": a.kind, "search": a.search,
            }))
        elif c == "road":
            show(call(admin, "build_road", {"from": a.frm, "to": a.to}))
        elif c == "depot":
            show(call(admin, "build_depot", {"near": a.near}))
        elif c == "airport":
            if a.tile is None and a.near is None:
                p.error("airport needs --tile or --near")
            show(call(admin, "build_airport", {
                "tile": a.tile, "near": a.near, "type": a.type, "search": a.search, "level": not a.no_level,
            }))
        elif c == "demolish":
            show(call(admin, c, {"tile": a.tile}))
        elif c == "engines":
            show(call(admin, c, {"type": a.type, "cargo": a.cargo}))
        elif c == "buy":
            show(call(admin, c, {
                "depot": a.depot, "engine": a.engine, "cargo": a.cargo, "count": a.count,
                "orders": a.orders, "share_with": a.share_with, "start": not a.no_start,
            }))
        elif c == "orders":
            if not a.orders and a.share_with is None:
                p.error("orders needs stops or --share-with")
            show(call(admin, c, {"vehicles": a.vehicles, "orders": a.orders, "share_with": a.share_with}))
        elif c == "vehicles":
            show(call(admin, c, {"station": a.station, "ids": a.ids}))
        elif c == "vehicle":
            show(call(admin, c, {"action": a.action, "ids": a.ids, "count": a.count, "depot": a.depot}))
        elif c == "loan":
            show(call(admin, c, {"amount": a.amount}))
        elif c == "wait":
            if not 1 <= a.days <= 366:
                p.error("wait from 1 to 366 days at a time")
            start = admin.poll_date()
            admin.rcon("unpause")
            try:
                # A day is 74 ticks of 27 ms: two seconds.
                admin.wait_date(start + a.days, timeout=a.days * 4 + 30)
            finally:
                admin.rcon("pause")
            show(call(admin, "report"))
        elif c == "save":
            print(admin.rcon(f"save {a.name}"))


if __name__ == "__main__":
    try:
        main()
    except AdminError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
