#!/usr/bin/env python3
"""Summarise the `fs_timing` lines the engine writes for every store call.

    # 1. drive the app: open the sources, walk the tree, open a few files
    npm run tauri:dev
    # 2. read what it cost
    python3 scripts/fs-timings.py

Each line the engine writes is one call into a store — `list`, `stat`, or `read` — with how
long it took and how much it was for. This groups them by operation and by the mount the
path is under, which is the split a caching decision is made on: a source whose listings
cost seconds wants a listing cache, one whose reads cost seconds wants the bytes.

Only the *window's* calls are here. The agent reads through the FUSE mount, which does not
pass through the code that writes these lines.
"""

import glob
import os
import re
import sys
from collections import defaultdict

DEFAULT_LOGS = os.path.expanduser(
    "~/Library/Application Support/com.brekkylab.ailoy/logs/ailoy.log*"
)

# `op="list" path="/notion" ms=1843 n=12`, with or without the quotes tracing's fmt layer
# puts on a string field depending on how it was recorded.
FIELD = re.compile(r'(\w+)=(?:"([^"]*)"|([^\s]+))')


def rows(paths):
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "fs_timing" not in line:
                    continue
                # `findall` gives "" for the alternative that did not match, not None.
                fields = {k: (q or b) for k, q, b in FIELD.findall(line)}
                if "op" not in fields or "ms" not in fields:
                    continue
                try:
                    fields["ms"] = int(fields["ms"])
                except ValueError:
                    continue
                yield fields


def source_of(path):
    """The mount a path is under — the app mounts a source at a top-level directory."""
    parts = [p for p in path.split("/") if p]
    return "/" + parts[0] if parts else "/"


def pct(values, p):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))]


def human_ms(ms):
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms}ms"


def main(argv):
    top = 10
    if "--top" in argv:
        i = argv.index("--top")
        top = int(argv[i + 1])
        del argv[i : i + 2]
    paths = argv[1:] or sorted(glob.glob(DEFAULT_LOGS))
    if not paths:
        sys.exit(f"no log files: pass them as arguments, or run the app first ({DEFAULT_LOGS})")

    calls = list(rows(paths))
    if not calls:
        sys.exit(
            "no fs_timing lines in "
            + ", ".join(paths)
            + "\nthe app writes them as it reads; drive the file tree first"
        )

    groups = defaultdict(list)
    for c in calls:
        groups[(c["op"], source_of(c.get("path", "")))].append(c)

    print(f"{len(calls)} calls in {', '.join(os.path.basename(p) for p in paths)}\n")
    head = f"{'op':6} {'source':14} {'calls':>6} {'p50':>8} {'p90':>8} {'max':>8} {'total':>8} {'errors':>7}"
    print(head)
    print("-" * len(head))
    for (op, source), group in sorted(groups.items(), key=lambda kv: -sum(c["ms"] for c in kv[1])):
        times = [c["ms"] for c in group]
        errors = sum(1 for c in group if "err" in c)
        print(
            f"{op:6} {source:14} {len(group):>6} {human_ms(pct(times, 50)):>8} "
            f"{human_ms(pct(times, 90)):>8} {human_ms(max(times)):>8} "
            f"{human_ms(sum(times)):>8} {errors:>7}"
        )

    print(f"\nslowest {top}:")
    for c in sorted(calls, key=lambda c: -c["ms"])[:top]:
        size = f" n={c['n']}" if "n" in c else ""
        err = f" err={c['err']}" if "err" in c else ""
        print(f"  {human_ms(c['ms']):>8}  {c['op']:5} {c.get('path', '?')}{size}{err}")


if __name__ == "__main__":
    main(sys.argv)
