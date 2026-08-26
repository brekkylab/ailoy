"""Whether the `headhunting` commands written in the documentation actually run.

**An example must not teach a command that does not run.** Reading it by hand will not
catch that: a column name that is gone or an option that was removed still looks right,
and only the person who followed it fails. That happened in this project — the
README screen taught a query using `naive_years` and `city`, neither of which is in
`candidate_tenure`.

It is caught only by running it. So this script pulls the commands out of the
documentation and hands them to **the same `Executable` the agent reaches**
(`src/bin/headhunting.rs`).

    python3 eval/check_transcript.py
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent
DB = HERE / "data" / "headhunter.db"

# What to check, in documents people read and copy from.
DOCS = ["README.md"]

# A command can start anywhere on a line: in a transcribed screen block it is preceded by
# `$ `, and in prose it is wrapped in backticks.
COMMAND = re.compile(r"(?:^|[`$]\s*)(headhunting\s+[^`\n]+)", re.M)

# An example carrying a placeholder cannot be run. It explains syntax rather than showing
# a run, so trying to run it would produce a false failure.
PLACEHOLDER = re.compile(r"[…<>]|\.\.\.")


def binary() -> Path:
    """The by-hand binary. If it is absent, say how to build it and stop."""
    for target in (HERE.parent.parent / "target" / "debug" / "headhunting",):
        if target.is_file():
            return target
    sys.exit(
        "no headhunting binary.\n"
        "  cargo build -p headhunter --bin headhunting"
    )


# A bare subcommand name is not a run example but a mention — a table cell or a phrase in
# prose. Run with no arguments it fails with "no conditions at all", which does not mean
# the documentation is wrong.
SUBCOMMANDS = {"search", "read", "query", "distribution"}


def commands(text: str) -> list[str]:
    """The runnable commands pulled from a document."""
    import shlex

    out = []
    for raw in COMMAND.findall(text):
        cmd = raw.strip().rstrip("`").strip()
        if PLACEHOLDER.search(cmd):
            continue
        try:
            parts = shlex.split(cmd)
        except ValueError:
            # Unbalanced quotes: a fragment cut out of the document, not something to check.
            continue
        if len(parts) < 2:
            continue
        if len(parts) == 2 and parts[1] in SUBCOMMANDS:
            continue
        if cmd not in out:
            out.append(cmd)
    return out


def main() -> None:
    if not DB.is_file():
        sys.exit(f"no pool at {DB}. Build it with `python3 sql/load.py`")
    exe = binary()

    import shlex

    failures = 0
    checked = 0
    for name in DOCS:
        path = HERE / name
        if not path.is_file():
            continue
        found = commands(path.read_text())
        if not found:
            # Finding nothing means the screens went stale or a command was renamed and
            # the document never followed. Neither may pass quietly.
            failures += 1
            checked += 1
            print(f"{name}: **not one runnable command.** The screens are stale")
            continue
        print(f"\n{name} — {len(found)}")
        for cmd in found:
            checked += 1
            # `headhunting` is stripped and only the arguments are passed. No shell is
            # involved because letting one reinterpret the document's quotes would run a
            # different command from the one written.
            args = shlex.split(cmd)[1:]
            result = subprocess.run(
                [str(exe), *args],
                capture_output=True,
                text=True,
                cwd=HERE,
                env={"HEADHUNTER_DB": str(DB), "PATH": "/usr/bin:/bin"},
            )
            if result.returncode == 0:
                rows = result.stdout.count("\n")
                print(f"  OK   {cmd}   ({rows} lines)")
            else:
                failures += 1
                why = (result.stderr or "(no reason given)").strip().splitlines()[0]
                print(f"  **   {cmd}\n       {why}")

    print(f"\n{failures} of {checked} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
