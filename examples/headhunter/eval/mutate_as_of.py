"""Mutation experiment for the AS_OF check — a real mistake changes one side only.

**Whether the mutation actually applied is checked first.** Otherwise the original is put
back and reported as "passed quietly" — which is how this went wrong once.
"""
import subprocess
import sys
from pathlib import Path
H = Path(__file__).resolve().parent.parent
V = H / "sql/views.sql"
PY_ = H / "datagen" / ".venv" / "bin" / "python"
orig = V.read_text()
SQL_LINE = "COALESCE(end_year*12 + COALESCE(end_month,1), 2026*12+8)"
assert SQL_LINE in orig, "the reference string is not in views.sql"

def run(label, mutated):
    if mutated == orig:
        print(f"  [{label}] **mutation failed — identical to the original**")
        return
    V.write_text(mutated)
    r = subprocess.run([str(PY_), "sql/load.py"], cwd=H, capture_output=True, text=True)
    V.write_text(orig)
    caught = r.returncode != 0
    print(f"  [{label}] {'caught' if caught else '**passed quietly**'}  exit={r.returncode}")
    if caught:
        print(f"      {(r.stdout + r.stderr).strip().splitlines()[-1][:95]}")

# Real mistake 1: the SQL changed and the comment left alone — the data diverges. Must be caught.
run("SQL only 2027 (comment stays 2026)", orig.replace(SQL_LINE, SQL_LINE.replace("2026*12+8", "2027*12+8")))
# Real mistake 2: the month off by one — every current role is a month out. Must be caught.
run("SQL only 2026*12+9", orig.replace(SQL_LINE, SQL_LINE.replace("2026*12+8", "2026*12+9")))
# Real mistake 3: the literal pre-computed — it reads the same but the check cannot find it.
# Must be caught.
run("SQL as 24320 (literal only in the comment)", orig.replace(SQL_LINE, SQL_LINE.replace("2026*12+8", "24320")))
# Changing only the comment does not affect the data. Passing is right.
run("comment only 2027 (SQL stays 2026)", orig.replace("-- `2026*12+8` has", "-- `2027*12+8` has"))

ok = V.read_text() == orig
print(f"\n  restored: {'OK' if ok else '**broken — restore views.sql from git**'}")
sys.exit(0 if ok else 1)
