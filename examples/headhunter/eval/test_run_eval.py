"""Tests for the scorer itself.

    python3 -m pytest eval/test_run_eval.py

The pool has to be built first: `python3 sql/load.py`.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_eval import score  # noqa: E402

DB = HERE.parent / "data" / "headhunter.db"


@pytest.fixture
def con():
    if not DB.is_file():
        pytest.skip(f"no pool at {DB} — build it with `python3 sql/load.py`")
    c = sqlite3.connect(DB)
    yield c
    c.close()


def _run_dir(tmp_path: Path, slug: str) -> Path:
    """A run directory carrying the one artifact every posting must produce."""
    run = tmp_path / slug
    run.mkdir()
    (run / "00-shortlist.md").write_text(
        "## Picks\n"
        "1. urn:li:person:b3tq9wmk — picked\n\n"
        "<!-- rejected -->\n"
        "- nobody\n"
    )
    return run


def test_prose_beside_the_artifacts_is_not_scored_as_a_mail(tmp_path, con):
    """`SCENARIO.md` sits in the run directory and is written for a person to read.

    The mails are `NN-<slug>.md`. Scoring every `*.md` reads the scenario as a cold mail,
    and since it is English prose about a `ko` candidate it fails the language check —
    a document that is doing exactly what it should makes the run look broken.
    """
    run = _run_dir(tmp_path, "backend-rust")
    (run / "SCENARIO.md").write_text(
        "This posting tests whether a headline is read as evidence.\n"
        "urn:li:person:b3tq9wmk is the control.\n"
    )

    auto, _ = score(con, run)

    assert not any("SCENARIO.md" in message for message in auto), auto


def test_a_cold_mail_is_still_scored(tmp_path, con):
    """The narrowing must not swallow the mails themselves."""
    run = _run_dir(tmp_path, "backend-rust")
    # b3tq9wmk is a `ko` profile, so an English mail is a failure.
    (run / "01-someone.md").write_text(
        "urn:li:person:b3tq9wmk\n\nHello, I am writing about a role.\n"
    )

    auto, _ = score(con, run)

    assert any("01-someone.md" in message for message in auto), auto
