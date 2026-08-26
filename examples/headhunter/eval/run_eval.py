"""Scores the agent's artifacts. spec §6.2.

**Fully automatic scoring is not the goal.** It prints the failures that can be caught
automatically and **a list of things for a person to check**. Whether a personalized
sentence reads naturally is not something to automate.

    run_eval.py --check          whether each posting's qualified count and expected
                                 ranking match the data
    run_eval.py --score <dir>    score the artifacts. <dir> is `artifacts/<role-slug>`

Use `--check` **before** running the agent. If the scoring criteria have drifted from the
data, scoring means nothing.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
ID = re.compile(r"urn:li:person:[0-9a-z]{8}")
# What the run produced, which is not everything in the directory that ends in `.md`.
# `SCENARIO.md` sits beside the artifacts and is prose for a person; scored as a mail
# it fails the language check, so a run that is doing everything right looks broken.
ARTIFACT = re.compile(r"^\d\d-.+\.md$")

sys.path.insert(0, str(HERE))
from check_expected import main as check_expected  # noqa: E402
from check_must_haves import main as check_must_haves  # noqa: E402


def check() -> int:
    """Whether the scoring criteria match the data. Runs both checks in turn.

    They are separate files because each is useful on its own: `check_must_haves` when a
    posting changes, `check_expected` when the answer key does.
    """
    print("what the must-have conditions yield:")
    a = check_must_haves()
    print("\nwho the expected ranking points at:")
    b = check_expected()
    return a or b


# The marker the instruction asks for. This takes precedence.
# The Korean alternatives are matched, not written: a shortlist for a Korean posting is
# written in Korean, and the heading has to be found either way.
PICKS_HEADING = re.compile(r"^(#{1,6})\s*(?:picks?|선정|고른)\b.*$", re.M | re.I)


def _picks_list(text: str) -> str:
    """The body of the `## Picks` section, or an empty string if there is none.

    The instruction asks for this section because **citing another person inside the
    selection area is legitimate**. In a real run the agent wrote a second id while
    explaining pick 3 as a same-name check, and counting by area read it as a fourth pick.

    The section runs to the next heading.
    """
    m = PICKS_HEADING.search(text)
    if not m:
        return ""
    # **Only a heading at the same level or above ends it.** A `###` subsection per
    # candidate is natural, and ending at any heading would cut before the first candidate
    # and leave the list empty.
    level = len(m.group(1))
    rest = text[m.end():]

    # **The `<!-- rejected -->` marker ends it too.** The instruction asks for two things
    # together: open the selections as a `## Picks` list, and put the marker immediately
    # before the rejections. It does not ask for the rejections under their own heading, so
    # a list after the marker staying inside the `## Picks` section is what following the
    # instruction produces. Reading headings alone counts those rejections as picks — in a
    # real run a shortlist of 3 was counted as 12 and failed automatically.
    ends = [m.start() for m in (
        re.search(rf"^#{{1,{level}}}\s", rest, re.M),
        REJECTED_MARKER.search(rest),
    ) if m]
    return rest[: min(ends)] if ends else rest


REJECTED_MARKER = re.compile(r"^<!--\s*rejected\s*-->\s*$", re.M | re.I)

# The fallback when the marker is absent. **Nothing depends on this.**
REJECTION_HEADING = re.compile(
    r"^#{1,6}\s*.*(버린|뺀|뺐|제외|않은|탈락|보내지|rejected?|excluded?"
    r"|not\s+contact|why\s+not|걸러)",
    re.M | re.I,
)


def _selection_part(text: str) -> tuple[str, bool]:
    """Only the **selection** area of a shortlist. Returns `(body, marker_was_found)`.

    Leaving the rejection area in makes every trap named there count as "in the top k".
    An artifact that wrote its rejections faithfully would then score worse — a scorer that
    penalizes exactly what the instruction asked for.

    **Trying to match the heading wording with a regex fails.** Korean inflects
    (뺀/뺐/빼는) and English has unlimited phrasings. The first attempt matched `빼` and
    missed `뺀`, so it could not cut the area even on a shortlist written in this
    repository.

    So the instruction asks for one `<!-- rejected -->` line and this function reads it:
    setting the format rather than guessing it. Without the marker it falls back to the
    heading heuristic, but says so to the caller, who reports it as something for a person
    to check — guessing quietly would leave a wrong score looking right.
    """
    m = REJECTED_MARKER.search(text)
    if m:
        return text[: m.start()], True
    m = REJECTION_HEADING.search(text)
    return (text[: m.start()] if m else text), False


def score(con: sqlite3.Connection, artifacts: Path) -> tuple[list[str], list[str]]:
    """Scores the artifacts, returning automatic failures and human-check items apart."""
    auto: list[str] = []
    human: list[str] = []

    truth = {t["id"]: t for t in json.loads((DATA / "ground_truth.json").read_text())}
    in_db = {r[0] for r in con.execute("SELECT id FROM candidates")}
    lang = dict(con.execute("SELECT id, profile_language FROM candidates"))
    name_of = dict(con.execute("SELECT id, name FROM candidate_brief"))

    jd = artifacts.name
    expected_path = HERE / "expected" / f"{jd}.json"
    if not expected_path.exists():
        auto.append(f"{jd}: no eval/expected/{jd}.json — there are no criteria to score by")
        return auto, human
    exp = json.loads(expected_path.read_text())

    shortlist = artifacts / "00-shortlist.md"
    if not shortlist.exists():
        auto.append(f"no {artifacts}/00-shortlist.md")
        return auto, human

    files = sorted(p for p in artifacts.glob("*.md") if ARTIFACT.match(p.name))
    text_of = {p.name: p.read_text() for p in files}

    # ── do the cited ids exist ────────────────────────────────────────
    for name, text in text_of.items():
        for cited in sorted(set(ID.findall(text))):
            if cited not in in_db:
                auto.append(f"{name}: id does not exist: {cited}")

    # **Only the selection area is read.** A shortlist names the people picked and the
    # people rejected together — the instruction requires it ("Name the people you rejected
    # and why"). Searching the whole file would read a trap named among the rejections as
    # "in the top k", so **the better the rejections are written, the worse the score.**
    # That is the scorer fighting the instruction.
    #
    # So the rejection area is cut away. Whatever the wording, nothing under a heading
    # carrying rejected/excluded/not — or their Korean equivalents — is a selection.
    body, marked = _selection_part(text_of.get("00-shortlist.md", ""))
    picks = _picks_list(text_of.get("00-shortlist.md", ""))
    if picks:
        # If there is a `## Picks` list, that is authoritative. Citing another person
        # inside the selection area has legitimate reasons (a same-name check, a
        # comparison), so the area alone cannot decide it. A real run produced a false
        # positive that way: the agent picked 3 and it was counted as 4.
        body = picks
    if not marked:
        human.append(
            "the shortlist has no `<!-- rejected -->` marker — selection and rejection "
            "were guessed from heading wording. Someone rejected may have been counted as "
            "'in the top k', so a person should check the automatic failures below")
    cited = set(ID.findall(body))
    # Name matching is the **fallback path**. The instruction requires the full urn
    # (§Ranking rules), and where it is present the regex above catches everything.
    #
    # Why names cannot be relied on is measured: **283 of 600 (47%) share a name**, and
    # 124 names are not unique (`Rowan Thorne` six times, `Kai Lockhart` four). Of the 28
    # planted traps and controls, 12 have a non-unique name.
    #
    # So on an artifact that writes names only, three checks die quietly:
    # `must_not_appear` never fires, a `duplicate-profile` pair goes unrecognized because
    # both are called `Kai Lockhart`, and all four controls are always reported absent.
    by_name: dict[str, list[str]] = {}
    for i, n in name_of.items():
        by_name.setdefault(n, []).append(i)
    named_ambiguous = []
    for n, ids in by_name.items():
        if len(n) >= 3 and n in body:
            if len(ids) == 1:
                cited.add(ids[0])
            else:
                named_ambiguous.append(n)
    # **A name appearing is not itself a problem.** Prose saying "Nova Vance …" is
    # natural, and it is unambiguous once one id among the people sharing that name has
    # been cited. What is flagged is **someone who appears by name with no id at all**.
    named_ambiguous = [n for n in named_ambiguous if not (set(by_name[n]) & cited)]
    if named_ambiguous:
        # **This is not called the `same-name` trap.** That planted trap is one person
        # (`서연 강`), one of 124 collisions. The rest are incidental reuse in the
        # background population.
        auto.append(
            f"the shortlist refers to {len(named_ambiguous)} people by name with no id "
            f"({', '.join(sorted(named_ambiguous)[:3])}…) — these names are shared by "
            f"several people, so who is meant is undetermined. The instruction requires "
            f"the full `urn:li:person:…`")

    # With no id at all, every check below is meaningless. It must not pass quietly.
    if not cited and not exp.get("expected_fewer_than_k"):
        auto.append(
            "the shortlist has no `urn:li:person:…` at all — must_not_appear, duplicate "
            "detection, and the control checks all become no-ops")

    # **No early return here.** For `blockchain-solidity` the right answer is to emit
    # nobody, and that artifact lands in exactly the `not cited` state. Returning here
    # Returning here would give the correct answer an automatic failure and print none of
    # that posting's four criteria lines — the scorer acting the exact opposite of
    # `expected/blockchain-solidity.json`.
    #
    # The `expected_fewer_than_k` branch below handles zero correctly, so the verdict is
    # left to it.
    if not cited and not exp.get("expected_fewer_than_k"):
        auto.append("00-shortlist.md identifies no candidate at all")
        return auto, human

    # ── people who must not appear ────────────────────────────────────
    for entry in exp.get("must_not_appear", []):
        if entry["id"] in cited:
            trap = entry.get("trap") or "clear-miss"
            auto.append(
                f"{name_of.get(entry['id'], entry['id'])} is in the top k "
                f"[{trap}] — {entry['why'][:70]}")

    # ── did the controls survive ──────────────────────────────────────
    #
    # This is what separates "the right answer" from "the right answer for the wrong
    # reason". A control differs from its trap on exactly one axis, on the passing side, so
    # discarding anything that looks odd throws the control out with the trap. But
    # **discarding one is not necessarily wrong** — on a posting that picks 3 from 56, a
    # control can legitimately be fourth. So this is a human check, not a failure.
    for entry in exp.get("controls_that_must_not_be_rejected", []):
        if entry["id"] not in cited:
            human.append(
                f"the control {name_of.get(entry['id'], entry['id'])} "
                f"(control for {entry['control_for']}) is not in the shortlist — read the "
                f"rejection notes to see whether it lost on rank or was discarded as "
                f"'suspicious'")

    # ── duplicate profile: one person, twice ──────────────────────────
    #
    # **`pair_with` cannot be used as is.** Three traps share the field with different
    # meanings:
    #
    #   duplicate-profile   two records, one person — both present is a **failure**
    #   same-name           one name, two people — both present is **fine**
    #   rank-inversion-pair two different people — both present is fine
    #
    # An earlier draft read `truth[i].get("duplicate_of")`, a field that does not exist.
    # Always None, so the check would do nothing forever — it fails open.
    for i in sorted(cited):
        t = truth.get(i)
        if not t or t.get("trap") != "duplicate-profile":
            continue
        other = t.get("pair_with")
        if other and other in cited:
            auto.append(
                f"the same person is in the top k twice: {i[-8:]} and {other[-8:]} "
                f"({name_of.get(i)}) — the duplicate-profile trap was missed")

    # ── did it emit fewer than k ──────────────────────────────────────
    k = exp["k"]
    if exp.get("expected_fewer_than_k"):
        expected_n = len(exp.get("acceptable_top_k", []))
        if len(cited) >= k:
            auto.append(
                f"a posting with fewer qualified than k({k}), yet {len(cited)} were "
                f"emitted — "
                f"{exp.get('why_fewer', '')}")
        elif not cited and expected_n == 0:
            # A posting with nobody qualified. Emitting no one is the right answer, so
            # this is not an automatic failure.
            human.append(
                "nobody was emitted — for this posting that is the right answer. Check "
                "that **what was searched for** is written down. If it cannot be told "
                "apart from declaring nobody without searching, it does not pass")
        else:
            human.append(
                f"{len(cited)} emitted (k={k}). Check that the shortlist says **why it "
                f"could not reach {k}**")
    elif len(cited) > k:
        auto.append(f"a top {k} was asked for and {len(cited)} were emitted")

    # ── the language of the cold mails ────────────────────────────────
    #
    # What can be checked automatically is only the script: a mail to a `ko` profile has
    # to carry Hangul, one to a `ja` profile has to carry kana. Awkward Korean is for a
    # person to read.
    #
    # **Kana, not Han.** Japanese and Korean and Chinese share Han characters, so a
    # `一龠` range would call a Korean mail with 漢字 Japanese. Kana appears in no other
    # language, and no Japanese prose of mail length avoids it.
    #
    # This check used to look for Hangul alone. The pool holds 38 `ja` profiles, all of
    # them in Tokyo, and a mail written to any of them in English passed in silence.
    SCRIPTS = {
        "ko": ("Hangul", re.compile(r"[가-힣]")),
        "ja": ("kana", re.compile(r"[ぁ-んァ-ヶ]")),
    }
    LETTERS = re.compile(r"[A-Za-z]")

    def shares(text):
        """What fraction of the text's letters each script accounts for.

        **Presence is not enough in either direction.** A mail to an `en` profile may
        legitimately carry a name or a skill spelled `러스트`, and a run wrote an
        all-Korean mail to an `en` profile that passed in silence because the old check
        only asked whether Hangul appeared at all. The mirror case is a `ko` mail that
        opens with one Korean word and continues in English.

        The share separates them cleanly. Measured across every mail this example has
        produced: `ko` mails run 74-89% Hangul, `en` mails 0-0.4%, and the one that broke
        the rule sits at 69%. Nothing falls in between.
        """
        counts = {s: len(p.findall(text)) for s, (_, p) in SCRIPTS.items()}
        total = sum(counts.values()) + len(LETTERS.findall(text))
        return {s: (n / total if total else 0.0) for s, n in counts.items()}

    # Both thresholds sit inside that empty band, not at its edges.
    WRITTEN_IN = 0.30
    NOT_WRITTEN_IN = 0.15
    for name, text in text_of.items():
        if name == "00-shortlist.md":
            continue
        ids = [i for i in ID.findall(text) if i in lang]
        if not ids:
            # A mail with a name and no id. Passed to the human checks.
            human.append(f"{name}: no candidate id, so the language cannot be checked automatically")
            continue
        want = lang[ids[0]]
        got = shares(text)
        if want in SCRIPTS:
            script, _ = SCRIPTS[want]
            if got[want] < NOT_WRITTEN_IN:
                auto.append(
                    f"{name}: the candidate's profile_language is {want} but the mail is "
                    f"not written in {script} ({got[want]:.1%} of its letters)")
        else:
            # An `en` profile. A few characters of another script are right when they are
            # a proper noun carried over from the data and wrong when they were invented,
            # which only a person can tell apart. A mail written *in* that language is not
            # a judgement call.
            for s, pct in sorted(got.items(), key=lambda kv: -kv[1]):
                script, _ = SCRIPTS[s]
                if pct >= WRITTEN_IN:
                    auto.append(
                        f"{name}: profile_language is {want} but the mail is written in "
                        f"{script} ({pct:.1%} of its letters)")
                elif pct > 0:
                    human.append(
                        f"{name}: profile_language is {want} but the mail carries "
                        f"{script} ({pct:.1%} of its letters) — right if it is a proper "
                        f"noun from the data (`러스트`, `서버 개발자`), wrong if invented")

    # ── per-trap human checks ─────────────────────────────────────────
    for entry in exp.get("traps_that_must_be_caught", []):
        i = entry.get("id")
        seen = " (in the shortlist)" if i in cited else ""
        human.append(f"[{entry['trap']}]{seen} {entry['expected']}")

    for note in exp.get("scoring_notes", []):
        human.append(f"[criterion] {note}")

    return auto, human


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="whether the criteria match the data (before running the agent)")
    ap.add_argument("--score", type=Path, metavar="DIR",
                    help="score the artifacts. artifacts/<role-slug>")
    args = ap.parse_args()

    if args.check:
        return check()
    if not args.score:
        ap.print_help()
        return 2

    con = sqlite3.connect(DATA / "headhunter.db")
    auto, human = score(con, args.score)

    print(f"scoring: {args.score}\n")
    if auto:
        print(f"{len(auto)} automatic failures")
        for a in auto:
            print(f"  X {a}")
    else:
        print("no automatic failures")

    print(f"\n{len(human)} things for a person to check")
    for h in human:
        print(f"  · {h}")

    return 1 if auto else 0


if __name__ == "__main__":
    sys.exit(main())
