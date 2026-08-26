"""The prose layer: sentences written from the structure layer's facts alone.

**It runs once.** LLM output is not deterministic, so a rerun writes different sentences.
The result is committed, and from then on the prose is part of the data.

Because the prompt carries only facts, no amount of good writing can change them — a
background profile with no Rust history does not clear a Rust must-have however well it
reads.

The core and the background are written by different models. The core's constraints are
demanding ("say they are interested, do not say they have experience"), and mixing the
models also mixes the prose style, so style cannot be used to spot the core.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from common.writer import dump

DATA = Path(__file__).parent.parent.parent / "data"

CORE_MODEL = "claude-opus-5"
BACKGROUND_MODEL = "claude-haiku-4-5-20251001"

# narration.json is rewritten after each batch, so a crash keeps everything before it.
BATCH_SIZE = 20


def is_core(truth: dict) -> bool:
    """Whether this is one of the core 65. **`control_for` has to be included.**

    A control carries neither `trap` nor `verdict`, so testing `trap or verdict` sends all
    11 controls to the background model — and then the writing style becomes a second axis
    on which a control differs from its trap. The agent could then separate them by prose
    density rather than by judgment, catching the trap for the wrong reason.
    """
    return bool(truth["trap"] or truth.get("control_for") or truth.get("verdict"))


SYSTEM = """\
You write LinkedIn profile prose for a synthetic dataset. You are given facts about one
person — companies, titles, dates, skills — and you write two things: a `summary` for the
profile, and a `description` for each position.

Rules you must not break:

- Use only the facts given. Never invent a company, a date, a title, or a skill.
- Never state or imply a total years-of-experience figure. It is computed from the dates,
  and a figure in prose that disagrees with the dates is a defect.
- Do not mention that positions overlap in time, even when they do.
- Write in the profile's language: `en` or `ko` as given.
- No email addresses, no phone numbers, no URLs.

Answer as JSON: {"summary": "...", "descriptions": ["...", "..."]} with one description per
position, in the order given.
"""


def prompt_for(candidate: dict, truth: dict) -> str:
    """One person's facts as a prompt. `truth` is their `ground_truth.json` entry, which
    is where the per-trap constraint lives — `candidates.json` does not carry it.
    """
    lines = [
        f"language: {candidate['profile_language']}",
        f"name: {candidate['first_name']} {candidate['last_name']}",
        f"headline: {candidate['headline']}",
        f"seniority: {candidate['seniority']}",
        f"skills: {', '.join(s['name'] for s in candidate['skills'])}",
        "positions:",
    ]
    for position in candidate["positions"]:
        end = (
            f"{position['end_year']}-{position['end_month']:02d}"
            if position.get("end_year")
            else "present"
        )
        lines.append(
            f"  - {position['title']} at {position['company_name']} "
            f"({position['company_size']} employees, {position['employment_type']}), "
            f"{position['start_year']}-{position['start_month']:02d} to {end}"
        )
    if truth.get("narrate_hint"):
        lines.append(f"\nAdditional constraint: {truth['narrate_hint']}")
    return "\n".join(lines)


def _model_for(truth: dict) -> str:
    return CORE_MODEL if is_core(truth) else BACKGROUND_MODEL


def _unfence(text: str) -> str:
    """Strips a ```json fence, which models add whether asked to or not."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    if "\n" in text:
        text = text[text.index("\n") + 1 :]
    return text.rstrip().removesuffix("```")


def _request_one(model: str, candidate: dict, truth: dict) -> dict | None:
    """Calls the API for one person and validates the result, or returns None.

    `anthropic` is imported here so `--dry-run` works without the SDK installed — checking
    what would be sent is the one inspection available before spending a call.
    """
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt_for(candidate, truth)}],
        )
        parsed = json.loads(_unfence(response.content[0].text))
    except Exception as exc:  # network, JSON parsing. Counts as this person failing, no more.
        print(f"  failed {candidate['id']}: {exc}", file=sys.stderr)
        return None

    descriptions = parsed.get("descriptions")
    if not isinstance(descriptions, list) or len(descriptions) != len(candidate["positions"]):
        # Assembly pairs positions to descriptions by order; a length mismatch pairs wrongly.
        print(
            f"  failed {candidate['id']}: descriptions {len(descriptions) if isinstance(descriptions, list) else '?'}"
            f" != positions {len(candidate['positions'])}",
            file=sys.stderr,
        )
        return None

    return {"summary": parsed.get("summary", ""), "descriptions": descriptions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N (for trials)")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the prompts and call no API"
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="only the 65 for which is_core() holds",
    )
    parser.add_argument(
        "--ids", type=str, default=None, help="only these ids, comma-separated"
    )
    args = parser.parse_args()

    candidates = json.loads((DATA / "candidates.json").read_text(encoding="utf-8"))
    truths = {t["id"]: t for t in json.loads((DATA / "ground_truth.json").read_text(encoding="utf-8"))}

    # Checked over the whole dataset regardless of filters — a wrong is_core() shows up here.
    n_core = sum(1 for c in candidates if is_core(truths[c["id"]]))
    print(f"is_core() split: core={n_core} background={len(candidates) - n_core}", file=sys.stderr)

    narration_path = DATA / "narration.json"
    narration: dict = (
        json.loads(narration_path.read_text(encoding="utf-8")) if narration_path.exists() else {}
    )

    # --ids and --core-only narrow what this run covers.
    scope = candidates
    if args.ids:
        wanted = set(args.ids.split(","))
        scope = [c for c in scope if c["id"] in wanted]
    if args.core_only:
        scope = [c for c in scope if is_core(truths[c["id"]])]

    # Skipping ids that already have prose is what makes an interrupted run resumable.
    already = [c for c in scope if c["id"] in narration]
    pending = [c for c in scope if c["id"] not in narration]

    # --limit counts what this run will do, not people already finished.
    if args.limit is not None:
        pending = pending[: args.limit]

    if args.dry_run:
        # stdout carries JSON Lines only; anything for a person goes to stderr.
        for candidate in pending:
            truth = truths[candidate["id"]]
            record = {
                "id": candidate["id"],
                "tier": "core" if is_core(truth) else "background",
                "model": _model_for(truth),
                "n_positions": len(candidate["positions"]),
                "prompt": prompt_for(candidate, truth),
            }
            print(json.dumps(record, ensure_ascii=False))
        print(
            f"dry-run: {len(pending)} to write, no API calls ({len(already)} skipped)",
            file=sys.stderr,
        )
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        raise SystemExit(1)

    processed = 0
    failed = 0
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        try:
            for candidate in batch:
                truth = truths[candidate["id"]]
                result = _request_one(_model_for(truth), candidate, truth)
                if result is None:
                    failed += 1
                    continue
                narration[candidate["id"]] = result
                processed += 1
        except Exception as exc:
            # A whole-batch failure. This batch stops; the batches already saved are safe.
            print(f"batch {start}-{start + len(batch)} failed: {exc}", file=sys.stderr)
        # Saved after every batch, so a crash keeps what came before.
        dump(narration_path, narration)

    print(
        f"{processed} written, {len(already)} skipped, {failed} failed",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
