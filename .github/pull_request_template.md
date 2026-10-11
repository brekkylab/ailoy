<!-- Describe the finished state, not the rounds of work that got here. Closes #<issue>.

Title: one sentence saying what Ailoy does once this is merged, with no issue number. Problem, Cause, Fix and Result are for a change that fixes something; drop the ones that do not apply. A change with no problem behind it, such as a new document or a refactor, has `## Changes` in place of Fix. Keep Verification and the three checkboxes. -->

## Problem

<!-- What was wrong or missing, as a user of the crate or a binding meets it. -->

## Cause

<!-- Where in the code it comes from, and why. Where a comment in the diff carries that reason, say so here rather than repeating it. -->

## Fix

<!-- What Ailoy does now, and the shape of the change: the crate, then each binding it reaches. -->

## Result

<!-- What a user sees after this, including anything that moved in a binding, the README or an example. -->

## Verification

<!-- Fill in the lines below rather than summarizing them. Tests that need a provider key skip without it, so name the keys that were set: a green run with none set covers less than the suite. AGENTS.md lists the commands. -->

- **Rust** — `cargo clippy --workspace && cargo test`, with keys:
- **Python** — `uv run pytest` in `bindings/python`:
- **Node** — `npm test` in `bindings/node`:

- [ ] A test fails without this change
- [ ] A behaviour the crate gained is reachable from both bindings, or this PR says which binding it is for
- [ ] Comments state what the code does now, not the history of the change
