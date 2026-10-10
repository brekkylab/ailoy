<!-- Describe the finished state: what Ailoy does now, and why that is the right answer. Not the
     rounds of work that got here. Closes #<issue>.

     Title: one sentence saying what Ailoy does once this is merged, with no issue number and no
     `fix:`-style prefix. Keep the three headings and the three checkboxes below. -->

## What changed

## Why this is right

<!-- The reason the new behaviour is the correct one: the vendor API it matches, the spec it
     follows, the bug it closes. Where a comment in the diff carries that reason, say so here
     rather than repeating it. -->

## Verification

<!-- Fill in the lines below rather than summarizing them. Tests that need a provider key skip
     without it, so name the keys that were set: a green run with none set covers less than the
     suite. AGENTS.md lists the commands. -->

- **Rust** — `cargo clippy --workspace && cargo test`, with keys:
- **Python** — `uv run pytest` in `bindings/python`:
- **Node** — `npm test` in `bindings/node`:

- [ ] A test fails without this change
- [ ] A behaviour the crate gained is reachable from both bindings, or this PR says which binding it is for and why
- [ ] Comments state what the code does now, not the history of the change
