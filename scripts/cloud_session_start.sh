#!/bin/bash
# What a cloud session needs before Claude starts working, run by the SessionStart hook in
# .claude/settings.json. A local session exits at once: the developer's own environment is theirs.
#
# The toolchain comes from rust-toolchain.toml on the first `cargo` call, and the tests fetch
# virtx's console server themselves, so this only warms the three dependency sets, each of which
# would otherwise be paid for while the model waits.
set -eu
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
cd "$CLAUDE_PROJECT_DIR"
cargo fetch --quiet
(cd bindings/python && uv sync --quiet)
(cd bindings/node && npm ci --silent)
