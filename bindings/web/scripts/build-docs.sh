#!/bin/bash

# Optional argument: output directory for the generated site
# Usage: ./build-docs.sh [output_dir]
# If not specified, typedoc will use the default location (./docs)

OUT_DIR="${1:-}"

# Convert OUT_DIR to absolute path if provided
if [ -n "$OUT_DIR" ]; then
  # Handle case where path is relative
  if [[ "$OUT_DIR" != /* ]]; then
    # Get current directory before any cd operations
    CURRENT_DIR="$(pwd)"
    OUT_DIR="$CURRENT_DIR/$OUT_DIR"
  fi
fi

TYPEDOC_ARGS="./src/index.ts --tsconfig ./tsconfig.json --favicon ../../docs/static/img/favicon.ico --gitRevision main"

if [ -n "$OUT_DIR" ]; then
  TYPEDOC_ARGS="$TYPEDOC_ARGS --out $OUT_DIR"
fi

# Change cwd to the base of web binding
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_BINDING_DIR="$(cd "$SCRIPT_DIR/../" && pwd)"
pushd "$WEB_BINDING_DIR" > /dev/null

npx typedoc $TYPEDOC_ARGS

# Restore cwd
popd > /dev/null
