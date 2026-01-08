#!/bin/bash

# Optional argument: output directory for the generated site
# Usage: ./build-docs.sh [output_dir]
# If not specified, mkdocs will use the default location (./site)

SITE_DIR="${1:-}"

# Convert SITE_DIR to absolute path if provided
if [ -n "$SITE_DIR" ]; then
  # Handle case where path is relative
  if [[ "$SITE_DIR" != /* ]]; then
    # Get current directory before any cd operations
    CURRENT_DIR="$(pwd)"
    SITE_DIR="$CURRENT_DIR/$SITE_DIR"
  fi
fi

MKDOCS_BUILD_ARGS=""

if [ -n "$SITE_DIR" ]; then
  MKDOCS_BUILD_ARGS="$MKDOCS_BUILD_ARGS --site-dir $SITE_DIR"
fi

# Change cwd to the base of python binding
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BINDING_DIR="$(cd "$SCRIPT_DIR/../" && pwd)"
pushd "$PYTHON_BINDING_DIR" > /dev/null

uv pip compile --group docs --quiet -o /tmp/requirements-docs.txt
uvx --with-requirements /tmp/requirements-docs.txt mkdocs build $MKDOCS_BUILD_ARGS
rm -f /tmp/requirements-docs.txt

# Restore cwd
popd > /dev/null
