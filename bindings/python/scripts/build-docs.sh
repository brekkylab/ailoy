#!/bin/bash

# Optional argument: output directory for the generated site
# Usage: ./build-docs.sh [output_dir]
# If not specified, mkdocs will use the default location (./site)

SITE_DIR="${1:-}"
MKDOCS_BUILD_ARGS=""

if [ -n "$SITE_DIR" ]; then
  MKDOCS_BUILD_ARGS="$MKDOCS_BUILD_ARGS --site-dir $SITE_DIR"
fi

uv pip compile --group docs --quiet -o /tmp/requirements-docs.txt
uvx --with-requirements /tmp/requirements-docs.txt mkdocs build $MKDOCS_BUILD_ARGS
rm -f /tmp/requirements-docs.txt
