#!/usr/bin/env bash
# Build cortex-local-console from the sibling checkout and place it where Tauri expects a
# sidecar: src-tauri/binaries/<name>-<target-triple>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CORTEX="${CORTEX_DIR:-$HERE/../../../cortex}"
PROFILE="${1:-release}"
# The default is a sibling of this repo, which is only true of a checkout laid out the way
# the README describes; say so here rather than letting `cd` fail with a bare path.
if [ ! -d "$CORTEX" ]; then
  echo "cortex checkout not found at $CORTEX; set CORTEX_DIR" >&2
  exit 2
fi
TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
if [ "$PROFILE" = release ]; then FLAG=--release; else FLAG=; fi
( cd "$CORTEX" && cargo build -p cortex-local-console $FLAG )
mkdir -p "$HERE/src-tauri/binaries"
cp "$CORTEX/target/$PROFILE/cortex-local-console" "$HERE/src-tauri/binaries/cortex-local-console-$TRIPLE"
echo "sidecar: $HERE/src-tauri/binaries/cortex-local-console-$TRIPLE"
