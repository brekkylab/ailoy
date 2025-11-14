#!/usr/bin/env bash

KIND=${1:-dylib}   # Target type: cdylib or dylib
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FILE="$SCRIPT_DIR/../Cargo.toml"
TMP="$FILE.tmp"

awk -v kind="$KIND" '
{
  sub(/\r$/, "")  # Remove CR if the file has Windows line endings
}

# Enter [lib] section
/^\[lib\][[:space:]]*(#.*)?$/ {
  inlib = 1
  print
  next
}

# Leave [lib] section when another section starts
/^\[.*\][[:space:]]*(#.*)?$/ && $0 !~ /^\[lib\]/ {
  inlib = 0
  print
  next
}

# Replace crate-type line inside [lib] section
inlib && /^[[:space:]]*crate-type[[:space:]]*=/ {
  printf "crate-type = [\"%s\"]\n", kind
  next
}

# Print all other lines as-is
{ print }
' "$FILE" > "$TMP" && mv "$TMP" "$FILE"
