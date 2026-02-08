#!/usr/bin/env bash
# Load .env into the current shell, skipping comments and invalid lines.
# Usage: source scripts/load_env.sh   OR   . scripts/load_env.sh
# Or from repo root: source planner-matter-inference/scripts/load_env.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/../.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "No .env at $ENV_FILE" >&2
  return 2 2>/dev/null || exit 2
fi
while IFS= read -r line; do
  # Skip empty lines and lines that start with #
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  # Strip inline comment and trim
  line="${line%%#*}"
  line="${line%"${line##*[![:space:]]}"}"
  # Export only lines that look like VAR=value
  if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    export "$line"
  fi
done < "$ENV_FILE"
