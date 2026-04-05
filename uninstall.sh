#!/usr/bin/env bash
# Remove the mlx-claude symlink and (if present) the bootstrapped lib dir.

set -euo pipefail

BIN_DIR="${MLX_CLAUDE_BIN_DIR:-$HOME/.local/bin}"
LIB_DIR="${MLX_CLAUDE_LIB_DIR:-$HOME/.local/share/mlx-claude}"
LINK="$BIN_DIR/mlx-claude"

removed_something=0

if [[ -L "$LINK" || -e "$LINK" ]]; then
  rm -f "$LINK"
  echo "Removed: $LINK"
  removed_something=1
fi

# Only remove lib dir if it contains our script (guard against user-placed files)
if [[ -f "$LIB_DIR/mlx-claude.py" ]]; then
  rm -rf "$LIB_DIR"
  echo "Removed: $LIB_DIR"
  removed_something=1
fi

if (( removed_something == 0 )); then
  echo "Nothing to remove."
fi
