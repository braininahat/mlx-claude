#!/usr/bin/env bash
# One-line installer for mlx-claude.
#
#   curl -sSfL https://raw.githubusercontent.com/braininahat/mlx-claude/master/bootstrap.sh | bash
#
# Env overrides:
#   MLX_CLAUDE_REF       git ref to fetch (default: master)
#   MLX_CLAUDE_BIN_DIR   symlink dir      (default: $HOME/.local/bin)
#   MLX_CLAUDE_LIB_DIR   script dir       (default: $HOME/.local/share/mlx-claude)
#
# Re-running = update to the latest ref.

set -euo pipefail

REF="${MLX_CLAUDE_REF:-master}"
BIN_DIR="${MLX_CLAUDE_BIN_DIR:-$HOME/.local/bin}"
LIB_DIR="${MLX_CLAUDE_LIB_DIR:-$HOME/.local/share/mlx-claude}"
REPO_BASE="https://raw.githubusercontent.com/braininahat/mlx-claude/${REF}"
LIB_PATH="$LIB_DIR/mlx-claude.py"
SANITIZER_PATH="$LIB_DIR/sse_sanitizer.py"
LINK="$BIN_DIR/mlx-claude"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is required. Install with:" >&2
  echo "       curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "WARNING: 'claude' CLI not on PATH. mlx-claude needs it to launch." >&2
  echo "         Setup: https://docs.claude.com/en/docs/claude-code/setup" >&2
fi

mkdir -p "$LIB_DIR" "$BIN_DIR"
curl -fsSL "$REPO_BASE/mlx-claude.py" -o "$LIB_PATH"
curl -fsSL "$REPO_BASE/sse_sanitizer.py" -o "$SANITIZER_PATH"
chmod +x "$LIB_PATH" "$SANITIZER_PATH"
ln -sfn "$LIB_PATH" "$LINK"

echo "Installed: $LINK -> $LIB_PATH (ref=$REF)"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo
    echo "NOTE: $BIN_DIR is not on your PATH."
    echo "Add to your shell rc (~/.zshrc, ~/.bashrc):"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac

echo
echo "Run: mlx-claude"
