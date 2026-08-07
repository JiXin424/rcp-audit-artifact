#!/bin/bash
# Download RCP audit checkpoints from ModelScope.
# Usage: bash scripts/download_checkpoints.sh [target_dir]
#   target_dir: where to put the checkpoints/ directory (default: .)
#
# Requires: git (with LFS)
# ModelScope repo: https://www.modelscope.cn/J1Xin424/rcp-audit-checkpoints

set -euo pipefail

TARGET="${1:-.}"
REPO_URL="https://www.modelscope.cn/J1Xin424/rcp-audit-checkpoints.git"
DEST="$TARGET/checkpoints"

if [ -d "$DEST/.git" ]; then
    echo "checkpoints/ already exists with a git repo; skipping clone."
    cd "$DEST" && git pull origin main || true
else
    echo "Cloning checkpoints from ModelScope..."
    if [ -d "$DEST" ]; then
        echo "Removing existing checkpoints/ (no .git)..."
        rm -rf "$DEST"
    fi
    git clone --depth=1 "$REPO_URL" "$DEST"
    echo "Downloaded checkpoints to $DEST"
fi

echo "Done. Checkpoints are at $DEST"
echo "Run 'make core-audit' to verify."
