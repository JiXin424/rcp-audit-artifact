#!/bin/bash
# Download RCP audit checkpoints (own-trained families) from ModelScope and
# fetch the RELEASED SLRTP2025 evaluator from its official source with
# SHA-256 verification.
#
# Usage: bash scripts/download_checkpoints.sh [target_dir]
#   target_dir: where to put the checkpoints/ directory (default: .)
#
# Requires: git (with LFS), curl, unzip, and gdown (pip install gdown).
#   ModelScope repo: https://www.modelscope.cn/J1Xin424/rcp-audit-checkpoints
#
# Licensing note: the ModelScope mirror contains ONLY checkpoints the authors
# trained themselves. The released evaluator is NOT redistributed there; it
# is fetched from the official challenge source below and verified against
# the SHA-256 the paper audits.

set -euo pipefail

TARGET="${1:-.}"
REPO_URL="https://www.modelscope.cn/J1Xin424/rcp-audit-checkpoints.git"
DEST="$TARGET/checkpoints"

RELEASED_SHA256="f081cfd2cca93a3610ed3a13d34bf9b3cb36db5390fda83181db84abab1aa428"
OFFICIAL_BUNDLE_URL="https://drive.google.com/file/d/1fjKHigsEWHwsMHnwwWdFYZ8dECXslTKi/view"
OFFICIAL_REPO="https://github.com/walsharry/SLRTP-Sign-Production-Evaluation"

# ---- 1. Own-trained checkpoints from ModelScope -----------------------------
if [ -d "$DEST/.git" ]; then
    echo "checkpoints/ already exists with a git repo; skipping clone."
    cd "$DEST" && git pull origin main || true
else
    echo "Cloning own-trained checkpoints from ModelScope..."
    if [ -d "$DEST" ]; then
        echo "Removing existing checkpoints/ (no .git)..."
        rm -rf "$DEST"
    fi
    git clone --depth=1 "$REPO_URL" "$DEST"
    echo "Downloaded own-trained checkpoints to $DEST"
fi

# ---- 2. Released evaluator from the official source ------------------------
if [ -f "$DEST/released/backTranslation_PHIX_model/best.ckpt" ]; then
    echo "Released evaluator already present; verifying checksum..."
else
    echo
    echo "The released SLRTP2025 evaluator is not redistributed by the authors"
    echo "(no redistribution licence identified). Fetching it from the official"
    echo "challenge source:"
    echo "  $OFFICIAL_REPO"
    mkdir -p "$DEST/released"
    TMP=$(mktemp -d)
    if command -v gdown >/dev/null 2>&1; then
        gdown "$OFFICIAL_BUNDLE_URL" -O "$TMP/bundle.zip" || {
            echo "gdown failed. Download the bundle manually from:"
            echo "  $OFFICIAL_BUNDLE_URL"
            echo "extract backTranslation_PHIX_model/ into $DEST/released/, then re-run."
            exit 1
        }
        unzip -o "$TMP/bundle.zip" -d "$TMP/extracted"
        SRC=$(find "$TMP/extracted" -type d -name "backTranslation_PHIX_model" | head -n1)
        [ -n "$SRC" ] || { echo "backTranslation_PHIX_model not found in bundle"; exit 1; }
        mv "$SRC" "$DEST/released/backTranslation_PHIX_model"
        rm -rf "$TMP"
    else
        echo "'gdown' not installed. Download the bundle manually from:"
        echo "  $OFFICIAL_BUNDLE_URL"
        echo "extract backTranslation_PHIX_model/ into $DEST/released/, then re-run."
        exit 1
    fi
fi

ACTUAL=$(sha256sum "$DEST/released/backTranslation_PHIX_model/best.ckpt" | cut -d' ' -f1)
if [ "$ACTUAL" != "$RELEASED_SHA256" ]; then
    echo "ERROR: released checkpoint SHA-256 mismatch."
    echo "  expected: $RELEASED_SHA256"
    echo "  actual:   $ACTUAL"
    exit 1
fi
echo "Released evaluator verified (SHA-256 $RELEASED_SHA256)."

echo "Done. Checkpoints are at $DEST"
echo "Run 'make core-audit' to verify."
