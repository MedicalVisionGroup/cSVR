#!/usr/bin/env bash
# Maintainer script: upload the two cSVR checkpoints to the Hugging Face Hub
# repo that download_checkpoints.sh pulls from. End users never need this.
#
# Prereqs: hf CLI (pip install -U huggingface_hub) logged in with a WRITE
# token: `hf auth login` (tokens: https://huggingface.co/settings/tokens).
#
# Usage:
#   ./upload_checkpoints_hf.sh            # repo is created private
#   ./upload_checkpoints_hf.sh --public   # repo is created public
# The visibility flag only matters on first creation; flip an existing repo
# at https://huggingface.co/<repo>/settings. Set CSVR_HF_REPO to override the
# default repo id (must then also be changed in download_checkpoints.sh).
# Each release gets a new repo (csvr_v1, csvr_v2, ...) instead of overwriting
# files that older checkouts verify against.
set -euo pipefail

REPO="${CSVR_HF_REPO:-mafirenze/csvr_v2}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${CSVR_CHECKPOINT_DIR:-$REPO_DIR/model_checkpoints}"

command -v hf >/dev/null 2>&1 || {
  echo "ERROR: 'hf' CLI not found - pip install -U huggingface_hub" >&2; exit 1; }
hf auth whoami >/dev/null 2>&1 || {
  echo "ERROR: not logged in to Hugging Face - run: hf auth login" >&2; exit 1; }

VISIBILITY=--private
[ "${1:-}" = "--public" ] && VISIBILITY=--public

# Errors if the repo already exists - that is fine, uploads below still work.
hf repo create "$REPO" --type model "$VISIBILITY" || true

for name in cSVR_SVR.ckpt cSVR_MLP.ckpt; do
  f="$SRC/$name"
  [ -e "$f" ] || { echo "ERROR: $f not found" >&2; exit 1; }
  echo "[up  ] $name ($(numfmt --to=iec "$(stat -Lc %s "$f")")) -> $REPO"
  hf upload "$REPO" "$f" "$name" --commit-message "Upload $name"
done

echo
echo "Done: https://huggingface.co/$REPO"
echo "Check each file's sha256 on its Hub page against the values in"
echo "download_checkpoints.sh, and make the repo public before release."
