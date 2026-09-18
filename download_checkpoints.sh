#!/usr/bin/env bash
# Download the two cSVR checkpoints from the Hugging Face Hub into
# $CSVR_CHECKPOINT_DIR (default: <repo>/model_checkpoints). Resumable:
# interrupted downloads keep a .part file and continue where they left off on
# the next run. Needs only curl and sha256sum.
#
# Each release has its own Hub repo: mafirenze/csvr_v2 is this one; the previous
# release's checkpoints stay in mafirenze/csvr_v1. Files already present are
# checked against the sha256 values below, not just by name.
#
# While the Hub repo is private, export HF_TOKEN=hf_... (a read token from
# https://huggingface.co/settings/tokens) before running.
set -euo pipefail

HF_REPO="${CSVR_HF_REPO:-mafirenze/csvr_v2}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${CSVR_CHECKPOINT_DIR:-$REPO_DIR/model_checkpoints}"
mkdir -p "$DEST"

# <final name> <size in bytes> <sha256>
FILES=(
  "cSVR_SVR.ckpt 7925758389 bdc2ce515c2eaf648f1ec3618ade9bc0240c35f32007340e835b0dc33d2ab6be"
  "cSVR_MLP.ckpt 5714560176 eb130bd5aa23cfac6a692c96a00ef9b5d9b21202cf91010032c91713a000a5f8"
)

AUTH=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH=(-H "Authorization: Bearer $HF_TOKEN")
fi

download() {
  local name="$1" size="$2" sha="$3"
  local out="$DEST/$name" part="$DEST/$name.part"
  local url="https://huggingface.co/${HF_REPO}/resolve/main/${name}"

  if [ -e "$out" ]; then
    # Verify rather than trust the name: every release ships the same filenames (and
    # both MLP releases have the same size), so a file left by an earlier release
    # would otherwise be kept silently.
    echo "[sha ] $name already exists, verifying ..."
    if echo "$sha *$out" | sha256sum -c --quiet - >/dev/null 2>&1; then
      echo "[skip] $name is up to date"
      return 0
    fi
    echo "ERROR: $out is not this release's $name (sha256 differs): probably a" >&2
    echo "       checkpoint from an earlier release or a custom one. Move it away" >&2
    echo "       and re-run to download the current file." >&2
    return 1
  fi

  if [ ! -e "$part" ] || [ "$(stat -c %s "$part")" -lt "$size" ]; then
    echo "[get ] $name ($(numfmt --to=iec "$size")) <- $url"
    # ${AUTH[@]+...} instead of a bare "${AUTH[@]}": empty-array expansion
    # trips `set -u` on bash < 4.4 (e.g. macOS /bin/bash).
    curl -L --fail --retry 5 --retry-delay 5 -C - ${AUTH[@]+"${AUTH[@]}"} -o "$part" "$url" || {
      echo "ERROR: $name: download failed. A 401/404 usually means the repo is" >&2
      echo "       private (export HF_TOKEN=hf_... and re-run) or CSVR_HF_REPO" >&2
      echo "       ('$HF_REPO') is wrong. Re-run to resume a partial download." >&2
      return 1
    }
  fi

  local got
  got="$(stat -c %s "$part")"
  if [ "$got" -ne "$size" ]; then
    echo "ERROR: $name: size mismatch (got $got, expected $size). Re-run to resume." >&2
    return 1
  fi

  echo "[sha ] verifying $name ..."
  if ! echo "$sha *$part" | sha256sum -c --quiet -; then
    rm -f "$part"
    echo "ERROR: $name: sha256 mismatch - corrupted download removed, re-run to retry." >&2
    return 1
  fi
  mv "$part" "$out"
  echo "[done] $name"
}

for entry in "${FILES[@]}"; do
  # shellcheck disable=SC2086  # fields are space-free by construction
  download $entry
done

echo "All checkpoints present in $DEST"
echo "(the MONAIfbs masking checkpoint is fetched automatically from Zenodo on first --preprocess run)"
