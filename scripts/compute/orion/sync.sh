#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <branch>" >&2
  exit 2
fi
branch=$1
repo_root=$(git rev-parse --show-toplevel)
storage=/home/mlw0719/doit_correction_storage
remote_repo=/home/mlw0719/doit_correction
# Sync only between finite queues so pending jobs never start with updated code.
tmp_bundle=$(mktemp "${TMPDIR:-/tmp}/doit-correction.XXXXXX.bundle")
trap 'rm -f "$tmp_bundle"' EXIT

git -C "$repo_root" show-ref --verify --quiet "refs/heads/$branch" || {
  echo "Local branch does not exist: $branch" >&2
  exit 1
}
git -C "$repo_root" bundle create "$tmp_bundle" "$branch"
ssh orion "mkdir -p '$storage'"
scp "$tmp_bundle" "orion:$storage/doit-correction.bundle"
ssh orion "bash -s -- '$remote_repo' '$storage/doit-correction.bundle' '$branch'" <<'REMOTE'
set -euo pipefail
repo=$1
bundle=$2
branch=$3
mkdir -p "$(dirname "$repo")"
if [[ ! -d "$repo/.git" ]]; then
  mkdir -p "$repo"
  git -C "$repo" init
fi
if ! git -C "$repo" diff --quiet || ! git -C "$repo" diff --cached --quiet; then
  echo "Refusing sync: tracked remote source/config/script changes are present." >&2
  exit 1
fi
git -C "$repo" fetch "$bundle" "refs/heads/$branch"
if git -C "$repo" show-ref --verify --quiet "refs/heads/$branch"; then
  git -C "$repo" checkout "$branch"
  git -C "$repo" merge --ff-only FETCH_HEAD
else
  git -C "$repo" checkout -b "$branch" FETCH_HEAD
fi
REMOTE
