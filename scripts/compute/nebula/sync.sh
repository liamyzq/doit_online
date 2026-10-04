#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <branch>" >&2
  exit 2
fi
branch=$1
repo_root=$(git rev-parse --show-toplevel)
storage=/home/mlw0719/doit_reliability_storage
remote_repo=/home/mlw0719/doit_reliability
tmp_bundle=$(mktemp "${TMPDIR:-/tmp}/doit-reliability.XXXXXX.bundle")
trap 'rm -f "$tmp_bundle"' EXIT
git -C "$repo_root" show-ref --verify --quiet "refs/heads/$branch"
git -C "$repo_root" bundle create "$tmp_bundle" "$branch"
ssh nebula "mkdir -p '$storage'"
scp "$tmp_bundle" "nebula:$storage/doit-reliability.bundle"
ssh nebula "bash -s -- '$remote_repo' '$storage/doit-reliability.bundle' '$branch'" <<'REMOTE'
set -euo pipefail
repo=$1
bundle=$2
branch=$3
mkdir -p "$repo"
if [[ ! -d "$repo/.git" ]]; then
  git -C "$repo" init
fi
if ! git -C "$repo" diff --quiet || ! git -C "$repo" diff --cached --quiet; then
  echo "Refusing sync: tracked remote changes are present." >&2
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
