#!/usr/bin/env bash
set -euo pipefail

branch=codex/doit-posterior-geometry
repo_root=$(git rev-parse --show-toplevel)
storage=/home/mlw0719/doit_geometry_storage
remote_repo=/home/mlw0719/doit_geometry
tmp_bundle=$(mktemp "${TMPDIR:-/tmp}/doit-geometry.XXXXXX.bundle")
trap 'rm -f "$tmp_bundle"' EXIT
git -C "$repo_root" bundle create "$tmp_bundle" "$branch"
ssh orion "mkdir -p '$storage'"
scp "$tmp_bundle" "orion:$storage/doit-geometry.bundle"
ssh orion "bash -s -- '$remote_repo' '$storage/doit-geometry.bundle' '$branch'" <<'REMOTE'
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
