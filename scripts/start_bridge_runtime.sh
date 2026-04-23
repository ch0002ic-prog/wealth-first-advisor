#!/bin/sh
set -eu

runtime_root="${WEALTH_FIRST_RUNTIME_ROOT:-/var/data}"
artifact_root="${WEALTH_FIRST_ARTIFACT_ROOT_PATH:-$runtime_root/artifacts}"
seed_artifact_root="${WEALTH_FIRST_SEED_ARTIFACT_ROOT:-/opt/wealth-first/seed-artifacts}"

mkdir -p "$runtime_root" "$artifact_root"

if [ -d "$seed_artifact_root" ]; then
  find "$seed_artifact_root" -maxdepth 1 -type f | while IFS= read -r seed_file; do
    target_file="$artifact_root/$(basename "$seed_file")"
    if [ ! -e "$target_file" ]; then
      cp "$seed_file" "$target_file"
    fi
  done
fi

exec python -m wealth_first.tradingview_bridge serve-env