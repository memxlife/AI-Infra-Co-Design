#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd "$(dirname "$0")" && pwd)
eve_dir=${EVE_CHECKOUT:-"$task_dir/.eve"}

if [[ ! -d "$eve_dir/.git" ]]; then
  git clone --depth 1 https://github.com/scaling-group/eve.git "$eve_dir"
fi
cp -R "$task_dir/configs/." "$eve_dir/configs/"
mkdir -p "$eve_dir/examples"
cp -R "$task_dir/examples/." "$eve_dir/examples/"
uv sync --project "$eve_dir" --python 3.12
printf 'Ready. Next: bash "%s/prepare.sh"\n' "$task_dir"
