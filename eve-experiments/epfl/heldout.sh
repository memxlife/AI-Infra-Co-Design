#!/usr/bin/env bash
set -euo pipefail
run_root=$(cd "${1:?Usage: bash heldout.sh /path/to/completed/run}" && pwd)
source "$(dirname "$0")/scripts/env.sh"
"$python_bin" "$task_dir/scripts/select_best.py" "$run_root"
"$python_bin" -m examples.epfl_synthesis.prepare --splits validation test
for split in validation test; do
  output="$run_root/heldout/$split"
  if [[ -f "$output/results.json" ]]; then
    printf 'Existing %s result: %s/results.json\n' "$split" "$output"
    continue
  fi
  "$python_bin" -m examples.epfl_synthesis.evaluate \
    --recipe "$run_root/best_recipe.json" \
    --data "$eve_dir/.runs/epfl-synthesis-data/$split" \
    --output "$output" || {
      # A scientifically invalid recipe has exit status 1 and a saved score of 0.
      [[ -f "$output/results.json" ]] || exit 1
    }
done
