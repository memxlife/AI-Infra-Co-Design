#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$EVE_EVAL_LOG_ROOT"
python -m examples.epfl_synthesis.evaluate \
  --recipe "$EVE_SOLVER_ROOT/recipe.json" \
  --data "$EVE_SOLVER_ROOT/train" --output "$EVE_EVAL_LOG_ROOT" || {
  if [[ ! -f "$EVE_EVAL_LOG_ROOT/score.yaml" ]]; then
    printf 'score: 0.0\nsummary: evaluator crashed\n' > "$EVE_EVAL_LOG_ROOT/score.yaml"
  fi
}
