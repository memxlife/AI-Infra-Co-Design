#!/usr/bin/env bash
set -euo pipefail
arm=${1:?Usage: bash run.sh A|B|C|D [Hydra arguments]}
case "$arm" in A|B|C|D) ;; *) echo 'Choose A, B, C or D.' >&2; exit 2 ;; esac
shift
source "$(dirname "$0")/scripts/env.sh"
exec "$python_bin" -m scaling_evolve.algorithms.eve.runner \
  --config-name="epfl_$arm" "$@"
