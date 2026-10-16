#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/scripts/env.sh"
exec "$python_bin" -m examples.epfl_synthesis.prepare --splits train
