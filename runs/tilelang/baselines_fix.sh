#!/bin/bash
# after the sweep, the baselines and the counters: the carver baseline (fixed mapping) and the cuBLAS second opinion, per shape
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
while pgrep -f "sweep_all.sh|baselines_all.sh|counters_all.sh" > /dev/null; do sleep 30; done
PY=$HOME/.venv/bin/python
for s in "1024 1024 1024 1024" "m32 32 4096 4096" "4096 4096 4096 4096"; do set -- $s
  echo "== fix $1 $(date)"; $PY spark_baseline.py --kernel gemm_basic_holes.py --m $2 --n $3 --k $4 --only cublas,carver --out baseline_fix_$1.json > baseline_fix_$1.log 2>&1; echo "== done $1 $(date)"
done
