#!/bin/bash
# TileLang's autotune list on the five new shapes (runs on the Spark), for the comparison with the loop's one-shot picks
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
PY=$HOME/.venv/bin/python
for s in "2048 2048 2048 2048" "m16 16 4096 4096" "m128 128 4096 4096" "k1024 4096 4096 1024" "8192 8192 8192 8192"; do set -- $s
  echo "== new $1 $(date)"; $PY spark_baseline.py --kernel gemm_basic_holes.py --m $2 --n $3 --k $4 --only cublas,autotune --out baseline_new_$1.json > baseline_new_$1.log 2>&1; echo "== done $1 $(date)"
done
