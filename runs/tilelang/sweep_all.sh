#!/bin/bash
# T3 sweep driver (runs on the Spark). Per shape: 8 compile workers in parallel, then one timing pass.
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
PY=$HOME/.venv/bin/python
run() {  # name M N K
  echo "== $1 compile $(date)"
  for i in 0 1 2 3 4 5 6 7; do
    $PY spark_sweep.py --kernel gemm_basic_holes.py --m $2 --n $3 --k $4 --configs configs_$1.json --out compile_$1_$i.jsonl --phase compile --slice $i/8 > compile_$1_$i.log 2>&1 &
  done
  wait
  echo "== $1 time $(date)"
  $PY spark_sweep.py --kernel gemm_basic_holes.py --m $2 --n $3 --k $4 --configs configs_$1.json --out sweep_$1.jsonl --phase time > time_$1.log 2>&1
  echo "== $1 done $(date)"
}
run 1024 1024 1024 1024
run m32 32 4096 4096
run 4096 4096 4096 4096
