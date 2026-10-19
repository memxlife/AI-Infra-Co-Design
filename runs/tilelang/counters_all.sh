#!/bin/bash
# waits for the sweep and the baselines, then the counter experiments (runs on the Spark)
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
while pgrep -f "sweep_all.sh|baselines_all.sh" > /dev/null; do sleep 30; done
echo "== counters $(date)"; $HOME/.venv/bin/python spark_counters.py --kernel gemm_basic_holes.py --plan counters_plan.json --out counters.json; echo "== done $(date)"
