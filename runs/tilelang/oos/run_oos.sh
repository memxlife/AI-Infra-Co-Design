#!/bin/bash
# R5 on the Spark: waits for the step-3 picks, then the out-of-sample shapes (4 picks + 4 x 30 samples)
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
while pgrep -f "run_picks.sh|spark_probe.py" > /dev/null; do sleep 20; done
rm -f oos.jsonl; $HOME/.venv/bin/python spark_probe.py --plan oos/plan.json --out oos.jsonl > oos.log 2>&1; echo "== oos done $(date)"
