#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
while pgrep -f "run_oos.sh|run_picks.sh|spark_probe.py" > /dev/null; do sleep 20; done
rm -f splitk.jsonl; $HOME/.venv/bin/python spark_probe.py --plan splitk/plan.json --out splitk.jsonl > splitk.log 2>&1; echo "== splitk done $(date)"
