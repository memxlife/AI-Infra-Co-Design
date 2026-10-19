#!/bin/bash
# waits for the 400-point sample, then times the solver's picks and round 1's picks in the same session (runs on the Spark)
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
while [ "$(wc -l < sample.jsonl 2>/dev/null || echo 0)" -lt 400 ] && pgrep -f "spark_probe.py --plan sample/plan.json" > /dev/null; do sleep 20; done
rm -f sample_picks.jsonl
$HOME/.venv/bin/python spark_probe.py --plan sample/plan_picks.json --out sample_picks.jsonl > sample_picks.log 2>&1
echo "== picks done $(date)"
