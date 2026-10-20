#!/bin/bash
# The whole tier study on the Spark, detached: wait for TileLang's autotuner, then all seven shapes.
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/cuda/bin:$PATH; source ~/.venv/bin/activate
echo "$(date) waiting for TileLang's autotuner" >> study_out/run_all.log
while pgrep -f tilelang_autotune.py > /dev/null; do sleep 30; done
echo "$(date) autotuner done; starting the study" >> study_out/run_all.log
python -m agentic.decide.study.run study_out >> study_out/run_all.log 2>&1
echo "$(date) study finished" >> study_out/run_all.log
