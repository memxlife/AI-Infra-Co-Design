#!/bin/bash
# Start the whole tier study on the Spark, detached, plus a watchdog that re-runs missing shapes.
# Waits and kills go by PID only: an earlier `pgrep -f NAME` wait matched its own ssh command line and stalled for hours.
# Run it as: setsid nohup agentic/decide/study/run_all.sh > /dev/null 2>&1 < /dev/null &
# then check study_out/run_all.log shows a "built in parallel" line before walking away.
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/cuda/bin:$PATH; source ~/.venv/bin/activate
mkdir -p study_out
echo "$(date) starting the study" >> study_out/run_all.log
nohup python -m agentic.decide.study.run study_out >> study_out/run_all.log 2>&1 < /dev/null &
PID=$!; echo $PID > study_out/study.pid
nohup agentic/decide/study/watchdog.sh $PID > /dev/null 2>&1 < /dev/null &
echo "$(date) study pid $PID, watchdog pid $!" >> study_out/run_all.log
