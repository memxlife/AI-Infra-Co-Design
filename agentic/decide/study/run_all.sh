#!/bin/bash
# Start the whole tier study on the Spark, detached, plus a watchdog that re-runs missing shapes.
# Waits and kills go by PID only: an earlier `pgrep -f NAME` wait matched its own ssh command line and stalled for hours.
# Run it as: setsid nohup agentic/decide/study/run_all.sh [OUT_DIR] > /dev/null 2>&1 < /dev/null &   (default study_out)
# then check OUT_DIR/run_all.log shows a "built in parallel" line before walking away.
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/cuda/bin:$PATH; source ~/.venv/bin/activate
O=${1:-study_out}; mkdir -p "$O"
echo "$(date) starting the study" >> "$O/run_all.log"
nohup python -m agentic.decide.study.run "$O" >> "$O/run_all.log" 2>&1 < /dev/null &
PID=$!; echo $PID > "$O/study.pid"
nohup agentic/decide/study/watchdog.sh $PID "$O" > /dev/null 2>&1 < /dev/null &
echo "$(date) study pid $PID, watchdog pid $!" >> "$O/run_all.log"
