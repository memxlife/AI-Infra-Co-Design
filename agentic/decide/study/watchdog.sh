#!/bin/bash
# Watchdog for the tier study: wait for the running study by PID (never by name), then re-run it (it skips finished
# shapes) up to two times if any of the seven shapes is missing. Log: OUT_DIR/watchdog.log
cd ~/work/research/AI-Infra-Co-Design-study
export PATH=/usr/local/cuda/bin:$PATH; source ~/.venv/bin/activate
PID=$1; O=${2:-study_out}; L=$O/watchdog.log
echo "$(date) watching study pid $PID" >> $L
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "$(date) study pid $PID ended" >> $L
for attempt in 1 2; do
  n=$(ls "$O"/*x*x*.json 2>/dev/null | wc -l)
  echo "$(date) shapes finished: $n of 7" >> $L
  [ "$n" -ge 7 ] && break
  echo "$(date) re-running the study (attempt $attempt)" >> $L
  python -m agentic.decide.study.run "$O" >> "$O/run_all.log" 2>&1
done
echo "$(date) watchdog done: $(ls "$O"/*x*x*.json 2>/dev/null | wc -l) of 7 shapes" >> $L
