#!/bin/bash
# Watchdog for the tier study: wait for the running study by PID (never by name), then re-run it (it skips finished
# shapes) up to two times if any of the seven shapes is missing. Log: study_out/watchdog.log
cd ~/work/research/AI-Infra-Co-Design-study
export PATH=/usr/local/cuda/bin:$PATH; source ~/.venv/bin/activate
L=study_out/watchdog.log; PID=$1
echo "$(date) watching study pid $PID" >> $L
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "$(date) study pid $PID ended" >> $L
for attempt in 1 2; do
  n=$(ls study_out/*x*x*.json 2>/dev/null | wc -l)
  echo "$(date) shapes finished: $n of 7" >> $L
  [ "$n" -ge 7 ] && break
  echo "$(date) re-running the study (attempt $attempt)" >> $L
  python -m agentic.decide.study.run study_out >> study_out/run_all.log 2>&1
done
echo "$(date) watchdog done: $(ls study_out/*x*x*.json 2>/dev/null | wc -l) of 7 shapes" >> $L
