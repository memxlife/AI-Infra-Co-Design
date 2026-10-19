#!/bin/bash
# On the Spark. Skeleton 3 timings first (GPU idle), then racecheck under compute-sanitizer.
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
PY=$HOME/.venv/bin/python
export TILELANG_CACHE_DIR=$(mktemp -d)
echo "=== skel3"; timeout 1500 $PY skel3.py 2>&1 | grep -v "INFO\|Loading\|WARNING"
for k in fam_cpasync_ws fam_cpasync_nows fam_tma_ws skel2_1024x1024x1024_16x128x32_st2_th64_sm48 frag; do
  echo "=== inputs $k"; timeout 900 $PY racecheck.py $k 30 2>&1 | grep -v "INFO\|Loading\|WARNING"
  echo "=== sanitizer $k"; timeout 1500 compute-sanitizer --tool racecheck --racecheck-report hazard --print-limit 20 $PY racecheck.py $k 1 2>&1 | grep -v "INFO\|Loading\|WARNING" | grep -i "racecheck\|hazard\|ERROR SUMMARY\|error" | head -30
done
echo "=== done"
