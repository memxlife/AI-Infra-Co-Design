#!/bin/bash
# One process under sustained load (sustain.cu) with nvidia-smi sampling SM clock, power and temperature every 500 ms.
cd "$(dirname "$0")"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 500 > sustain_smi.csv &
SMI=$!; sleep 1; ./sustain; sleep 1; kill $SMI
echo "--- nvidia-smi every 2 s"; awk 'NR%4==1' sustain_smi.csv
