#!/bin/bash
# Step 0: cuBLASLt and our pick under ncu, the same sections, one kernel launch each (runs on the Spark)
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
SEC="--section LaunchStats --section Occupancy --section SpeedOfLight --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis"
MET="--metrics lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks,launch__waves_per_multiprocessor,launch__registers_per_thread,launch__block_size,launch__grid_size,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,gpu__time_duration.sum"
for s in "1024 1024 1024 1024" "m32 32 4096 4096" "4096 4096 4096 4096"; do set -- $s
  echo "== cublaslt $1 $(date)"
  # warm-up launches of every candidate come first in the harness (20 each); the best is algo 0 for all three shapes: skip its 20 warm-ups, take the next launch
  ncu --launch-skip 20 --launch-count 2 $SEC $MET ./cublaslt_gemm $2 $3 $4 20 3 > ncu/cublaslt_$1.txt 2>&1
  echo "== ours $1 $(date)"
  ncu -k regex:matmul --launch-skip 1 --launch-count 1 $SEC $MET $HOME/.venv/bin/python ncu/once_$1.py > ncu/ours_$1.txt 2>&1
done
echo "== done $(date)"
