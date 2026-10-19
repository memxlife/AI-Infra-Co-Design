#!/bin/bash
# Step 2: our picks at the out-of-sample shapes under ncu (and cuBLASLt there too, since it costs seconds)
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
SEC="--section LaunchStats --section Occupancy --section SpeedOfLight --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis"
MET="--metrics lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__waves_per_multiprocessor,launch__registers_per_thread,launch__block_size,launch__grid_size,launch__shared_mem_per_block_dynamic,gpu__time_duration.sum,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active"
for s in "2048 2048 2048 2048" "8192 8192 8192 8192" "k1024 4096 4096 1024" "m128 128 4096 4096"; do set -- $s
  echo "== cublaslt $1 $(date)"; ncu --launch-skip 20 --launch-count 1 $SEC $MET ./cublaslt_gemm $2 $3 $4 20 3 > ncu/cublaslt_$1.txt 2>&1
  echo "== ours $1 $(date)"; ncu -k regex:matmul --launch-skip 1 --launch-count 1 $SEC $MET $HOME/.venv/bin/python ncu/once_$1.py > ncu/ours_$1.txt 2>&1
done
for s in "1024 1024 1024 1024" "m32 32 4096 4096" "4096 4096 4096 4096"; do set -- $s     # the stall metrics for the three step-0 shapes too
  echo "== stalls ours $1 $(date)"; ncu -k regex:matmul --launch-skip 1 --launch-count 1 $MET $HOME/.venv/bin/python ncu/once_$1.py > ncu/stalls_ours_$1.txt 2>&1
  echo "== stalls cublaslt $1 $(date)"; ncu --launch-skip 20 --launch-count 1 $MET ./cublaslt_gemm $2 $3 $4 20 3 > ncu/stalls_cublaslt_$1.txt 2>&1
done
echo "== done $(date)"
