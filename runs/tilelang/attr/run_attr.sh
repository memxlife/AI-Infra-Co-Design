#!/bin/bash
# step 2b: the seven in-sample picks under ncu, full counters, real clocks (no clock control), one launch each
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
STALLS="long_scoreboard short_scoreboard barrier mio_throttle lg_throttle math_pipe_throttle not_selected selected wait sleeping membar no_instruction imc_miss drain dispatch_stall branch_resolving tex_throttle misc"
M="gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.max,sm__cycles_active.avg,sm__cycles_active.max,sm__cycles_active.min,smsp__cycles_active.avg,sm__pipe_tensor_cycles_active.avg,sm__pipe_tensor_cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.per_cycle_active,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__waves_per_multiprocessor,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__warps_launched.sum,sm__ctas_launched.sum,gpc__cycles_elapsed.max,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__average_warp_latency_per_inst_issued.ratio"
for s in $STALLS; do M="$M,smsp__average_warps_issue_stalled_${s}_per_issue_active.ratio"; done
for n in 1024 4096 m32 2048 8192 k1024 m128; do
  echo "== $n $(date)"; ncu --clock-control none -k regex:matmul --launch-skip 1 --launch-count 1 --metrics $M $HOME/.venv/bin/python attr/once_$n.py > attr/ncu_$n.txt 2>&1
done
echo "== done $(date)"
