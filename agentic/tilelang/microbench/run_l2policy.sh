#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH; NCU=/opt/nvidia/nsight-compute/2025.3.1/ncu
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang/microbench && nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o l2policy l2policy.cu || exit 1
M="lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum"
q() { $NCU --cache-control none --clock-control none --metrics $M -k readk --launch-skip $1 --launch-count 1 ./l2policy $2 $3 $4 2>/dev/null | grep -E "lookup_(hit|miss)" | awk '{gsub(",","",$NF); printf "%s=%s ", ($1 ~ /hit/ ? "hit" : "miss"), $NF}'; }
for W in 4 8 12 14 16 18 20 22 24 28 32 40 48 64; do echo "cyclic W=${W}MB passes=4: $(q 1 cyclic $W 4)"; done
for H in 4 8; do for S in 4 8 12 16 24 32 48 64; do echo "hotstream H=${H}MB S=${S}MB: $(q 2 hotstream $H $S)"; done; done
