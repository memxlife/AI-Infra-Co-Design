#!/bin/bash
# On the Spark: the loader alone, L2-resident (8 MB, pitch 2048 B = K 1024 fp16) unless noted.
export PATH=/usr/local/cuda/bin:$PATH
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang/microbench
nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o tileload tileload.cu || exit 1
: > tileload.txt
M8=$((8*1024*1024))
echo "# segment length at fixed tile bytes (16 KB), 128 threads, 3 stages, one block per SM, then 2 and 4 per SM" >> tileload.txt
for B in 1 48 96 192; do for SR in "32 512" "64 256" "128 128" "256 64" "512 32"; do set -- $SR; ./tileload $M8 2048 $2 $1 128 3 $B 256 >> tileload.txt; done; done
echo "# threads and stages, seg 64 (block_K 32), 128 rows, 48 and 96 blocks" >> tileload.txt
for B in 48 96; do for T in 128 256 512; do for S in 2 3 4 5; do ./tileload $M8 2048 128 64 $T $S $B 256 >> tileload.txt; done; done; done
echo "# the ablated tiles as loads: (rows = block_M + block_N, seg = block_K*2), their threads, stages and grid at 1024^3" >> tileload.txt
for C in "192 64 128 3 128" "192 64 256 3 128" "256 64 128 3 64" "256 64 256 3 64" "128 64 128 3 256" "128 64 256 3 256" "192 128 128 3 128" "192 128 256 3 128" "256 128 128 3 64" "384 64 256 3 32" "384 128 256 2 32" "128 256 128 3 256"; do set -- $C; ./tileload $M8 2048 $1 $2 $4 $3 $5 32 >> tileload.txt; done
echo "# DRAM source (256 MB, read-flushed clean), seg 64 / 128 / 256, 48 blocks, 128 threads, stages 2-4" >> tileload.txt
for SEG in 64 128 256 512; do for S in 2 3 4; do ./tileload $((256*1024*1024)) 8192 $((16384/SEG)) $SEG 128 $S 48 128 2 >> tileload.txt; done; done
echo "# DRAM source, few SMs (decode-like): 8, 16, 32, 48 blocks, seg 256, 128 rows (32 KB tiles), stages 2-4" >> tileload.txt
for B in 8 16 32 48; do for S in 2 3 4; do ./tileload $((256*1024*1024)) 8192 64 512 128 $S $B 128 2 >> tileload.txt; done; done
echo done; grep -c RESULT tileload.txt
