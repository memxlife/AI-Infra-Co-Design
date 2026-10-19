#!/bin/bash
# On the Spark: (1) the ablated kernels' tiles as loads (rows = block_M + block_N, seg = block_K*2) with their threads,
# stages and grid; (2) DRAM source without wrap-around (pitch = seg x ktiles), clean L2, by blocks: decode-like few SMs.
export PATH=/usr/local/cuda/bin:$PATH
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang/microbench
: > tileload2.txt
M8=$((8*1024*1024))
echo "# ablated tiles as loads" >> tileload2.txt
# rows seg threads stages blocks
for C in "192 64 128 3 128" "192 64 256 3 128" "256 64 128 3 64" "256 64 256 3 64" "128 64 128 3 256" "128 64 256 3 256" "192 128 128 3 128" "192 128 256 3 128" "256 128 128 3 64" "256 128 256 3 64" "384 64 256 3 32" "384 128 256 2 32" "320 64 128 3 64" "320 64 256 3 64" "128 256 128 3 256"; do
  set -- $C; ./tileload $M8 2048 $1 $2 $3 $4 $5 32 >> tileload2.txt; done
echo "# one block alone by tile bytes (seg 64), 128 and 256 threads, 3 stages" >> tileload2.txt
for R in 64 128 192 256 384; do for T in 128 256; do ./tileload $M8 2048 $R 64 $T 3 1 256 >> tileload2.txt; done; done
echo "# DRAM source, clean, no wrap: pitch = seg x 256 k-tiles; seg 64 / 256; 128 rows; blocks 8..96; stages 2 and 4" >> tileload2.txt
for SEG in 64 256; do for B in 8 16 32 48 96; do for S in 2 4; do ./tileload $((512*1024*1024)) $((SEG*256)) 128 $SEG 128 $S $B 256 2 >> tileload2.txt; done; done; done
echo done
