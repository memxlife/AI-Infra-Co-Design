#!/bin/bash
# On the Spark: build and sweep the MLP stream microbenchmark, cold (256 MB, flushed) and warm (8 MB in L2). One JSON-ish RESULT line each.
export PATH=/usr/local/cuda/bin:$PATH
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang/microbench
nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o mlp_stream mlp_stream.cu || exit 1
: > mlp.txt
for B in 1 8 24 48 96 192 384; do for T in 128 256 512 1024; do for U in 1 2 4 8 16; do
  ./mlp_stream $((256*1024*1024)) $B $T $U 15 1 >> mlp.txt
done; done; done
for B in 48 96 192 384; do for T in 128 256 512 1024; do for U in 1 4 16; do
  ./mlp_stream $((8*1024*1024)) $B $T $U 40 0 >> mlp.txt
done; done; done
echo done; wc -l mlp.txt
