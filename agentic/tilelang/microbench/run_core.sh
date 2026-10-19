#!/bin/bash
# On the Spark: flush modes, warm L2 bandwidth, latency ladder, tensor rate against warps.
export PATH=/usr/local/cuda/bin:$PATH
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang/microbench
F="-O3 -gencode arch=compute_121a,code=sm_121a"
nvcc $F -o mlp_stream mlp_stream.cu && nvcc $F -o chase chase.cu && nvcc $F -o mma_rate mma_rate.cu || exit 1
: > core.txt
echo "# flush: read S MB after a write flush (1, dirty L2), a read flush (2, clean L2), no flush (0)" >> core.txt
for MB in 2 4 8 12 16 20 24 32 48 64; do for FL in 1 2 0; do ./mlp_stream $((MB*1024*1024)) 192 512 4 15 $FL 1 >> core.txt; done; done
echo "# warm L2 bandwidth: 8 MB streamed 20 times in one launch, by blocks" >> core.txt
for B in 1 2 4 8 16 24 32 48 96 144; do ./mlp_stream $((8*1024*1024)) $B 512 4 10 0 20 >> core.txt; done
for B in 48 96; do for T in 256 1024; do for U in 1 8; do ./mlp_stream $((8*1024*1024)) $B $T $U 10 0 20 >> core.txt; done; done; done
echo "# latency ladder" >> core.txt
for S in 16384 65536 262144 1048576 4194304 8388608 16777216 20971520 25165824 33554432 67108864 268435456; do ./chase $S 0 >> core.txt; done
for S in 16384 65536 131072; do ./chase $S 1 >> core.txt; done
echo "# tensor rate" >> core.txt
for B in 1 48; do for T in 32 64 128 256 512 1024; do for A in 1 2 4 8; do ./mma_rate $B $T $A >> core.txt; done; done; done
echo done
