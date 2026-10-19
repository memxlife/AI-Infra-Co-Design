#!/bin/bash
# R1: cuBLASLt on the three shapes. R3: one config five times, unpinned then pinned at the application clock. Runs on the Spark.
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang
PY=$HOME/.venv/bin/python
echo "== build cublaslt $(date)"; nvcc -O3 -arch=sm_121a -o cublaslt_gemm cublaslt_gemm.cu -lcublasLt 2>&1 | tail -3
for s in "1024 1024 1024" "32 4096 4096" "4096 4096 4096"; do set -- $s; echo "== cublaslt $1 $2 $3"; ./cublaslt_gemm $1 $2 $3 20 100 2>&1 | tail -3; done > cublaslt.log 2>&1
cat cublaslt.log
H='{"block_M": 256, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 256, "swizzle": 0}'
echo "== noise unpinned 1024 $(date)"; $PY spark_noise.py --kernel gemm_basic_holes.py --m 1024 --n 1024 --k 1024 --holes "$H" --out noise_1024_unpinned.json 2>&1 | grep -v INFO
H4='{"block_M": 256, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 256, "swizzle": 4}'
echo "== noise unpinned 4096 $(date)"; $PY spark_noise.py --kernel gemm_basic_holes.py --m 4096 --n 4096 --k 4096 --holes "$H4" --out noise_4096_unpinned.json 2>&1 | grep -v INFO
echo "== pin clocks"; sudo -n nvidia-smi -lgc 2418,2418 | tail -1
echo "== noise pinned 1024 $(date)"; $PY spark_noise.py --kernel gemm_basic_holes.py --m 1024 --n 1024 --k 1024 --holes "$H" --out noise_1024_pinned.json 2>&1 | grep -v INFO
echo "== noise pinned 4096 $(date)"; $PY spark_noise.py --kernel gemm_basic_holes.py --m 4096 --n 4096 --k 4096 --holes "$H4" --out noise_4096_pinned.json 2>&1 | grep -v INFO
echo "== cublaslt pinned 4096"; ./cublaslt_gemm 4096 4096 4096 20 100 2>&1 | tail -1
sudo -n nvidia-smi -rgc | tail -1
echo "== done $(date)"
