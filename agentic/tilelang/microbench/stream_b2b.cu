// Streaming reads timed back-to-back with no flush (how the kernels above were timed): each run reads S bytes once,
// 16-byte loads, many loads in flight per warp, all SMs. Rate against S: below the L2 it is the L2 rate; well above it
// the DRAM rate; in between, what survives in the L2 from the previous run.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
template <bool CG>
__global__ void rd(const int4* __restrict__ p, size_t n16, int* sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x * 8 + threadIdx.x; int4 acc = make_int4(0, 0, 0, 0);
  for (; i < n16; i += (size_t)gridDim.x * blockDim.x * 8) {
    int4 v[8];
#pragma unroll
    for (int u = 0; u < 8; u++) { size_t j = i + (size_t)u * blockDim.x; v[u] = j < n16 ? (CG ? __ldcg(p + j) : p[j]) : make_int4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < 8; u++) { acc.x ^= v[u].x; acc.y ^= v[u].y; }
  }
  if (acc.x == 0x1234567 && acc.y == 3) sink[0] = 1;
}
int main() {
  int* sink; CK(cudaMalloc(&sink, 4)); char* buf; size_t maxb = (size_t)512 << 20; CK(cudaMalloc(&buf, maxb)); CK(cudaMemset(buf, 1, maxb));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int cg = 0; cg < 2; cg++) for (int mb : {8, 16, 20, 24, 28, 32, 36, 40, 48, 64, 128, 256}) {
    size_t n16 = ((size_t)mb << 20) / 16; auto k = cg ? rd<true> : rd<false>;
    int blocks = 48 * 4, threads = 512, reps = mb <= 64 ? 200 : 40;
    for (int r = 0; r < 10; r++) k<<<blocks, threads>>>((const int4*)buf, n16, sink);
    CK(cudaEventRecord(e0)); for (int r = 0; r < reps; r++) k<<<blocks, threads>>>((const int4*)buf, n16, sink); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); double us = ms * 1e3 / reps;
    printf("RESULT stream_b2b load=%s MB=%d us=%.1f GBs=%.1f\n", cg ? "cg" : "default", mb, us, (double)mb * 1048576 / (us * 1e3));
  }
}
