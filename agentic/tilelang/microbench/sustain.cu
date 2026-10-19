// Sustained load from one process: (1) mma.sync fp16 on all SMs back-to-back for 20 s, (2) a DRAM read stream for 20 s.
// Per second: the rate reached (TFLOP/s, GB/s). Run with nvidia-smi sampling clock and power alongside.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__global__ void mma(int iters, float* out) {
  unsigned a0 = threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13; float c[8][4] = {};
  for (int it = 0; it < iters; it++)
#pragma unroll
    for (int j = 0; j < 8; j++)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  float s = 0; for (int j = 0; j < 8; j++) s += c[j][0]; if (s == 1.2345f) out[0] = s;
}
__global__ void rd(const int4* __restrict__ p, size_t n, int* sink) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { int4 v = __ldcg(p + i); acc.x ^= v.x; }
  if (acc.x == 0x1234567) sink[0] = 1;
}
int main() {
  float* o; int* sink; CK(cudaMalloc(&o, 4)); CK(cudaMalloc(&sink, 4)); size_t bytes = (size_t)256 << 20; int4* buf; CK(cudaMalloc(&buf, bytes)); CK(cudaMemset(buf, 1, bytes));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto t_start = std::chrono::steady_clock::now(); auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
  const int iters = 2000;   // one launch ~0.5 ms
  for (int sec = 0; sec < 20; sec++) {
    double flop = 0, ms_sum = 0; CK(cudaEventRecord(e0));
    int n = 0; while (el() < sec + 1) { mma<<<48, 256>>>(iters, o); n++; if (n % 50 == 0) CK(cudaDeviceSynchronize()); }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    flop = (double)n * 48 * 8 * iters * 8 * 4096; printf("RESULT sustained mma t=%d launches=%d TFs=%.2f\n", sec, n, flop / (ms * 1e-3) / 1e12); fflush(stdout);
  }
  for (int sec = 20; sec < 40; sec++) {
    CK(cudaEventRecord(e0)); int n = 0;
    while (el() < sec + 1) { rd<<<192, 512>>>(buf, bytes / 16, sink); n++; if (n % 20 == 0) CK(cudaDeviceSynchronize()); }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("RESULT sustained dram t=%d launches=%d GBs=%.1f\n", sec, n, (double)n * bytes / (ms * 1e-3) / 1e9); fflush(stdout);
  }
}
