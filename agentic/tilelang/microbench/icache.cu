// Instruction fetch: a loop whose body is U independent FFMAs (8 chains, unrolled), so the code is about U x 16 bytes.
// 8 warps on one SM run it many times; FFMA per clock per SM against the loop's code size. A drop marks where the
// loop no longer fits an instruction cache level.
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
template <int U>
__global__ void k(int reps, float s, unsigned long long* out, float* sink) {
  float f[8];
#pragma unroll
  for (int j = 0; j < 8; j++) f[j] = threadIdx.x + j;
  __syncthreads(); long long t0 = clock64();
  for (int r = 0; r < reps; r++) {
#pragma unroll
    for (int i = 0; i < U / 8; i++) {
#pragma unroll
      for (int j = 0; j < 8; j++) f[j] = fmaf(f[j], s, 0.25f);
    }
  }
  float t = 0; for (int j = 0; j < 8; j++) t += f[j];
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  if (t == 1.2345f) sink[0] = t;
}
template <int U> void run(unsigned long long* o, float* sink) {
  const int warps = 8; int reps = 4096 * 64 / U;
  k<U><<<1, 32 * warps>>>(reps, 1.0001f, o, sink); CK(cudaDeviceSynchronize()); k<U><<<1, 32 * warps>>>(reps, 1.0001f, o, sink); CK(cudaDeviceSynchronize());
  unsigned long long h; CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
  printf("RESULT icache loop_ffma=%d code_KB=%.0f ffma_per_clk_per_SM=%.3f\n", U, U * 16.0 / 1024, (double)U * reps * warps / h);
}
int main() {
  unsigned long long* o; float* sink; CK(cudaMalloc(&o, 8)); CK(cudaMalloc(&sink, 4));
  run<256>(o, sink); run<512>(o, sink); run<1024>(o, sink); run<2048>(o, sink); run<4096>(o, sink); run<8192>(o, sink); run<16384>(o, sink);
}
