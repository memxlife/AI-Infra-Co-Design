// The special function unit (MUFU) on one SM: for each function, the dependent-chain latency (one warp, one chain)
// and the throughput (W warps, 8 independent chains per thread), in lane results per clock per SM; FFMA for scale.
#include <cuda_runtime.h>
#include <cstdio>
#define OPS(X) X(0, "ex2.f32", "ex2.approx.ftz.f32 %0, %0;") X(1, "rcp.f32", "rcp.approx.ftz.f32 %0, %0;") \
  X(2, "rsqrt.f32", "rsqrt.approx.ftz.f32 %0, %0;") X(3, "sin.f32", "sin.approx.ftz.f32 %0, %0;") \
  X(4, "lg2.f32", "lg2.approx.ftz.f32 %0, %0;") X(5, "tanh.f32", "tanh.approx.f32 %0, %0;") X(6, "ffma.f32", "fma.rn.f32 %0, %0, 0f3F800001, 0f3A000000;")
template <int OP, int CH>
__global__ void k(int iters, float* out, long long* cyc) {
  float v[CH];
#pragma unroll
  for (int c = 0; c < CH; c++) v[c] = 0.3f + 0.001f * (threadIdx.x + c);
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int c = 0; c < CH; c++) {
#define CASE(n, name, asmstr) if (OP == n) asm volatile(asmstr : "+f"(v[c]));
      OPS(CASE)
#undef CASE
    }
  }
  __syncthreads(); long long t1 = clock64();
  float s = 0; for (int c = 0; c < CH; c++) s += v[c]; if (s == 1.2345f) out[0] = s;
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
}
template <int OP> void run(const char* name, float* o, long long* c) {
  long long h; int it = 4096;
  k<OP, 1><<<1, 32>>>(it, o, c); cudaDeviceSynchronize(); cudaMemcpy(&h, c, 8, cudaMemcpyDeviceToHost);
  printf("RESULT %-9s latency_cycles=%.1f", name, (double)h / it);
  for (int w : {1, 4, 8, 16, 32}) {
    k<OP, 8><<<1, 32 * w>>>(it, o, c); cudaDeviceSynchronize(); cudaMemcpy(&h, c, 8, cudaMemcpyDeviceToHost);
    printf(" | w%d %.1f", w, 32.0 * w * 8 * it / h);
  }
  printf("  (lane results per clock per SM)\n");
}
int main() {
  float* o; long long* c; cudaMalloc(&o, 4); cudaMalloc(&c, 8);
#define RUN(n, name, asmstr) run<n>(name, o, c);
  OPS(RUN)
}
