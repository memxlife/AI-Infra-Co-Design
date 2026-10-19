// Tensor-pipe rate against warps: each warp runs a long chain of mma.sync m16n8k16 (fp16 in, fp32 accumulate) on
// registers only, with A independent accumulators so A mmas are in flight per warp. Blocks = 48 (one per SM) or 1.
// FLOP per SM per clock = 4096 x mmas / (clock x time). No memory traffic.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
template <int A>
__global__ void mma_kernel(int iters, float* out) {
  unsigned a0 = threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
  float c[A][4];
#pragma unroll
  for (int j = 0; j < A; j++) for (int k = 0; k < 4; k++) c[j][k] = 0.f;
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int j = 0; j < A; j++)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  float s = 0; for (int j = 0; j < A; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 1.2345f) out[0] = s;
}
template <int A> float run(int blocks, int threads, int iters, float* o) {
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  mma_kernel<A><<<blocks, threads>>>(iters, o); cudaDeviceSynchronize();
  cudaEventRecord(e0); mma_kernel<A><<<blocks, threads>>>(iters, o); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); return ms;
}
int main(int argc, char** argv) {
  int blocks = atoi(argv[1]), threads = atoi(argv[2]), A = atoi(argv[3]); int iters = 200000 / A; float* o; cudaMalloc(&o, 4);
  float ms = A == 1 ? run<1>(blocks, threads, iters, o) : A == 2 ? run<2>(blocks, threads, iters, o) : A == 4 ? run<4>(blocks, threads, iters, o) : run<8>(blocks, threads, iters, o);
  double mmas = (double)blocks * (threads / 32) * iters * A; double tf = mmas * 4096 / (ms * 1e-3) / 1e12;
  printf("RESULT blocks=%d threads=%d warps_per_sm=%d acc=%d ms=%.3f TFs=%.2f TFs_per_SM=%.3f\n", blocks, threads, threads / 32, A, ms, tf, tf / blocks);
}
