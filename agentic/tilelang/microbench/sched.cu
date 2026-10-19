// Warp scheduler policy on one sub-partition: warps 0 and 4 share sub-partition 0. Warp 0 runs a long stream of
// independent FFMAs (always ready to issue). Warp 4 runs K dependent integer adds and records when it finishes.
// Greedy-then-oldest would starve warp 4 until warp 0 stalls; round-robin interleaves them (~2 cycles per add).
// Also: warp 4 alone (baseline).
#include <cuda_runtime.h>
#include <cstdio>
__global__ void k(int n_ffma, int n_add, int with_hog, unsigned long long* out) {
  int w = threadIdx.x / 32;
  __syncthreads();
  long long t0 = clock64();
  if (w == 0 && with_hog) {
    float a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    for (int i = 0; i < n_ffma; i++) { a0 = a0 * 1.0001f + 0.5f; a1 = a1 * 1.0001f + 0.5f; a2 = a2 * 1.0001f + 0.5f; a3 = a3 * 1.0001f + 0.5f; a4 = a4 * 1.0001f + 0.5f; a5 = a5 * 1.0001f + 0.5f; a6 = a6 * 1.0001f + 0.5f; a7 = a7 * 1.0001f + 0.5f; }
    if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 0.123f) out[3] = 1;
    if (threadIdx.x == 0) out[0] = clock64() - t0;
  } else if (w == 4) {
    unsigned x = threadIdx.x;
    for (int i = 0; i < n_add; i++) asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(i));
    if (threadIdx.x == 128) { out[1] = clock64() - t0; out[2] = x; }
  }
}
int main() {
  unsigned long long* o, h[4]; cudaMalloc(&o, 32);
  for (int with : {0, 1}) {
    k<<<1, 160>>>(200000, 20000, with, o); cudaDeviceSynchronize(); cudaMemcpy(h, o, 32, cudaMemcpyDeviceToHost);
    printf("RESULT hog=%d warp4_cycles=%llu (%.2f per add) warp0_cycles=%llu\n", with, h[1], (double)h[1] / 20000, with ? h[0] : 0ULL);
  }
}
