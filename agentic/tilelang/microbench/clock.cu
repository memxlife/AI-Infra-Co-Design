// SM clock under load: one block per SM spins on dependent FMAs for a fixed number of clock64 ticks and records
// %globaltimer at both ends; clock = ticks / ns. Run while idle and right after a heavy mma load.
#include <cuda_runtime.h>
#include <cstdio>
__global__ void spin(long long ticks, unsigned long long* out, float* sink) {
  unsigned long long g0, g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long c0 = clock64(); float a = threadIdx.x;
  while (clock64() - c0 < ticks) { a = a * 1.0001f + 0.5f; }
  long long c1 = clock64(); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (threadIdx.x == 0) { out[2 * blockIdx.x] = c1 - c0; out[2 * blockIdx.x + 1] = g1 - g0; }
  if (a == 0.f) sink[0] = a;
}
int main() {
  unsigned long long *o, h[96]; float* s; cudaMalloc(&o, 96 * 8); cudaMalloc(&s, 4);
  for (int r = 0; r < 3; r++) {
    spin<<<48, 128>>>(200000000LL, o, s); cudaDeviceSynchronize(); cudaMemcpy(h, o, 96 * 8, cudaMemcpyDeviceToHost);
    double mn = 1e9, mx = 0; for (int b = 0; b < 48; b++) { double f = (double)h[2 * b] / h[2 * b + 1]; mn = f < mn ? f : mn; mx = f > mx ? f : mx; }
    printf("RESULT run=%d clock_GHz_min=%.3f max=%.3f\n", r, mn, mx);
  }
}
