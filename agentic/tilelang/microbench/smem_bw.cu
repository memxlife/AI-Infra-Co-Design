// Shared-memory rate per SM by instruction and by row stride. Each address adds (previous result & zero), zero = 0 at
// run time, so the compiler cannot drop, merge or hoist a load; 32 warps hide the latency this chain adds.
//  ldmatrix x4 / x4.trans: lane l gives row l%8 of matrix l/8 at (l%8)*stride + (l/8)*16 (a linear tile: A rows
//    block_K*2 apart, B rows block_N*2 apart)
//  LDS.128, STS.128, stmatrix x4: each lane 16 contiguous bytes (a warp 512 B)
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
template <int KIND>
__global__ void k(int iters, int stride, unsigned zero, unsigned long long* out, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  unsigned base = (unsigned)__cvta_generic_to_shared(buf) + (warp % 4) * 8192;
  unsigned lm = base + (lane % 8) * stride + (lane / 8) * 16;
  unsigned ls = base + lane * 16;
  unsigned r0 = lane, r1 = 1, r2 = 2, r3 = 3;
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      unsigned off = (u & 3) * 128 + (r0 & zero); unsigned x0, x1, x2, x3;
      if (KIND == 0) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(lm + off) : "memory"); r0 ^= x0; r1 ^= x1; r2 ^= x2; r3 ^= x3; }
      if (KIND == 1) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(lm + off) : "memory"); r0 ^= x0; r1 ^= x1; r2 ^= x2; r3 ^= x3; }
      if (KIND == 2) { asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(ls + off) : "memory"); r0 ^= x0; r1 ^= x1; r2 ^= x2; r3 ^= x3; }
      if (KIND == 3) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(ls + off), "r"(r0 + u), "r"(r1), "r"(r2), "r"(r3) : "memory"); }
      if (KIND == 4) { asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(ls + off), "r"(r0 + u), "r"(r1), "r"(r2), "r"(r3) : "memory"); }
    }
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  if (r0 == 12345 && r1 == r2 && r3 == 7) sink[0] = r0;
}
int main() {
  unsigned long long* o; unsigned* sink; CK(cudaMalloc(&o, 8)); CK(cudaMalloc(&sink, 4)); unsigned long long h;
  const char* names[5] = {"ldmatrix.x4", "ldmatrix.x4.trans", "LDS.128", "STS.128", "stmatrix.x4"};
  void (*fs[5])(int, int, unsigned, unsigned long long*, unsigned*) = {k<0>, k<1>, k<2>, k<3>, k<4>};
  auto run = [&](int f, int warps, int stride) {
    const int iters = 1000;
    fs[f]<<<1, 32 * warps, 32 * 1024>>>(iters, stride, 0u, o, sink); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
    return (double)h / (iters * 8.0 * warps);
  };
  for (int f = 0; f < 5; f++) CK(cudaFuncSetAttribute(fs[f], cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024));
  for (int f = 0; f < 5; f++) for (int warps : {8, 16, 32}) { double c = run(f, warps, 48); printf("RESULT %s warps=%d stride=48 cycles_per_instr_per_SM=%.2f bytes_per_clk=%.0f\n", names[f], warps, c, 512 / c); }
  for (int f = 0; f < 2; f++) for (int stride : {16, 32, 48, 64, 80, 128, 144, 256, 512}) { double c = run(f, 32, stride); printf("RESULT %s warps=32 stride=%d cycles_per_instr_per_SM=%.2f\n", names[f], stride, c); }
}
