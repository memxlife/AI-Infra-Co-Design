// Interference inside one SM: warps 0-3 (one per sub-partition) run independent HMMA.16816 chains at the tensor pipe's
// rate; warps 4-7 (one more per sub-partition) issue a memory instruction stream of kind KIND: 0 nothing, 1 cp.async
// 16 B (L2-resident, commit + wait every 8), 2 ldmatrix.x4, 3 ld.global 16 B, 4 integer adds. clock64 around the mma
// loop gives mma cycles; the memory warps count their instructions. Lost tensor cycles per memory instruction =
// (mma cycles with - without) / memory instructions issued during the mma loop.
#include <cuda_runtime.h>
#include <cstdio>
__device__ unsigned long long res[8];
__device__ volatile int stop_flag;
template <int KIND>
__global__ void k(const char* g, int iters) {
  extern __shared__ __align__(128) char sm[];
  int w = threadIdx.x / 32;
  if (w < 4) {
    unsigned a0 = threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
    float c[4][4] = {};
    __syncthreads(); long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
#pragma unroll
      for (int j = 0; j < 4; j++)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    long long t1 = clock64();
    if (threadIdx.x % 32 == 0) res[w] = t1 - t0;
    if (threadIdx.x == 0) stop_flag = 1;
    float s = 0; for (int j = 0; j < 4; j++) s += c[j][0]; if (s == 1.234f) res[7] = 1;
  } else {
    unsigned sa = (unsigned)__cvta_generic_to_shared(sm) + (threadIdx.x % 128) * 16;
    const char* p = g + (threadIdx.x % 128) * 16; unsigned long long n = 0; unsigned acc = 0;
    __syncthreads();
    while (!stop_flag) {
#pragma unroll
      for (int u = 0; u < 8; u++) {
        if (KIND == 1) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa + (u % 4) * 2048), "l"(p + ((n + u) % 512) * 2048) : "memory");
        if (KIND == 2) { unsigned r0, r1, r2, r3; asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sa + (u % 4) * 2048) : "memory"); acc += r0 ^ r3; }
        if (KIND == 3) { uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + ((n + u) % 512) * 2048)); acc += v.x; }
        if (KIND == 4) asm volatile("add.u32 %0, %0, %1;" : "+r"(acc) : "r"(u));
      }
      if (KIND == 1) { asm volatile("cp.async.commit_group;" ::: "memory"); asm volatile("cp.async.wait_group 0;" ::: "memory"); }
      n += 8;
    }
    if (threadIdx.x % 32 == 0) res[w] = n;
    if (acc == 0x12345u) res[7] = acc;
  }
}
template <int KIND> void run(const char* g, const char* name) {
  int iters = 20000, smem = 4 * 2048 + 128 * 16; unsigned long long h[8], z = 0; int zero = 0;
  cudaMemcpyToSymbol(stop_flag, &zero, 4); cudaFuncSetAttribute(k<KIND>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  k<KIND><<<1, 256, smem>>>(g, iters); cudaDeviceSynchronize();
  cudaMemcpyToSymbol(stop_flag, &zero, 4); k<KIND><<<1, 256, smem>>>(g, iters); cudaDeviceSynchronize(); cudaMemcpyFromSymbol(h, res, 64);
  double mma_cyc = (h[0] + h[1] + h[2] + h[3]) / 4.0, ideal = iters * 4 * 16.0, mem = (h[4] + h[5] + h[6] + h[7]) / 4.0;
  printf("RESULT kind=%-9s mma_cycles=%.0f ideal=%.0f lost=%.0f mem_instr_per_warp=%.0f lost_per_mem_instr=%.2f\n", name, mma_cyc, ideal, mma_cyc - ideal, mem, mem > 0 ? (mma_cyc - ideal) / mem : 0.0);
}
int main() {
  char* g; cudaMalloc(&g, 4 << 20); cudaMemset(g, 0, 4 << 20);
  run<0>(g, "none"); run<4>(g, "int-add"); run<2>(g, "ldmatrix"); run<3>(g, "ld.global"); run<1>(g, "cp.async");
}
