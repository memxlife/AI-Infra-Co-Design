// Instruction-level latencies and throughputs the scheduling control bits do not state (for the GB10 SM simulator).
//  ldsm_lat     one warp, dependent chain of ldmatrix.x4 (the address comes from the previous result): cycles per hop
//  ldsm_tput    W warps on one SM, independent ldmatrix.x4: cycles per instruction per SM (bytes per clock = 512 / that)
//  bar_lat      W warps, back-to-back __syncthreads: cycles per barrier
//  cpasync_lat  one thread, 16-byte cp.async + commit + wait_group 0, from an L2-resident line: cycles per round trip
//  cpasync_tput W warps on one SM, each lane 16 B cp.async in groups of G, wait each group: bytes per clock per SM
// All timed with clock64 inside one block (one SM).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__global__ void ldsm_lat(int iters, unsigned long long* out) {
  __shared__ __align__(128) unsigned short buf[64 * 64];
  for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) buf[i] = 0;    // all zeros: the loaded value adds 0 to the next address
  __syncthreads();
  unsigned a = smem_addr(buf) + (threadIdx.x % 32) * 16, r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
    a += r0;                                                             // dependent: the next address waits for this result
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = r1 + r2 + r3; }
}
__global__ void ldsm_tput(int iters, unsigned long long* out) {
  __shared__ __align__(128) unsigned short buf[16384 + 8192];
  unsigned a = smem_addr(buf) + ((threadIdx.x / 32) % 8) * 1024 + (threadIdx.x % 32) * 16; unsigned s = 0;
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
    unsigned b = a + (i & 7) * 2048;                                    // a new address every iteration: no hoisting
#pragma unroll
    for (int u = 0; u < 8; u++) { unsigned r0, r1, r2, r3; asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(b + u * 128) : "memory"); s += r0 ^ r1 ^ r2 ^ r3; }
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = s; }
}
__global__ void bar_lat(int iters, unsigned long long* out) {
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
}
__global__ void cpasync_lat(const char* g, int iters, unsigned long long* out) {
  __shared__ __align__(128) char s[256];
  if (threadIdx.x) return;
  unsigned sa = smem_addr(s);
  for (int i = 0; i < 64; i++) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(g + (i % 64) * 128)); asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_group 0;"); }
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(g + (i % 64) * 128)); asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_group 0;"); }
  long long t1 = clock64(); out[0] = t1 - t0;
}
template <int G>
__global__ void cpasync_tput(const char* g, int iters, unsigned long long* out) {
  extern __shared__ __align__(128) char s[];
  unsigned sa = smem_addr(s) + threadIdx.x * 16 * G;
  const char* p = g + threadIdx.x * 16;
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < G; u++) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa + u * 16), "l"(p + ((i * G + u) % 256) * blockDim.x * 16));
    asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_group 0;");
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
}
int main() {
  unsigned long long *o, h[2]; char* g; cudaMalloc(&o, 16); cudaMalloc(&g, 64 << 20); cudaMemset(g, 0, 64 << 20);
  int it = 20000;
  ldsm_lat<<<1, 32>>>(it, o); cudaDeviceSynchronize(); cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost); printf("RESULT ldsm_lat cycles=%.1f\n", (double)h[0] / it);
  for (int w : {1, 2, 4, 8, 16}) { ldsm_tput<<<1, 32 * w>>>(2000, o); cudaDeviceSynchronize(); cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost); double c = (double)h[0] / (2000.0 * 8 * w); printf("RESULT ldsm_tput warps=%d cycles_per_ldsm_per_sm=%.2f bytes_per_clk=%.1f\n", w, c, 512 / c); }
  for (int w : {1, 4, 8, 16, 32}) { bar_lat<<<1, 32 * w>>>(it, o); cudaDeviceSynchronize(); cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost); printf("RESULT bar_lat warps=%d cycles=%.1f\n", w, (double)h[0] / it); }
  cpasync_lat<<<1, 32>>>(g, 2000, o); cudaDeviceSynchronize(); cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost); printf("RESULT cpasync_lat cycles=%.1f\n", (double)h[0] / 2000);
  for (int w : {1, 2, 4, 8, 16}) for (int G : {1, 4, 8}) {
    int smem = 32 * w * 16 * G; int its = 400;
    auto k = G == 1 ? cpasync_tput<1> : G == 4 ? cpasync_tput<4> : cpasync_tput<8>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    k<<<1, 32 * w, smem>>>(g, 20, o); cudaDeviceSynchronize();
    k<<<1, 32 * w, smem>>>(g, its, o); cudaDeviceSynchronize(); cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost);
    double cyc = (double)h[0] / its; double bytes = 32.0 * w * 16 * G;
    printf("RESULT cpasync_tput warps=%d per_thread=%d cycles_per_group=%.1f bytes_per_clk=%.2f\n", w, G, cyc, bytes / cyc);
  }
}
