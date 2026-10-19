// The cp.async GEMM loader loop built up one step at a time, one block alone on the chip (L2-resident source):
//  mode 0: copies only (8 x 16 B per thread per k-tile, 16 KB tiles: 256 rows x 64 B, rows 2 KB apart)
//  mode 1: + commit_group
//  mode 2: + wait_group(STAGES-1)
//  mode 3: + one __syncthreads after the wait
//  mode 4: + a second __syncthreads (TileLang's loop: barrier, issue, commit, wait, barrier)
//  mode 5: mode 4 with contiguous 512 B per warp instruction (rows 64 B apart: no row stride)
// Prints cycles per k-tile (clock64, thread 0).
#include <cuda_runtime.h>
#include <cstdio>
__device__ __forceinline__ void cp16(unsigned s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g) : "memory"); }
template <int MODE, int ST>
__global__ void loop(const char* __restrict__ src, int ktiles, int pitch, unsigned long long* out) {
  extern __shared__ __align__(128) char sm[];
  const int rows = 256, seg = 64, tile = rows * seg, per = tile / 16;
  unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
  long long t0 = 0; unsigned long long g0 = 0, g1;
  for (int kt = -ST; kt < ktiles; kt++) {
    if (kt == 0) { __syncthreads(); t0 = clock64(); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0)); }
    int k = kt + ST;
    if (MODE >= 4) __syncthreads();
    unsigned dst = sbase + (k % ST) * tile;
    for (int i = threadIdx.x; i < per; i += blockDim.x) {
      int r = i / 4, c = i % 4;
      const char* g = src + (size_t)r * pitch + (size_t)(k % 32) * seg + c * 16;
      cp16(dst + r * seg + c * 16, g);
    }
    if (MODE >= 1) asm volatile("cp.async.commit_group;" ::: "memory");
    if (MODE >= 2) asm volatile("cp.async.wait_group %0;" :: "n"(ST - 1) : "memory");
    if (MODE >= 3) __syncthreads();
  }
  asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
  long long t1 = clock64(); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = g1 - g0; }
}
template <int MODE> void run(const char* g, unsigned long long* o, int threads, int pitch, int blocks = 1) {
  const int ST = 3, kt = 512; unsigned long long h[2]; int smem = ST * 256 * 64;
  cudaFuncSetAttribute(loop<MODE, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  loop<MODE, ST><<<blocks, threads, smem>>>(g, kt, pitch, o); cudaDeviceSynchronize();
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  loop<MODE, ST><<<blocks, threads, smem>>>(g, kt, pitch, o); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); cudaDeviceSynchronize(); cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost);
  printf("RESULT mode=%d threads=%d blocks=%d cycles_per_ktile=%.0f ns_per_ktile=%.0f clock_GHz=%.2f block0_GBs=%.1f chip_GBs=%.0f per_SM_GBs=%.1f\n", MODE, threads, blocks, (double)h[0] / kt, (double)h[1] / kt, (double)h[0] / h[1], 16384.0 * kt / h[1], 16384.0 * kt * blocks / (ms * 1e6), 16384.0 * kt * blocks / (ms * 1e6) / (blocks < 48 ? blocks : 48));
}
int main() {
  char* g; unsigned long long* o; cudaMalloc(&g, 8 << 20); cudaMalloc(&o, 64); cudaMemset(g, 1, 8 << 20);
  for (int b : {1, 8, 16, 24, 32, 40, 48, 96}) run<4>(g, o, 128, 2048, b);
  for (int b : {32, 48}) run<4>(g, o, 256, 2048, b);
}
