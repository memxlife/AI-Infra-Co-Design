// The chip's L2 rate for the cp.async loop as blocks per SM grow: the lean TileLang-shaped loop (barrier, copies,
// commit, wait(ST-1), barrier; compile-time addressing) with tiles of ROWS x 64 B (rows 2 KB apart, as block_K 32),
// launched as 48 x R blocks so that every SM holds R. Each block streams its own panel (L2-resident, 8 MB total).
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
__device__ unsigned smid_of[512];
__device__ __forceinline__ void cp16(unsigned s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g) : "memory"); }
template <int ROWS, int ST, int T>
__global__ void __launch_bounds__(T) loop(const char* __restrict__ src, int ktiles) {
  extern __shared__ __align__(128) char sm[];
  constexpr int TILE = ROWS * 64, PER = TILE / 16;
  unsigned base = (unsigned)__cvta_generic_to_shared(sm);
  const char* panel = src + (size_t)(blockIdx.x % 32) * ROWS * 2048;
  for (int kt = -(ST - 1); kt < ktiles; kt++) {
    int k = kt + ST - 1;
    __syncthreads();
    unsigned d = base + (k % ST) * TILE;
    for (int i = threadIdx.x; i < PER; i += T) cp16(d + i * 16, panel + (size_t)(i / 4) * 2048 + (k % 32) * 64 + (i % 4) * 16);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group %0;" :: "n"(ST - 1) : "memory");
    __syncthreads();
  }
  if (threadIdx.x == 0) { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); smid_of[blockIdx.x] = s; }
}
template <int ROWS, int ST, int T> void run(const char* g, int R) {
  int blocks = 48 * R, kt = 512, smem = ST * ROWS * 64;
  auto k = loop<ROWS, ST, T>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int occ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, T, smem);
  k<<<blocks, T, smem>>>(g, kt); cudaDeviceSynchronize();
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); std::vector<float> v;
  for (int r = 0; r < 7; r++) { cudaEventRecord(e0); k<<<blocks, T, smem>>>(g, kt); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms); }
  std::sort(v.begin(), v.end()); float ms = v[3];
  unsigned s[512]; cudaMemcpyFromSymbol(s, smid_of, 4 * blocks); int cnt[64] = {0}; for (int b = 0; b < blocks; b++) cnt[s[b] % 64]++;
  int mx = 0; for (int i = 0; i < 64; i++) mx = std::max(mx, cnt[i]);
  double bytes = (double)blocks * kt * ROWS * 64;
  printf("RESULT tile_KB=%d threads=%d st=%d R=%d occupancy_limit=%d max_blocks_on_one_SM=%d chip_TBs=%.2f per_SM_GBs=%.1f\n", ROWS * 64 / 1024, T, ST, R, occ, mx, bytes / (ms * 1e-3) / 1e12, bytes / (ms * 1e-3) / 1e9 / 48);
}
int main() {
  char* g; cudaMalloc(&g, 32 << 20); cudaMemset(g, 1, 32 << 20);
  for (int R : {1, 2, 3, 4}) run<128, 3, 128>(g, R);     // 8 KB tiles (64x64x32's A+B), 24 KB of stages
  for (int R : {1, 2, 3}) run<192, 3, 128>(g, R);        // 12 KB tiles (64x128x32)
  for (int R : {1, 2}) run<256, 3, 128>(g, R);           // 16 KB tiles (128x128x32)
  for (int R : {1, 2, 3, 4}) run<128, 3, 256>(g, R);
}
