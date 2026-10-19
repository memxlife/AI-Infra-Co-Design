// GEMM-shaped loads, nothing else: a grid of tiles_m x tiles_n blocks (TileLang's default order: columns fastest);
// every k-tile, block (r, c) copies its A tile (BM rows x BK*2 bytes, rows K*2 apart) and its B tile (BK rows x BN*2
// bytes, rows N*2 apart) into a ring of ST shared-memory stages with cp.async, then commit, wait, two barriers (the loop
// TileLang generates, with compile-time addressing). Blocks in a row share A, blocks in a column share B.
//   lockstep: every block walks k from 0 (as the real kernels do): neighbours read the same lines at the same time
//   stagger:  block b starts at k-tile (b * 7) mod KT and wraps: the same lines in total, read at different times
// L2-resident (M = N = K = 1024: A and B are 2 MB each). Per-block %globaltimer spans; the chip rate from events.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
__device__ unsigned long long span[1024];
__device__ __forceinline__ void cp16(unsigned s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g) : "memory"); }
template <int BM, int BN, int BK, int ST, int T>
__global__ void __launch_bounds__(T) load(const char* __restrict__ A, const char* __restrict__ B, int K, int N, int tiles_n, int stagger) {
  extern __shared__ __align__(128) char sm[];
  constexpr int AB = BM * BK * 2, BB = BK * BN * 2, TILE = AB + BB, ACH = AB / 16, BCH = BB / 16, ASEG = BK * 2 / 16, BSEG = BN * 2 / 16;
  const int r = blockIdx.x / tiles_n, c = blockIdx.x % tiles_n, KT = K / BK;
  const int k0 = stagger ? (blockIdx.x * 7) % KT : 0;
  unsigned base = (unsigned)__cvta_generic_to_shared(sm);
  unsigned long long g0, g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  for (int it = -(ST - 1); it < KT; it++) {
    int k = (k0 + it + ST - 1) % KT; int slot = (it + ST - 1) % ST;
    __syncthreads();
    if (it + ST - 1 < KT) {
      unsigned d = base + slot * TILE;
      const char* a = A + (size_t)(r * BM) * K * 2 + (size_t)k * BK * 2;
      const char* b = B + (size_t)(k * BK) * N * 2 + (size_t)c * BN * 2;
      for (int i = threadIdx.x; i < ACH; i += T) cp16(d + i * 16, a + (size_t)(i / ASEG) * K * 2 + (i % ASEG) * 16);
      for (int i = threadIdx.x; i < BCH; i += T) cp16(d + AB + i * 16, b + (size_t)(i / BSEG) * N * 2 + (i % BSEG) * 16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group %0;" :: "n"(ST - 1) : "memory");
    __syncthreads();
  }
  asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (threadIdx.x == 0) span[blockIdx.x] = g1 - g0;
}
template <int BM, int BN, int BK, int ST, int T> void run(const char* A, const char* B, int stagger) {
  const int M = 1024, N = 1024, K = 1024; int tm = M / BM, tn = N / BN, blocks = tm * tn; int smem = ST * (BM * BK + BK * BN) * 2;
  auto k = load<BM, BN, BK, ST, T>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  for (int w = 0; w < 3; w++) k<<<blocks, T, smem>>>(A, B, K, N, tn, stagger);
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); std::vector<float> ts;
  for (int rep = 0; rep < 9; rep++) { cudaEventRecord(e0); k<<<blocks, T, smem>>>(A, B, K, N, tn, stagger); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms); }
  std::sort(ts.begin(), ts.end()); float ms = ts[ts.size() / 2];
  std::vector<unsigned long long> sp(blocks); cudaMemcpyFromSymbol(sp.data(), span, 8 * blocks);
  double mean = 0; for (auto v : sp) mean += v; mean /= blocks;
  double per_block = (double)(K / BK) * (BM * BK + BK * BN) * 2;
  printf("RESULT tile=%dx%dx%d st=%d thr=%d blocks=%d %s event_us=%.1f block_mean_us=%.1f per_block_GBs=%.1f chip_GBs=%.0f\n", BM, BN, BK, ST, T, blocks, stagger ? "stagger " : "lockstep",
         ms * 1e3, mean / 1e3, per_block / (mean * 1e-9) / 1e9, per_block * blocks / (ms * 1e-3) / 1e9);
}
int main() {
  char *A, *B; cudaMalloc(&A, 2 << 20); cudaMalloc(&B, 2 << 20); cudaMemset(A, 1, 2 << 20); cudaMemset(B, 2, 2 << 20);
  for (int s : {0, 1}) {
    run<256, 128, 32, 3, 256>(A, B, s);   // 32 blocks: the case the simulator misses
    run<128, 256, 32, 3, 256>(A, B, s);
    run<128, 128, 64, 3, 128>(A, B, s);   // 64 blocks, one per SM by shared memory
    run<128, 128, 32, 3, 128>(A, B, s);   // 64 blocks, two per SM
    run<64, 128, 32, 3, 256>(A, B, s);    // 128 blocks
    run<64, 64, 32, 3, 128>(A, B, s);     // 256 blocks
  }
}
