// The GEMM loader alone: each block streams tiles of `rows` x `seg` bytes (a k-slice of `rows` rows, rows `pitch` bytes
// apart, like an A tile: block_M rows x block_K*2 bytes) into a ring of `stages` shared-memory stages with cp.async 16 B,
// the same issue/commit/wait/__syncthreads pattern TileLang generates. No compute. Blocks take distinct row panels.
// Source: `bytes` total (8 MB: L2-resident, warm; 256 MB: DRAM), read-flushed or not.
// Run: ./tileload <bytes> <pitch> <rows> <seg> <threads> <stages> <blocks> <ktiles> [flush 0/2]
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { auto e = (x); if (e != 0) { fprintf(stderr, "error %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__device__ __forceinline__ void cp16(void* s, const void* g) {
  unsigned sa = (unsigned)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(g));
}
template <int STAGES>
__global__ void tileload(const char* __restrict__ src, size_t bytes, int pitch, int rows, int seg, int ktiles, unsigned* out) {
  const int stages = STAGES;
  extern __shared__ __align__(128) char sm[];
  const int tile = rows * seg, per = tile / 16, segs16 = seg / 16;
  const size_t panel = (size_t)rows * pitch, npanels = bytes / panel;
  const char* base = src + (blockIdx.x % npanels) * panel;
  const int kmax = pitch / seg;
  auto issue = [&](int kt) {
    char* dst = sm + (kt % stages) * tile; const char* g = base + (size_t)(kt % kmax) * seg;
    for (int i = threadIdx.x; i < per; i += blockDim.x) { int r = i / segs16, c = i % segs16; cp16(dst + r * seg + c * 16, g + (size_t)r * pitch + c * 16); }
    asm volatile("cp.async.commit_group;");
  };
  for (int s = 0; s < stages - 1; s++) issue(s);
  unsigned acc = 0;
  for (int kt = 0; kt < ktiles; kt++) {
    issue(kt + stages - 1);
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 1));   // tile kt has landed; stages-1 newer groups stay in flight
    __syncthreads();
    acc ^= ((unsigned*)(sm + (kt % stages) * tile))[threadIdx.x % (tile / 4)];
    __syncthreads();
  }
  if (acc == 0x12345678u) out[blockIdx.x] = acc;
}

__global__ void flush_read(const uint4* f, size_t n16, unsigned* out) {
  unsigned a = 0; for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) a ^= f[i].x;
  if (a == 0xFFFFFFFFu) out[0] = a;
}
int main(int argc, char** argv) {
  size_t bytes = strtoull(argv[1], 0, 10); int pitch = atoi(argv[2]), rows = atoi(argv[3]), seg = atoi(argv[4]), threads = atoi(argv[5]), stages = atoi(argv[6]), blocks = atoi(argv[7]), ktiles = atoi(argv[8]);
  int flush = argc > 9 ? atoi(argv[9]) : 0; int reps = 15;
  char* src; uint4* f; unsigned* out; CK(cudaMalloc(&src, bytes)); CK(cudaMalloc(&f, 256u << 20)); CK(cudaMalloc(&out, 65536 * 4));
  CK(cudaMemset(src, 1, bytes));
  int smem = stages * rows * seg;
  auto k = stages == 2 ? tileload<2> : stages == 3 ? tileload<3> : stages == 4 ? tileload<4> : tileload<5>;
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int per_sm = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, threads, smem));
  for (int i = 0; i < 3; i++) k<<<blocks, threads, smem>>>(src, bytes, pitch, rows, seg, ktiles, out);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); std::vector<float> t;
  for (int r = 0; r < reps; r++) {
    if (flush == 2) { flush_read<<<1024, 256>>>(f, (256u << 20) / 16, out); CK(cudaDeviceSynchronize()); }
    CK(cudaEventRecord(e0)); k<<<blocks, threads, smem>>>(src, bytes, pitch, rows, seg, ktiles, out); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms);
  }
  std::sort(t.begin(), t.end()); double us = t[t.size() / 2] * 1e3; double moved = (double)blocks * ktiles * rows * seg;
  int resident = std::min(blocks, per_sm * 48);
  printf("RESULT bytes=%zu pitch=%d rows=%d seg=%d threads=%d stages=%d blocks=%d per_sm=%d ktiles=%d flush=%d us=%.2f GBs=%.1f GBs_per_SM=%.1f\n",
         bytes, pitch, rows, seg, threads, stages, blocks, per_sm, ktiles, flush, us, moved / (us * 1e-6) / 1e9, moved / (us * 1e-6) / 1e9 / std::min(48, resident));
}
