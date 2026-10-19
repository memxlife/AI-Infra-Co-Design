// DRAM latency under load, for the TMA path. 47 streamer blocks read a 256 MB buffer at a set rate (each thread keeps
// 4 loads in flight, then sleeps `ns`); meanwhile one block on another SM either
//  chase: follows pointers through 64 MB (random inside 2 MB pages: TLB hits, L2 misses), ns per hop, or
//  tma:   loads one TMA box of B bytes from a fresh place in a 512 MB buffer (never in the L2), waits on its mbarrier,
//         and records issue -> complete, 200 boxes, median.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__device__ volatile int stop;
__global__ void streamer(const uint4* __restrict__ p, size_t n16, int ns, unsigned long long* bytes) {
  unsigned acc = 0; unsigned long long b = 0; size_t st = (size_t)gridDim.x * blockDim.x, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  while (!stop) {
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; u++) { size_t j = (i + u * st) % n16; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + j)); }
#pragma unroll
    for (int u = 0; u < 4; u++) acc ^= v[u].x;
    i = (i + 4 * st) % n16; b += 64;
    if (ns) __nanosleep(ns);
  }
  if (acc == 1) bytes[1] = acc;
  atomicAdd(bytes, b);
}
__global__ void chaser(const unsigned* __restrict__ c, int hops, unsigned long long* out) {
  unsigned i = 0; unsigned long long g0, g1;
  for (int h = 0; h < 500; h++) i = __ldcg(c + i);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  for (int h = 0; h < hops; h++) i = __ldcg(c + i);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = g1 - g0; out[1] = i; stop = 1;
}
__device__ unsigned long long tma_t[256];
__global__ void tmabox(const __grid_constant__ CUtensorMap map, int rows, int nbox, int bytes, int run) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) unsigned long long mb;
  unsigned bar = (unsigned)__cvta_generic_to_shared(&mb), dst = (unsigned)__cvta_generic_to_shared(sm);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int k = 0; k < nbox; k++) {
      // every box of every run at its own place: never touched before, so it comes from DRAM
      unsigned long long g0, g1; int lin = run * nbox + k; int x = (lin % 64) * 64, y = (lin / 64) * rows;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                   :: "r"(dst), "l"(&map), "r"(x), "r"(y), "r"(bar) : "memory");
      asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(bar), "r"(k & 1) : "memory");
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
      if (k < 256) tma_t[k] = g1 - g0;
    }
    stop = 1;
  }
}
int main() {
  // chase ring: 64 MB
  size_t S = 64u << 20, n = S / 128; std::vector<unsigned> h(S / 4, 0), ord(n); size_t per = (2u << 20) / 128; std::mt19937 g(5);
  for (size_t p = 0; p < n; p += per) { std::vector<unsigned> blk(per); for (size_t k = 0; k < per; k++) blk[k] = p + k; std::shuffle(blk.begin() + (p == 0), blk.end(), g); for (size_t k = 0; k < per; k++) ord[p + k] = blk[k]; }
  for (size_t k = 0; k < n; k++) h[ord[k] * 32] = ord[(k + 1) % n] * 32;
  unsigned* c; uint4* big; unsigned long long* o; CK(cudaMalloc(&c, S)); CK(cudaMalloc(&big, 256u << 20)); CK(cudaMalloc(&o, 32));
  CK(cudaMemcpy(c, h.data(), S, cudaMemcpyHostToDevice)); CK(cudaMemset(big, 1, 256u << 20));
  // TMA source: 512 MB of fp16, 4096 cols wide (8 KB rows), boxes of 64 cols x rows
  const size_t W = 4096, H = (512u << 20) / (W * 2); unsigned short* src; CK(cudaMalloc(&src, W * H * 2)); CK(cudaMemset(src, 0, W * H * 2));
  cudaStream_t s1, s2; CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  CK(cudaFuncSetAttribute(tmabox, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
  auto bg = [&](int ns) { int z = 0; CK(cudaMemcpyToSymbol(stop, &z, 4)); CK(cudaMemset(o, 0, 32)); CK(cudaDeviceSynchronize());
                          if (ns >= 0) streamer<<<47, 256, 0, s1>>>(big, (256u << 20) / 16, ns, o + 2); for (volatile int d = 0; d < 2000000; d++); };
  for (int ns : {-1, 20000, 5000, 2000, 1000, 500, 200, 0}) {
    bg(ns); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventRecord(e0, s2));
    int hops = 20000; chaser<<<1, 1, 0, s2>>>(c, hops, o); CK(cudaEventRecord(e1, s2)); CK(cudaDeviceSynchronize());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); unsigned long long hh[3]; CK(cudaMemcpy(hh, o, 24, cudaMemcpyDeviceToHost));
    printf("RESULT chase sleep_ns=%d ns_per_hop=%.1f background_GBs=%.1f\n", ns, (double)hh[0] / hops, ns >= 0 ? hh[2] / (ms * 1e-3) / 1e9 : 0.0);
    for (int rows : {32, 64, 128, 256}) {
      int bytes = rows * 64 * 2; CUtensorMap map; cuuint64_t dims[2] = {W, H}; cuuint64_t strides[1] = {W * 2}; cuuint32_t box[2] = {64, (cuuint32_t)rows}; cuuint32_t es[2] = {1, 1};
      CK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, src, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
      static int run = 0;
      bg(ns); CK(cudaEventRecord(e0, s2)); tmabox<<<1, 32, 64 * 1024, s2>>>(map, rows, 200, bytes, run++); CK(cudaEventRecord(e1, s2)); CK(cudaDeviceSynchronize());
      CK(cudaEventElapsedTime(&ms, e0, e1)); CK(cudaMemcpy(hh, o, 24, cudaMemcpyDeviceToHost));
      std::vector<unsigned long long> t(200); CK(cudaMemcpyFromSymbol(t.data(), tma_t, 8 * 200)); std::sort(t.begin(), t.end());
      printf("RESULT tma sleep_ns=%d box_KB=%d median_ns=%llu p10=%llu p90=%llu (background as in the chase line)\n", ns, bytes / 1024, t[100], t[20], t[180]);
    }
  }
}
