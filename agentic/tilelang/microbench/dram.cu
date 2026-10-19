// DRAM behavior on the GB10.
//  load: one block chases pointers through 64 MB (random within each 2 MB page, pages in order: TLB hits, L2 misses)
//        while N other blocks stream-read a separate 256 MB buffer; ns per hop and the streamers' GB/s
//  mix:  48 x 1024 threads read R and write W bytes per element (read-only, write-only, copy, 3:1); total GB/s
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
__device__ volatile int stop;
__global__ void streamer(const uint4* __restrict__ p, size_t n16, unsigned long long* bytes) {
  unsigned acc = 0; unsigned long long b = 0; size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  while (!stop) {
    for (int u = 0; u < 64; u++) { uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i)); acc ^= v.x; i += st; if (i >= n16) i -= n16; b += 16; }
  }
  if (acc == 1) bytes[1] = acc;
  atomicAdd(bytes, b);
}
__global__ void chaser(const unsigned* __restrict__ c, int hops, unsigned long long* out) {
  unsigned i = 0; unsigned long long g0, g1;
  for (int h = 0; h < 2000; h++) i = __ldcg(c + i);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  for (int h = 0; h < hops; h++) i = __ldcg(c + i);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = g1 - g0; out[1] = i; stop = 1;
}
__global__ void mix(uint4* __restrict__ rd, uint4* __restrict__ wr, size_t n16, int r, int w, unsigned* o) {
  unsigned acc = 0; size_t st = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += st) {
    for (int k = 0; k < r; k++) { uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(rd + i + k * n16)); acc ^= v.x; }
    for (int k = 0; k < w; k++) asm volatile("st.global.cg.v4.u32 [%0], {%1,%1,%1,%1};" :: "l"(wr + i + k * n16), "r"(acc + k));
  }
  if (acc == 7) o[0] = acc;
}
int main(int argc, char** argv) {
  if (!strcmp(argv[1], "load")) {
    size_t S = 64u << 20; size_t n = S / 128; std::vector<unsigned> h(S / 4, 0), ord(n);
    size_t per = (2u << 20) / 128; std::mt19937 g(5);
    for (size_t p = 0; p < n; p += per) { std::vector<unsigned> blk(per); for (size_t k = 0; k < per; k++) blk[k] = p + k; std::shuffle(blk.begin() + (p == 0), blk.end(), g); for (size_t k = 0; k < per; k++) ord[p + k] = blk[k]; }
    for (size_t k = 0; k < n; k++) h[ord[k] * 32] = ord[(k + 1) % n] * 32;
    unsigned* c; uint4* big; unsigned long long* o; CK: cudaMalloc(&c, S); cudaMalloc(&big, 256u << 20); cudaMalloc(&o, 32);
    cudaMemcpy(c, h.data(), S, cudaMemcpyHostToDevice); cudaMemset(big, 1, 256u << 20);
    cudaStream_t s1, s2; cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    for (int N : {0, 4, 8, 16, 24, 32, 40, 47}) {
      int zero = 0; cudaMemcpyToSymbol(stop, &zero, 4); cudaMemset(o, 0, 32); cudaDeviceSynchronize();
      unsigned long long g0 = 0;
      if (N) streamer<<<N, 1024, 0, s1>>>(big, (256u << 20) / 16, o + 2);
      for (volatile int d = 0; d < 2000000; d++);
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, s2);
      int hops = 40000; chaser<<<1, 1, 0, s2>>>(c, hops, o); cudaEventRecord(e1, s2); cudaDeviceSynchronize();
      float ms; cudaEventElapsedTime(&ms, e0, e1); unsigned long long hh[3]; cudaMemcpy(hh, o, 24, cudaMemcpyDeviceToHost);
      printf("RESULT load streamers=%d ns_per_hop=%.1f streamer_GBs=%.1f\n", N, (double)hh[0] / hops, N ? hh[2] / (ms * 1e-3) / 1e9 : 0.0);
    }
  } else {
    size_t n16 = (64u << 20) / 16; uint4 *rd, *wr; unsigned* o; cudaMalloc(&rd, 4 * (64u << 20)); cudaMalloc(&wr, 4 * (64u << 20)); cudaMalloc(&o, 4);
    cudaMemset(rd, 1, 4 * (64u << 20));
    int cfg[][2] = {{1, 0}, {0, 1}, {1, 1}, {3, 1}, {1, 3}};
    for (auto& c2 : cfg) {
      mix<<<48, 1024>>>(rd, wr, n16, c2[0], c2[1], o); cudaDeviceSynchronize();
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
      for (int r = 0; r < 5; r++) mix<<<48, 1024>>>(rd, wr, n16, c2[0], c2[1], o);
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
      double bytes = 5.0 * (c2[0] + c2[1]) * (64u << 20);
      printf("RESULT mix read=%d write=%d total_GBs=%.1f\n", c2[0], c2[1], bytes / (ms * 1e-3) / 1e9);
    }
  }
}
