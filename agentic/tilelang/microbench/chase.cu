// Load latency ladder: one thread follows a random cyclic chain of 128-byte-spaced pointers through a buffer of S bytes.
// S below the L1 gives L1 latency, below the L2 gives L2-hit latency, far above gives DRAM latency (cold). ns per hop.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
__global__ void chase(const unsigned* __restrict__ p, int hops, unsigned* out, unsigned long long* ns) {
  unsigned i = 0; unsigned long long g0, g1;
  for (int h = 0; h < 1024; h++) i = p[i];                                  // warm the path (TLB, and the cache for small S)
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  for (int h = 0; h < hops; h++) i = __ldcg(p + i);                         // .cg: skip L1 (L2 or DRAM latency)
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = i; ns[0] = g1 - g0;
}
__global__ void chase_l1(const unsigned* __restrict__ p, int hops, unsigned* out, unsigned long long* ns) {
  unsigned i = 0; unsigned long long g0, g1;
  for (int h = 0; h < 1024; h++) i = p[i];
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  for (int h = 0; h < hops; h++) i = __ldca(p + i);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = i; ns[0] = g1 - g0;
}
int main(int argc, char** argv) {
  size_t S = strtoull(argv[1], 0, 10); int l1 = argc > 2 ? atoi(argv[2]) : 0; int hops = 20000;
  size_t n = S / 128; std::vector<unsigned> order(n); for (size_t k = 0; k < n; k++) order[k] = k;
  std::shuffle(order.begin() + 1, order.end(), std::mt19937(7));
  std::vector<unsigned> h(S / 4, 0); for (size_t k = 0; k < n; k++) h[order[k] * 32] = order[(k + 1) % n] * 32;
  unsigned *p, *o; unsigned long long* ns; cudaMalloc(&p, S); cudaMalloc(&o, 4); cudaMalloc(&ns, 8);
  cudaMemcpy(p, h.data(), S, cudaMemcpyHostToDevice); unsigned long long t;
  if (l1) chase_l1<<<1, 1>>>(p, hops, o, ns); else chase<<<1, 1>>>(p, hops, o, ns);
  cudaDeviceSynchronize(); cudaMemcpy(&t, ns, 8, cudaMemcpyDeviceToHost);
  printf("RESULT S=%zu l1=%d ns_per_hop=%.1f\n", S, l1, (double)t / hops);
}
