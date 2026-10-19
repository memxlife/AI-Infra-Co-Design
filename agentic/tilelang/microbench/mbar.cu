// mbarrier costs on one SM. (1) wake: warp 1 spins until clock64 passes a start time, records it and arrives; warp 0
// waits on the phase (try_wait, as TileLang emits it, or a test_wait spin) and records when it gets through. Same SM, so
// the two clocks agree. (2) arrive: one warp's lane 0 arrives N times on a barrier expecting N: cycles per arrive.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__device__ __forceinline__ void init(unsigned bar, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory"); }
__device__ __forceinline__ void arrive(unsigned bar) { asm volatile("{ .reg .b64 st; mbarrier.arrive.shared::cta.b64 st, [%0]; }" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void arrive_relaxed(unsigned bar) { asm volatile("{ .reg .b64 st; mbarrier.arrive.relaxed.cta.shared::cta.b64 st, [%0]; }" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void wait_try(unsigned bar, int phase) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(bar), "r"(phase) : "memory");
}
__device__ __forceinline__ void wait_test(unsigned bar, int phase) {
  asm volatile("{ .reg .pred p; W: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(bar), "r"(phase) : "memory");
}
template <bool TRY>
__global__ void wake(long long delay, long long* out) {
  __shared__ __align__(8) unsigned long long b; unsigned bar = (unsigned)__cvta_generic_to_shared(&b);
  if (threadIdx.x == 0) init(bar, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); __syncthreads();
  long long start = clock64() + delay;
  // the arriving warp keeps its time in a register and stores it after arriving (an arrive releases earlier stores)
  if (threadIdx.x == 32) { long long t; do { t = clock64(); } while (t < start); long long ta = clock64(); arrive(bar); out[0] = ta; }
  if (threadIdx.x == 33) { long long t; do { t = clock64(); } while (t < start); }
  if (threadIdx.x / 32 == 0) { if (TRY) wait_try(bar, 0); else wait_test(bar, 0); long long t = clock64(); if (threadIdx.x == 0) out[1] = t; }
}
__global__ void arrives(int n, long long* out) {
  __shared__ __align__(8) unsigned long long b; unsigned bar = (unsigned)__cvta_generic_to_shared(&b);
  if (threadIdx.x == 0) init(bar, n);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); __syncthreads();
  if (threadIdx.x == 0) { long long t0 = clock64(); for (int i = 0; i < n; i++) arrive(bar); wait_test(bar, 0); long long t1 = clock64(); out[0] = t1 - t0; }
}
int main() {
  long long* o; CK(cudaMalloc(&o, 16)); long long h[2];
  for (int t = 0; t < 2; t++) {
    std::vector<long long> v;
    for (int r = 0; r < 21; r++) {
      if (t) wake<true><<<1, 64>>>(20000, o); else wake<false><<<1, 64>>>(20000, o);
      CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost)); v.push_back(h[1] - h[0]);
    }
    std::sort(v.begin(), v.end());
    printf("RESULT mbarrier_wake wait=%s median_cycles=%lld min=%lld max=%lld\n", t ? "try_wait" : "test_wait_spin", v[10], v[0], v[20]);
  }
  for (int n : {1, 8, 64, 512}) {
    arrives<<<1, 32>>>(n, o); CK(cudaDeviceSynchronize()); arrives<<<1, 32>>>(n, o); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, o, 8, cudaMemcpyDeviceToHost)); printf("RESULT mbarrier_arrive n=%d cycles=%lld per_arrive=%.1f\n", n, h[0], (double)h[0] / n);
  }
}
