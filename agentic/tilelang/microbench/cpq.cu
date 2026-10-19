// Does cp.async issue block when the copy path is busy? W warps each issue K 16-byte cp.async back to back (L2-resident
// sources, distinct lines), with clock64 read before the first and after the K-th issue (no commit/wait in between).
// If issue never blocks, the time is ~K issue slots; if the path has a finite queue, it grows by ~32 cycles per copy
// past the queue depth. Then the round trip of the whole batch (commit + wait 0).
#include <cuda_runtime.h>
#include <cstdio>
template <int K>
__global__ void issue(const char* g, unsigned long long* out) {
  extern __shared__ __align__(128) char s[];
  unsigned sa = (unsigned)__cvta_generic_to_shared(s) + threadIdx.x * 16;
  const char* p = g + threadIdx.x * 16;
  for (int r = 0; r < 2; r++) {                    // second pass: TLB and L2 warm
    __syncthreads();
    long long t0 = clock64();
#pragma unroll
    for (int u = 0; u < K; u++) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa + (u % 8) * blockDim.x * 16), "l"(p + (size_t)u * blockDim.x * 16) : "memory");
    long long t1 = clock64();
    asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_group 0;" ::: "memory");
    long long t2 = clock64();
    if (r == 1 && threadIdx.x % 32 == 0) { out[2 * (threadIdx.x / 32)] = t1 - t0; out[2 * (threadIdx.x / 32) + 1] = t2 - t0; }
  }
}
template <int K> void run(int warps, const char* g, unsigned long long* o) {
  unsigned long long h[64]; int smem = warps * 32 * 16 * 8;
  cudaFuncSetAttribute(issue<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  issue<K><<<1, 32 * warps, smem>>>(g, o); cudaDeviceSynchronize(); cudaMemcpy(h, o, 16 * warps, cudaMemcpyDeviceToHost);
  double iss = 0, tot = 0, mx = 0; for (int w = 0; w < warps; w++) { iss += h[2 * w]; tot += h[2 * w + 1]; mx = mx > h[2 * w] ? mx : h[2 * w]; }
  printf("RESULT warps=%d K=%d issue_cycles_mean=%.0f issue_cycles_max=%.0f roundtrip_mean=%.0f per_copy_issue=%.1f\n", warps, K, iss / warps, mx, tot / warps, iss / warps / K);
}
int main() {
  char* g; unsigned long long* o; cudaMalloc(&g, 64 << 20); cudaMalloc(&o, 1024); cudaMemset(g, 0, 64 << 20);
  for (int w : {1, 4, 8}) { run<1>(w, g, o); run<2>(w, g, o); run<4>(w, g, o); run<8>(w, g, o); run<16>(w, g, o); run<32>(w, g, o); run<64>(w, g, o); }
}
