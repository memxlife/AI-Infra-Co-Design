// How many stores or atomics an SM takes before issue waits for the drain: one warp (or 4) issues N back-to-back
// RED.F32x2 in the mma accumulator layout (8 rows x 32 B per instruction), RED.F32x2 contiguous, or STG.128 contiguous,
// and reads clock64 right after the last issue (no fence). Cycles against N: flat while the queue takes them, then the
// drain rate.
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
template <int KIND>
__global__ void k(float* C, int n, unsigned long long* out) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float* base = C + (size_t)warp * (1 << 20);
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < n; i++) {
    if (KIND == 0) { int row = (i % 16) * 8 + lane / 4, col = (i / 16) * 8 + (lane % 4) * 2; atomicAdd(reinterpret_cast<float2*>(base + (size_t)row * 1024 + col), make_float2(1.f, 1.f)); }
    if (KIND == 1) atomicAdd(reinterpret_cast<float2*>(base) + i * 32 + lane, make_float2(1.f, 1.f));
    if (KIND == 2) reinterpret_cast<float4*>(base)[i * 32 + lane] = make_float4(i, 1.f, 2.f, 3.f);
  }
  long long t1 = clock64();
  if (lane == 0) out[warp] = t1 - t0;
}
int main() {
  float* C; CK(cudaMalloc(&C, (size_t)8 << 22)); CK(cudaMemset(C, 0, (size_t)8 << 22)); unsigned long long* o; CK(cudaMalloc(&o, 64)); unsigned long long h[8];
  const char* names[3] = {"red_f32x2_frag", "red_f32x2_contig", "stg128_contig"};
  void (*fs[3])(float*, int, unsigned long long*) = {k<0>, k<1>, k<2>};
  for (int f = 0; f < 3; f++) for (int warps : {1, 4}) for (int n : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
    fs[f]<<<1, 32 * warps>>>(C, n, o); CK(cudaDeviceSynchronize()); fs[f]<<<1, 32 * warps>>>(C, n, o); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, o, 8 * warps, cudaMemcpyDeviceToHost)); unsigned long long m = 0; for (int w = 0; w < warps; w++) m = m > h[w] ? m : h[w];
    printf("RESULT %s warps=%d n=%d issue_cycles=%llu per_instr=%.1f\n", names[f], warps, n, m, (double)m / n);
  }
}
