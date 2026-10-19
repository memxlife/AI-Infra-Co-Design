// L2-resident 16-byte loads (ld.global.cg) where a warp's 32 lanes are split over R rows `stride` bytes apart (32/R
// lanes per row, contiguous within a row), as a GEMV with BLOCK_N = R reads B. 48 blocks x 8 warps, each warp walking
// its own rows along k. Chip rate against R and the stride: a power-of-two stride shows whether rows pile onto the
// same L2 slices.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__global__ void rs(const char* __restrict__ base, int rows_per_warp, int stride, int steps, unsigned long long* out, int* sink) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32, gw = blockIdx.x * (blockDim.x / 32) + warp;
  int lpr = 32 / rows_per_warp; int r = lane / lpr, c = lane % lpr;
  const char* row = base + (size_t)(gw * rows_per_warp + r) * stride;
  int4 acc = make_int4(0, 0, 0, 0);
  __syncthreads(); long long t0 = clock64();
  for (int s = 0; s < steps; s += 8) {
    int4 v[8];
#pragma unroll
    for (int u = 0; u < 8; u++) v[u] = __ldcg(reinterpret_cast<const int4*>(row + ((size_t)(s + u) * lpr * 16 + c * 16) % stride));   // 8 in flight
#pragma unroll
    for (int u = 0; u < 8; u++) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
  if (acc.x == 1234 && acc.y == acc.z) sink[0] = acc.w;
}
int main() {
  const int blocks = 48, warps = 16; unsigned long long* o; int* sink; CK(cudaMalloc(&o, 8 * blocks)); CK(cudaMalloc(&sink, 4));
  std::vector<unsigned long long> h(blocks);
  for (int stride : {4096, 8192, 4160}) {
    for (int R : {1, 2, 4, 8, 16, 32}) {
      // the set: blocks*warps*R rows of `stride` bytes; keep it L2-resident (<= 8 MB) by limiting the walk to the row
      size_t bytes = (size_t)blocks * warps * R * stride; if (bytes > (size_t)12 << 20) { printf("RESULT skip stride=%d R=%d\n", stride, R); continue; }
      char* b; CK(cudaMalloc(&b, bytes)); CK(cudaMemset(b, 1, bytes));
      int steps = 2000;
      rs<<<blocks, 32 * warps>>>(b, R, stride, steps, o, sink); CK(cudaDeviceSynchronize());
      rs<<<blocks, 32 * warps>>>(b, R, stride, steps, o, sink); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), o, 8 * blocks, cudaMemcpyDeviceToHost)); double m = 0; for (auto x : h) m = std::max(m, (double)x);
      printf("RESULT rowsplit stride=%d rows_per_warp=%d set_MB=%.1f cycles_per_instr_per_SM=%.2f chip_B_per_clk=%.0f\n", stride, R, bytes / 1048576.0, m / (steps * warps), 512.0 * steps * warps * blocks / m);
      CK(cudaFree(b));
    }
  }
}
