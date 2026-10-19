// Plain global loads, atomics and register spills on the GB10.
//  chase:   dependent LDG (one lane) through a ring that fits L1 (16 KB) or only L2 (4 MB): latency in cycles
//  ldg:     LDG.E.128, each warp 512 contiguous bytes per instruction, W warps on one SM, from a 16 KB set (L1 hits)
//           or a 4 MB set (L2 hits): cycles per instruction per SM; and 48 blocks for the chip
//  red:     atomicAdd (float, result unused -> RED) per lane to distinct consecutive addresses (128 B per warp), one SM and
//           48 SMs; and every lane of every warp to one address (contention)
//  spill:   a per-thread array indexed at run time lives in local memory: STL and LDL per instruction
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__global__ void chase(const unsigned* __restrict__ next, int hops, unsigned long long* out) {
  unsigned p = 0; long long t0 = clock64();
  for (int i = 0; i < hops; i++) p = __ldcg(next + p) * 0 + next[p];
  long long t1 = clock64(); out[0] = t1 - t0; out[1] = p;
}
template <bool CG>
__global__ void ldg(const int4* __restrict__ src, int words, int iters, unsigned long long* out, int* sink) {
  int4 acc = make_int4(0, 0, 0, 0); int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int nw = words / 32; int base = (blockIdx.x * 977 + warp * 31) % nw;
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      int w = (base + i * 8 + u) % nw; const int4* q = src + w * 32 + lane + (acc.x & 0);   // the &0 keeps every load
      int4 v = CG ? __ldcg(q) : *q;                                                        // CG: cached in L2 only
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
  if (acc.x == 12345 && acc.y == acc.z) sink[0] = acc.w;
}
__global__ void red(float* dst, int iters, int same, unsigned long long* out) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float* p = same ? dst : dst + ((size_t)blockIdx.x * 64 + warp) * 32 * 8 + lane;
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) atomicAdd(same ? p : p + u * 32, 1.0f);
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}
// split-K's C update with S blocks adding into the same 128x128 fp32 tile at the same time: block b adds its 64 KB
// into tile b / S with RED.F32x2, 4 warps, each warp 64 instructions of 256 B (as TileLang's split-K epilogue)
__global__ void redtile(float2* dst, int share, unsigned long long* out) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float2* t = dst + (size_t)(blockIdx.x / share) * 8192;
  __syncthreads(); unsigned long long g0, g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
#pragma unroll 8
  for (int i = 0; i < 64; i++) atomicAdd(t + (warp * 64 + i) * 32 + lane, make_float2(1.0f, 1.0f));
  __syncthreads(); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (threadIdx.x == 0) out[blockIdx.x] = g1 - g0;
}
// split-K's C update in the mma accumulator layout: lane l holds rows l/4 (+8) and columns (l%4)*2..+1 of each 16x8
// piece, so one RED.F32x2 covers 8 rows x 32 bytes (8 lines, one sector each). Each block adds a 128x128 fp32 tile of
// a row-major C (row stride `ld` floats), 4 warps x 64 instructions. Blocks own distinct tiles; repeated `reps` times.
__global__ void redfrag(float* C, int ld, int reps, unsigned long long* out) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int tiles_n = ld / 128; int tr = blockIdx.x / tiles_n, tc = blockIdx.x % tiles_n;
  float* t = C + (size_t)tr * 128 * ld + tc * 128;
  __syncthreads(); long long c0 = clock64();
  for (int r = 0; r < reps; r++) {
#pragma unroll 8
    for (int i = 0; i < 64; i++) {
      int row = (warp * 32 + (i / 16) * 8) % 128 + lane / 4 + ((i / 8) & 1) * 0, col = (i % 16) * 8 + (lane % 4) * 2;
      atomicAdd(reinterpret_cast<float2*>(t + (size_t)row * ld + col), make_float2(1.0f, 1.0f));
    }
  }
  __syncthreads(); long long c1 = clock64();
  if (threadIdx.x == 0) out[blockIdx.x] = c1 - c0;
}
// paired-float atomics as split-K's C update emits them (RED.E.ADD.F32x2): 256 B per warp instruction
__global__ void red2(float2* dst, int iters, unsigned long long* out) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float2* p = dst + ((size_t)blockIdx.x * 64 + warp) * 32 * 8 + lane;
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) atomicAdd(p + u * 32, make_float2(1.0f, 1.0f));
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}
__global__ void spill(int iters, int k, unsigned long long* out, int* sink) {
  int a[64];
#pragma unroll
  for (int j = 0; j < 64; j++) a[j] = threadIdx.x + j;
  __syncthreads(); long long t0 = clock64(); int idx = k;      // the same index in every lane, like a spill slot
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) { idx = (idx + k) & 63; a[idx] += a[(idx + 7) & 63]; }   // run-time index: local memory
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  int s = 0; for (int j = 0; j < 64; j++) s += a[j]; if (s == 7) sink[0] = s;
}
int main() {
  unsigned long long* o; int* sink; CK(cudaMalloc(&o, 8 * 256)); CK(cudaMalloc(&sink, 4)); std::vector<unsigned long long> h(256);
  // pointer chase
  for (int kb : {16, 4096}) {
    int n = kb * 1024 / 4; std::vector<unsigned> ring(n); int stride = 128 / 4 + 1;   // one hop per 132 bytes
    for (int i = 0; i < n; i++) ring[i] = (unsigned)((i + stride) % n);
    unsigned* d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemcpy(d, ring.data(), n * 4, cudaMemcpyHostToDevice));
    chase<<<1, 1>>>(d, 20000, o); CK(cudaDeviceSynchronize()); chase<<<1, 1>>>(d, 20000, o); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), o, 16, cudaMemcpyDeviceToHost)); printf("RESULT chase set_KB=%d cycles_per_hop=%.1f\n", kb, h[0] / 20000.0); CK(cudaFree(d));
  }
  // LDG.128 throughput
  for (int kb : {16, 4096}) {
    int words = kb * 1024 / 16; int4* src; CK(cudaMalloc(&src, (size_t)words * 16)); CK(cudaMemset(src, 1, (size_t)words * 16));
    auto kern = kb == 16 ? ldg<false> : ldg<true>;    // 16 KB: L1 hits; 4 MB: L2 hits (ld.global.cg)
    for (int blocks : {1, 48}) for (int warps : {4, 8, 16, 32}) {
      const int iters = 500;
      kern<<<blocks, 32 * warps>>>(src, words, iters, o, sink); CK(cudaDeviceSynchronize());
      kern<<<blocks, 32 * warps>>>(src, words, iters, o, sink); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), o, 8 * blocks, cudaMemcpyDeviceToHost)); double m = 0; for (int b = 0; b < blocks; b++) m = std::max(m, (double)h[b]);
      double c = m / (iters * 8.0 * warps);
      printf("RESULT ldg128 set_KB=%d blocks=%d warps=%d cycles_per_instr_per_SM=%.2f bytes_per_clk_per_SM=%.0f\n", kb, blocks, warps, c, 512 / c);
    }
    CK(cudaFree(src));
  }
  // atomics
  float* dst; CK(cudaMalloc(&dst, 48 * 64 * 32 * 8 * 4)); CK(cudaMemset(dst, 0, 48 * 64 * 32 * 8 * 4));
  for (int same : {0, 1}) for (int blocks : {1, 48}) for (int warps : {4, 16, 32}) {
    const int iters = same ? 20 : 200;
    red<<<blocks, 32 * warps>>>(dst, iters, same, o); CK(cudaDeviceSynchronize());
    red<<<blocks, 32 * warps>>>(dst, iters, same, o); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), o, 8 * blocks, cudaMemcpyDeviceToHost)); double m = 0; for (int b = 0; b < blocks; b++) m = std::max(m, (double)h[b]);
    double c = m / (iters * 8.0 * warps);
    printf("RESULT red_f32 same_address=%d blocks=%d warps=%d cycles_per_warp_instr_per_SM=%.2f lane_ops_per_clk_chip=%.1f\n", same, blocks, warps, c, 32.0 * blocks / c);
  }
  float2* dst2; CK(cudaMalloc(&dst2, 48 * 64 * 32 * 8 * 8)); CK(cudaMemset(dst2, 0, 48 * 64 * 32 * 8 * 8));
  for (int blocks : {1, 48}) for (int warps : {4, 16, 32}) {
    const int iters = 200;
    red2<<<blocks, 32 * warps>>>(dst2, iters, o); CK(cudaDeviceSynchronize()); red2<<<blocks, 32 * warps>>>(dst2, iters, o); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), o, 8 * blocks, cudaMemcpyDeviceToHost)); double m = 0; for (int b = 0; b < blocks; b++) m = std::max(m, (double)h[b]);
    double c = m / (iters * 8.0 * warps);
    printf("RESULT red_f32x2 blocks=%d warps=%d cycles_per_warp_instr_per_SM=%.2f bytes_per_clk_chip=%.0f\n", blocks, warps, c, 256.0 * blocks / c);
  }
  float2* tiles; CK(cudaMalloc(&tiles, (size_t)512 * 8192 * 8)); CK(cudaMemset(tiles, 0, (size_t)512 * 8192 * 8));
  for (int blocks : {48, 96}) for (int share : {1, 2, 4, 8}) {
    std::vector<double> v;
    for (int r = 0; r < 7; r++) {
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      CK(cudaEventRecord(e0)); redtile<<<blocks, 128>>>(tiles, share, o); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms * 1e3);
    }
    std::sort(v.begin(), v.end());
    printf("RESULT redtile blocks=%d blocks_per_tile=%d event_us=%.2f chip_GBs=%.0f\n", blocks, share, v[3], blocks * 65536.0 / (v[3] * 1e3));
  }
  float* Cf; const int ld = 1024; CK(cudaMalloc(&Cf, (size_t)1024 * ld * 4 * 4)); CK(cudaMemset(Cf, 0, (size_t)1024 * ld * 4 * 4));
  for (int blocks : {1, 2, 48, 96}) {
    const int reps = 20; redfrag<<<blocks, 128>>>(Cf, ld, reps, o); CK(cudaDeviceSynchronize()); redfrag<<<blocks, 128>>>(Cf, ld, reps, o); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), o, 8 * blocks, cudaMemcpyDeviceToHost)); double m = 0; for (int b = 0; b < blocks; b++) m = std::max(m, (double)h[b]);
    double per_instr = m / (reps * 64.0 * 4) * std::min(blocks, 48) / std::min(blocks, 48);
    printf("RESULT redfrag blocks=%d cycles_per_warp_instr_per_block=%.2f chip_bytes_per_clk=%.0f\n", blocks, per_instr, 256.0 * 4 * 64 * reps * blocks / m);
  }
  // spills
  for (int warps : {1, 4, 16}) {
    const int iters = 1000; spill<<<1, 32 * warps>>>(iters, 3, o, sink); CK(cudaDeviceSynchronize()); spill<<<1, 32 * warps>>>(iters, 3, o, sink); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), o, 8, cudaMemcpyDeviceToHost)); printf("RESULT spill warps=%d cycles_per_step_per_SM=%.2f\n", warps, (double)h[0] / (iters * 8.0 * warps));
  }
}
