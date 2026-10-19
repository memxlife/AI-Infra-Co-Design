// The staged C store's two steps on the GB10.
//  stmatrix: W warps store x4 fragments (512 B per warp instruction) into conflict-free shared memory; cycles per
//            instruction per SM
//  TMA store: each block writes a 256 x 128 fp16 tile (64 KB) from shared memory to global with
//            cp.async.bulk.tensor.2d (boxes of 64 cols x 256 rows), then waits; per-block time and chip rate at 1..96 blocks
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__global__ void stsm(int iters, unsigned long long* out) {
  __shared__ __align__(128) unsigned short buf[16 * 1024];
  unsigned a = (unsigned)__cvta_generic_to_shared(buf) + ((threadIdx.x / 32) % 16) * 2048 + (threadIdx.x % 32) * 16;
  unsigned r0 = threadIdx.x, r1 = r0 + 1, r2 = r0 + 2, r3 = r0 + 3;
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(a + ((i + u) & 3) * 512), "r"(r0 + u), "r"(r1), "r"(r2), "r"(r3) : "memory");
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
}
__device__ unsigned long long spans[256];
template <bool READ>
__global__ void tstore(const __grid_constant__ CUtensorMap map, int tiles_n) {
  extern __shared__ __align__(1024) unsigned short tile[];                 // 256 rows x 128 cols fp16, stored as 2 boxes of 64 cols
  for (int i = threadIdx.x; i < 256 * 128; i += blockDim.x) tile[i] = i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); __syncthreads();
  unsigned long long g0, g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  if (threadIdx.x == 0) {
    int by = blockIdx.x / tiles_n, bx = blockIdx.x % tiles_n;
    for (int b = 0; b < 2; b++) {
      unsigned s = (unsigned)__cvta_generic_to_shared(tile + b * 256 * 64);
      asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" :: "l"(&map), "r"(bx * 128 + b * 64), "r"(by * 256), "r"(s) : "memory");
    }
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    if (READ) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // TileLang's default: smem has been read
    else asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");            // the writes are done
  }
  __syncthreads(); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (threadIdx.x == 0) spans[blockIdx.x] = g1 - g0;
}
int main() {
  unsigned long long* o; CK(cudaMalloc(&o, 8)); unsigned long long h;
  for (int w : {1, 2, 4, 8, 16}) { stsm<<<1, 32 * w>>>(2000, o); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost)); double c = (double)h / (2000.0 * 8 * w); printf("RESULT stmatrix warps=%d cycles_per_instr_per_SM=%.2f bytes_per_clk=%.1f\n", w, c, 512 / c); }
  // C: 4096 x 4096 fp16 so up to 512 tiles of 256 x 128 fit
  const int M = 4096, N = 4096; unsigned short* C; CK(cudaMalloc(&C, (size_t)M * N * 2));
  CUtensorMap map; cuuint64_t dims[2] = {N, M}; cuuint64_t strides[1] = {(cuuint64_t)N * 2}; cuuint32_t box[2] = {64, 256}; cuuint32_t es[2] = {1, 1};
  CK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, C, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  int smem = 256 * 128 * 2;
  CK(cudaFuncSetAttribute(tstore<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); CK(cudaFuncSetAttribute(tstore<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  for (int rd = 0; rd < 2; rd++) for (int blocks : {1, 2, 8, 24, 48, 96}) {
    auto k = rd ? tstore<true> : tstore<false>;
    k<<<blocks, 256, smem>>>(map, N / 128); CK(cudaDeviceSynchronize());
    std::vector<double> v;
    for (int r = 0; r < 9; r++) { k<<<blocks, 256, smem>>>(map, N / 128); CK(cudaDeviceSynchronize()); std::vector<unsigned long long> s(blocks); CK(cudaMemcpyFromSymbol(s.data(), spans, 8 * blocks)); double m = 0; for (auto x : s) m += x; v.push_back(m / blocks); }
    std::sort(v.begin(), v.end()); double us = v[4] / 1e3;
    printf("RESULT tma_store wait=%s blocks=%d block_us=%.2f GBs_per_block=%.1f chip_GBs=%.0f\n", rd ? "read" : "full", blocks, us, 65536 / (us * 1e3), 65536.0 * std::min(blocks, 48) / (us * 1e3));
  }
}
