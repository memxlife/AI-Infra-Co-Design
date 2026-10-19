// Ablation harness (round 3, step 2c): the TileLang-generated kernel included below, launched with the same tensor maps,
// grid, block and dynamic shared memory TileLang uses, timed with CUDA events (median of REPS) and with per-block
// %globaltimer stamps the patched kernel writes. Build: nvcc -O3 -arch=sm_121a -std=c++20 --expt-relaxed-constexpr -I... -lcuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
__device__ unsigned long long __tl_ts[2 * 65536];
#include "full.kernel.cu"
__global__ void fill(half_t* p, size_t n, unsigned seed) {           // uniform in [-1, 1), one LCG per element
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = half_t((float)(x & 0xffff) / 32768.0f - 1.0f);
  }
}
#define CK(x) do { auto e = (x); if (e != 0) { fprintf(stderr, "error %d at line %d\n", (int)e, __LINE__); exit(1); } } while (0)
static CUtensorMap make_map(void* ptr, unsigned long long d0, unsigned long long d1, unsigned long long stride1, unsigned box0, unsigned box1, int swz) {
  CUtensorMap m; cuuint64_t dims[2] = {d0, d1}; cuuint64_t strides[1] = {stride1}; cuuint32_t box[2] = {box0, box1}; cuuint32_t es[2] = {1, 1};
  CUtensorMapSwizzle sw = swz == 3 ? CU_TENSOR_MAP_SWIZZLE_128B : swz == 2 ? CU_TENSOR_MAP_SWIZZLE_64B : swz == 1 ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE;
  CK(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}
int main(int argc, char** argv) {
  int gx = atoi(argv[1]), gy = atoi(argv[2]), reps = argc > 3 ? atoi(argv[3]) : 50;
  const int M = 32, N = 4096, K = 4096, threads = 256, smem = 81920;
  half_t *A, *B, *C; CK(cudaMalloc(&A, (size_t)M * K * 2)); CK(cudaMalloc(&B, (size_t)K * N * 2)); CK(cudaMalloc(&C, (size_t)M * N * 2));
  fill<<<1024, 256>>>(A, (size_t)M * K, 0x9E3779B9u); fill<<<1024, 256>>>(B, (size_t)K * N, 0x7F4A7C15u); CK(cudaDeviceSynchronize());   // random data: zeros run cooler and faster
  CUtensorMap mA = make_map(A, 4096, 32, 8192, 64, 32, 3);
  CUtensorMap mB = make_map(B, 4096, 4096, 8192, 64, 128, 3);
  CK(cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(gx, gy), block(threads);
  for (int i = 0; i < 5; i++) matmul_kernel<<<grid, block, smem>>>(mA, mB, C);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); std::vector<float> t;
  for (int r = 0; r < reps; r++) { CK(cudaEventRecord(e0)); matmul_kernel<<<grid, block, smem>>>(mA, mB, C); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms); }
  std::sort(t.begin(), t.end());
  // per-block stamps of the last launch: first-start, last-end, mean block duration, max block duration
  int nb = gx * gy; std::vector<unsigned long long> ts(2 * nb); CK(cudaMemcpyFromSymbol(ts.data(), __tl_ts, sizeof(unsigned long long) * 2 * nb));
  unsigned long long t0 = ~0ull, t1 = 0; double sum = 0; unsigned long long mx = 0;
  for (int b = 0; b < nb; b++) { t0 = std::min(t0, ts[2 * b]); t1 = std::max(t1, ts[2 * b + 1]); unsigned long long d = ts[2 * b + 1] - ts[2 * b]; sum += d; mx = std::max(mx, d); }
  if (argc > 4) { FILE* f = fopen(argv[4], "w"); for (int b = 0; b < nb; b++) fprintf(f, "%d %llu %llu\n", b, ts[2 * b] - t0, ts[2 * b + 1] - t0); fclose(f); }
  printf("RESULT grid=%dx%d blocks=%d median_us=%.3f min_us=%.3f span_us=%.3f block_mean_us=%.3f block_max_us=%.3f\n", gx, gy, nb, t[t.size() / 2] * 1e3, t[0] * 1e3, (t1 - t0) / 1e3, sum / nb / 1e3, mx / 1e3);
  return 0;
}
