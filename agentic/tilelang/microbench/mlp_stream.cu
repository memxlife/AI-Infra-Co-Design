// Memory-level parallelism microbenchmark (round 3, model): the rate a stream of 16-byte loads reaches as a function of
// the bytes a chip keeps in flight. Each thread issues U independent 16-byte loads (addresses coalesced across the
// warp, strided by the whole grid), then consumes them, so U x 16 B per thread are outstanding at once. Bytes in flight
// per SM = resident blocks x threads x U x 16. The buffer is read once per launch: 256 MB (cold: DRAM) or a size that
// fits the L2 (warm: L2), timed with events, median of REPS. A flush kernel writes 256 MB between cold launches.
// Build: nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o mlp_stream mlp_stream.cu
// Run:   ./mlp_stream <bytes> <blocks> <threads> <U> [reps] [flush 0/1]   -> RESULT ... GBs=..
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { auto e = (x); if (e != 0) { fprintf(stderr, "error %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)

template <int U>
__global__ void __launch_bounds__(1024) stream_kernel(const uint4* __restrict__ p, size_t n16, unsigned* out, int passes) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (int pass = 0; pass < passes; pass++) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) v[u] = __ldcs(p + i + u * stride);      // streaming loads, U independent ones in flight
#pragma unroll
    for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) { uint4 v = __ldcs(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  }
  if (acc == 0xFFFFFFFFu) out[blockIdx.x] = acc;                        // never true for random data: keeps the loads
}
__global__ void flush_read_kernel(const uint4* f, size_t n16, unsigned* out) {
  unsigned a = 0; for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) { uint4 v = f[i]; a ^= v.x; }
  if (a == 0xFFFFFFFFu) out[0] = a;
}
__global__ void flush_kernel(uint4* f, size_t n16) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) f[i] = make_uint4(i, i, i, i);
}
static int g_passes = 1;
template <int U> void launch(int blocks, int threads, const uint4* p, size_t n16, unsigned* out) { stream_kernel<U><<<blocks, threads>>>(p, n16, out, g_passes); }

int main(int argc, char** argv) {
  size_t bytes = strtoull(argv[1], 0, 10); int blocks = atoi(argv[2]), threads = atoi(argv[3]), U = atoi(argv[4]);
  int reps = argc > 5 ? atoi(argv[5]) : 20; int flush = argc > 6 ? atoi(argv[6]) : 1; g_passes = argc > 7 ? atoi(argv[7]) : 1;
  size_t n16 = bytes / 16; uint4 *p, *f; unsigned* out;
  CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&f, 256u << 20)); CK(cudaMalloc(&out, 65536 * sizeof(unsigned)));
  std::vector<unsigned> h(bytes / 4); for (size_t i = 0; i < h.size(); i++) h[i] = (unsigned)(i * 2654435761u);
  CK(cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice));
  auto run = [&]() {
    switch (U) { case 1: launch<1>(blocks, threads, p, n16, out); break; case 2: launch<2>(blocks, threads, p, n16, out); break; case 4: launch<4>(blocks, threads, p, n16, out); break;
                 case 8: launch<8>(blocks, threads, p, n16, out); break; case 16: launch<16>(blocks, threads, p, n16, out); break; default: fprintf(stderr, "U\n"); exit(1); }
  };
  for (int i = 0; i < 3; i++) run(); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); std::vector<float> t;
  for (int r = 0; r < reps; r++) {
    if (flush == 1) { flush_kernel<<<1024, 256>>>(f, (256u << 20) / 16); CK(cudaDeviceSynchronize()); }
    if (flush == 2) { flush_read_kernel<<<1024, 256>>>(f, (256u << 20) / 16, out); CK(cudaDeviceSynchronize()); }
    CK(cudaEventRecord(e0)); run(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms);
  }
  std::sort(t.begin(), t.end()); float med = t[t.size() / 2];
  printf("RESULT bytes=%zu blocks=%d threads=%d U=%d flush=%d passes=%d median_us=%.3f min_us=%.3f GBs=%.1f inflight_per_block_B=%d\n", bytes, blocks, threads, U, flush, g_passes, med * 1e3, t[0] * 1e3, (double)bytes * g_passes / (med * 1e-3) / 1e9, threads * U * 16);
  return 0;
}
