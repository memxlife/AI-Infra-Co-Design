// Launch costs on the GB10. (1) empty kernels back-to-back from one stream: microseconds per launch, by grid size;
// (2) the same 100 launches as one CUDA graph; (3) a kernel with ~5 us of work back-to-back: time per kernel minus its
// own span (from %globaltimer inside) = the gap between kernels; (4) one launch alone: host call to kernel start.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__global__ void empty() {}
__device__ unsigned long long span_t[2];
__global__ void work(int iters, float* o, int rec) {
  unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  float f = threadIdx.x; for (int i = 0; i < iters; i++) f = fmaf(f, 1.0001f, 0.5f);
  if (f == 1.2345f) o[0] = f;
  __syncthreads(); unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (rec && threadIdx.x == 0 && blockIdx.x == 0) { span_t[0] = g0; span_t[1] = g1; }
}
__global__ void stamp(unsigned long long* t) { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); t[0] = g; }
int main() {
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); float ms; float* o; CK(cudaMalloc(&o, 4));
  for (int blocks : {1, 48, 1024}) {
    for (int i = 0; i < 100; i++) empty<<<blocks, 128>>>(); CK(cudaDeviceSynchronize());
    const int n = 10000; CK(cudaEventRecord(e0)); for (int i = 0; i < n; i++) empty<<<blocks, 128>>>(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1)); printf("RESULT back_to_back empty blocks=%d us_per_launch=%.2f\n", blocks, ms * 1e3 / n);
  }
  for (int blocks : {1, 48, 1024}) {
    cudaStream_t s; CK(cudaStreamCreate(&s)); cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)); for (int i = 0; i < 100; i++) empty<<<blocks, 128, 0, s>>>(); CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0)); CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(e0, s)); for (int r = 0; r < 100; r++) CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1)); printf("RESULT graph empty blocks=%d us_per_kernel=%.2f\n", blocks, ms * 1e3 / 10000);
  }
  for (int blocks : {48, 1024}) {
    int iters = 12000;   // ~5 us of FMA chain per thread
    work<<<blocks, 128>>>(iters, o, 1); CK(cudaDeviceSynchronize()); unsigned long long sp[2]; CK(cudaMemcpyFromSymbol(sp, span_t, 16));
    const int n = 2000; CK(cudaEventRecord(e0)); for (int i = 0; i < n; i++) work<<<blocks, 128>>>(iters, o, 0); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1)); double per = ms * 1e3 / n, own = (sp[1] - sp[0]) / 1e3;
    printf("RESULT back_to_back work blocks=%d us_per_kernel=%.2f block0_span_us=%.2f gap_us=%.2f\n", blocks, per, own, per - own);
  }
  unsigned long long* t; CK(cudaMallocManaged(&t, 8)); std::vector<double> lat;
  for (int r = 0; r < 50; r++) {
    CK(cudaDeviceSynchronize()); unsigned long long h0; stamp<<<1, 1>>>(t); CK(cudaDeviceSynchronize()); h0 = t[0];
    auto c0 = std::chrono::steady_clock::now(); stamp<<<1, 1>>>(t); CK(cudaDeviceSynchronize()); auto c1 = std::chrono::steady_clock::now();
    lat.push_back(std::chrono::duration<double, std::micro>(c1 - c0).count()); (void)h0;
  }
  std::sort(lat.begin(), lat.end()); printf("RESULT single launch+sync host_us median=%.1f min=%.1f\n", lat[25], lat[0]);
}
