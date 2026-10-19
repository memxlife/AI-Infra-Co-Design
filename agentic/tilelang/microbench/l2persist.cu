// Tier 4 payoff: keep a layer's weights in the L2 across other work. A decode-style kernel reads W bytes of weights
// (16-byte loads, all SMs); before every timed run a 64 MB stream (other layers) passes through the L2. Timed with
// and without a stream access-policy window that marks the weights persisting (hitRatio 1) and everything else
// streaming, after setting aside part of the L2 (cudaLimitPersistingL2CacheSize).
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d (%s) line %d\n", (int)e, cudaGetErrorString((cudaError_t)e), __LINE__); exit(1); } } while (0)
__global__ void rd(const int4* __restrict__ p, size_t n16, int* sink) {
  int4 a = make_int4(0, 0, 0, 0);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) { int4 v = p[i]; a.x ^= v.x; a.y ^= v.y; }
  if (a.x == 0x1234567 && a.y == 7) sink[0] = 1;
}
int main() {
  int* sink; CK(cudaMalloc(&sink, 4)); size_t FL = (size_t)64 << 20; int4* fl; CK(cudaMalloc(&fl, FL)); CK(cudaMemset(fl, 1, FL));
  cudaStream_t s; CK(cudaStreamCreate(&s)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int mb : {4, 8, 12, 16}) {
    size_t W = (size_t)mb << 20; int4* w; CK(cudaMalloc(&w, W)); CK(cudaMemset(w, 2, W));
    for (int persist = 0; persist < 2; persist++) {
      CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist ? (size_t)18 << 20 : 0));
      cudaStreamAttrValue a = {};
      if (persist) { a.accessPolicyWindow.base_ptr = w; a.accessPolicyWindow.num_bytes = W; a.accessPolicyWindow.hitRatio = 1.0f;
                     a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; }
      else { a.accessPolicyWindow.num_bytes = 0; }
      CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &a));
      CK(cudaCtxResetPersistingL2Cache());
      std::vector<float> t;
      for (int r = 0; r < 60; r++) {
        rd<<<192, 512, 0, s>>>(fl, FL / 16, sink);                       // other layers' traffic
        CK(cudaEventRecord(e0, s)); rd<<<192, 512, 0, s>>>(w, W / 16, sink); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (r >= 10) t.push_back(ms * 1e3f);
      }
      std::sort(t.begin(), t.end());
      printf("RESULT weights_MB=%d persisting_window=%d median_us=%.1f GBs=%.0f\n", mb, persist, t[t.size() / 2], W / (t[t.size() / 2] * 1e3));
    }
    CK(cudaFree(w));
  }
}
