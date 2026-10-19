// Which instruction kinds share a pipe: 4 chains of kind A and 4 of kind B per thread, 32 warps on one SM. Two kinds on
// one half-rate pipe total 2 warp instructions per clock per SM; on different pipes up to 4 (the issue limit).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
#define N_IT 512
__device__ __forceinline__ unsigned op(int kind, unsigned u, unsigned s, float& f, __half2& h) {
  switch (kind) {
    case 0: f = fmaf(f, 1.0001f, 0.5f); return u;                                  // FFMA
    case 1: return u * 3u + s;                                                      // IMAD
    case 2: return (u ^ s) & (u | 0x0f0f0f0fu);                                     // LOP3
    case 3: h = __hfma2(h, h, h); return u;                                         // HFMA2
    case 4: return __byte_perm(u, s, 0x5410);                                       // PRMT
    case 5: { f += __half2float(__ushort_as_half((unsigned short)u)); return u + 1u; } // HADD2.F32 (+FADD)
  }
  return u;
}
template <int A, int B>
__global__ void k(int seed, unsigned long long* out, int* sink) {
  unsigned u[8]; float f[8]; __half2 h[8];
#pragma unroll
  for (int j = 0; j < 8; j++) { u[j] = threadIdx.x * 7 + j + seed; f[j] = (float)u[j]; h[j] = __floats2half2_rn(f[j], 1.f); }
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < N_IT; i++) {
#pragma unroll
    for (int j = 0; j < 8; j++) u[j] = op(j < 4 ? A : B, u[j], (unsigned)seed, f[j], h[j]);
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  unsigned s = 0; for (int j = 0; j < 8; j++) s += u[j] + (unsigned)f[j] + (unsigned)__low2float(h[j]);
  if (s == 0x12345) sink[0] = s;
}
int main() {
  unsigned long long* o; int* sink; CK(cudaMalloc(&o, 8)); CK(cudaMalloc(&sink, 4)); unsigned long long hc;
  const char* nm[6] = {"FFMA", "IMAD", "LOP3", "HFMA2", "PRMT", "HADD2.F32+FADD"};
#define RUN(a, b) { k<a, b><<<1, 1024>>>(3, o, sink); CK(cudaDeviceSynchronize()); k<a, b><<<1, 1024>>>(3, o, sink); CK(cudaDeviceSynchronize()); \
    CK(cudaMemcpy(&hc, o, 8, cudaMemcpyDeviceToHost)); printf("RESULT mix %s + %s cycles=%llu steps_per_clk=%.3f\n", nm[a], nm[b], hc, (double)N_IT * 8 * 32 / hc); }
  RUN(1, 2) RUN(1, 0) RUN(2, 0) RUN(3, 2) RUN(3, 1) RUN(3, 0) RUN(4, 1) RUN(4, 2) RUN(5, 1) RUN(5, 2) RUN(1, 1) RUN(2, 2) RUN(0, 0)
}
