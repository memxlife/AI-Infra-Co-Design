// Sustained rate of single instruction kinds: 8 independent chains per thread so latency is hidden, 4..32 warps on one
// SM. Warp instructions per clock per SM (4 sub-partitions; 4.0 = one per clock per sub-partition). The machine code is
// checked: each kind's instruction count in the loop is printed by the run script.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
#define N_IT 512
template <int K>
__global__ void k(int seed, unsigned long long* out, int* sink) {
  unsigned u[8]; float f[8]; __half2 h[8];
#pragma unroll
  for (int j = 0; j < 8; j++) { u[j] = threadIdx.x * 7 + j + seed; f[j] = (float)u[j]; h[j] = __floats2half2_rn(f[j], f[j] + 1.f); }
  __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < N_IT; i++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      if (K == 0) f[j] = fmaf(f[j], 1.0001f, 0.5f);                                      // FFMA
      if (K == 1) u[j] = u[j] * 3u + (unsigned)seed;                                      // IMAD
      if (K == 2) u[j] = u[j] + (unsigned)seed + 5u;                                      // IADD3
      if (K == 3) u[j] = (u[j] ^ (unsigned)seed) & (u[j] | 0x0f0f0f0fu);                 // LOP3
      if (K == 4) u[j] = __funnelshift_r(u[j], u[j], seed);                               // SHF
      if (K == 5) u[j] = __byte_perm(u[j], (unsigned)seed, 0x5410);                       // PRMT
      if (K == 6) h[j] = __hfma2(h[j], h[j], h[j]);                                       // HFMA2
      if (K == 7) { float a = __low2float(h[j]); h[j] = __halves2half2(__float2half_rn(a + 1.f), __high2half(h[j])); }  // HADD2.F32 + F2F
      if (K == 8) { f[j] = __half2float(__ushort_as_half((unsigned short)u[j])); u[j] += (unsigned)f[j]; }             // HADD2.F32 (half->float)
      if (K == 9) { h[j] = __floats2half2_rn(f[j], f[j]); f[j] += __low2float(h[j]); }    // F2FP.PACK_AB
      if (K == 10) { f[j] = (float)(int)u[j]; u[j] += (unsigned)seed; }                   // I2F
      if (K == 11) { u[j] = (unsigned)(int)f[j]; f[j] += 1.f; }                           // F2I
    }
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  unsigned s = 0; for (int j = 0; j < 8; j++) s += u[j] + (unsigned)f[j] + (unsigned)__low2float(h[j]);
  if (s == 0x12345) sink[0] = s;
}
int main() {
  unsigned long long* o; int* sink; CK(cudaMalloc(&o, 8)); CK(cudaMalloc(&sink, 4)); unsigned long long h;
  const char* names[12] = {"FFMA", "IMAD", "IADD3", "LOP3", "SHF", "PRMT", "HFMA2", "F2F.F16.F32+HADD2.F32", "HADD2.F32(h2f)", "F2FP.PACK_AB", "I2F", "F2I"};
  void (*fs[12])(int, unsigned long long*, int*) = {k<0>, k<1>, k<2>, k<3>, k<4>, k<5>, k<6>, k<7>, k<8>, k<9>, k<10>, k<11>};
  for (int x = 0; x < 12; x++) for (int warps : {4, 8, 16, 32}) {
    fs[x]<<<1, 32 * warps>>>(3, o, sink); CK(cudaDeviceSynchronize()); fs[x]<<<1, 32 * warps>>>(3, o, sink); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
    printf("RESULT %s warps=%d loop_warp_steps_per_clk_per_SM=%.3f\n", names[x], warps, (double)N_IT * 8 * warps / h);
  }
}
