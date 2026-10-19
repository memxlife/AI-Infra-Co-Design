// mma.sync rate by input type on sm_121a: 8 independent accumulators per warp, 8 warps per SM on all 48 SMs, timed
// with events around the whole kernel (a clock64 read inside the kernel is not ordered after register-only mma
// instructions: the compiler may move it, and a first version read 8x the true rate). Dense FLOP per clock per SM.
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do { auto e = (x); if (e) { fprintf(stderr, "err %d line %d\n", (int)e, __LINE__); exit(1); } } while (0)
#define ITERS 8192
#define MMA_F32(ASM, NA, NB)                                                                                   \
  {                                                                                                            \
    float d[8][4]; unsigned a[NA], b[NB];                                                                      \
    for (int i = 0; i < NA; i++) a[i] = threadIdx.x * 3 + i; for (int i = 0; i < NB; i++) b[i] = threadIdx.x + i; \
    for (int j = 0; j < 8; j++) for (int q = 0; q < 4; q++) d[j][q] = 0.f;                                   \
    __syncthreads(); long long t0 = clock64();                                                                 \
    for (int it = 0; it < ITERS; it++) {                                                                       \
      _Pragma("unroll") for (int j = 0; j < 8; j++) asm volatile(ASM : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3]) : ARGS_##NA##_##NB); \
    }                                                                                                          \
    __syncthreads(); long long t1 = clock64();                                                                 \
    if (threadIdx.x == 0) out[0] = t1 - t0;                                                                    \
    float s = 0; for (int j = 0; j < 8; j++) s += d[j][0] + d[j][3]; if (s == 1234.5f) sink[0] = 1;            \
  }
#define ARGS_4_2 "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
#define ARGS_2_1 "r"(a[0]), "r"(a[1]), "r"(b[0])
#define ARGS_4_2S "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
__global__ void f16f32(unsigned long long* out, int* sink) MMA_F32("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", 4, 2)
__global__ void bf16f32(unsigned long long* out, int* sink) MMA_F32("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", 4, 2)
__global__ void tf32f32(unsigned long long* out, int* sink) MMA_F32("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", 4, 2)
__global__ void e4m3f32(unsigned long long* out, int* sink) MMA_F32("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", 4, 2)
__global__ void e5m2f32(unsigned long long* out, int* sink) MMA_F32("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", 4, 2)
__global__ void e2m1f32(unsigned long long* out, int* sink) MMA_F32("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", 4, 2)
// f16 accumulate: two 32-bit registers of f16x2 per thread
__global__ void f16f16(unsigned long long* out, int* sink) {
  unsigned d[8][2]; unsigned a[4], b[2];
  for (int i = 0; i < 4; i++) a[i] = threadIdx.x * 3 + i; for (int i = 0; i < 2; i++) b[i] = threadIdx.x + i;
  for (int j = 0; j < 8; j++) d[j][0] = d[j][1] = 0;
  __syncthreads(); long long t0 = clock64();
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int j = 0; j < 8; j++) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" : "+r"(d[j][0]), "+r"(d[j][1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  unsigned s = 0; for (int j = 0; j < 8; j++) s += d[j][0]; if (s == 1234) sink[0] = 1;
}
__global__ void s8s32(unsigned long long* out, int* sink) {
  int d[8][4]; unsigned a[4], b[2];
  for (int i = 0; i < 4; i++) a[i] = threadIdx.x * 3 + i; for (int i = 0; i < 2; i++) b[i] = threadIdx.x + i;
  for (int j = 0; j < 8; j++) for (int q = 0; q < 4; q++) d[j][q] = 0;
  __syncthreads(); long long t0 = clock64();
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int j = 0; j < 8; j++) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(d[j][0]), "+r"(d[j][1]), "+r"(d[j][2]), "+r"(d[j][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __syncthreads(); long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  int s = 0; for (int j = 0; j < 8; j++) s += d[j][0]; if (s == 1234) sink[0] = 1;
}
int main() {
  unsigned long long* o; int* sink; CK(cudaMalloc(&o, 8)); CK(cudaMalloc(&sink, 4)); unsigned long long h;
  struct T { const char* n; void (*f)(unsigned long long*, int*); int m, n_, k; } ts[] = {
    {"f16->f32 m16n8k16", f16f32, 16, 8, 16}, {"f16->f16 m16n8k16", f16f16, 16, 8, 16}, {"bf16->f32 m16n8k16", bf16f32, 16, 8, 16},
    {"tf32->f32 m16n8k8", tf32f32, 16, 8, 8}, {"e4m3->f32 m16n8k32", e4m3f32, 16, 8, 32}, {"e5m2->f32 m16n8k32", e5m2f32, 16, 8, 32},
    {"e2m1->f32 m16n8k32 (f8f6f4)", e2m1f32, 16, 8, 32}, {"s8->s32 m16n8k32", s8s32, 16, 8, 32}};
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (auto& t : ts) {
    const int warps = 8, blocks = 48;
    t.f<<<blocks, 32 * warps>>>(o, sink); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); t.f<<<blocks, 32 * warps>>>(o, sink); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); double n = (double)ITERS * 8 * warps;        // mmas per SM
    double flop_s = 2.0 * t.m * t.n_ * t.k * n / (ms * 1e-3);
    printf("RESULT %s dense_TFs_per_SM=%.3f flop_per_clk_per_SM_at_2.54GHz=%.0f cycles_per_mma_per_SMSP=%.2f\n", t.n, flop_s / 1e12, flop_s / 2.54e9, (ms * 1e-3 * 2.54e9) / (n / 4));
  }
}
