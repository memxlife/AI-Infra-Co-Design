// Conversion rates on the GB10 (issue #5: a 4-bit GEMV's loop is 32 I2F.F16 per pass, and the simulator charges them
// at the SFU rate, 8 cycles per warp instruction per sub-partition, never measured: alu.cu's I2F row was folded).
//
// Each thread runs 8 independent chains. Per element and loop pass: one conversion, then its result is folded back
// into the chain's input (so nothing is loop-invariant and nothing can be hoisted or dropped), e.g.
//   cvt.rn.f16.s32 h, x;  mov.b32 r, {h, h};  add.s32 x, x, r;
// 32 warps per SM (one 1024-thread block on each of the 48 SMs), timed with events around the whole kernel (clock64
// reads move past register-only instructions). MUFU.EX2 (8 cycles per instruction per sub-partition, sfu.cu) runs in
// the same binary as a ruler: cycles per element = 8 x t(mode) / t(ex2) needs no clock. Read the loop's SASS
// (cuobjdump -sass cvt) to confirm the instruction counts before trusting a rate.
// nvcc -O3 -arch=sm_121a -o cvt cvt.cu && ./cvt
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int E = 8;          // independent chains per thread
constexpr int ITERS = 4096;

template <int MODE>
__global__ void __launch_bounds__(1024, 1) probe(const unsigned* in, unsigned* out) {
  unsigned x[E];
  for (int i = 0; i < E; i++) x[i] = in[(blockIdx.x * blockDim.x + threadIdx.x) * E + i];
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < E; i++) {
      unsigned r;
      if (MODE == 0) {          // scaffold only: the feedback instructions with a move in place of the conversion
        asm volatile("{ .reg .b16 h; mov.b16 h, %1; mov.b32 %0, {h, h}; }" : "=r"(r) : "h"((unsigned short)x[i]));
      } else if (MODE == 1) {   // I2F.F16 from s32
        asm volatile("{ .reg .b16 h; cvt.rn.f16.s32 h, %1; mov.b32 %0, {h, h}; }" : "=r"(r) : "r"(x[i]));
      } else if (MODE == 2) {   // I2F.F16 from u32
        asm volatile("{ .reg .b16 h; cvt.rn.f16.u32 h, %1; mov.b32 %0, {h, h}; }" : "=r"(r) : "r"(x[i]));
      } else if (MODE == 3) {   // int to fp32
        asm volatile("{ .reg .f32 f; cvt.rn.f32.s32 f, %1; mov.b32 %0, f; }" : "=r"(r) : "r"(x[i]));
      } else if (MODE == 4) {   // fp32 to fp16
        asm volatile("{ .reg .b16 h; cvt.rn.f16.f32 h, %1; mov.b32 %0, {h, h}; }" : "=r"(r) : "f"(__uint_as_float(x[i] | 0x3f800000u)));
      } else if (MODE == 5) {   // ruler: MUFU.EX2, 8 cycles per instruction per sub-partition (sfu.cu)
        asm volatile("{ .reg .f32 f; ex2.approx.ftz.f32 f, %1; mov.b32 %0, f; }" : "=r"(r) : "f"(__uint_as_float((x[i] & 0x007fffffu) | 0x3f800000u)));
      } else if (MODE == 6) {   // I2F.F16 and MUFU.EX2 on alternate chains: one unit or two?
        if (i & 1) asm volatile("{ .reg .f32 f; ex2.approx.ftz.f32 f, %1; mov.b32 %0, f; }" : "=r"(r) : "f"(__uint_as_float((x[i] & 0x007fffffu) | 0x3f800000u)));
        else asm volatile("{ .reg .b16 h; cvt.rn.f16.s32 h, %1; mov.b32 %0, {h, h}; }" : "=r"(r) : "r"(x[i]));
      }
      x[i] += r;
    }
  }
  unsigned s = 0;
  for (int i = 0; i < E; i++) s ^= x[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int MODE>
float run(const unsigned* in, unsigned* out) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  probe<MODE><<<48, 1024>>>(in, out); cudaDeviceSynchronize();   // warm-up
  float best = 1e30f;
  for (int r = 0; r < 5; r++) {
    cudaEventRecord(a); probe<MODE><<<48, 1024>>>(in, out); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
  }
  return best;
}

int main() {
  unsigned *in, *out; size_t n = 48 * 1024 * E;
  cudaMalloc(&in, n * 4); cudaMalloc(&out, 48 * 1024 * 4);
  unsigned* h = new unsigned[n]; for (size_t i = 0; i < n; i++) h[i] = (unsigned)(i * 2654435761u) >> 8;
  cudaMemcpy(in, h, n * 4, cudaMemcpyHostToDevice);
  const char* name[] = {"scaffold only (mov)", "I2F.F16 from s32", "I2F.F16 from u32", "int to fp32", "fp32 to fp16",
                        "MUFU.EX2 ruler", "I2F.F16 + MUFU.EX2 alternating"};
  float t[7] = {run<0>(in, out), run<1>(in, out), run<2>(in, out), run<3>(in, out), run<4>(in, out), run<5>(in, out), run<6>(in, out)};
  // per sub-partition: 8 warps x ITERS x E elements
  double elems = 8.0 * ITERS * E;
  printf("mode | ms | cycles per element per sub-partition at 2.52 GHz | by the EX2 ruler (8 x t / t_ex2)\n");
  for (int m = 0; m < 7; m++)
    printf("%-32s | %8.3f | %6.2f | %6.2f\n", name[m], t[m], t[m] * 1e-3 * 2.52e9 / elems, 8.0 * t[m] / t[5]);
  cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) printf("error: %s\n", cudaGetErrorString(e));
  return 0;
}
