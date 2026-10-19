#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void matmul_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) matmul_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  const int __tl_bid = (int)(blockIdx.y * gridDim.x + blockIdx.x);
  if ((int)threadIdx.x == 0) { unsigned long long __t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(__t)); __tl_ts[2 * __tl_bid] = __t; }
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  float C_local[128];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
  }
  tl::cp_async_commit();
  for (int k = 0; k < 0; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    { }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  { }
  #pragma unroll
  for (int i_11 = 0; i_11 < 0; ++i_11) {
    half_t C_local_cast[2];
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_11 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 262144) + (((((int)threadIdx.x) & 63) >> 5) * 131072)) + ((i_11 >> 3) * 16384)) + ((i_11 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_11 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
  }
  if ((int)threadIdx.x == 0) { unsigned long long __t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(__t)); __tl_ts[2 * __tl_bid + 1] = __t; }
}
