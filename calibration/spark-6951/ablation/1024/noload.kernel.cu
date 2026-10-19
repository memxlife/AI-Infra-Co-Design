#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void matmul_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) matmul_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C) {
  const int __tl_bid = (int)(blockIdx.y * gridDim.x + blockIdx.x);
  if ((int)threadIdx.x == 128) { unsigned long long __t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(__t)); __tl_ts[2 * __tl_bid] = __t; }
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) < 32) {
      for (int k = 0; k < 8; ++k) {
        mbarrier[((k & 1) + 2)].wait((((k & 3) >> 1) ^ 1));
        if (tl::tl_shuffle_elect<32>()) {
          #pragma unroll
          for (int i = 0; i < 2; ++i) {
          }
          mbarrier[(k & 1)].arrive();
          #pragma unroll
          for (int i_1 = 0; i_1 < 2; ++i_1) {
          }
        }
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i_2 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 8; ++k_1) {
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      {
        half_t A_local[16];
        half_t B_local[32];
        for (int ki = 0; ki < 8; ++ki) {
          #pragma unroll
          for (int i_3 = 0; i_3 < 2; ++i_3) {
            tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((((k_1 & 1) * 8192) + ((ki >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
          }
          #pragma unroll
          for (int i_4 = 0; i_4 < 4; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((k_1 & 1) * 16384) + (((((int)threadIdx.x) & 127) >> 6) * 8192)) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_5 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_5 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      mbarrier[((k_1 & 1) + 2)].arrive();
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      half_t C_local_cast[2];
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_6 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + ((((((((((((int)blockIdx.y) * 65536) + (((((int)threadIdx.x) & 63) >> 5) * 32768)) + ((i_6 >> 4) * 16384)) + ((i_6 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_6 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128)) = *(uint1*)(C_local_cast + 0);
    }
  }
  if ((int)threadIdx.x == 128) { unsigned long long __t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(__t)); __tl_ts[2 * __tl_bid + 1] = __t; }
}
