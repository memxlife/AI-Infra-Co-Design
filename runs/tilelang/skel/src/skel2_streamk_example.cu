#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/atomic.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tl_matmul_streamk_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(64, 1) tl_matmul_streamk_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared_full_tiles = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared_full_tiles = ((void*)((char*)buf_dyn_shmem + 1024));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  int start_iter[1];
  int end_iter[1];
  float C_local[32];
  start_iter[0] = ((((int)blockIdx.x) * 2) + min(((int)blockIdx.x), 104));
  while (1) {
    if (!((start_iter[0] < (((((int)blockIdx.x) * 2) + min((((int)blockIdx.x) + 1), 104)) + 2)))) { break; }
    end_iter[0] = min(((start_iter[0] + 16) - (start_iter[0] & 15)), (((((int)blockIdx.x) * 2) + min((((int)blockIdx.x) + 1), 104)) + 2));
    int tile_id = (start_iter[0] >> 4);
    int remain_iters = (start_iter[0] & 15);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    if (0 < (end_iter[0] - start_iter[0])) {
      if (0 < (end_iter[0] - start_iter[0])) {
        tl::cp_async_gs_conditional<16>((&(((half_t*)A_shared)[((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int64_t)tile_id) >> (int64_t)3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)512)) + ((((int64_t)start_iter[(int64_t)0]) & (int64_t)15) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8))])), ((((tile_id < 128) && (0 <= tile_id)) && (tile_id < 128)) && (0 <= tile_id)));
      }
      if (0 < (end_iter[0] - start_iter[0])) {
        #pragma unroll
        for (int i_1 = 0; i_1 < 8; ++i_1) {
          tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((i_1 * 512) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((tile_id & 7) * 65536) + (i_1 * 8192)) + ((((int)threadIdx.x) >> 2) * 512)) + ((start_iter[0] & 15) * 32)) + ((((int)threadIdx.x) & 3) * 8))])));
        }
      }
      tl::cp_async_commit();
    }
    for (int k = 0; k < ((end_iter[0] - start_iter[0]) - 1); ++k) {
      __syncthreads();
      tl::cp_async_gs_conditional<16>((&(((half_t*)A_shared)[((((((k + 1) & 1) * 512) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int64_t)tile_id) >> (int64_t)3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)512)) + (((int64_t)k) * (int64_t)32)) + ((((int64_t)start_iter[(int64_t)0]) & (int64_t)15) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32)])), ((((((tile_id < 128) && (0 <= tile_id)) && ((k + (start_iter[0] & 15)) < 15)) && (tile_id < 128)) && (0 <= tile_id)) && ((k + (start_iter[0] & 15)) < 15)));
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        tl::cp_async_gs_conditional<16>((&(((half_t*)B_shared)[(((((((k + 1) & 1) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((((int64_t)tile_id) & (int64_t)7) * (int64_t)65536) + (((int64_t)i_2) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)512)) + (((int64_t)k) * (int64_t)32)) + ((((int64_t)start_iter[(int64_t)0]) & (int64_t)15) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32)])), (((k + (start_iter[0] & 15)) < 15) && ((k + (start_iter[0] & 15)) < 15)));
      }
      tl::cp_async_commit();
      tl::cp_async_wait<1>();
      __syncthreads();
      {
        half_t A_local[8];
        half_t B_local[32];
        for (int ki = 0; ki < 2; ++ki) {
          tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((k & 1) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[0])));
          #pragma unroll
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[((((((((k & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_3 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[(i_3 * 8)])));
          }
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    if (1 <= (end_iter[0] - start_iter[0])) {
      {
        half_t A_local_1[8];
        half_t B_local_1[32];
        for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
          tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((end_iter[0] + 1) - start_iter[0]) & 1) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
          #pragma unroll
          for (int i_4 = 0; i_4 < 4; ++i_4) {
            tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[((((((((((end_iter[0] + 1) - start_iter[0]) & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_4 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_1[(i_4 * 8)])));
          }
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    if ((remain_iters == 0) && ((end_iter[0] % 16) == 0)) {
      if ((tile_id < 128) && (0 <= tile_id)) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 16; ++i_5) {
          *(float2*)(C + ((((((((tile_id >> 3) * 16384) + ((i_5 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + ((tile_id & 7) * 128)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(C_local + (i_5 * 2));
        }
      }
    } else {
      if (0 <= tile_id) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 16; ++i_6) {
          if (tile_id < 128) {
            AtomicAddx2((&(C[((((((((tile_id >> 3) * 16384) + ((i_6 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + ((tile_id & 7) * 128)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_6 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))])), *(float2*)(C_local + (i_6 * 2)));
          }
        }
      }
    }
    start_iter[0] = end_iter[0];
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 8; ++i_7) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i_7 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  __syncthreads();
  tl::cp_async_gs<16>((&(((half_t*)A_shared_full_tiles)[((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int)blockIdx.x) + 20) >> 3) * 8192) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 8))])));
  #pragma unroll
  for (int i_8 = 0; i_8 < 8; ++i_8) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared_full_tiles)[((((i_8 * 512) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((int)blockIdx.x) + 4) & 7) * 65536) + (i_8 * 8192)) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  tl::cp_async_commit();
  for (int k_1 = 0; k_1 < 15; ++k_1) {
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      half_t A_local_2[8];
      half_t B_local_2[32];
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared_full_tiles)[((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_2[0])));
        #pragma unroll
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((half_t*)B_shared_full_tiles)[(((((((((int)threadIdx.x) >> 5) * 2048) + (i_9 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_2[(i_9 * 8)])));
        }
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
    __syncthreads();
    tl::cp_async_gs<16>((&(((half_t*)A_shared_full_tiles)[((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int)blockIdx.x) + 20) >> 3) * 8192) + ((((int)threadIdx.x) >> 2) * 512)) + (k_1 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      tl::cp_async_gs<16>((&(((half_t*)B_shared_full_tiles)[((((i_10 * 512) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((((int)blockIdx.x) + 4) & 7) * 65536) + (i_10 * 8192)) + ((((int)threadIdx.x) >> 2) * 512)) + (k_1 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_3[8];
    half_t B_local_3[32];
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      tl::ptx_ldmatrix_x4((&(((half_t*)A_shared_full_tiles)[((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_3[0])));
      #pragma unroll
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared_full_tiles)[(((((((((int)threadIdx.x) >> 5) * 2048) + (i_11 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_3[(i_11 * 8)])));
      }
      for (int j_3 = 0; j_3 < 4; ++j_3) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 16; ++i_12) {
    *(float2*)(C + (((((((((((int)blockIdx.x) + 20) >> 3) * 16384) + ((i_12 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((((int)blockIdx.x) + 4) & 7) * 128)) + ((((int)threadIdx.x) >> 5) * 64)) + ((i_12 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(C_local + (i_12 * 2));
  }
}

