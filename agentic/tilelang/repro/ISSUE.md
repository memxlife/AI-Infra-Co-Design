# Wrong results: explicit linear shared-memory layout on GEMM operands when copies lower to TMA (sm_121a)

**Commit:** fe6d319c78c0738186fc6e83eb8436ddef65204d (0.1.15+cuda.gitfe6d319c), built from source.
**Host:** NVIDIA GB10 (DGX Spark), `sm_121a`, CUDA 13.0 (nvcc 13.0.r13.0), torch 2.11.0+cu128, Python 3.12.

## What happens

The plain GEMM (`examples/gemm/example_gemm.py` shape) with `T.annotate_layout({A_s: make_linear_layout(A_s), B_s: make_linear_layout(B_s)})`
computes wrong results when the copies into `A_s`/`B_s` are lowered to TMA (the default on `sm_121a`). Max abs error
against `A @ B` is ~20–150 (FP16 in, FP32 accumulate, 1024³ and 32×4096×4096), where the inferred layout gives ~0.06.
With the same kernel and TMA disabled per copy (`tilelang.cuda.language.copy(..., disable_tma=True)`), or with the
producer warp group disabled (`tl.disable_warp_specialized`), the linear layout is correct (and slower, as expected).
`make_swizzled_layout` and the inferred layout are correct with TMA.

In a random sample of 400 legal configurations of this kernel (tiles 64–256, K slices 32–128, 2–5 stages, 128–512
threads, panels 0–16), every one of the 31 wrong results had `linear` layout + TMA + the producer group; 0 of 96 linear
configurations without TMA were wrong.

## Minimal repro

Output on the host above (three layouts x TMA on/off; only linear + TMA is wrong):

```
layout=inferred  TMA=on   max|err|=   0.062  ok   (CUtensorMap in source: True)
layout=inferred  TMA=off  max|err|=   0.062  ok   (CUtensorMap in source: False)
layout=swizzled  TMA=on   max|err|=   0.062  ok   (CUtensorMap in source: True)
layout=swizzled  TMA=off  max|err|=   0.062  ok   (CUtensorMap in source: False)
layout=linear    TMA=on   max|err|=  23.293  WRONG   (CUtensorMap in source: True)
layout=linear    TMA=off  max|err|=   0.062  ok   (CUtensorMap in source: False)
```

<details><summary>linear_layout_tma_wrong.py</summary>

```python
"""Minimal repro: an explicit linear (row-major) shared-memory layout on the GEMM operands gives wrong results when the
copies are lowered to TMA (sm_121a, TileLang fe6d319c). The same kernel with the inferred layout, or with
make_swizzled_layout, or with TMA disabled per copy, is correct.
Run: python linear_layout_tma_wrong.py"""
import torch, tilelang, tilelang.language as T
from tilelang.layout import make_linear_layout, make_swizzled_layout
from tilelang.cuda.language import copy as cuda_copy

M = N = K = 1024; BM, BN, BK, STAGES, THREADS = 128, 128, 32, 3, 128


def build(layout, disable_tma):
    @tilelang.jit
    def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
            A_s = T.alloc_shared((BM, BK), dtype); B_s = T.alloc_shared((BK, BN), dtype); C_l = T.alloc_fragment((BM, BN), accum_dtype)
            if layout == "linear":
                T.annotate_layout({A_s: make_linear_layout(A_s), B_s: make_linear_layout(B_s)})
            elif layout == "swizzled":
                T.annotate_layout({A_s: make_swizzled_layout(A_s), B_s: make_swizzled_layout(B_s)})
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=STAGES):
                cuda_copy(A[by * BM, k * BK], A_s, disable_tma=disable_tma)
                cuda_copy(B[k * BK, bx * BN], B_s, disable_tma=disable_tma)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * BM, bx * BN])
        return C
    return matmul.compile(M=M, N=N, K=K)


torch.manual_seed(0)
A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); ref = (A @ B).float()
for layout in ("inferred", "swizzled", "linear"):
    for disable_tma in (False, True):
        k = build(layout, disable_tma); C = k(A, B).float(); err = (C - ref).abs().max().item()
        src = k.get_kernel_source()
        print(f"layout={layout:9s} TMA={'off' if disable_tma else 'on ':3s}  max|err|={err:8.3f}  {'WRONG' if err > 1 else 'ok'}   (CUtensorMap in source: {'CUtensorMap' in src})")
        if layout == "linear" and not disable_tma:
            open("failing_linear_tma.cu", "w").write(src)        # the generated CUDA of the failing point
```
</details>

<details><summary>Generated CUDA of the failing point (linear layout, TMA on)</summary>

```cuda
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
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_s = ((void*)((char*)buf_dyn_shmem + 24576));
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_l[128];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) < 32) {
      for (int k = 0; k < 32; ++k) {
        mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
        if (tl::tl_shuffle_elect<32>()) {
          mbarrier[(k % 3)].expect_transaction(8192);
          tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((half_t*)A_s)[((k % 3) * 4096)])), (k * 32), (((int)blockIdx.y) * 128));
          mbarrier[(k % 3)].arrive_and_expect_tx(8192);
          tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((half_t*)B_s)[((k % 3) * 4096)])), (((int)blockIdx.x) * 128), (k * 32));
        }
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_l + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      {
        half_t A_local[32];
        half_t B_local[32];
        for (int ki = 0; ki < 2; ++ki) {
          #pragma unroll
          for (int i_1 = 0; i_1 < 4; ++i_1) {
            tl::ptx_ldmatrix_x4((&(((half_t*)A_s)[(((((((k_1 % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (ki * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(A_local[(i_1 * 8)])));
          }
          #pragma unroll
          for (int i_2 = 0; i_2 < 4; ++i_2) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_s)[(((((((k_1 % 3) * 4096) + (ki * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + (((((int)threadIdx.x) & 127) >> 6) * 64)) + (i_2 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), (&(B_local[(i_2 * 8)])));
          }
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_l + ((i_3 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_l + (((i_3 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      mbarrier[((k_1 % 3) + 3)].arrive();
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      half_t C_local_cast[2];
      uint1 __1;
      float2 v_ = *(float2*)(C_l + (i_4 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + ((((((((((((int)blockIdx.y) * 131072) + (((((int)threadIdx.x) & 63) >> 5) * 65536)) + ((i_4 >> 4) * 16384)) + ((i_4 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_4 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128)) = *(uint1*)(C_local_cast + 0);
    }
  }
}
```
</details>

## Expected

Either the explicit linear layout is honoured by the TMA lowering (tensor map without swizzle), or `annotate_layout`
with a layout the TMA path cannot serve raises at lowering time instead of producing a kernel that runs and is wrong.
