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
