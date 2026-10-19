# The C store as a choice: direct (T.copy(C_local, C)) or staged through shared memory (T.copy(C_local, C_shared);
# T.copy(C_shared, C)), TileLang's own example_gemm_autotune / example_gemm_persistent epilogue. Same tiles; correctness,
# warm time, store instructions in the SASS.
import re, subprocess, sys, tempfile, torch, tilelang, tilelang.language as T
from tilelang.cuda.language import copy as cuda_copy

def make(bm, bn, bk, st, th, staged, tma):
    pc = {} if tma else {"tl.disable_warp_specialized": True}
    @tilelang.jit(pass_configs=pc)
    def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=th) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), dtype)
            B_shared = T.alloc_shared((bk, bn), dtype)
            C_local = T.alloc_fragment((bm, bn), accum_dtype)
            if staged:
                C_shared = T.alloc_shared((bm, bn), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=st):
                if tma:
                    T.copy(A[by * bm, k * bk], A_shared); T.copy(B[k * bk, bx * bn], B_shared)
                else:
                    cuda_copy(A[by * bm, k * bk], A_shared, disable_tma=True); cuda_copy(B[k * bk, bx * bn], B_shared, disable_tma=True)
                T.gemm(A_shared, B_shared, C_local)
            if staged:
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * bm, bx * bn])
            else:
                T.copy(C_local, C[by * bm, bx * bn])
        return C
    return matmul

