import tilelang
import tilelang.language as T
from tilelang.cuda.language import copy as cuda_copy


@tilelang.jit(pass_configs={'tl.disable_warp_specialized': True})
def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=256) as (bx, by):
        A_shared = T.alloc_shared((64, 128), dtype)
        B_shared = T.alloc_shared((128, 64), dtype)
        C_local = T.alloc_fragment((64, 64), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 128), num_stages=3):
            cuda_copy(A[by * 64, k * 128], A_shared, disable_tma=True)
            cuda_copy(B[k * 128, bx * 64], B_shared, disable_tma=True)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * 64, bx * 64])

    return C