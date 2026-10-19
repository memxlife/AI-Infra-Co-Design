import tilelang
import tilelang.language as T
from tilelang.cuda.language import copy as cuda_copy


@tilelang.jit(pass_configs={'tl.disable_warp_specialized': True})
def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
        A_shared = T.alloc_shared((128, 32), dtype)
        B_shared = T.alloc_shared((32, 128), dtype)
        C_local = T.alloc_fragment((128, 128), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 32), num_stages=4):
            cuda_copy(A[by * 128, k * 32], A_shared, disable_tma=True)
            cuda_copy(B[k * 32, bx * 128], B_shared, disable_tma=True)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * 128, bx * 128])

    return C