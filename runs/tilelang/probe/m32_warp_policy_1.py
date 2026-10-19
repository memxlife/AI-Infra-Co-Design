import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy


@tilelang.jit
def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, 256), T.ceildiv(M, 16), threads=256) as (bx, by):
        A_shared = T.alloc_shared((16, 32), dtype)
        B_shared = T.alloc_shared((32, 256), dtype)
        C_local = T.alloc_fragment((16, 256), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 32), num_stages=5):
            T.copy(A[by * 16, k * 32], A_shared)
            T.copy(B[k * 32, bx * 256], B_shared)
            T.gemm(A_shared, B_shared, C_local, policy=GemmWarpPolicy.FullRow)

        T.copy(C_local, C[by * 16, bx * 256])

    return C