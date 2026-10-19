import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
        A_shared = T.alloc_shared((64, 32), dtype)
        B_shared = T.alloc_shared((32, 64), dtype)
        C_local = T.alloc_fragment((64, 64), accum_dtype)

        T.use_swizzle(panel_size=16)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
            T.copy(A[by * 64, k * 32], A_shared)
            T.copy(B[k * 32, bx * 64], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * 64, bx * 64])

    return C