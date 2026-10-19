import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 32), threads=128) as (bx, by):
        A_shared = T.alloc_shared((32, 128), dtype)
        B_shared = T.alloc_shared((128, 128), dtype)
        C_local = T.alloc_fragment((32, 128), accum_dtype)

        T.use_swizzle(panel_size=4)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 128), num_stages=2):
            T.copy(A[by * 32, k * 128], A_shared)
            T.copy(B[k * 128, bx * 128], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * 32, bx * 128])

    return C