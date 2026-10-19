import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, C, dtype=T.float16, accum_dtype=T.float32):
    # TileLang's examples/gemm_splitk/example_tilelang_gemm_splitk.py with the holes of gemm_basic plus split_k.
    # C is FP32, zeroed by the caller, and every z-block adds its part of K into it with atomics.
    M, N, K = T.const("M, N, K")
    splitK = K // 2

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), accum_dtype)

    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 16), 2, threads=64) as (bx, by, bz):
        A_shared = T.alloc_shared((16, 64), dtype)
        B_shared = T.alloc_shared((64, 128), dtype)
        C_local = T.alloc_fragment((16, 128), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(splitK, 64), num_stages=5):
            T.copy(A[by * 16, bz * splitK + ko * 64], A_shared)
            T.copy(B[bz * splitK + ko * 64, bx * 128], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.atomic_add(C[by * 16, bx * 128], C_local)