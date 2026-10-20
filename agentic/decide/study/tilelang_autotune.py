# TileLang's own result for each benchmark shape, on the Spark:
#  default:   examples/gemm/example_gemm.py as written (128x128x32, 3 stages, 128 threads, direct C store), fp16
#  autotuned: TileLang's AutoTuner over the 144-config grid of examples/gemm/example_gemm_autotune.py
#             (block_M/N {64,128,256}, block_K {32,64}, num_stages {0..3}, threads {128,256}, swizzle panel 10 on/off,
#             staged C store as the example writes it), on the same fp16 A (MxK), B (KxN) problem as every other column;
#             the AutoTuner times with TileLang's do_bench (event backend) as it ships.
# Writes the autotuner's pick and its own latency; the study re-times every pick with one shared protocol.
# python tilelang_autotune.py OUT.json
import itertools, json, sys, time, torch, tilelang, tilelang.language as T
from tilelang.autotuner import AutoTuner

SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192), (4096, 4096, 1024), (32, 4096, 4096), (128, 4096, 4096)]

def configs():
    return [{"block_M": a, "block_N": b, "block_K": c, "num_stages": d, "thread_num": e, "enable_rasteration": f}
            for a, b, c, d, e, f in itertools.product([64, 128, 256], [64, 128, 256], [32, 64], [0, 1, 2, 3], [128, 256], [True, False])]

def kernel_factory(M, N, K):
    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None, enable_rasteration=None):
        dtype, accum = T.float16, T.float32
        @T.prim_func
        def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])
        return main
    return kernel

def ref_program(A, B): return (A.float() @ B.float()).half()

out = {}
for (M, N, K) in SHAPES:
    t0 = time.time()
    try:
        at = (AutoTuner.from_kernel(kernel=kernel_factory(M, N, K), configs=configs())
              .set_compile_args(out_idx=[-1], target="auto")
              .set_profile_args(supply_type=tilelang.TensorSupplyType.Auto, ref_prog=ref_program, skip_check=False, backend="event"))
        res = at.run(warmup=3, rep=20)
        cfg = dict(res.config); out[f"{M}x{N}x{K}"] = {"config": cfg, "latency_ms_tilelang": float(res.latency), "tune_s": round(time.time() - t0, 1)}
    except Exception as e:
        out[f"{M}x{N}x{K}"] = {"error": str(e).strip().splitlines()[-1][:300], "tune_s": round(time.time() - t0, 1)}
    print(f"{M}x{N}x{K}", json.dumps(out[f"{M}x{N}x{K}"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
