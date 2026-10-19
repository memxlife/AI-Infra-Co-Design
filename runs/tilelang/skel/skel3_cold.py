# Cold (do_bench with the 256 MB flush) and warm timings for skeleton 3 kernels, so both readings are on record (Finding 0).
import importlib.util, sys, torch
from tilelang.profiler import do_bench
TL = "/home/lishang/work/research/tilelang-work/tilelang"; sys.path.insert(0, TL)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def warm_us(fn, reps=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3
def cold_us(fn): return do_bench(fn, warmup=10, rep=50, backend="event") * 1e3
N = K = 4096
g = load("example_gemv", f"{TL}/examples/gemv/example_gemv.py")
A = torch.randn(K, device="cuda").half(); B = torch.randn(N, K, device="cuda").half()
k = g.splitk_gemv_vectorized_tvm.compile(N=N, K=K, BLOCK_N=2, reduce_threads=32); print(f"gemv splitk_gemv_vectorized_tvm M=1: cold {cold_us(lambda: k(A, B)):.1f} us | warm {warm_us(lambda: k(A, B)):.1f} us", flush=True)
x1 = torch.randn(1, K, device="cuda").half(); print(f"torch fp16 M=1: cold {cold_us(lambda: x1 @ B.T):.1f} us | warm {warm_us(lambda: x1 @ B.T):.1f} us", flush=True)
pr = load("sm120_nvfp4_w4a16_gemm", "pr3290/sm120_nvfp4_w4a16_gemm.py")
from examples.dequantize_gemm.quantize.nvfp4 import quantize_bf16_to_nvfp4_blockscaled
torch.manual_seed(0); w_bf16 = (torch.randn(N, K, device="cuda") * 0.02 * 448).to(torch.bfloat16)
packed, _, sb = quantize_bf16_to_nvfp4_blockscaled(w_bf16, return_scale_bytes=True); weight = pr.prepare_nvfp4_weight(packed, sb, 1 / 448.0)
for M in (32, 128):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    cfgs = [("heuristic FRAG", pr.FRAG)] + [(str(tuple(c)), c) for c in (pr.FragConfig(min(M, 64), 4, 64, 1, 3), pr.FragConfig(min(M, 64), 4, 64, 2, 3), pr.FragConfig(min(M, 64), 4, 64, 4, 3), pr.FragConfig(min(M, 64), 8, 64, 2, 3), pr.FragConfig(min(M, 64), 2, 64, 2, 3))]
    for label, c in cfgs:
        try:
            fn = lambda: pr.launch(x, y, weight, c); fn(); torch.cuda.synchronize()
            print(f"frag M={M} {label}: cold {cold_us(fn):.1f} us | warm {warm_us(fn):.1f} us", flush=True)
        except Exception as e: print(f"frag M={M} {label}: FAILED {str(e).splitlines()[-1][:120]}", flush=True)
    xb = x; print(f"torch bf16 M={M}: cold {cold_us(lambda: xb @ w_bf16.T):.1f} us | warm {warm_us(lambda: xb @ w_bf16.T):.1f} us", flush=True)
