# Skeleton 3: examples/gemv (unchanged) at M=1, N=K=4096; PR 3290 fragment kernel (unchanged) at M=32 and M=128, N=K=4096.
import importlib.util, sys, torch, tilelang, tilelang.language as T
TL = "/home/lishang/work/research/tilelang-work/tilelang"; sys.path.insert(0, TL); sys.path.insert(0, f"{TL}/examples/gemv")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def warm_us(fn, reps=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3
N = K = 4096
g = load("example_gemv", f"{TL}/examples/gemv/example_gemv.py")
A = torch.randn(K, device="cuda").half(); B = torch.randn(N, K, device="cuda").half(); ref = (A.float() @ B.float().T)
for label, fn, cfg in [("splitk_gemv_vectorized", g.splitk_gemv_vectorized, dict(BLOCK_N=2, reduce_threads=32)),
                       ("splitk_gemv_vectorized_tvm", g.splitk_gemv_vectorized_tvm, dict(BLOCK_N=2, reduce_threads=32)),
                       ("splitk_gemv", g.splitk_gemv, dict(BLOCK_N=32, BLOCK_K=32, reduce_threads=32)),
                       ("gemv_alloc_reducer", g.gemv_alloc_reducer, dict(block_M=128, block_N=128))]:
    try:
        if label == "gemv_alloc_reducer":
            k = fn.compile(M=N, N=K, **cfg); run = lambda: k(B, A)
        else:
            k = fn.compile(N=N, K=K, **cfg); run = lambda: k(A, B)
        out = run(); torch.cuda.synchronize(); err = (out.float() - ref).abs().max().item(); rel = ((out.float() - ref).norm() / ref.norm()).item()
        cold = k.get_profiler().do_bench(backend="event", warmup=25, rep=100, return_mode="median") * 1e3 if label != "gemv_alloc_reducer" else float("nan")
        us = warm_us(run); print(f"skel3 gemv {label} {cfg} M=1 N=K=4096: max err {err:.3f} rel {rel:.2e} | cold {cold:.1f} us | warm {us:.1f} us | B stream {N*K*2/us/1e3:.0f} GB/s", flush=True)
    except Exception as e:
        import traceback; print(f"skel3 gemv {label}: FAILED " + traceback.format_exc().strip().splitlines()[-1], flush=True)
x1 = torch.randn(1, K, device="cuda").half(); us = warm_us(lambda: x1 @ B.T); print(f"torch fp16 matmul M=1 N=K=4096: warm {us:.1f} us | {N*K*2/us/1e3:.0f} GB/s", flush=True)

# PR 3290 fragment kernel, unchanged module (W4A16: bf16 activations, NVFP4 weights)
pr = load("sm120_nvfp4_w4a16_gemm", "pr3290/sm120_nvfp4_w4a16_gemm.py")
from examples.dequantize_gemm.quantize.nvfp4 import decode_packed_fp4_e2m1, decode_ue4m3_scale_bytes, quantize_bf16_to_nvfp4_blockscaled
torch.manual_seed(0); gs = 448.0
w_bf16 = (torch.randn(N, K, device="cuda") * 0.02 * gs).to(torch.bfloat16)
packed, _, scale_bytes = quantize_bf16_to_nvfp4_blockscaled(w_bf16, return_scale_bytes=True)
weight = pr.prepare_nvfp4_weight(packed, scale_bytes, 1.0 / gs)
w_ref = decode_packed_fp4_e2m1(packed) * decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1) * (1.0 / gs)
for M in (32, 128):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    cfg = pr.frag_config(N, K, min(M, 64), 48); ref = x.float() @ w_ref.T
    try:
        pr.launch(x, y, weight, pr.FRAG); torch.cuda.synchronize(); rel = ((y.float() - ref).norm() / ref.norm()).item()
        us = warm_us(lambda: pr.launch(x, y, weight, pr.FRAG)); passes = -(-M // 64)
        wb = (N * K // 2 + N * K // 16) * passes
        print(f"skel3 pr3290 frag M={M} N=K=4096 cfg(per 64-row pass)={tuple(cfg)} passes={passes}: rel err {rel:.2e} | warm {us:.1f} us | weight bytes {wb/1e6:.1f} MB -> {wb/us/1e3:.0f} GB/s | {2*M*N*K/us/1e6:.1f} TFLOP/s", flush=True)
        for alt in [pr.FragConfig(min(M,64), 8, 64, 1, 3), pr.FragConfig(min(M,64), 4, 64, 1, 3), pr.FragConfig(min(M,64), 4, 64, 4, 3)]:
            try:
                pr.launch(x, y, weight, alt); torch.cuda.synchronize(); rel = ((y.float() - ref).norm() / ref.norm()).item(); us = warm_us(lambda: pr.launch(x, y, weight, alt))
                print(f"   alt {tuple(alt)}: rel err {rel:.2e} | warm {us:.1f} us", flush=True)
            except Exception as e: print(f"   alt {tuple(alt)}: FAILED {str(e).splitlines()[-1][:200]}", flush=True)
        us_t = warm_us(lambda: x @ w_bf16.T); print(f"   torch bf16 matmul M={M}: warm {us_t:.1f} us", flush=True)
    except Exception as e:
        import traceback; print(f"skel3 pr3290 frag M={M}: FAILED " + traceback.format_exc()[-1500:], flush=True)
