# On the Spark: skinny sweeps. (a) PR 3290 fragment kernel over FragConfig knobs at M=32 and M=128, N=K=4096, cold and warm;
# (b) the GEMV example's splitk_gemv_vectorized_tvm over (BLOCK_N, reduce_threads) at M=1. One JSON line each to argv[1].
import importlib.util, itertools, json, sys, torch
from tilelang.profiler import do_bench
TL = "/home/lishang/work/research/tilelang-work/tilelang"; sys.path.insert(0, TL)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def warm_us(fn, reps=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3
def cold_us(fn): return do_bench(fn, warmup=5, rep=30, backend="event") * 1e3
out = open(sys.argv[1], "a"); N = K = 4096
pr = load("sm120_nvfp4_w4a16_gemm", "pr3290/sm120_nvfp4_w4a16_gemm.py")
from examples.dequantize_gemm.quantize.nvfp4 import quantize_bf16_to_nvfp4_blockscaled, decode_packed_fp4_e2m1, decode_ue4m3_scale_bytes
torch.manual_seed(0); w_bf16 = (torch.randn(N, K, device="cuda") * 0.02 * 448).to(torch.bfloat16)
packed, _, sb = quantize_bf16_to_nvfp4_blockscaled(w_bf16, return_scale_bytes=True); weight = pr.prepare_nvfp4_weight(packed, sb, 1 / 448.0)
w_ref = decode_packed_fp4_e2m1(packed) * decode_ue4m3_scale_bytes(sb).repeat_interleave(16, dim=1) / 448.0
for M in (32, 128):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16); ref = x.float() @ w_ref.T
    bm = 32 if M == 32 else 64
    for nw, bk, sk, st in itertools.product((2, 4, 8), (32, 64, 128), (1, 2, 4, 8), (2, 3, 4)):
        if (K // bk) % sk: continue
        smem = st * (bm * bk * 2 + nw * (bk // 32) * 576) + 4
        if smem > 99 * 1024: continue
        c = pr.FragConfig(bm, nw, bk, sk, st); rec = {"kind": "frag", "M": M, "cfg": list(c), "smem": smem, "ctas": (N // (32 * nw)) * sk}
        try:
            fn = lambda: pr.launch(x, y, weight, c); fn(); torch.cuda.synchronize(); rel = ((y.float() - ref).norm() / ref.norm()).item()
            rec.update(ok=rel < 5e-3, rel_err=rel, cold_us=cold_us(fn), warm_us=warm_us(fn)); print(f"frag M={M} {tuple(c)} ctas {rec['ctas']}: rel {rel:.1e} cold {rec['cold_us']:.1f} warm {rec['warm_us']:.1f}", flush=True)
        except Exception as e:
            rec.update(ok=False, error=str(e).strip().splitlines()[-1][:200]); print(f"frag M={M} {tuple(c)}: FAILED {rec['error'][:120]}", flush=True)
        out.write(json.dumps(rec) + "\n"); out.flush()
g = load("example_gemv", f"{TL}/examples/gemv/example_gemv.py"); A = torch.randn(K, device="cuda").half(); B = torch.randn(N, K, device="cuda").half(); ref = A.float() @ B.float().T
for bn, rt in itertools.product((1, 2, 4, 8, 16), (4, 8, 16, 32)):
    rec = {"kind": "gemv", "M": 1, "cfg": [bn, rt], "blocks": N // bn, "threads": bn * rt}
    try:
        k = g.splitk_gemv_vectorized_tvm.compile(N=N, K=K, BLOCK_N=bn, reduce_threads=rt); fn = lambda: k(A, B); o = fn(); torch.cuda.synchronize(); rel = ((o.float() - ref).norm() / ref.norm()).item()
        rec.update(ok=rel < 5e-3, rel_err=rel, cold_us=cold_us(fn), warm_us=warm_us(fn)); print(f"gemv ({bn},{rt}) blocks {rec['blocks']} threads {rec['threads']}: rel {rel:.1e} cold {rec['cold_us']:.1f} warm {rec['warm_us']:.1f}", flush=True)
    except Exception as e:
        rec.update(ok=False, error=str(e).strip().splitlines()[-1][:200]); print(f"gemv ({bn},{rt}): FAILED {rec['error'][:120]}", flush=True)
    out.write(json.dumps(rec) + "\n"); out.flush()
