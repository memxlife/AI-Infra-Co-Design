# Skeletons 1 and 2, unchanged kernel bodies from the TileLang repo, with warp specialization off through the pass config.
import importlib.util, re, sys, time, torch, tilelang, tilelang.language as T
TL = "/home/lishang/work/research/tilelang-work/tilelang"

def timed(label, k, A, B, ref, args=()):
    C = k(A, B, *args); torch.cuda.synchronize()
    out = args[0] if args else C
    err = (out.float() - ref).abs().max().item()
    cold = k.get_profiler().do_bench(backend="event", warmup=25, rep=100, return_mode="median") * 1e3
    s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
    for _ in range(20): k(A, B, *args)
    torch.cuda.synchronize(); s0.record()
    for _ in range(200): k(A, B, *args)
    s1.record(); torch.cuda.synchronize(); warm = s0.elapsed_time(s1) / 200 * 1e3
    src = k.get_kernel_source(); th = re.search(r"__launch_bounds__\((\d+)", src)
    print(f"{label}: max err {err:.3f} | cold {cold:.1f} us | warm {warm:.1f} us | threads {th.group(1) if th else '?'} | TMA {'CUtensorMap' in src} cp.async {'cp.async' in src}", flush=True)
    return src

# --- skeleton 1: examples/gemm/example_gemm.py, body unchanged, decorator with warp specialization off
src = open(f"{TL}/examples/gemm/example_gemm.py").read()
body = src[src.index("def matmul"):src.index("\n\n\ndef main")]
hdr = "import tilelang\nimport tilelang.language as T\n"
open("skel1_asis.py", "w").write(hdr + "@tilelang.jit\n" + body + "\n")
open("skel1_nows.py", "w").write(hdr + "@tilelang.jit(pass_configs={'tl.disable_warp_specialized': True})\n" + body + "\n")
def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m_asis = load("skel1_asis").matmul; m_nows = load("skel1_nows").matmul
M = N = K = 1024
A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); ref = (A @ B).float()
for bm, bn, bk in ((128, 128, 64), (64, 128, 32), (64, 64, 32)):
    k1 = m_asis.compile(M=M, N=N, K=K, block_M=bm, block_N=bn, block_K=bk); timed(f"skel1 as-is TMA+WS {bm}x{bn}x{bk} st3 th128", k1, A, B, ref)
    k2 = m_nows.compile(M=M, N=N, K=K, block_M=bm, block_N=bn, block_K=bk); s2 = timed(f"skel1 WS off (cp.async)  {bm}x{bn}x{bk} st3 th128", k2, A, B, ref)
    if (bm, bn, bk) == (128, 128, 32): open("skel1_cpasync_128x128x32.cu", "w").write(s2)

# --- skeleton 2: examples/gemm_streamk, kernel body unchanged, warp specialization off; at the example's own shape first
sys.path.insert(0, f"{TL}/examples/gemm_streamk")
ssrc = open(f"{TL}/examples/gemm_streamk/example_tilelang_gemm_streamk.py").read()
ssrc = ssrc.replace("@tilelang.jit\ndef tl_matmul_streamk", "@tilelang.jit(pass_configs={'tl.disable_warp_specialized': True})\ndef tl_matmul_streamk")
ssrc = ssrc.split("\n\ndef main")[0]
open("skel2_streamk_nows.py", "w").write(ssrc + "\n")
ns2 = vars(load("skel2_streamk_nows"))
mm, nn, kk = ns2["m"], ns2["n"], ns2["k"]; total_sm = ns2["total_sm"]
print("stream-K example shape", mm, nn, kk, "total_sm (hard-coded)", total_sm, "streamk_tiles", ns2["streamk_tiles"], "sm_patition_factor", ns2["sm_patition_factor"])
A2, B2 = ns2["A"], ns2["B"]; C2 = torch.zeros((mm, nn), device="cuda", dtype=torch.float32)
k3 = ns2["tl_matmul_streamk"].compile(ns2["streamk_tiles"], ns2["BLOCK_SIZE_M"], ns2["BLOCK_SIZE_N"], ns2["BLOCK_SIZE_K"], False, True, T.float16, T.float32, T.float32, 2, 64) if False else None
try:
    fn = ns2["tl_matmul_streamk"]
    k3 = fn.compile(M=mm, N=nn, K=kk, streamk_tiles=ns2["streamk_tiles"], block_M=ns2["BLOCK_SIZE_M"], block_N=ns2["BLOCK_SIZE_N"], block_K=ns2["BLOCK_SIZE_K"], trans_A=False, trans_B=True, dtypeAB=T.float16, dtypeC=T.float32, accum_dtype=T.float32, num_stages=2, threads=64)
    ref2 = (A2 @ B2.T).float()
    def run_sk(): C2.zero_(); k3(A2, B2, C2)
    run_sk(); torch.cuda.synchronize(); err = (C2 - ref2).abs().max().item()
    s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
    for _ in range(20): k3(A2, B2, C2)
    torch.cuda.synchronize(); s0.record()
    for _ in range(200): k3(A2, B2, C2)
    s1.record(); torch.cuda.synchronize(); warm = s0.elapsed_time(s1) / 200 * 1e3
    src3 = k3.get_kernel_source(); print(f"skel2 stream-K WS off at {mm}x{nn}x{kk}: max err {err:.3f} | warm {warm:.1f} us (incl. no zeroing) | TMA {'CUtensorMap' in src3} cp.async {'cp.async' in src3}")
    open("skel2_streamk_example.cu", "w").write(src3)
except Exception as e:
    import traceback; print("skel2 stream-K WS off: FAILED", traceback.format_exc()[-900:])
