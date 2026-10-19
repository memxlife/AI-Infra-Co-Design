# Skeleton 1: TileLang's examples/gemm/example_gemm.py kernel, unchanged, with warp specialization off (pass config): cp.async loads,
# every warp computes. Run at 1024^3 with the example's own tile (128x128x32, 3 stages, 128 threads). Timed cold (do_bench) and warm (events, back-to-back).
import importlib.util, sys, time, torch, tilelang, tilelang.language as T
sys.path.insert(0, "/home/lishang/work/research/tilelang-work/tilelang/examples/gemm")
spec = importlib.util.spec_from_file_location("ex", "/home/lishang/work/research/tilelang-work/tilelang/examples/gemm/example_gemm.py"); ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)

def run(label, fn, M, N, K, **holes):
    k = fn.compile(M=M, N=N, K=K, **holes)
    A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); C = k(A, B); torch.cuda.synchronize()
    err = (C.float() - (A @ B).float()).abs().max().item()
    cold = k.get_profiler().do_bench(backend="event", warmup=25, rep=100, return_mode="median") * 1e3
    s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
    for _ in range(20): k(A, B)
    torch.cuda.synchronize(); s0.record()
    for _ in range(200): k(A, B)
    s1.record(); torch.cuda.synchronize(); warm = s0.elapsed_time(s1) / 200 * 1e3
    src = k.get_kernel_source()
    print(f"{label}: max err {err:.3f} | cold (do_bench) {cold:.1f} us | warm (events, back-to-back) {warm:.1f} us | threads {src.split('__launch_bounds__(')[1].split(',')[0] if '__launch_bounds__' in src else '?'} | TMA {'CUtensorMap' in src} cp.async {'cp.async' in src or 'LDGSTS' in src}", flush=True)
    return k

m_ws = ex.matmul                                                   # unchanged: TileLang picks TMA + producer warps on sm_121a
run("example as-is (TMA+WS)", m_ws, 1024, 1024, 1024, block_M=128, block_N=128, block_K=32)
m_no = tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})(ex.matmul.fn if hasattr(ex.matmul, "fn") else ex.matmul.__wrapped__ if hasattr(ex.matmul, "__wrapped__") else None) if False else None
# the same kernel with warp specialization off: re-decorate the example's function body
import inspect, textwrap
src = inspect.getsource(ex.matmul.fn if hasattr(ex.matmul, "fn") else ex.matmul)
