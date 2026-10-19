# Is the "TileLang is 1.8x behind cuBLASLt at 1024^3" gap in the kernels, or in how they were timed? Same session, same
# kernels, four ways:
#  idle:   synchronize, event, call, event, synchronize (the S1 method): on an idle GPU the first event fires at once,
#          so the host time TileLang's Python wrapper spends before the launch is counted
#  busy:   the GPU is kept busy (torch.cuda._sleep) before the first event, so the wrapper's host time is hidden
#  graph:  the call captured in a CUDA graph and replayed between two events (kernel time plus a graph launch)
#  b2b:    300 calls back-to-back between two events, divided by 300
# plus the host time of one call (perf_counter, no synchronize). cuBLASLt from its C++ harness (idle method).
import re, statistics, subprocess, sys, time, torch, tilelang, tilelang.language as T
from tilelang.cuda.language import copy as cuda_copy
sys.path.insert(0, "../epilogue"); from epi import make as make_epi
sys.path.insert(0, "../decode_stages"); from sweep import make as make_tma

def t_idle(fn, n=100):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); t = []
    for _ in range(n): torch.cuda.synchronize(); e0.record(); fn(); e1.record(); e1.synchronize(); t.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(t)
def t_busy(fn, n=100):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); t = []
    for _ in range(n): torch.cuda._sleep(400000); e0.record(); fn(); e1.record(); e1.synchronize(); t.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(t)
def t_graph(fn, n=100):
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); t = []
        for _ in range(n): torch.cuda._sleep(400000); e0.record(); g.replay(); e1.record(); e1.synchronize(); t.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(t)
    except Exception as e:
        return float("nan")
def t_b2b(fn, n=300):
    for _ in range(20): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(n): fn()
    e1.record(); e1.synchronize(); return e0.elapsed_time(e1) * 1e3 / n
def t_host(fn, n=200):
    t = []
    for _ in range(n): torch.cuda.synchronize(); a = time.perf_counter(); fn(); t.append((time.perf_counter() - a) * 1e6)
    return statistics.median(t)

for size in (1024, 32):
    M, N, K = (1024, 1024, 1024) if size == 1024 else (32, 4096, 4096)
    A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); ref = A.float() @ B.float()
    ks = {}
    if size == 1024:
        for (bm, bn, bk, st, th, tma, staged) in [(128, 256, 32, 3, 256, 0, 0), (64, 128, 32, 3, 256, 0, 1), (128, 128, 32, 3, 128, 1, 1), (64, 128, 128, 2, 128, 1, 0)]:
            ks[f"{'TMA' if tma else 'cp.async'} {bm}x{bn}x{bk} st{st} th{th} {'staged' if staged else 'direct'}"] = make_epi(bm, bn, bk, st, th, staged, tma).compile(M=M, N=N, K=K)
    else:
        for (bm, bn, bk, st) in [(32, 128, 128, 2), (32, 128, 64, 4)]:
            ks[f"TMA {bm}x{bn}x{bk} st{st}"] = make_tma(bm, bn, bk, st).compile(M=M, N=N, K=K)
    out = subprocess.run(["../cublaslt_gemm", str(M), str(N), str(K), "20", "100"], capture_output=True, text=True).stdout
    cub = float(re.search(r"cublaslt_best_ms=([0-9.]+)", out).group(1)) * 1e3
    print(f"== {M}x{N}x{K}: cuBLASLt (C++ harness, idle method) {cub:.1f} us", flush=True)
    for name, k in ks.items():
        fn = (lambda k=k: k(A, B)); C = fn(); torch.cuda.synchronize(); err = ((C.float() - ref).norm() / ref.norm()).item()
        print(f"   {name:36s} err {err:.1e} | idle {t_idle(fn):6.1f} | busy {t_busy(fn):6.1f} | graph {t_graph(fn):6.1f} | b2b {t_b2b(fn):6.1f} | host call {t_host(fn):6.1f} us", flush=True)
