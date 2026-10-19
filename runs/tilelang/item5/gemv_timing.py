# GEMV timing three ways: back-to-back events (300 launches), one launch at a time (sync between, median of 200), and
# the GPU-side span from %globaltimer is not available here, so also a CUDA graph of 50 launches.
import statistics, sys, torch
from kernels import gemv
for n in (2048,):
    A = torch.randn(n, device="cuda").half(); B = torch.randn(n, n, device="cuda").half()
    for (bn, rt) in ((2, 32), (4, 32), (8, 32), (4, 64)):
        k = gemv(bn, rt).compile(N=n, K=n)
        out = torch.empty(n, device="cuda", dtype=torch.float16)
        for _ in range(30): k(A, B)
        torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(300): k(A, B)
        e1.record(); torch.cuda.synchronize(); b2b = e0.elapsed_time(e1) / 300 * 1e3
        one = []
        for _ in range(200):
            torch.cuda.synchronize(); e0.record(); k(A, B); e1.record(); torch.cuda.synchronize(); one.append(e0.elapsed_time(e1) * 1e3)
        import time
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(300): k(A, B)
        host = (time.perf_counter() - t) / 300 * 1e6; torch.cuda.synchronize()
        print(f"gemv n={n} bn={bn} rt={rt}: back-to-back {b2b:6.1f} us | one launch median {statistics.median(one):6.1f} us | host call {host:6.1f} us", flush=True)
