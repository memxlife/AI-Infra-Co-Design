# One session, three interleaved rounds: cuBLASLt (its harness: each rep timed alone with events, median, fastest
# heuristic candidate) against the decode sweep's best TileLang kernels and the old picks, timed the same way
# (each launch alone between two events, median of 100, after warm-up, no flush). python vs_cublaslt.py
import re, statistics, subprocess, torch
from sweep import make
FLUSH = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
def per_launch_us(fn, warm=20, reps=100, flush=False):
    for _ in range(warm): fn()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); t = []
    for _ in range(reps):
        if flush: FLUSH.sum()
        e0.record(); fn(); e1.record(); e1.synchronize(); t.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(t)
N = K = 4096; ks = {}
for M, cfgs in {32: [(32, 128, 128, 2), (32, 128, 64, 4)], 128: [(256, 64, 32, 2), (128, 128, 32, 4)]}.items():
    A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half()
    for (bm, bn, bk, st) in cfgs:
        k = make(bm, bn, bk, st).compile(M=M, N=N, K=K); ks[(M, f"TileLang {bm}x{bn}x{bk} st{st}")] = (lambda k=k, A=A, B=B: k(A, B))
res = {}
for r in range(3):
    for M in (32, 128):
        for fl in (0, 1):
            tag = " flushed" if fl else " back-to-back"
            out = subprocess.run(["../cublaslt_gemm", str(M), str(N), str(K), "20", "100", str(fl)], capture_output=True, text=True).stdout
            best = float(re.search(r"cublaslt_best_ms=([\d.]+)", out).group(1)) * 1e3
            res.setdefault((M, "cuBLASLt" + tag), []).append(best)
            for key, fn in ks.items():
                if key[0] == M: res.setdefault((M, key[1] + tag), []).append(per_launch_us(fn, flush=bool(fl)))
for (M, name), v in sorted(res.items()):
    print(f"M={M:4d} {name:32s} {' '.join(f'{x:6.1f}' for x in v)}  median {statistics.median(v):6.1f} us", flush=True)
