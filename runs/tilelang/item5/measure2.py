# Item-5 kernels again in a new session, two ways: back-to-back (warm, three rounds, median) and with the L2 cleared
# before every run by reading a 64 MB buffer (clean flush; events around the kernel only; median of 100).
import json, statistics, torch
from kernels import gemv, splitk, warm_us
flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
def flushed_us(fn, reps=100):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); t = []
    for _ in range(10): fn()
    for _ in range(reps):
        flush.sum(); e0.record(); fn(); e1.record(); torch.cuda.synchronize(); t.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(t)
ks = {}
for n in (2048, 4096):
    A = torch.randn(n, device="cuda").half(); B = torch.randn(n, n, device="cuda").half()
    for (bn_, rt) in ((2, 32), (4, 32), (8, 32), (4, 64)):
        k = gemv(bn_, rt).compile(N=n, K=n); ks[f"gemv_n{n}_bn{bn_}_rt{rt}"] = (lambda k=k, A=A, B=B: k(A, B))
for (M, N, K, bm, bn, bk, st, th) in ((1024, 1024, 1024, 128, 128, 32, 3, 128), (128, 4096, 4096, 128, 128, 32, 3, 128)):
    A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); C = torch.zeros(M, N, device="cuda")
    for split in (1, 2, 4, 8):
        k = splitk(bm, bn, bk, st, th, split).compile(M=M, N=N, K=K); ks[f"splitk_{M}x{N}x{K}_{bm}x{bn}x{bk}_s{split}"] = (lambda k=k, A=A, B=B, C=C: k(A, B, C))
w = {n: [] for n in ks}
for r in range(3):
    for n, f in ks.items(): w[n].append(warm_us(f))
out = {}
for n, f in ks.items():
    out[n] = {"warm_rounds": w[n], "warm_us": statistics.median(w[n]), "flushed_us": flushed_us(f)}
    print(f"{n:40s} warm {' '.join(f'{x:7.1f}' for x in w[n])} median {out[n]['warm_us']:7.1f} | flushed {out[n]['flushed_us']:7.1f}", flush=True)
json.dump(out, open("measured2.json", "w"), indent=1)
