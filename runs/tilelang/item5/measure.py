# Re-time the 16 item-5 kernels: compile all, then three interleaved rounds of back-to-back timing (300 launches);
# the median per kernel. One timing right after a compile was noisy (gemv bn8: 30.5 us once, 13.5 us after).
import json, statistics, torch
from kernels import gemv, splitk, warm_us
ks = {}
for n in (2048, 4096):
    A = torch.randn(n, device="cuda").half(); B = torch.randn(n, n, device="cuda").half()
    for (bn_, rt) in ((2, 32), (4, 32), (8, 32), (4, 64)):
        k = gemv(bn_, rt).compile(N=n, K=n); ks[f"gemv_n{n}_bn{bn_}_rt{rt}"] = (lambda k=k, A=A, B=B: k(A, B))
for (M, N, K, bm, bn, bk, st, th) in ((1024, 1024, 1024, 128, 128, 32, 3, 128), (128, 4096, 4096, 128, 128, 32, 3, 128)):
    A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); C = torch.zeros(M, N, device="cuda")
    for split in (1, 2, 4, 8):
        k = splitk(bm, bn, bk, st, th, split).compile(M=M, N=N, K=K); ks[f"splitk_{M}x{N}x{K}_{bm}x{bn}x{bk}_s{split}"] = (lambda k=k, A=A, B=B, C=C: k(A, B, C))
t = {name: [] for name in ks}
for r in range(3):
    for name, fn in ks.items(): t[name].append(warm_us(fn))
out = {name: {"rounds": v, "median_us": statistics.median(v)} for name, v in t.items()}
for name, v in out.items(): print(f"{name:40s} {' '.join(f'{x:7.1f}' for x in v['rounds'])}  median {v['median_us']:7.1f}", flush=True)
json.dump(out, open("measured.json", "w"), indent=1)
