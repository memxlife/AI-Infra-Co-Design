# One session, interleaved: the M=128 GEMMs of the five-shape set (fp16 C store) and split-K s1 (fp32 C by atomics)
# at 128x4096x4096, back-to-back timing, three rounds. Is the 135 vs 158 us gap the kernel or the session?
import importlib.util, statistics, sys, torch
sys.path.insert(0, "../shapes"); from kernels import splitk, warm_us
M, N, K = 128, 4096, 4096
A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); C = torch.zeros(M, N, device="cuda")
ks = {}
for name in ("s128x4096x4096_128x128x64_st3_th128_p0", "s128x4096x4096_64x128x32_st3_th128_p0", "s128x4096x4096_128x256x32_st3_th256_p0"):
    spec = importlib.util.spec_from_file_location(name, f"../shapes/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    k = m.matmul.compile(M=M, N=N, K=K); ks[name] = (lambda k=k: k(A, B))
k1 = splitk(128, 128, 32, 3, 128, 1).compile(M=M, N=N, K=K); ks["splitk_s1_128x128x32"] = (lambda: k1(A, B, C))
t = {n: [] for n in ks}
for r in range(3):
    for n, f in ks.items(): t[n].append(warm_us(f))
for n, v in t.items(): print(f"{n:42s} {' '.join(f'{x:7.1f}' for x in v)} median {statistics.median(v):7.1f}", flush=True)
