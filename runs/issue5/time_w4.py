"""Time issue #5's 4-bit GEMV (1 x 14336 x 4096, int4 weights) three ways: their protocol (a CUDA graph of 30 calls on
the same weights), warm back to back, and rotating weight copies (each call reads the next of 9 copies, 264 MB)."""
import statistics, torch
from w4_candidate import build

M, N, K = 1, 14336, 4096
kern = build({"M": M, "N": N, "K": K}, {"n_partition": 2, "reduce_thread": 16})
A = torch.randn(M, K, device="cuda").half()
B = torch.randint(-128, 128, (N, K // 2), dtype=torch.int8, device="cuda")
Bu = B.to(torch.int32) & 0xFF
deq = torch.stack([Bu & 15, (Bu >> 4) & 15], dim=-1).reshape(N, K).float()   # element 2j = low nibble, 2j+1 = high
ref = A.float() @ deq.t(); out = kern(A, B).float(); torch.cuda.synchronize()
print("rel err", ((out - ref).norm() / ref.norm()).item(), "| weight MB", B.numel() / 1e6)
Bs = [B] + [B.clone() for _ in range(8)]

def seq(fn, n=200):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(n): fn()
    e1.record(); e1.synchronize(); return e0.elapsed_time(e1) * 1e3 / n

def graph(fns, n=30):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:3]: f()
    torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n): fns[i % len(fns)]()
    for _ in range(10): g.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); t = []
    for _ in range(7): e0.record(); g.replay(); e1.record(); e1.synchronize(); t.append(e0.elapsed_time(e1) * 1e3 / n)
    return statistics.median(t)

same = lambda: kern(A, B)
fns = [lambda b=b: kern(A, b) for b in Bs]; i = [0]
def rot():
    i[0] = (i[0] + 1) % len(fns); return fns[i[0]]()
for _ in range(20): same()
print(f"their protocol, graph of 30 on the same weights: {graph([same]):7.1f} us")
print(f"warm, back to back on the same weights:          {statistics.median([seq(same) for _ in range(3)]):7.1f} us")
print(f"rotating 9 weight copies, back to back:          {statistics.median([seq(rot) for _ in range(3)]):7.1f} us")
print(f"rotating 9 weight copies, graph of 30:           {graph(fns):7.1f} us")
print(f"DRAM floor for the weights at 237 GB/s:          {B.numel() / 237e3:7.1f} us")
