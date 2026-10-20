# Tier 4, concurrency: a DRAM-bound decode GEMM (M=32, 32 blocks, ~143 us cold) and a compute-bound GEMM (1024^3,
# 128 blocks, ~31 us), run one after the other on one stream, then on two streams at once (with and without stream
# priorities). Weights flushed from the L2 before each pair. Median of 30 pairs.
import statistics, sys, torch
sys.path.insert(0, "../epilogue"); from epi import make as make_epi
sys.path.insert(0, "../decode_stages"); from sweep import make as make_tma
Ad = torch.randn(32, 4096, device="cuda").half(); Bd = torch.randn(4096, 4096, device="cuda").half()
Ac = torch.randn(1024, 1024, device="cuda").half(); Bc = torch.randn(1024, 1024, device="cuda").half()
kd = make_tma(32, 128, 64, 4).compile(M=32, N=4096, K=4096); kc = make_epi(64, 128, 32, 3, 256, 1, 0).compile(M=1024, N=1024, K=1024)
FL = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream(); hp = torch.cuda.Stream(priority=-5)
def run(mode, reps=30):
    t = []
    for _ in range(reps + 3):
        FL.sum(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        if mode == "serial":
            kd(Ad, Bd); kc(Ac, Bc)
        else:
            a, b = (s1, s2) if mode == "two_streams" else (s1, hp)
            a.wait_stream(torch.cuda.current_stream()); b.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(a): kd(Ad, Bd)
            with torch.cuda.stream(b): kc(Ac, Bc)
            torch.cuda.current_stream().wait_stream(a); torch.cuda.current_stream().wait_stream(b)
        e1.record(); torch.cuda.synchronize(); t.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(t[3:])
for _ in range(3): kd(Ad, Bd); kc(Ac, Bc)
alone_d = run_d = None
res = {m: run(m) for m in ("serial", "two_streams", "priority_compute")}
for m, v in res.items(): print(f"RESULT {m:18s} pair_us={v:7.1f}", flush=True)
