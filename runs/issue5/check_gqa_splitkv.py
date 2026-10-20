"""Issue 5's gqa_splitkv: the contributor's three checks (random, zeros, Q and K x20) against an fp32 reference,
for the kernel as written and with the per-split log-sum-exp kept in fp32."""
import importlib, sys, torch
case = {"B": 4, "Hq": 32, "Hkv": 8, "L": 16384, "D": 128}; p = {"BKV": 64, "BH": 64, "split": 8, "stages": 1, "threads": 128}

def inputs(seed, pattern):
    torch.manual_seed(seed); b, hq, hkv, L, d = (case[k] for k in ("B", "Hq", "Hkv", "L", "D"))
    q = torch.randn((b, hq, d), device="cuda", dtype=torch.float16); k = torch.randn((b, L, hkv, d), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k); mask = torch.ones((b, L, hkv), device="cuda", dtype=torch.uint8)
    if pattern == "zeros": q.zero_(); k.zero_(); v.zero_()
    if pattern == "x20": q.mul_(20); k.mul_(20)
    g = hq // hkv; kr = k.float().repeat_interleave(g, 2).permute(0, 2, 1, 3); vr = v.float().repeat_interleave(g, 2).permute(0, 2, 1, 3)
    s = (q.float().unsqueeze(2) @ kr.transpose(-1, -2)) * d ** -0.5
    return (q, k, v, mask), (torch.softmax(s, -1) @ vr).squeeze(2).half()

for mod in sys.argv[1:]:
    kern = importlib.import_module(mod).build(case, p)
    for seed, pat in ((17, "random"), (29, "zeros"), (41, "x20")):
        args, ref = inputs(seed, pat); out = kern(*args); torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs().max().item()
        ok = torch.isfinite(out).all().item() and torch.allclose(out, ref, rtol=0.01, atol=0.0 if pat == "zeros" else 0.01)
        print(f"{mod:20s} {pat:7s} max abs error {err:.6f}  {'pass' if ok else 'FAIL'}", flush=True)
