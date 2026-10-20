"""Issue 5's block_sparse_attention with the contributor's own mask (diagonal blocks plus column 0), inputs
(random, zeros, x3) and pass rule (rtol = atol = 0.01, all finite), then a timing loop like theirs."""
import time, torch
from bsa_candidate import build
B, H, L, D = 1, 32, 4096, 128; n = L // 64
kern = build({"B": B, "heads": H, "L": L, "D": D, "causal": True}, {"stages": 1, "threads": 128})
blk = torch.eye(n, device="cuda", dtype=torch.bool); blk[:, 0] = True; blk = blk.expand(B, H, n, n).contiguous()
dense = blk.repeat_interleave(64, -2).repeat_interleave(64, -1) & torch.ones(L, L, device="cuda", dtype=torch.bool).tril()
for seed, pat in ((17, "random"), (29, "zeros"), (41, "x3")):
    torch.manual_seed(seed); q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.float16) for _ in range(3))
    if pat == "zeros": q.zero_(); k.zero_(); v.zero_()
    if pat == "x3": q.mul_(3); k.mul_(3); v.mul_(3)
    s = (q.float() @ k.float().transpose(-1, -2)) * D ** -0.5; s.masked_fill_(~dense, float("-inf")); ref = (s.softmax(-1) @ v.float()).half()
    t = time.time(); out = kern(q, k, v, blk.to(torch.int32)); torch.cuda.synchronize(); dt = time.time() - t
    ok = bool(torch.isfinite(out).all()) and torch.allclose(out, ref, rtol=0.01, atol=0.01)
    print(f"{pat:6s} max abs error {(out.float() - ref.float()).abs().max().item():.4f}  NaN {torch.isnan(out).any().item()}  {'pass' if ok else 'FAIL'}  ({dt:.2f} s)", flush=True)
q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.float16) for _ in range(3)); m = blk.to(torch.int32)
for _ in range(10): kern(q, k, v, m)
torch.cuda.synchronize(); t = time.time()
for _ in range(30): kern(q, k, v, m)
torch.cuda.synchronize(); print(f"timing: {(time.time() - t) / 30 * 1e6:.1f} us per call")
