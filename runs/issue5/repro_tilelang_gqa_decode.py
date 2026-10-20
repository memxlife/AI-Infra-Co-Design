"""TileLang's own examples/flash_decoding/example_gqa_decode.py: split-KV output against the example's ref_program,
at normal scale and with Q and K x 20, as shipped and with glse / lse_local in fp32."""
import importlib.util, sys, torch
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
for label, path in (("as shipped", sys.argv[1]), ("glse, lse_local fp32", sys.argv[2])):
    ex = load(path, label.replace(" ", "_").replace(",", ""))
    b, h, g, L, d = 4, 32, 8, 16384, 128
    kern = ex.flashattn(b, h, g, L, d, block_N=64, block_H=64, num_split=8, num_stages=1, threads=128)
    for s in (1, 20):
        torch.manual_seed(0)
        q = torch.randn(b, h, d, device="cuda", dtype=torch.float16) * s; k = torch.randn(b, L, g, d, device="cuda", dtype=torch.float16) * s
        v = torch.randn(b, L, g, d, device="cuda", dtype=torch.float16); mask = torch.ones(b, L, g, device="cuda", dtype=torch.uint8)
        out = kern(q, k, v, mask); ref = ex.ref_program(q, k, v, mask); torch.cuda.synchronize()
        gq = h // g; kr = k.float().repeat_interleave(gq, 2).permute(0, 2, 1, 3); vr = v.float().repeat_interleave(gq, 2).permute(0, 2, 1, 3)
        ref32 = (torch.softmax((q.float().unsqueeze(2) @ kr.transpose(-1, -2)) * d ** -0.5, -1) @ vr).squeeze(2)
        print(f"   fp32 reference: kernel max abs error {(out.float() - ref32).abs().max().item():.4f}; example ref_program vs fp32 {(ref.float() - ref32).abs().max().item():.4f}")
        print(f"{label:22s} Q,K x{s:<3d} max abs error {(out.float() - ref.float()).abs().max().item():.4f}", flush=True)
