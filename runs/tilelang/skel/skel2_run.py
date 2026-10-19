# On the Spark: build and time rendered gemm_streamk modules (one per knob point). argv: a JSON list file of {name, shape}; prints one JSON line each.
import importlib.util, json, re, sys, torch
def load(name):
    spec = importlib.util.spec_from_file_location(name, name + ".py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def median_kernel_us(fn, pre, reps=100):
    ts = []
    for _ in range(10): pre(); fn()
    torch.cuda.synchronize()
    for _ in range(reps):
        pre(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record(); fn(); e1.record(); ts.append((e0, e1))
    torch.cuda.synchronize(); v = sorted(a.elapsed_time(b) for a, b in ts); return v[len(v) // 2] * 1e3
items = json.load(open(sys.argv[1])); out = open(sys.argv[2], "a")
for it in items:
    s = it["shape"]; M, N, K = s["M"], s["N"], s["K"]; rec = {**it}
    try:
        m = load(it["name"]); k = m.build()
        torch.manual_seed(0); A = torch.randn(M, K, device="cuda").half(); B = torch.randn(N, K, device="cuda").half(); C = torch.zeros(M, N, device="cuda"); ref = A.float() @ B.float().T
        C.zero_(); k(A, B, C); torch.cuda.synchronize(); err = (C - ref).abs().max().item()
        us = median_kernel_us(lambda: k(A, B, C), lambda: C.zero_())
        src = k.get_kernel_source(); rec.update(ok=err < 0.5, max_err=err, kernel_us=us, streamk_tiles=m.streamk_tiles, dp_tiles=m.blocking_tiles, partition=m.sm_patition_factor, launch_bounds=int(re.search(r"__launch_bounds__\((\d+)", src).group(1)))
        print(f"{it['name']}: err {err:.3f} | {us:.1f} us | sk {m.streamk_tiles} dp {m.blocking_tiles} x{m.sm_patition_factor}", flush=True)
    except Exception as e:
        rec.update(ok=False, error=str(e).strip().splitlines()[-1][:300]); print(f"{it['name']}: FAILED {rec['error'][:160]}", flush=True)
    out.write(json.dumps(rec) + "\n"); out.flush()
