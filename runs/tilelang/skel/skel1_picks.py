# On the Spark: the pipelined family's solver picks, measured warm and cold, plus the stage variants of the 1024^3 pick. argv: items json, out jsonl.
import importlib.util, json, re, sys, torch
from tilelang.profiler import do_bench
def load(name):
    spec = importlib.util.spec_from_file_location(name, name + ".py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def warm_us(fn, reps=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3
items = json.load(open(sys.argv[1])); out = open(sys.argv[2], "a")
for it in items:
    s = it["shape"]; M, N, K = s["M"], s["N"], s["K"]; rec = dict(it)
    try:
        m = load(it["name"]); k = m.matmul.compile(M=M, N=N, K=K, **it.get("open", {}))
        torch.manual_seed(0); A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); C = k(A, B); torch.cuda.synchronize()
        err = (C.float() - A.float() @ B.float()).abs().max().item(); rel = ((C.float() - A.float() @ B.float()).norm() / (A.float() @ B.float()).norm()).item()
        rec.update(ok=rel < 1e-2, max_err=err, rel_err=rel, warm_us=warm_us(lambda: k(A, B)), cold_us=do_bench(lambda: k(A, B), warmup=10, rep=50, backend="event") * 1e3)
        src = k.get_kernel_source(); rec["launch_bounds"] = int(re.search(r"__launch_bounds__\((\d+)", src).group(1))
        print(f"{it['name']}: rel {rel:.1e} | warm {rec['warm_us']:.1f} us | cold {rec['cold_us']:.1f} us | expected {it.get('expected_us', 0):.1f}", flush=True)
    except Exception as e:
        rec.update(ok=False, error=str(e).strip().splitlines()[-1][:300]); print(f"{it['name']}: FAILED {rec['error'][:160]}", flush=True)
    out.write(json.dumps(rec) + "\n"); out.flush()
