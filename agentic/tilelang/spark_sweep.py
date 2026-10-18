"""The T3 sweep, run on the Spark. Two phases so the GPU is never shared while timing:
  compile  N worker processes each compile a slice of the configs (TileLang's disk cache keeps the result);
  time     one process compiles again (cache hits), checks correctness and times each config in turn.
Every config lands as one JSON line in --out as soon as it is done, so the Mac can read partial results.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, re, subprocess, sys, time, traceback


def load(kernel, func):
    spec = importlib.util.spec_from_file_location("km", kernel); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, func)


def one(fn, a, holes, timing: bool):
    import torch
    rep = {"holes": holes, "shape": {"M": a.m, "N": a.n, "K": a.k}}
    t = time.time()
    try:
        k = fn.compile(M=a.m, N=a.n, K=a.k, **holes)
    except Exception as e:                     # a config TileLang's passes refuse: that is a finding, not a crash
        rep.update(ok=False, stage="compile", error=str(e).splitlines()[-1][:400], compile_s=round(time.time() - t, 2))
        return rep
    rep["compile_s"] = round(time.time() - t, 2)
    if not timing:
        rep["ok"] = True; return rep
    try:
        src = k.get_kernel_source(); host = k.get_host_source()
        m = re.search(r"__launch_bounds__\((\d+)", src); rep["threads"] = int(m.group(1)) if m else None
        from spark_runner import dynamic_smem
        rep["dynamic_smem_bytes"], rep["dynamic_smem_source"] = dynamic_smem(k, holes)
        from spark_runner import ptxas_report
        px = ptxas_report(src, a.arch); f = next(iter(px["functions"].values()), {}) if px.get("ok") else {}
        rep["n_regs"] = f.get("registers"); rep["n_spills"] = (f.get("spill_stores", 0) + f.get("spill_loads", 0)) if f else None
        rep["static_smem_bytes"] = f.get("static_smem_bytes"); rep["smem_bytes"] = (rep["static_smem_bytes"] or 0) + rep["dynamic_smem_bytes"]
        rep["uses_tma"] = "CUtensorMap" in src
        A = torch.randn(a.m, a.k, device="cuda").half(); B = torch.randn(a.k, a.n, device="cuda").half()
        C = k(A, B); torch.cuda.synchronize(); ref = A @ B
        diff = (C.float() - ref.float()).abs(); tol = 1e-2 + 1e-2 * ref.float().abs()
        rep["correct"] = bool((diff <= tol).all()); rep["max_abs_err"] = float(diff.max())
        p = k.get_profiler()
        lat = p.do_bench(backend="event", warmup=a.warmup, rep=a.reps, return_mode="median")
        rep["latency_ms"] = float(lat); rep["tflops"] = 2.0 * a.m * a.n * a.k / (lat * 1e-3) / 1e12; rep["ok"] = True
    except Exception as e:
        rep.update(ok=False, stage="run", error=str(e).splitlines()[-1][:400])
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True); ap.add_argument("--func", default="matmul")
    ap.add_argument("--m", type=int, required=True); ap.add_argument("--n", type=int, required=True); ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--configs", required=True, help="JSON list of hole dicts"); ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["compile", "time"], default="time"); ap.add_argument("--slice", default="0/1", help="i/n: this worker's share in the compile phase")
    ap.add_argument("--reps", type=int, default=100); ap.add_argument("--warmup", type=int, default=25); ap.add_argument("--arch", default="sm_121a")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    cfgs = json.load(open(a.configs)); i, n = map(int, a.slice.split("/"))
    fn = load(a.kernel, a.func)
    done = set()
    if a.phase == "time" and os.path.exists(a.out):
        done = {json.dumps(json.loads(l)["holes"], sort_keys=True) for l in open(a.out) if l.strip()}
    with open(a.out, "a") as f:
        for j, holes in enumerate(cfgs):
            if j % n != i or json.dumps(holes, sort_keys=True) in done:
                continue
            rep = one(fn, a, holes, timing=a.phase == "time"); rep["i"] = j
            f.write(json.dumps(rep) + "\n"); f.flush()
            print(j, rep.get("ok"), rep.get("latency_ms"), rep.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
