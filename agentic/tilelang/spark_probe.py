"""Round 2 knob curves, on the Spark: a list of (kernel file, label, holes) rendered on the Mac; each is compiled (fresh
cache so the attributes are present), checked against torch, timed (median of CUDA events), with registers and spills
from ptxas. One JSON line per item, written as it finishes. The GPU must be idle."""
from __future__ import annotations
import argparse, importlib.util, json, os, re, sys, tempfile, time, traceback


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=100); ap.add_argument("--arch", default="sm_121a")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TILELANG_CACHE_DIR", tempfile.mkdtemp(prefix="tl_cache_probe_"))
    import torch
    from spark_runner import ptxas_report, dynamic_smem, smi_state
    plan = json.load(open(a.plan))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["label"] for l in open(a.out) if l.strip()}
    with open(a.out, "a") as f:
        for item in plan:
            if item["label"] in done:
                continue
            rep = {"label": item["label"], "shape": item["shape"], "holes": item["holes"], "kernel": item["kernel"]}
            t0 = time.time()
            try:
                spec = importlib.util.spec_from_file_location("km_" + re.sub(r"\W", "_", item["label"]), item["kernel"]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                M, N, K = item["shape"]["M"], item["shape"]["N"], item["shape"]["K"]
                k = mod.matmul.compile(M=M, N=N, K=K)
                rep["compile_s"] = round(time.time() - t0, 2)
                src = k.get_kernel_source()
                m = re.search(r"__launch_bounds__\((\d+)", src); rep["threads"] = int(m.group(1)) if m else None
                rep["dynamic_smem_bytes"], rep["dynamic_smem_source"] = dynamic_smem(k, item["holes"])
                px = ptxas_report(src, a.arch); fn = next(iter(px["functions"].values()), {}) if px.get("ok") else {}
                rep["n_regs"] = fn.get("registers"); rep["n_spills"] = (fn.get("spill_stores", 0) + fn.get("spill_loads", 0)) if fn else None
                rep["uses"] = {"tma": "CUtensorMap" in src, "cp_async": "cp.async" in src or "LDGSTS" in (px.get("sass_text") or ""), "mma": "mma" in src.lower()}
                A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half()
                if item.get("inout_c"):                 # split-K: the kernel adds into a zeroed FP32 C
                    C = torch.zeros(M, N, device="cuda", dtype=torch.float32); k(A, B, C)
                else:
                    C = k(A, B)
                torch.cuda.synchronize(); ref = A @ B
                diff = (C.float() - ref.float()).abs(); tol = 1e-2 + 1e-2 * ref.float().abs()
                rep["correct"] = bool((diff <= tol).all()); rep["max_abs_err"] = float(diff.max())
                p = k.get_profiler(); rep["clock_before"] = smi_state()
                lat = p.do_bench(backend="event", warmup=25, rep=a.reps, return_mode="median")
                rep["clock_after"] = smi_state()
                rep["latency_us"] = float(lat) * 1e3; rep["tflops"] = 2.0 * M * N * K / (lat * 1e-3) / 1e12; rep["ok"] = True
            except Exception as e:
                rep["ok"] = False; rep["error"] = traceback.format_exc()[-600:]
            f.write(json.dumps(rep) + "\n"); f.flush()
            print(item["label"], rep.get("ok"), rep.get("latency_us"), rep.get("n_regs"), (rep.get("error") or "").splitlines()[-1][:160] if not rep.get("ok") else "", flush=True)


if __name__ == "__main__":
    main()
