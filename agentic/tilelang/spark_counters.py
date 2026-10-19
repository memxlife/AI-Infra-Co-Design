"""The counter experiments of the brief's calibration table, run on the Spark after the sweep (the GPU must be idle):
  F_tc      the best big-square config under ncu: tensor-pipe utilization
  B_HBM     the best decode config under ncu: DRAM bytes (L2 misses x 32) and time
  SWZ       one tile config across the four swizzle panels: bank conflicts, L2/DRAM bytes
Writes one JSON with every run's metrics. Uses spark_runner's ncu path (kernel-name filter, one launch)."""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile


def run(kernel, func, m, n, k, holes, outdir):
    out = os.path.join(outdir, f"r_{m}x{n}x{k}_{'_'.join(str(v) for v in holes.values())}.json")
    py = os.path.expanduser("~/.venv/bin/python")
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [py, os.path.join(here, "spark_runner.py"), "--kernel", kernel, "--func", func, "--m", str(m), "--n", str(n), "--k", str(k),
           "--holes", json.dumps(holes), "--reps", "100", "--warmup", "25", "--ncu", "--out", out]
    subprocess.run(cmd, check=False, capture_output=True, text=True)
    r = json.load(open(out)) if os.path.exists(out) else {"ok": False}
    return {k2: r.get(k2) for k2 in ("ok", "shape", "holes", "latency_ms", "tflops", "n_regs", "n_spills", "smem_bytes", "dynamic_smem_bytes", "threads", "ncu", "error")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kernel", required=True); ap.add_argument("--plan", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.load(open(a.plan)); outdir = tempfile.mkdtemp(prefix="tl_counters_")
    results = []
    for item in plan:
        r = run(a.kernel, "matmul", item["M"], item["N"], item["K"], item["holes"], outdir)
        r["label"] = item["label"]; results.append(r)
        print(item["label"], r.get("ok"), r.get("latency_ms"), (r.get("ncu") or {}).get("ok"), flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
