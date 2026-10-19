"""R3: the noise band. One configuration timed five times, each time a fresh compile-from-cache and a fresh run, with
the SM clock and the temperature recorded before and after; optionally with the clock pinned. Writes JSON."""
from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys, time


def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(", ")
    return {"sm_mhz": float(out[0]), "temp_c": float(out[1]), "power_w": float(out[2])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kernel", required=True); ap.add_argument("--m", type=int); ap.add_argument("--n", type=int); ap.add_argument("--k", type=int)
    ap.add_argument("--holes", required=True); ap.add_argument("--times", type=int, default=5); ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    import torch
    spec = importlib.util.spec_from_file_location("km", a.kernel); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    holes = json.loads(a.holes)
    k = mod.matmul.compile(M=a.m, N=a.n, K=a.k, **holes)
    A = torch.randn(a.m, a.k, device="cuda").half(); B = torch.randn(a.k, a.n, device="cuda").half()
    p = k.get_profiler(); runs = []
    for i in range(a.times):
        before = smi()
        lat = p.do_bench(backend="event", warmup=25, rep=a.reps, return_mode="median")
        after = smi()
        runs.append({"i": i, "latency_us": float(lat) * 1e3, "before": before, "after": after})
        print(i, round(float(lat) * 1e3, 2), "us", after, flush=True)
        time.sleep(2)
    us = [r["latency_us"] for r in runs]
    json.dump({"shape": {"M": a.m, "N": a.n, "K": a.k}, "holes": holes, "runs": runs, "min_us": min(us), "max_us": max(us), "spread": (max(us) - min(us)) / min(us)}, open(a.out, "w"), indent=1)
    print("spread", round((max(us) - min(us)) / min(us) * 100, 2), "%")


if __name__ == "__main__":
    main()
