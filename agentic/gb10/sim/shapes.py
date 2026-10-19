"""Simulate the cp.async kernels measured at 2048^3, 4096^3, 4096x4096x1024, M=32 and M=128 (runs/gb10/shapes) and
compare with their warm times. python -m agentic.gb10.sim.shapes"""
import json, statistics as st
from pathlib import Path
from scipy.stats import spearmanr
from .kernel import simulate_kernel
ROOT = Path(__file__).resolve().parents[3]; D = ROOT / "runs" / "gb10" / "shapes"

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(D / "measured.jsonl")]; rows = [r for r in rows if r.get("ok")]; by = {}; out = []
    for r in rows:
        p = simulate_kernel(D / f"{r['name']}.sass", r["holes"], r["shape"], r["regs"])
        tag = "x".join(str(r["shape"][k]) for k in "MNK"); by.setdefault(tag, []).append((r, p)); h = r["holes"]
        out.append({"name": r["name"], "sim_us": p["us"], "meas_us": r["warm_us"], "dram_MB": p["dram_MB"], "waves": p["waves"]})
        print(f"{tag:14s} {h['block_M']}x{h['block_N']}x{h['block_K']} st{h['num_stages']} th{h['threads']} p{h['swizzle']} | sim {p['us']:8.1f} vs meas {r['warm_us']:8.1f} ({(p['us']/r['warm_us']-1)*100:+4.0f}%) | dram {p['dram_MB']:5.0f} MB", flush=True)
    allerr = []
    for tag, v in by.items():
        e = [abs(p["us"] / r["warm_us"] - 1) for r, p in v]; allerr += e
        rho = spearmanr([p["us"] for r, p in v], [r["warm_us"] for r, p in v])[0] if len(v) > 2 else float("nan")
        best = min(v, key=lambda x: x[0]["warm_us"]); pick = min(v, key=lambda x: x[1]["us"])
        print(f"{tag:14s} n={len(v)} median |err| {st.median(e)*100:3.0f}%  worst {max(e)*100:3.0f}%  rank {rho:5.2f}  regret {(pick[0]['warm_us']/best[0]['warm_us']-1)*100:4.1f}%")
    print(f"all {len(allerr)}: median |err| {st.median(allerr)*100:.0f}%, within 10 %: {sum(x <= .1 for x in allerr)}/{len(allerr)}")
    json.dump(out, open(ROOT / "runs" / "gb10" / "sim_shapes.json", "w"), indent=1)
