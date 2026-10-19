"""The shared-memory layout as a choice, on the simulator: five tiles at 1024^3 on the cp.async path, each with
TileLang's inferred layout and with a linear one (same machine code apart from addresses), against warm timings
(runs/tilelang/layout/pairs_1024.txt). python -m agentic.gb10.sim.layout"""
from __future__ import annotations
import json
from pathlib import Path
from .kernel import simulate_kernel, ldsm_ways

ROOT = Path(__file__).resolve().parents[3]; D = ROOT / "runs" / "tilelang" / "layout" / "pairs"


def run(size: int = 1024) -> list:
    meta = json.load(open(D / "meta.json")); shape = {"M": size, "N": size, "K": size}; rows = []
    for name, m in meta.items():
        h = {"block_M": m["bm"], "block_N": m["bn"], "block_K": m["bk"], "num_stages": m["stages"], "threads": m["threads"],
             "swizzle": 0, "smem_layout": m["layout"]}
        us = simulate_kernel(D / f"{name}.sass", h, shape, m["regs"])["us"]
        rows.append({"kernel": name, "ways": ldsm_ways(h), "sim_us": round(us, 2), "meas_us": round(m["warm_us"], 2)})
        print(f"{name:34s} ways A,B {ldsm_ways(h)} sim {us:6.1f} vs meas {m['warm_us']:6.1f} ({(us / m['warm_us'] - 1) * 100:+4.0f}%)", flush=True)
    by = {r["kernel"]: r for r in rows}
    print("linear costs (sim vs measured):")
    for name, r in by.items():
        if name.endswith("_inferred"):
            l = by[name.replace("_inferred", "_linear")]
            print(f"  {name[:-9]:24s} sim {l['sim_us'] / r['sim_us']:4.2f}x  meas {l['meas_us'] / r['meas_us']:4.2f}x")
    json.dump(rows, open(ROOT / "runs" / "gb10" / "sim_layout.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    run()
