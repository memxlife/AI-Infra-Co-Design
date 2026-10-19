"""Decode-shape TMA GEMMs against the simulator: 26 kernels (tiles x pipeline depth) at M=32 and M=128, N=K=4096
(runs/tilelang/decode_stages), against their times after an L2 flush. The test for the TMA path's DRAM model: the
measured curves fall with depth until DRAM is saturated. python -m agentic.gb10.sim.decode_sweep"""
from __future__ import annotations
import json, sys
from pathlib import Path
from scipy.stats import spearmanr
from .ws_validate import kernel_us

ROOT = Path(__file__).resolve().parents[3]; D = ROOT / "runs" / "tilelang" / "decode_stages"


def run(names=None, quiet=False) -> list:
    meta = json.load(open(D / "sweep.json")); regs = json.load(open(D / "sass" / "regs.json")); rows = []
    for name, m in meta.items():
        if names and name not in names: continue
        h = {"block_M": m["bm"], "block_N": m["bn"], "block_K": m["bk"], "num_stages": m["stages"], "threads": 128, "swizzle": 0, "tma": 1}
        shape = {"M": m["M"], "N": 4096, "K": 4096}
        us = kernel_us(name, h, shape, warm=False, sass_path=D / "sass" / f"{name}.sass")
        rows.append({"kernel": name, "M": m["M"], "sim_us": round(us, 1), "meas_us": round(m["flushed_us"], 1)})
        if not quiet: print(f"{name:24s} sim {us:7.1f} vs flushed {m['flushed_us']:7.1f} ({(us / m['flushed_us'] - 1) * 100:+4.0f}%)", flush=True)
    for M in (32, 128):
        r = [x for x in rows if x["M"] == M]
        if len(r) < 3: continue
        e = sorted(abs(x["sim_us"] / x["meas_us"] - 1) for x in r); best = min(r, key=lambda x: x["sim_us"])
        print(f"M={M}: n={len(r)} median |err| {e[len(e) // 2] * 100:.0f}% worst {e[-1] * 100:.0f}% rank {spearmanr([x['sim_us'] for x in r], [x['meas_us'] for x in r])[0]:.2f} "
              f"regret {(best['meas_us'] / min(x['meas_us'] for x in r) - 1) * 100:.1f}% (sim picks {best['kernel']})")
    return rows


if __name__ == "__main__":
    rows = run(sys.argv[1:] or None)
    json.dump(rows, open(ROOT / "runs" / "gb10" / "sim_decode_sweep.json", "w"), indent=1)
