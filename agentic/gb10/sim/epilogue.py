"""The C store as a choice, on the simulator: five tiles at 1024^3, each with the direct store (T.copy(C_local, C):
4-byte STG.E) and the staged store (T.copy(C_local, C_shared); T.copy(C_shared, C): stmatrix, then a TMA store),
against warm timings (runs/tilelang/epilogue/epi_1024.txt). python -m agentic.gb10.sim.epilogue"""
from __future__ import annotations
import json, re
from pathlib import Path
from .kernel import simulate_kernel
from .ws_validate import kernel_us

ROOT = Path(__file__).resolve().parents[3]; D = ROOT / "runs" / "tilelang" / "epilogue"


def measured() -> dict:
    out = {}
    for line in open(D / "epi_1024.txt"):
        m = re.match(r"(cp\.async|TMA)\s+(\d+)x(\d+)x(\d+) st\d+ th\d+ (direct|staged): err [\d.]+ warm\s+([\d.]+) us", line)
        if m: out[f"{'tma' if m.group(1) == 'TMA' else 'cp'}_{m.group(2)}x{m.group(3)}x{m.group(4)}_{m.group(5)}"] = float(m.group(6))
    return out


def run(size: int = 1024) -> list:
    meta = json.load(open(D / "sass" / "meta.json")); meas = measured(); shape = {"M": size, "N": size, "K": size}; rows = []
    for name, m in meta.items():
        h = {"block_M": m["bm"], "block_N": m["bn"], "block_K": m["bk"], "num_stages": m["stages"], "threads": m["threads"], "swizzle": 0}
        path = D / "sass" / f"{name}.sass"
        us = kernel_us(name, h, shape, sass_path=path) if m["tma"] else simulate_kernel(path, h, shape, m["regs"])["us"]
        rows.append({"kernel": name, "sim_us": round(us, 2), "meas_us": meas.get(name)})
        print(f"{name:24s} sim {us:6.1f} vs meas {meas.get(name, float('nan')):6.1f} ({(us / meas[name] - 1) * 100:+4.0f}%)", flush=True)
    by = {r["kernel"]: r for r in rows}
    print("staged saves (sim vs measured):")
    for name in by:
        if name.endswith("_direct"):
            s = name.replace("_direct", "_staged"); d, t = by[name], by[s]
            print(f"  {name[:-7]:18s} sim {d['sim_us'] - t['sim_us']:+5.1f} us  meas {d['meas_us'] - t['meas_us']:+5.1f} us")
    json.dump(rows, open(ROOT / "runs" / "gb10" / "sim_epilogue.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    run()
