"""Functional fidelity: does the model name the unit the GPU actually spends its time on?
(a) the 22 ablated cp.async kernels: the ablation's dominant part (compute alone, loads under the full grid, the sync
    left over) against the model's units per k-tile;
(b) the seven TMA picks: ncu's tensor-pipe share and DRAM rate against the model's bottleneck;
(c) DRAM traffic: the model's launch-order L2 simulation against ncu's DRAM bytes.
python -m agentic.gb10.fidelity -> runs/gb10/fidelity.json"""
from __future__ import annotations
import json
from pathlib import Path
from .model import predict, dram_bytes, SMS
from .evaluate import kern

ROOT = Path(__file__).resolve().parents[2]


def ablation_check():
    ab = ROOT / "calibration" / "spark-6951" / "ablation_cpasync"
    m = json.load(open(ab / "mechanisms.json")); regs = dict(l.split() for l in open(ab / "regs.txt"))
    rows = []
    for name, r in sorted(m.items()):
        h = r["pick"]; o = r["overlap"]; smem = h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
        p = predict(kern(r["shape"], h, h["threads"], int(regs[name]), smem, "warm")); u = p["per_ktile_us"]
        meas = "load" if o["load_part_us_under_full_grid"] > o["compute_part_us"] else "compute"
        mod = "load" if max(u["port"], u["latency"]) > max(u["tensor"], u["smem"]) else "compute"
        rows.append({"kernel": name, "ablation": meas, "model": mod, "compute_us": o["compute_part_us"], "loads_us": o["load_part_us_under_full_grid"], "sync_us": o["sync_cost_us"]})
    return rows


def ncu_check():
    att = json.load(open(ROOT / "runs" / "tilelang" / "attr" / "attribution.json")); picks = json.load(open(ROOT / "runs" / "tilelang" / "attr" / "picks.json"))
    rows = []
    for name, a in att.items():
        pk = picks[name]; h = pk["pick"]; s = pk["shape"]
        smem = h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
        regs = {128: 240, 256: 168, 512: 96}[h["threads"]]
        p = predict(kern(s, h, h["threads"] + 128, regs, smem, "cold"))   # ncu replays on flushed, clean caches
        meas = "dram" if a["dram_rate_GBps"] > 0.8 * 237 else ("tensor" if a["tensor_pct_active"] > 55 else "load/latency")
        mod = p["bottleneck"]; mod = "load/latency" if mod in ("port", "latency", "bubble", "smem") else mod
        rows.append({"pick": name, "ncu_tensor_pct": a["tensor_pct_active"], "ncu_dram_GBs": a["dram_rate_GBps"], "ncu": meas, "model": mod,
                     "sim_dram_MB": dram_bytes(h, s, SMS) / 1e6, "ncu_dram_MB": a["dram_MB"], "old_count_MB": a["model_dram_MB"]})
    return rows


if __name__ == "__main__":
    a = ablation_check(); b = ncu_check()
    print(f"(a) ablated cp.async kernels: model names the dominant part in {sum(r['ablation'] == r['model'] for r in a)}/{len(a)}")
    for r in a:
        if r["ablation"] != r["model"]: print(f"    differs: {r['kernel']:24s} ablation {r['ablation']:7s} (compute {r['compute_us']:.1f}, loads {r['loads_us']:.1f} us) model {r['model']}")
    print(f"(b) TMA picks: model names ncu's bottleneck in {sum(r['ncu'] == r['model'] for r in b)}/{len(b)}")
    for r in b: print(f"    {r['pick']:6s} ncu tensor {r['ncu_tensor_pct']:5.1f} % DRAM {r['ncu_dram_GBs']:6.1f} GB/s -> {r['ncu']:12s} model {r['model']:12s}")
    print("(c) DRAM traffic, simulated against ncu (old formula in brackets):")
    for r in b: print(f"    {r['pick']:6s} sim {r['sim_dram_MB']:7.1f} MB  ncu {r['ncu_dram_MB']:7.1f}  ratio {r['sim_dram_MB']/r['ncu_dram_MB']:4.2f}  (old {r['old_count_MB']:.0f})")
    (ROOT / "runs" / "gb10").mkdir(parents=True, exist_ok=True); json.dump({"ablation": a, "ncu": b}, open(ROOT / "runs" / "gb10" / "fidelity.json", "w"), indent=1)
