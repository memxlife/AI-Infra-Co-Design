"""Round 3 step 2b: attribution, not accuracy. The solver's pick at each of the seven shapes under ncu (runs/tilelang/attr/):
the model's terms in microseconds beside the counter each should correspond to, converted to microseconds the same way,
plus the stall decomposition of the active time and the 'unexplained' remainder. No refitting here.
python -m agentic.tilelang.attribution   -> runs/tilelang/attr/attribution.json and the tables on stdout"""
from __future__ import annotations
import json, re
from pathlib import Path
from ..chips import load_chip
from .calibrate import fit, SHAPES7, rows

ROOT = Path(__file__).resolve().parents[2]; ATTR = ROOT / "runs" / "tilelang" / "attr"
UNIT = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}
DEFAULTS = {"prefetch": 0, "warp_policy": 0, "vec_width": 0, "tma": 1, "warp_spec": 1, "smem_layout": "inferred"}


def metrics(path: Path) -> dict:
    out = {}
    for m in re.finditer(r"^\s+(\S+)\s+(\S*)\s+([\d.,]+)\s*$", path.read_text(), re.M):
        try:
            v = float(m.group(3).replace(",", ""))
        except ValueError:
            continue
        u = m.group(2)
        if m.group(1) == "gpu__time_duration.sum":
            v *= UNIT.get(u, 1.0)                     # to microseconds
        elif u.startswith("K"):
            v *= 1e3
        elif u.startswith("M"):
            v *= 1e6
        out[m.group(1)] = v
    return out


def build(chip, terms) -> dict:
    picks = json.load(open(ATTR / "picks.json")); B_HBM, B_L2 = terms["B_HBM"], terms["B_L2_chip"]
    tables = {}
    for name, shape in SHAPES7.items():
        n = metrics(ATTR / f"ncu_{name}.txt"); pk = picks[name]; t = pk["terms"]; h = pk["pick"]
        dur = n["gpu__time_duration.sum"]; f = n["sm__cycles_elapsed.avg"] / dur
        act = n["sm__cycles_active.avg"] / f; idle = (n["sm__cycles_elapsed.max"] - n["sm__cycles_active.avg"]) / f
        tensor = n["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"] / 100 * act
        dram_b = (n["lts__t_sectors_op_read_lookup_miss.sum"] + n["lts__t_sectors_op_write_lookup_miss.sum"]) * 32
        l2_b = (n["lts__t_sectors_op_read.sum"] + n["lts__t_sectors_op_write.sum"]) * 32
        startup = terms["tile_startup"].get(f"{h['block_M']}x{h['block_N']}", 0.0) * pk["waves"] * 1e6
        stalls = {k.split("stalled_")[1].split("_per")[0]: v for k, v in n.items() if "issue_stalled_" in k}
        tot = sum(stalls.values()) + 1.0
        st_us = {k: v / tot * act for k, v in sorted(stalls.items(), key=lambda x: -x[1])}
        model = {"prologue (once per launch)": terms["t_prologue"] * 1e6, "tile start-up x waves": startup,
                 "compute (tensor) x waves": t["t_compute_tile"] * 1e6 * pk["waves"], "copy x waves": t["t_copy_tile"] * 1e6 * pk["waves"],
                 "DRAM (bytes / B_HBM, per-block cap)": t["t_dram"] * 1e6, "L2 (bytes / B_L2)": t["t_l2"] * 1e6, "launch": terms["launch_s"] * 1e6}
        counter = {"prologue (once per launch)": idle, "tile start-up x waves": None, "compute (tensor) x waves": tensor, "copy x waves": None,
                   "DRAM (bytes / B_HBM, per-block cap)": dram_b / B_HBM * 1e6, "L2 (bytes / B_L2)": l2_b / B_L2 * 1e6, "launch": None}
        meas_event = None
        for r in rows(name):
            if all(r["holes"].get(k, DEFAULTS.get(k)) == v for k, v in h.items()):
                meas_event = r["latency_ms"] * 1e3; break
        attributed = idle + max(tensor, dram_b / B_HBM * 1e6, l2_b / B_L2 * 1e6)
        tables[name] = {"shape": shape, "pick": h, "duration_ncu_us": dur, "measured_event_us": meas_event, "predicted_us": pk["predicted_us"], "bound": pk["bound"],
                        "waves": pk["waves"], "tiles": pk["tiles"], "clock_MHz": f, "active_us": act, "idle_us": idle, "tensor_us": tensor,
                        "tensor_pct_active": n["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"], "dram_MB": dram_b / 1e6, "dram_rate_GBps": dram_b / (dur * 1e-6) / 1e9,
                        "l2_MB": l2_b / 1e6, "l2_hit_pct": n.get("lts__t_sector_hit_rate.pct"), "occupancy_pct": n.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
                        "warps_per_sm": n.get("sm__warps_active.avg.per_cycle_active"), "blocks_limit_regs": n.get("launch__occupancy_limit_registers"), "blocks_limit_smem": n.get("launch__occupancy_limit_shared_mem"),
                        "issue_active_pct": n.get("smsp__issue_active.avg.pct_of_peak_sustained_active"), "cpi": n.get("smsp__average_warp_latency_per_inst_issued.ratio"), "bank_conflicts": n.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"),
                        "model": model, "counter": counter, "stall_us": st_us, "attributed_us": attributed, "unexplained_us": dur - attributed,
                        "model_dram_MB": t["dram_bytes"] / 1e6, "model_l2_MB": t["l2_read_bytes"] / 1e6}
    return tables


if __name__ == "__main__":
    chip = load_chip("spark-6951-gb10"); terms = fit(chip)
    tables = build(chip, terms); json.dump(tables, open(ATTR / "attribution.json", "w"), indent=1)
    for name, T in tables.items():
        h = T["pick"]
        print(f"\n=== {name} {h['block_M']}x{h['block_N']}x{h['block_K']} st{h['num_stages']} th{h['threads']} swz{h['swizzle']} | ncu {T['duration_ncu_us']:.1f} us (event {T['measured_event_us']}) predicted {T['predicted_us']:.1f} bound {T['bound']} waves {T['waves']} tiles {T['tiles']} clock {T['clock_MHz']:.0f} MHz")
        print(f"   active {T['active_us']:.1f} idle {T['idle_us']:.1f} tensor {T['tensor_us']:.1f} ({T['tensor_pct_active']:.0f}%) | DRAM {T['dram_MB']:.0f} MB @ {T['dram_rate_GBps']:.0f} GB/s (model {T['model_dram_MB']:.0f}) | L2 {T['l2_MB']:.0f} MB hit {T['l2_hit_pct']:.0f}% (model {T['model_l2_MB']:.0f}) | occ {T['occupancy_pct']:.0f}% warps/SM {T['warps_per_sm']:.1f} blocks limit regs {T['blocks_limit_regs']} smem {T['blocks_limit_smem']} | issue {T['issue_active_pct']:.0f}% cpi {T['cpi']:.1f} | bank conf {T['bank_conflicts']:.0f}")
        for k in T["model"]:
            c = T["counter"][k]; print(f"   {k:38s} model {T['model'][k]:8.1f}   counter {'(none)' if c is None else f'{c:8.1f}'}")
        print("   stalls (us of active, per warp):", {k: round(v, 1) for k, v in list(T["stall_us"].items())[:6]}, "| attributed", round(T["attributed_us"], 1), "unexplained", round(T["unexplained_us"], 1))
