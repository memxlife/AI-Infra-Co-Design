"""Mechanisms of the cp.async (warp-specialization-off) kernels from their ablation (round 3, S2 revised, skeleton 1):
the same differences as analyze.py, over calibration/spark-6951/ablation_cpasync/ablation.jsonl, for every (tile, threads)
config at 1024^3. Writes mechanisms.json there and the family block of chip_tilelang.json (families.gemm_pipelined)."""
from __future__ import annotations
import json, statistics
from pathlib import Path
from .analyze import mech
from ..model.mech import blocks_per_sm
ROOT = Path(__file__).resolve().parents[3]; AB = ROOT / "calibration" / "spark-6951" / "ablation_cpasync"; CHIP = ROOT / "calibration" / "spark-6951" / "chip_tilelang.json"


def load():
    out = {}
    for l in (AB / "ablation.jsonl").read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        if r.get("ok") and r.get("result"):
            r["blocks"] = int(r["blocks"]); r["full_blocks"] = int(r["full_blocks"]); out.setdefault(r["name"], {})[(r["variant"], r["blocks"])] = r
    return out


def main():
    R = load(); out = {}
    for name, rr in R.items():
        any_rec = next(iter(rr.values())); out[name] = mech(name, rr, {"shape": any_rec["shape"], "pick": any_rec["pick"]})
    json.dump(out, open(AB / "mechanisms.json", "w"), indent=1)
    print(f"{'config':24s} {'KT':>3} {'blocks':>6} | {'setup':>5} {'store':>5} | {'cmp/kt':>6} {'TF/SM':>5} | {'ld/kt':>6} {'GB/s':>5} | {'full':>7} {'max':>6} {'sum':>6} {'sync':>6}")
    tiles = {}
    for name, m in sorted(out.items()):
        o = m["overlap"]; h = m["pick"]
        print(f"{name:24s} {m['KT']:3d} {m['blocks']:6d} | {m['block_setup_us']:5.2f} {m['store_per_block_us']:5.2f} | {m['compute_per_ktile_us']:6.3f} {m['tensor_rate_TFs_per_SM'] or 0:5.2f} | {m['load_per_ktile_us_1block']:6.3f} {m['load_rate_GBs_1block'] or 0:5.0f} | {m['full_event_us'] or 0:7.1f} {o['max_parts_us']:6.2f} {o['sum_parts_us']:6.2f} {o['sync_cost_us'] or 0:6.2f}")
        tiles[name] = {"tile": f"{h['block_M']}x{h['block_N']}x{h['block_K']}", "threads": h["threads"], "num_stages": h["num_stages"], "compute_per_ktile_us": m["compute_per_ktile_us"], "tensor_rate_TFs_per_SM": m["tensor_rate_TFs_per_SM"],
                       "load_per_ktile_alone_us": m["load_per_ktile_us_1block"], "load_rate_GBs_1block": m["load_rate_GBs_1block"], "store_per_block_us": m["store_per_block_us"], "block_setup_us": m["block_setup_us"],
                       "overlap_sync_per_ktile_us": (o["sync_cost_us"] or 0) / m["KT"], "load_curve": m["load_curve"], "blocks_per_sm": blocks_per_sm(h), "load_rate_chip_GBs_full_grid": (m["load_curve"].get(m["blocks"]) or m["load_curve"].get(str(m["blocks"])) or {}).get("GBs_chip"), "load_rate_chip_GBs_48_blocks": (m["load_curve"].get(48) or m["load_curve"].get("48") or {}).get("GBs_chip"), "full_event_us": m["full_event_us"], "launch_us_event_1block_nostore": m["launch_us_event_1block_nostore"]}
    chip = json.load(open(CHIP))
    rates = [t["tensor_rate_TFs_per_SM"] for t in tiles.values() if t["tensor_rate_TFs_per_SM"]]; lds = [t["load_rate_GBs_1block"] for t in tiles.values() if t["load_rate_GBs_1block"]]
    syncs = [t["overlap_sync_per_ktile_us"] for t in tiles.values()]; stores = [t["store_per_block_us"] / (int(t["tile"].split("x")[0]) * int(t["tile"].split("x")[1]) * 2 / 1024) for t in tiles.values()]
    chip.setdefault("families", {})["gemm_pipelined"] = {
        "source": "ablation of the cp.async kernels (no producer group) at 1024^3, calibration/spark-6951/ablation_cpasync/, one build per variant, grids 1 / 48 / full; analyze_cpasync.py",
        "tensor_rate_TFs_per_SM": {"median": statistics.median(rates), "min": min(rates), "max": max(rates)},
        "load_rate_alone_GBs_per_block": {"median": statistics.median(lds), "min": min(lds), "max": max(lds)},
        "sync_per_ktile_us": {"median": statistics.median(syncs), "min": min(syncs), "max": max(syncs)},
        "store_us_per_KB_per_block": {"median": statistics.median(stores)},
        "block_setup_us": {"median": statistics.median([t["block_setup_us"] for t in tiles.values()])},
        "blocks": tiles}
    json.dump(chip, open(CHIP, "w"), indent=1); print("families.gemm_pipelined written:", len(tiles), "configs")


if __name__ == "__main__":
    main()
