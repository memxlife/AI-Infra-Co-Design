"""Turn the ablation timings into mechanism measurements per pick (round 3, step 2c). Every number here is a difference
of two timings of the same kernel, or a timing of one variant; nothing is fitted.
python -m agentic.tilelang.ablation.analyze  -> calibration/spark-6951/ablation/mechanisms.json and a table"""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; AB = ROOT / "calibration" / "spark-6951" / "ablation"


def load():
    rows = [json.loads(l) for l in (AB / "ablation.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for r in rows:
        if r.get("ok") and r.get("result"):
            r["blocks"] = int(r["blocks"]); r["full_blocks"] = int(r["full_blocks"])
            out.setdefault(r["name"], {})[(r["variant"], r["blocks"])] = r
    return out


def mech(name: str, R: dict, pick: dict) -> dict:
    s = pick["shape"]; h = pick["pick"]; KT = s["K"] // h["block_K"]
    full_blocks = next(iter(R.values()))["full_blocks"]
    def g(v, b): return R.get((v, b))
    def ev(v, b): r = g(v, b); return r["median_us"] if r else None
    def span(v, b): r = g(v, b); return r["span_us"] if r else None
    def bmean(v, b): r = g(v, b); return r["block_mean_us"] if r else None
    m = {"shape": s, "pick": h, "KT": KT, "blocks": full_blocks, "waves": -(-full_blocks // 48)}
    # 1. launch + per-launch prologue: no k-tiles, no store, 1 block (event time) ; the block's own setup from its stamps
    m["launch_us_event_1block_nostore"] = ev("kt0_nostore", 1); m["block_setup_us"] = bmean("kt0_nostore", 1)
    m["launch_us_event_48blocks_nostore"] = ev("kt0_nostore", 48) or ev("kt0_nostore", min(k[1] for k in R if k[0] == "kt0_nostore" and k[1] > 1))
    m["launch_us_event_full_nostore"] = ev("kt0_nostore", full_blocks)
    # 2. epilogue: store on minus store off, per block (stamps) and for the full grid (events)
    m["store_per_block_us"] = (bmean("kt0", 1) or 0) - (bmean("kt0_nostore", 1) or 0)
    m["store_full_grid_us"] = (ev("kt0", full_blocks) or 0) - (ev("kt0_nostore", full_blocks) or 0)
    # 3. compute per k-tile, one block alone (loads removed): block span minus its setup, over KT
    m["compute_per_ktile_us"] = ((bmean("noload", 1) or 0) - (bmean("kt0", 1) or 0)) / KT
    flops_ktile = 2.0 * h["block_M"] * h["block_N"] * h["block_K"]
    m["tensor_rate_TFs_per_SM"] = flops_ktile / (m["compute_per_ktile_us"] * 1e-6) / 1e12 if m["compute_per_ktile_us"] > 0 else None
    m["compute_full_grid_us"] = ev("noload", full_blocks)
    # 4. data movement per k-tile, one block alone (compute removed)
    m["load_per_ktile_us_1block"] = ((bmean("nocompute", 1) or 0) - (bmean("kt0", 1) or 0)) / KT
    bytes_ktile = (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    m["bytes_per_ktile"] = bytes_ktile; m["load_rate_GBs_1block"] = bytes_ktile / (m["load_per_ktile_us_1block"] * 1e-6) / 1e9 if m["load_per_ktile_us_1block"] > 0 else None
    # 5. shared rate under load: loads only at 48, 96, full blocks: bytes moved by the grid over the kernel span
    curve = {}
    for (v, b), r in R.items():
        if v == "nocompute":
            curve[b] = {"event_us": r["median_us"], "block_mean_us": r["block_mean_us"], "GBs_chip": b * KT * bytes_ktile / (r["span_us"] * 1e-6) / 1e9 if r["span_us"] else None,
                        "GBs_per_block": KT * bytes_ktile / ((r["block_mean_us"] - (bmean("kt0", 1) or 0)) * 1e-6) / 1e9 if r["block_mean_us"] else None}
    m["load_curve"] = dict(sorted(curve.items()))
    # 7. overlap: the full kernel's per-block time against max and sum of its parts
    fb = bmean("full", full_blocks); setup = bmean("kt0", 1) or 0
    c = m["compute_per_ktile_us"] * KT; l = (bmean("nocompute", full_blocks) or 0) - setup
    m["full_block_us"] = fb; m["full_event_us"] = ev("full", full_blocks); m["full_span_us"] = span("full", full_blocks)
    m["overlap"] = {"max_parts_us": setup + max(c, l), "sum_parts_us": setup + c + l, "measured_block_us": fb, "compute_part_us": c, "load_part_us_under_full_grid": l, "setup_us": setup,
                    "sync_cost_us": fb - (setup + max(c, l)) if fb else None}
    return m


if __name__ == "__main__":
    R = load(); picks = json.load(open(ROOT / "runs" / "tilelang" / "attr" / "picks.json")); out = {}
    for name in picks:
        if name not in R: continue
        out[name] = mech(name, R[name], picks[name]); m = out[name]
        print(f"\n=== {name} {m['pick']['block_M']}x{m['pick']['block_N']}x{m['pick']['block_K']} st{m['pick']['num_stages']} KT={m['KT']} blocks={m['blocks']} waves={m['waves']}")
        print(f"   launch+setup: 1 block {m['launch_us_event_1block_nostore']} us (block setup {m['block_setup_us']:.2f}), 48 blocks {m['launch_us_event_48blocks_nostore']}, full {m['launch_us_event_full_nostore']} | store per block {m['store_per_block_us']:.2f} us, full grid {m['store_full_grid_us']:.2f}")
        print(f"   compute per k-tile (1 block) {m['compute_per_ktile_us']:.3f} us -> {m['tensor_rate_TFs_per_SM']:.2f} TF/s per SM | load per k-tile alone {m['load_per_ktile_us_1block']:.3f} us ({m['bytes_per_ktile'] // 1024} KB) -> {m['load_rate_GBs_1block']:.0f} GB/s per block")
        print("   load curve (blocks: chip GB/s, per-block GB/s):", {b: (round(v['GBs_chip'] or 0), round(v['GBs_per_block'] or 0)) for b, v in m["load_curve"].items()})
        o = m["overlap"]; print(f"   full kernel: event {m['full_event_us']} us, span {m['full_span_us']}, block mean {o['measured_block_us']:.2f} | max(parts) {o['max_parts_us']:.2f} sum(parts) {o['sum_parts_us']:.2f} (compute {o['compute_part_us']:.2f}, loads-under-load {o['load_part_us_under_full_grid']:.2f}, setup {o['setup_us']:.2f}) -> sync/other {o['sync_cost_us']:.2f}")
    json.dump(out, open(AB / "mechanisms.json", "w"), indent=1)
