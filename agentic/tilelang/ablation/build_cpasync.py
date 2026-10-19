"""Ablation variants of the cp.async (warp-specialization-off) kernels (round 3, S2 revised, skeleton 1).
Input: calibration/spark-6951/ablation_cpasync/sources.json (one entry per config: name, holes, shape, threads, smem) and
<name>.kernel.cu next to it (TileLang's get_kernel_source on the Spark). Output: <name>/{full,kt0,kt0_nostore,noload,
nocompute,nocompute_nostore}.cu and plan.json for spark_ablate.sh (ABL_DIR=ablation_cpasync)."""
from __future__ import annotations
import json, re
from pathlib import Path
from .variants import patch_cpasync
ROOT = Path(__file__).resolve().parents[3]; AB = ROOT / "calibration" / "spark-6951" / "ablation_cpasync"
HARNESS = (Path(__file__).parent / "harness_ptr.cu.in").read_text()


def main():
    srcs = json.load(open(AB / "sources.json")); plan = []
    for e in srcs:
        name, h, s = e["name"], e["holes"], e["shape"]
        src = (AB / f"{name}.kernel.cu").read_text()
        threads = int(re.search(r"__launch_bounds__\((\d+)", src).group(1)); smem = int(e["smem"])
        d = AB / name; d.mkdir(exist_ok=True)
        variants = {"full": {}, "kt0": {"kt": 0}, "kt0_nostore": {"kt": 0, "store": False}, "noload": {"loads": False}, "nocompute": {"compute": False}, "nocompute_nostore": {"compute": False, "store": False}}
        for vname, kw in variants.items():
            (d / f"{vname}.kernel.cu").write_text(patch_cpasync(src, **kw))
            h_ = HARNESS.replace("@KERNEL@", f"{vname}.kernel.cu").replace("@M@", str(s["M"])).replace("@N@", str(s["N"])).replace("@K@", str(s["K"])).replace("@THREADS@", str(threads)).replace("@SMEM@", str(smem))
            (d / f"{vname}.cu").write_text(h_)
        gx, gy = -(-s["N"] // h["block_N"]), -(-s["M"] // h["block_M"]); full = gx * gy
        g48 = (min(gx, 48), 1) if gx >= 48 else (gx, max(1, 48 // gx))
        grids = {"full": [(gx, gy)], "kt0": [(1, 1), (gx, gy)], "kt0_nostore": [(1, 1), g48, (gx, gy)], "noload": [(1, 1), (gx, gy)], "nocompute": [(1, 1), g48, (gx, gy)], "nocompute_nostore": [(1, 1)]}
        for vname, gs in grids.items():
            for g in dict.fromkeys(gs):
                plan.append({"name": name, "variant": vname, "grid": list(g), "blocks": g[0] * g[1], "full_blocks": full, "threads": threads, "smem": smem, "kt": s["K"] // h["block_K"], "shape": s, "pick": h})
        print(name, f"grid {gx}x{gy}", "threads", threads, "smem", smem, "kt", s["K"] // h["block_K"])
    json.dump(plan, open(AB / "plan.json", "w"), indent=0); print(len(plan), "runs planned")


if __name__ == "__main__":
    main()
