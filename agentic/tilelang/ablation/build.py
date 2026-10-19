"""Generate the ablation variants and harnesses for the seven picks (round 3, step 2c). Output: calibration/spark-6951/ablation/<name>/
with one .cu per variant (kernel patched + harness) and a plan.json the Spark driver runs. Grids per variant follow the brief."""
from __future__ import annotations
import json, re
from pathlib import Path
from .variants import patch

ROOT = Path(__file__).resolve().parents[3]; AB = ROOT / "calibration" / "spark-6951" / "ablation"
HARNESS = (Path(__file__).resolve().parent / "harness.cu.in").read_text()
VARIANTS = {"full": {}, "kt0": {"kt": 0}, "kt0_nostore": {"kt": 0, "store": False}, "noload": {"loads": False}, "nocompute": {"compute": False},
            "nocompute_nostore": {"compute": False, "store": False}}


def smem_of(name: str) -> int:
    m = re.search(r"launch__shared_mem_per_block_dynamic\s+(\S+)/block\s+([\d.,]+)", (ROOT / "runs" / "tilelang" / "attr" / f"ncu_{name}.txt").read_text())
    v = float(m.group(2).replace(",", "")); return int(round(v * (1e3 if m.group(1).startswith("K") else 1)))


def main():
    picks = json.load(open(ROOT / "runs" / "tilelang" / "attr" / "picks.json")); maps = json.load(open(AB / "tensormaps.json"))
    plan = []
    for name, pk in picks.items():
        src = (AB / f"pick_{name}.cu").read_text(); s = pk["shape"]; h = pk["pick"]
        threads = int(re.search(r"__launch_bounds__\((\d+)", src).group(1)); smem = smem_of(name)
        gx, gy = -(-s["N"] // h["block_N"]), -(-s["M"] // h["block_M"])
        (A, B) = maps[name]                    # [dtype, rank, ptr, d0, d1, esize, stride1, box0, box1, es0, es1, interleave, swizzle, l2, oob]
        d = AB / name; d.mkdir(exist_ok=True)
        for vname, kw in VARIANTS.items():
            kern = patch(src, **kw); (d / f"{vname}.kernel.cu").write_text(kern)
            h_ = HARNESS.replace("@KERNEL@", f"{vname}.kernel.cu").replace("@M@", str(s["M"])).replace("@N@", str(s["N"])).replace("@K@", str(s["K"])).replace("@THREADS@", str(threads)).replace("@SMEM@", str(smem))
            for tag, mp in (("A", A), ("B", B)):
                h_ = h_.replace(f"@{tag}_D0@", str(mp[3])).replace(f"@{tag}_D1@", str(mp[4])).replace(f"@{tag}_S1@", str(mp[6])).replace(f"@{tag}_B0@", str(mp[7])).replace(f"@{tag}_B1@", str(mp[8])).replace(f"@{tag}_SWZ@", str(mp[12]))
            (d / f"{vname}.cu").write_text(h_)
        full = gx * gy
        grids = {"full": [(gx, gy)], "kt0": [(1, 1), (min(gx, 48), 1) if gx >= 48 else (gx, max(1, 48 // gx)), (gx, gy)],
                 "kt0_nostore": [(1, 1), (min(gx, 48), 1) if gx >= 48 else (gx, max(1, 48 // gx)), (gx, gy)],
                 "noload": [(1, 1), (gx, gy)], "nocompute": [(1, 1), (gx, max(1, min(gy, 48 // gx))) if gx <= 48 else (48, 1), (gx, max(1, min(gy, 96 // gx))) if gx <= 96 else (96, 1), (gx, gy)],
                 "nocompute_nostore": [(1, 1), (gx, gy)]}
        for vname, gs in grids.items():
            for g in dict.fromkeys(gs):
                plan.append({"name": name, "variant": vname, "grid": list(g), "blocks": g[0] * g[1], "full_blocks": full, "threads": threads, "smem": smem, "kt": s["K"] // h["block_K"], "shape": s, "pick": h})
        print(name, f"grid {gx}x{gy}", "threads", threads, "smem", smem, "kt", s["K"] // h["block_K"])
    json.dump(plan, open(AB / "plan.json", "w"), indent=1); print(len(plan), "runs planned")


if __name__ == "__main__":
    main()
