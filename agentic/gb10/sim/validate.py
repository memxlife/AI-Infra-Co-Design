"""Run the SM simulator on the 22 ablated cp.async kernels and their variants and compare with the ablation's timings.
python -m agentic.gb10.sim.validate [name ...]"""
from __future__ import annotations
import json, math, re, sys, time
from pathlib import Path
from .sm import SM, kernel_sass
from .sass import main_loop

ROOT = Path(__file__).resolve().parents[3]
AB = ROOT / "calibration" / "spark-6951" / "ablation_cpasync"; SASS = ROOT / "calibration" / "spark-6951" / "sass"
F = 2.52e3   # cycles per us


def trips(name: str, variant: str) -> int:
    src = (AB / name / f"{variant}.kernel.cu").read_text()
    m = re.search(r"for \(int k = 0; k < (\d+); \+\+k\)", src); return int(m.group(1)) if m else 1


def sass_trips(ins, h: dict, src_trips: int) -> int:
    """The machine loop may be unrolled: its trip count is the source count over the k-tiles one iteration covers,
    found from the mma (or copies) one iteration holds against what one k-tile needs per warp."""
    lp = main_loop(ins)
    if not lp: return 1
    body = ins[lp[0]:lp[1] + 1]; warps = h["threads"] // 32
    n_mma = sum(x.base == "HMMA" for x in body); n_cp = sum(x.base == "LDGSTS" for x in body)
    if n_mma: per_kt = h["block_M"] * h["block_N"] * h["block_K"] / (16 * 8 * 16) / warps; u = n_mma / per_kt
    elif n_cp: per_kt = (h["block_M"] + h["block_N"]) * h["block_K"] * 2 / 16 / h["threads"]; u = n_cp / per_kt
    else: u = 1
    return max(1, round(src_trips / max(1, round(u))))


def simulate(name: str, variant: str, blocks: int, active: int = 1) -> dict:
    ins = kernel_sass(SASS / f"{name}__{variant}.sass")
    h = json.load(open(AB / "mechanisms.json"))[name]["pick"]
    sm = SM(ins, h["threads"], blocks, sass_trips(ins, h, trips(name, variant)), active_sms=active); t = time.time(); r = sm.run(); r["wall_s"] = round(time.time() - t, 2)
    r["us"] = r["cycles"] / F; return r


if __name__ == "__main__":
    m = json.load(open(AB / "mechanisms.json")); regs = dict(l.split() for l in open(AB / "regs.txt"))
    names = sys.argv[1:] or sorted(m); rows = []
    print(f"{'kernel':24s} R | full block us: sim vs meas | mma alone (noload, 1 block) | loads alone (nocompute, R blocks)")
    for name in names:
        r = m[name]; h = r["pick"]; tiles = r["blocks"]
        smem = h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
        warps = h["threads"] // 32; rw = math.ceil(int(regs[name]) * 32 / 256) * 256
        Rmax = max(1, min(65536 // (warps * rw), 102400 // (smem + 1024), 1536 // h["threads"], 24))
        R = min(Rmax, math.ceil(tiles / 48))
        act = min(48, tiles)
        full = simulate(name, "full", R, act); nol = simulate(name, "noload", 1, 1); noc = simulate(name, "nocompute", R, act)
        o = r["overlap"]; KT = r["KT"]
        meas_full = o["measured_block_us"]; meas_c = r["compute_per_ktile_us"] * KT + o["setup_us"]; meas_l = o["load_part_us_under_full_grid"] + o["setup_us"]
        print(f"{name:24s} {R} | {full['us']:6.1f} vs {meas_full:6.1f} ({(full['us']/meas_full-1)*100:+4.0f}%) | {nol['us']:6.1f} vs {meas_c:6.1f} ({(nol['us']/meas_c-1)*100:+4.0f}%) | {noc['us']:6.1f} vs {meas_l:6.1f} ({(noc['us']/meas_l-1)*100:+4.0f}%)   [{full['wall_s']}s]", flush=True)
        rows.append({"kernel": name, "R": R, "full": (full["us"], meas_full), "noload": (nol["us"], meas_c), "nocompute": (noc["us"], meas_l)})

    import statistics as st
    for part in ("full", "noload", "nocompute"):
        e = sorted(abs(r[part][0] / r[part][1] - 1) for r in rows)
        print(f"{part:10s} median |error| {st.median(e)*100:3.0f} %  worst {e[-1]*100:3.0f} %  within 10 %: {sum(x <= .1 for x in e)}/{len(e)}")
    from scipy.stats import spearmanr
    print("rank correlation of whole-block times:", round(spearmanr([r["full"][0] for r in rows], [r["full"][1] for r in rows])[0], 2))
    json.dump(rows, open(ROOT / "runs" / "gb10" / "sim_validate.json", "w"), indent=1)
