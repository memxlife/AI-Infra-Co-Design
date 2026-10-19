"""The warp-specialized TMA picks (step 2c ablation) on the simulator: one block alone (mma-only, loads-only, whole)
and the whole kernel at its grid, against the ablation's timings. python -m agentic.gb10.sim.ws_validate [name ...]"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from .ws import WSSM
from .sm import kernel_sass
from .sass import main_loop
from ..model import dram_split_kslice, C as MC
from .kernel import dram_clk, wave_dram

ROOT = Path(__file__).resolve().parents[3]; AB = ROOT / "calibration" / "spark-6951" / "ablation"; SASS = ROOT / "calibration" / "spark-6951" / "sass"
F = 2.54e3


def ws_sim(name: str, variant: str, h: dict, shape: dict, blocks_eq: float = 1.0, miss: float = 0.0, dclk: float | None = None,
           sass_path=None, nblocks: int = 1, grid: int | None = None) -> dict:
    ins = kernel_sass(sass_path or SASS / f"tma_{name}__{variant}.sass"); KT = math.ceil(shape["K"] / h["block_K"])
    lp = main_loop(ins); warps = h["threads"] // 32
    if lp:
        body = ins[lp[0]:lp[1] + 1]; n_mma = sum(x.base in ("HMMA", "IMMA", "QMMA", "OMMA") for x in body)
        per_kt = h["block_M"] * h["block_N"] * h["block_K"] / (16 * 8 * 16) / warps
        u = max(1, round(n_mma / per_kt)) if n_mma else 1; trips = max(1, KT // u)
    else:
        trips = 1
    waits = [i for i, x in enumerate(ins) if x.op.startswith("SYNCS.PHASECHK") and "TRYWAIT" in x.op]
    tmas = [i for i, x in enumerate(ins) if x.base == "UTMALDG"]
    if tmas and len(waits) >= 2:
        w0 = max(i for i in waits if i < tmas[0]); w1 = min((i for i in waits if i > tmas[0]), default=len(ins))
        tma_per_kt = max(1, sum(1 for i in tmas if w0 < i < w1))
    else:
        tma_per_kt = 1
    # out-of-range rows and columns of a box are zero-filled by the TMA unit without moving data
    tile = (min(h["block_M"], shape["M"]) + min(h["block_N"], shape["N"])) * h["block_K"] * 2
    sm = WSSM(ins, h["threads"] + 128, nblocks, trips, h["threads"], h["num_stages"], tile, tma_per_kt, kt=KT, active_sms=blocks_eq,
              miss_frac=miss, dram_clk_per_512B=dclk if dclk is not None else dram_clk(0.0, 0.0, blocks_eq),
              ctile_bytes=min(h["block_M"], shape["M"]) * min(h["block_N"], shape["N"]) * 2, resident=1, grid_blocks=grid)
    t = time.time(); r = sm.run(limit=3_000_000 * nblocks); r["us"] = r["cycles"] / F; r["wall_s"] = round(time.time() - t, 1)
    r["warps_done"] = sum(w.done for w in sm.warps); r["warps"] = len(sm.warps); return r


def kernel_us(name, h, shape, variant="full", warm=True, sass_path=None):
    tiles = math.ceil(shape["M"] / h["block_M"]) * math.ceil(shape["N"] / h["block_N"]); KT = math.ceil(shape["K"] / h["block_K"])
    sp = dram_split_kslice(h, shape, 48)
    from .kernel import FLOW
    if FLOW:
        rb = secs = 0.0
        for wv in sp["waves"]:
            r_, s_ = wave_dram(h, shape, wv, warm); rb += r_; secs += s_
        # TMA moves only rows and columns inside the matrix (out-of-range rows of a box are zero-filled, no traffic)
        cb = tiles * KT * (min(h["block_M"], shape["M"]) + min(h["block_N"], shape["N"])) * h["block_K"] * 2; eq = min(48.0, float(tiles))
        return 2.6 + ws_sim(name, variant, h, shape, eq, min(1.0, rb / cb) if cb else 0.0, dram_clk(rb, secs, eq), sass_path, nblocks=math.ceil(tiles / 48), grid=tiles)["us"]
    left, t, w = tiles, 2.6, 0
    while left > 0:
        n = min(left, 48); cb = n * KT * (min(h["block_M"], shape["M"]) + min(h["block_N"], shape["N"])) * h["block_K"] * 2
        wv = sp["waves"][w] if w < len(sp["waves"]) else {"A": 0.0, "B": 0.0, "cols": 1, "blocks": n}
        rb, secs = wave_dram(h, shape, wv, warm)
        t += ws_sim(name, variant, h, shape, n, min(1.0, rb / cb) if cb else 0.0, dram_clk(rb, secs, n), sass_path)["us"]; left -= n; w += 1
    return t


if __name__ == "__main__":
    m = json.load(open(AB / "mechanisms.json")); names = sys.argv[1:] or list(m)
    print(f"{'pick':6s} | one block: mma-only sim vs meas | whole sim vs meas | whole kernel at grid: sim vs meas")
    for name in names:
        r = m[name]; h = r["pick"]; s = r["shape"]; KT = r["KT"]; o = r["overlap"]
        nol = ws_sim(name, "noload", h, s); full1 = ws_sim(name, "full", h, s)
        meas_c = r["compute_per_ktile_us"] * KT + (r.get("block_setup_us") or 0.1)
        k = kernel_us(name, h, s)
        print(f"{name:6s} | {nol['us']:7.1f} vs {meas_c:7.1f} ({(nol['us']/meas_c-1)*100:+4.0f}%) [{nol['warps_done']}/{nol['warps']}] | {full1['us']:7.1f} | {k:8.1f} vs {r['full_event_us']:8.1f} ({(k/r['full_event_us']-1)*100:+4.0f}%)", flush=True)
