"""Plain-load kernels on the simulator: TileLang's vectorized split-K GEMV (examples/gemv) and split-K GEMM with global
atomics (examples/gemm_splitk), against warm timings (runs/tilelang/item5/measured.json). python -m agentic.gb10.sim.gemv

GEMV, what the machine code does: each thread issues 16-byte loads of the vector A and of its row of B for every
k-step (unrolled), multiplies, then adds its partial sum into a shared-memory slot with a compare-and-swap retry loop.
Which load reads what comes from the code itself: a load whose address depends on the block index reads B (each block
its own rows), the others read A (the same for every block, so L1 holds it after the first touch). Lines per load
come from the lane layout: a warp is BLOCK_N rows x (32 / BLOCK_N) lanes along k, 16 bytes per lane."""
from __future__ import annotations
import json, math
from pathlib import Path
from .sm import SM, kernel_sass
from .kernel import F, occupancy, simulate_kernel, dram_clk, dram_peak
from ..model import C as MC

ROOT = Path(__file__).resolve().parents[3]; D = ROOT / "runs" / "tilelang" / "item5"


def block_dependent_loads(ins) -> list:
    """For each LDG in program order: does its address depend on the block index (S2R SR_CTAID)?"""
    taint = set(); out = []
    for x in ins:
        args = [a.strip() for a in x.args.split(",")]
        if x.base == "S2R" and "SR_CTAID" in x.args: taint.add(args[0]); continue
        if x.base == "LDG":
            regs = [r for r in __import__("re").findall(r"\bR\d+", x.args.split(",", 1)[1] if "," in x.args else x.args)]
            out.append(any(r in taint for r in regs)); continue
        if not args or not args[0].startswith("R"): continue
        srcs = __import__("re").findall(r"\bR\d+", ",".join(args[1:]))
        dst = args[0]
        if any(r in taint for r in srcs): taint.add(dst); taint.add("R%d" % (int(dst[1:]) + 1))   # wide ops write a pair
        else: taint.discard(dst)
    return out


def gemv_us(name: str, m: dict, warm: bool = True) -> float:
    ins = kernel_sass(D / "sass" / f"{name}.sass", kernel=m["kernel_name"])
    bn, rt, n, k = m["BLOCK_N"], m["reduce_threads"], m["N"], m["K"]
    per_row = max(1, 32 // bn) * 16                                 # bytes of one row a warp's lanes read per k-step
    a_lines = max(1, math.ceil(per_row / 128)); b_lines = min(bn, 32) * a_lines
    lines = [(b_lines, 512, False) if dep else (a_lines, per_row, True) for dep in block_dependent_loads(ins)]
    h = {"threads": m["threads"]}; R = occupancy({"threads": m["threads"]}, m["regs"], 0)
    # loops: the k-loop (if the compiler kept one) runs K / BLOCK_K / (B loads per pass) times; the shared-memory
    # reduction is a compare-and-swap loop: each of the reduce_threads lanes adding into one slot can make another
    # fail, so up to reduce_threads rounds (ncu: 30 rounds per warp at BLOCK_N 2, 19 at 8)
    deps = block_dependent_loads(ins); ldg_idx = [i for i, x in enumerate(ins) if x.base == "LDG"]
    loops = {}; main = None
    for i, x in enumerate(ins):
        if x.base == "BRA" and x.target is not None and x.target < x.addr and (x.pred or x.op.startswith("BRA.U")):
            s0 = next(j for j, y in enumerate(ins) if y.addr == x.target)
            nb_loads = sum(1 for j, d in zip(ldg_idx, deps) if s0 <= j <= i and d)
            if nb_loads: loops[i] = max(1, k // (rt * 8 * 2 // 2) // nb_loads)        # BLOCK_K = reduce_threads x 8 halves
            elif any(y.base == "ATOMS" for y in ins[s0:i + 1]): loops[i] = rt
    trips = 1
    # B from DRAM unless it stays in the L2 between back-to-back runs (probe stream_b2b.cu)
    b_bytes = n * k * 2; curve = MC["l2_carryover_b2b"]; mb = b_bytes / 2 ** 20; hit = 1.0
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= mb <= x1: hit = y0 + (y1 - y0) * (mb - x0) / (x1 - x0); break
    else:
        hit = 1.0 if mb < curve[0][0] else curve[-1][1]
    if not warm: hit = 0.0                                          # the L2 cleared before the run
    blocks = m["blocks"]; left, t = blocks, 2.6
    from .kernel import FLOW
    if FLOW:
        eq = min(blocks, 48 * R) / min(R, math.ceil(blocks / 48)); rb = blocks * bn * k * 2 * (1 - hit); secs = rb / (dram_peak(4096) * 1e9)
        sm = SM(ins, m["threads"], math.ceil(blocks / 48), 1, active_sms=eq, ldg_lines=lines, miss_frac=1 - hit,
                dram_clk_per_512B=dram_clk(rb, secs, eq), loop_trips=loops, resident=R, grid_blocks=blocks)
        sm.loop_end = -1
        return t + sm.run()["cycles"] / F
    while left > 0:
        nb = min(left, 48 * R); Rw = math.ceil(nb / 48); eq = nb / Rw
        rb = nb * bn * k * 2 * (1 - hit); secs = rb / (dram_peak(4096) * 1e9)
        sm = SM(ins, m["threads"], Rw, trips, active_sms=eq, ldg_lines=lines, miss_frac=1 - hit, dram_clk_per_512B=dram_clk(rb, secs, eq), loop_trips=loops)
        sm.loop_end = -1
        t += sm.run()["cycles"] / F; left -= nb
    return t


def splitk_us(name: str, m: dict, warm: bool = True) -> float:
    h = {"block_M": m["bm"], "block_N": m["bn"], "block_K": m["bk"], "num_stages": m["stages"], "threads": m["threads"], "swizzle": 0}
    shape = {"M": m["M"], "N": m["N"], "K": m["K"]}
    return simulate_kernel(D / "sass" / f"{name}.sass", h, shape, m["regs"], split=m["split"], kernel=m["kernel_name"], warm=warm)["us"]


def run() -> list:
    """Two timing conditions, both measured (runs/tilelang/item5/measured2.json): back-to-back (warm) and after a clean
    L2 flush (cold). Back-to-back timing of working sets just above the L2 varies by about 10 % between sessions
    (what survives in the L2 from the previous run), so the cold column is the stable check there."""
    meta = json.load(open(D / "sass" / "meta.json")); meas = json.load(open(D / "measured2.json")); rows = []
    for name, m in meta.items():
        f = gemv_us if m["kind"] == "gemv" else splitk_us
        w, c = f(name, m, True), f(name, m, False); mw, mc = meas[name]["warm_us"], meas[name]["flushed_us"]
        rows.append({"kernel": name, "sim_warm_us": round(w, 2), "meas_warm_us": round(mw, 2), "sim_cold_us": round(c, 2), "meas_cold_us": round(mc, 2)})
        print(f"{name:40s} warm {w:7.1f} vs {mw:7.1f} ({(w / mw - 1) * 100:+4.0f}%) | cold {c:7.1f} vs {mc:7.1f} ({(c / mc - 1) * 100:+4.0f}%)", flush=True)
    json.dump(rows, open(ROOT / "runs" / "gb10" / "sim_item5.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    run()
