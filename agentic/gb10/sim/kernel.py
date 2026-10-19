"""Simulate a whole TileLang cp.async GEMM kernel on the GB10: occupancy from the driver's limits, waves as the block
scheduler fills SMs, each wave on the SM simulator with its blocks per SM, its share of the chip's L2 and store limits
(blocks in flight), and its DRAM misses from the launch-order L2 simulation (agentic.gb10.model.dram_waves, checked
against ncu). Plus the launch. Timing condition: warm back-to-back runs (operands that fit the L2 stay there)."""
from __future__ import annotations
import json, math
from pathlib import Path
from .sm import SM, kernel_sass, P
from .sass import main_loop
from ..model import dram_split_kslice, bank_conflict_ways, C as MC

F = 2.54e3   # cycles per us under load (clock probes 2.52-2.54 GHz)
FLOW = True  # block replacement on the busiest SM (a new block starts when one exits) instead of separate waves


def launch_us(blocks: int = 48, graph: bool = False) -> float:
    """Per-kernel overhead in a back-to-back sequence: 2.6 us from one stream (ablation harness; probe launch.cu: the
    host submits at 2.05 us per kernel, 3.1 us between kernels with work), 0.43 us per kernel inside a CUDA graph plus
    about 0.43 ns per block."""
    return 0.43 + 0.00043 * blocks if graph else 2.6


def copy_lines(h: dict) -> list:
    def lines(row):
        rows = max(1, 512 // row); return rows * max(1, min(row, 512) // 128)
    nA = h["block_M"] * h["block_K"] * 2 // 16 // h["threads"]; nB = h["block_K"] * h["block_N"] * 2 // 16 // h["threads"]
    return [lines(h["block_K"] * 2)] * max(1, nA) + [lines(h["block_N"] * 2)] * max(1, nB)


def sass_trips(ins, h: dict, src_trips: int) -> int:
    lp = main_loop(ins)
    if not lp: return 1
    body = ins[lp[0]:lp[1] + 1]; warps = h["threads"] // 32
    n_mma = sum(x.base in ("HMMA", "IMMA", "QMMA", "OMMA") for x in body); n_cp = sum(x.base == "LDGSTS" for x in body)
    if n_mma: u = n_mma / (h["block_M"] * h["block_N"] * h["block_K"] / (16 * 8 * 16) / warps)
    elif n_cp: u = n_cp / ((h["block_M"] + h["block_N"]) * h["block_K"] * 2 / 16 / h["threads"])
    else: u = 1
    return max(1, round(src_trips / max(1, round(u))))


def occupancy(h: dict, regs: int, smem: int) -> int:
    warps = h["threads"] // 32; rw = math.ceil(regs * 32 / 256) * 256
    return max(1, min(65536 // (warps * rw), 102400 // (smem + 1024), 1536 // h["threads"], 24))


CPU_DRAM_GBs = 0.0     # environment: DRAM bandwidth the host CPU is using at the same time (cpuhog.c measured up to ~150)


def dram_peak(run_bytes: float) -> float:
    """DRAM read rate for misses that arrive as contiguous runs of run_bytes (probe tileload.cu, DRAM source, 48-96
    blocks: 64-byte runs 201-204 GB/s, 256-byte 230-234; mlp_stream contiguous 237), less what the CPU is using."""
    pts = MC["dram_peak_by_segment_GBs"]; peak = pts[-1][1]
    if run_bytes <= pts[0][0]: peak = pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 < run_bytes <= x1: peak = y0 + (y1 - y0) * (math.log(run_bytes) - math.log(x0)) / (math.log(x1) - math.log(x0)); break
    return max(10.0, peak - CPU_DRAM_GBs * (237.0 / 273.0))


def wave_dram(h: dict, shape: dict, wave: dict, warm: bool = True) -> tuple[float, float]:
    """One wave's DRAM work: (read-miss bytes, seconds of DRAM time for the whole chip).
    A misses come as block_K*2-byte pieces of rows far apart; B misses come as pieces of the same B rows that the
    neighbouring column tiles in flight read together, one run of (columns in flight) x block_N*2 bytes. C tiles go
    back to DRAM when operands and output do not stay in the L2 together (a write-back cache: a line written in one run
    is evicted before the next run writes it), at the write rate; the controller serves the mix as each part at its own
    rate (probe dram.cu: 1 / (f_read/242 + f_write/197) within 4 %)."""
    M, N, K = shape["M"], shape["N"], shape["K"]; bm, bn, bk = h["block_M"], h["block_N"], h["block_K"]
    if warm and (M * K + K * N + M * N) * 2 <= MC["l2_effective_bytes"]: return 0.0, 0.0
    run_a = min(bk, K) * 2 if K > bk else min(bm, M) * K * 2
    run_b = min(N, wave["cols"] * bn) * 2
    c_w = wave.get("c_bytes", wave["blocks"] * min(bm, M) * min(bn, N) * 2)
    if not warm:
        # timed after a flush: C ends the kernel dirty in the L2 and goes back to DRAM later (in the next flush), except
        # what the kernel itself pushes out because C does not fit beside the stream
        cap = MC["l2_effective_bytes"]; c_all = M * N * 2
        c_w *= max(0.0, c_all - cap) / c_all if c_all else 0.0
    wpeak = max(10.0, MC["dram_write_GBs"] - CPU_DRAM_GBs * (197.0 / 273.0))
    c_r = 0.0 if warm else wave.get("c_read", 0.0)                # C rows are contiguous: streamed like B
    secs = (wave["A"] / dram_peak(run_a) + (wave["B"] + c_r) / dram_peak(run_b) + c_w / wpeak) / 1e9
    return wave["A"] + wave["B"] + c_r, secs


def dram_clk(read_bytes: float, secs: float, blocks_eq: float) -> float:
    """Cycles per 512 B of read misses for one SM: the wave's DRAM time spread over its read misses (writes folded in),
    shared by the blocks in flight."""
    if read_bytes <= 0: return 512 / (237e9 / 2.54e9 / blocks_eq)
    return 512 * (secs / read_bytes) * 2.54e9 * blocks_eq


def ldsm_ways(h: dict) -> tuple:
    """Bank-conflict ways of ldmatrix on A (rows block_K*2 bytes apart) and B (read transposed, rows block_N*2 apart):
    TileLang's inferred and swizzled layouts are conflict-free; a linear layout keeps the row stride."""
    if h.get("smem_layout") != "linear": return (1, 1)
    return (bank_conflict_ways(h["block_K"] * 2), bank_conflict_ways(h["block_N"] * 2))


def simulate_kernel(sass_path, h: dict, shape: dict, regs: int, smem: int | None = None, warm: bool = True, split: int = 1,
                    kernel: str = "matmul_kernel", atomic_c: bool = False) -> dict:
    """split > 1: split-K, a third grid dimension (outermost in launch order); each block runs K/split and adds its C
    tile with global atomics (in the machine code)."""
    ins = kernel_sass(sass_path, kernel=kernel)
    M, N, K = shape["M"], shape["N"], shape["K"]; KT = math.ceil(K / split / h["block_K"])
    smem = smem if smem is not None else h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    tiles = math.ceil(M / h["block_M"]) * math.ceil(N / h["block_N"]) * split; Rmax = occupancy(h, regs, smem)
    trips = sass_trips(ins, h, KT - (h["num_stages"] - 1))
    if split == 1:
        sp = dram_split_kslice(h, shape, 48 * Rmax)
        if atomic_c:   # C added with atomics (fp32): lines not in the L2 are read from DRAM first
            sp = {"waves": [dict(w_, c_read=w_["blocks"] * min(h["block_M"], M) * min(h["block_N"], N) * 4) for w_ in sp["waves"]], "c_over": sp["c_over"]}
    else:
        # each split is its own GEMM over K/split with its own A columns and B rows; spread its misses over its blocks
        one = dram_split_kslice(h, {"M": M, "N": N, "K": K // split}, 48 * Rmax); t1 = tiles // split
        a_b = sum(w["A"] for w in one["waves"]) / t1; b_b = sum(w["B"] for w in one["waves"]) / t1
        sp = {"waves": [], "c_over": one["c_over"]}; left_ = tiles
        while left_ > 0:
            # the fp32 C stays in the L2 while the splits add into it; it goes back to DRAM once per run
            # atomics on C lines not in the L2 make it read them from DRAM first (probe ldg_atomic.cu redtile): after a
            # flush the first split to touch a C line pays a read of it
            nb_ = min(left_, 48 * Rmax); sp["waves"].append({"A": nb_ * a_b, "B": nb_ * b_b, "cols": min(nb_, math.ceil(N / h["block_N"])), "blocks": nb_,
                                                           "c_bytes": nb_ / split * min(h["block_M"], M) * min(h["block_N"], N) * 4,
                                                           "c_read": nb_ / split * min(h["block_M"], M) * min(h["block_N"], N) * 4}); left_ -= nb_
    traffic = 0.0
    if FLOW:
        rb = secs = 0.0
        for wv in sp["waves"]:
            r_, s_ = wave_dram(h, shape, wv, warm); rb += r_; secs += s_
        copy_bytes = tiles * KT * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
        miss = min(1.0, rb / copy_bytes) if copy_bytes else 0.0; eq = min(tiles, 48 * Rmax) / min(Rmax, math.ceil(tiles / 48))
        sm = SM(ins, h["threads"], math.ceil(tiles / 48), trips, active_sms=eq, copy_lines=copy_lines(h), miss_frac=miss,
                dram_clk_per_512B=dram_clk(rb, secs, eq), ctile_bytes=min(h["block_M"], M) * min(h["block_N"], N) * 2,
                ldsm_ways=ldsm_ways(h), resident=Rmax, grid_blocks=tiles,
                grid_dim=(math.ceil(N / h["block_N"]), math.ceil(M / h["block_M"]), split))
        r = sm.run(); us = r["cycles"] / F
        part = {"counts": {k: v for k, v in r.items() if k in ("STSM", "UTMASTG")}, "blocks": tiles, "per_sm": Rmax, "miss_frac": round(miss, 3),
                "dram_GBs": round(rb / secs / 1e9, 1) if secs else None, "us": round(us, 2), "busy": {k: round(v / max(1, r["cycles"]), 2) for k, v in r["busy"].items()}}
        return {"us": 2.6 + us, "tiles": tiles, "blocks_per_sm": Rmax, "waves": math.ceil(tiles / (48 * Rmax)), "dram_MB": rb / 1e6, "parts": [part]}
    left, t, w, parts = tiles, 2.6, 0, []
    while left > 0:
        n = min(left, 48 * Rmax); R = math.ceil(n / 48); eq = n / R
        copy_bytes = n * KT * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
        wv = sp["waves"][w] if w < len(sp["waves"]) else {"A": 0.0, "B": 0.0, "cols": 1, "blocks": n}
        rb, secs = wave_dram(h, shape, wv, warm); traffic += rb
        miss = min(1.0, rb / copy_bytes) if copy_bytes else 0.0
        sm = SM(ins, h["threads"], R, trips, active_sms=eq, copy_lines=copy_lines(h), miss_frac=miss, dram_clk_per_512B=dram_clk(rb, secs, eq),
                ctile_bytes=min(h["block_M"], M) * min(h["block_N"], N) * 2, ldsm_ways=ldsm_ways(h))
        r = sm.run(); us = r["cycles"] / F; t += us
        parts.append({"counts": {k: v for k, v in r.items() if k in ("STSM", "UTMASTG")}, "blocks": n, "per_sm": R, "miss_frac": round(miss, 3), "dram_GBs": round(rb / secs / 1e9, 1) if secs else None, "us": round(us, 2), "busy": {k: round(v / max(1, r["cycles"]), 2) for k, v in r["busy"].items()}})
        left -= n; w += 1
    return {"us": t, "tiles": tiles, "blocks_per_sm": Rmax, "waves": w, "dram_MB": traffic / 1e6, "parts": parts}
