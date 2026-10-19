"""Simulate a whole TileLang cp.async GEMM kernel on the GB10: occupancy from the driver's limits, waves as the block
scheduler fills SMs, each wave on the SM simulator with its blocks per SM, its share of the chip's L2 and store limits
(blocks in flight), and its DRAM misses from the launch-order L2 simulation (agentic.gb10.model.dram_waves, checked
against ncu). Plus the launch. Timing condition: warm back-to-back runs (operands that fit the L2 stay there)."""
from __future__ import annotations
import json, math
from pathlib import Path
from .sm import SM, kernel_sass, P
from .sass import main_loop
from ..model import dram_waves_kslice as dram_waves, C as MC

F = 2.54e3   # cycles per us under load (clock probes 2.52-2.54 GHz)


def copy_lines(h: dict) -> list:
    def lines(row):
        rows = max(1, 512 // row); return rows * max(1, min(row, 512) // 128)
    nA = h["block_M"] * h["block_K"] * 2 // 16 // h["threads"]; nB = h["block_K"] * h["block_N"] * 2 // 16 // h["threads"]
    return [lines(h["block_K"] * 2)] * max(1, nA) + [lines(h["block_N"] * 2)] * max(1, nB)


def sass_trips(ins, h: dict, src_trips: int) -> int:
    lp = main_loop(ins)
    if not lp: return 1
    body = ins[lp[0]:lp[1] + 1]; warps = h["threads"] // 32
    n_mma = sum(x.base == "HMMA" for x in body); n_cp = sum(x.base == "LDGSTS" for x in body)
    if n_mma: u = n_mma / (h["block_M"] * h["block_N"] * h["block_K"] / (16 * 8 * 16) / warps)
    elif n_cp: u = n_cp / ((h["block_M"] + h["block_N"]) * h["block_K"] * 2 / 16 / h["threads"])
    else: u = 1
    return max(1, round(src_trips / max(1, round(u))))


def occupancy(h: dict, regs: int, smem: int) -> int:
    warps = h["threads"] // 32; rw = math.ceil(regs * 32 / 256) * 256
    return max(1, min(65536 // (warps * rw), 102400 // (smem + 1024), 1536 // h["threads"], 24))


def dram_clk(h: dict, shape: dict, blocks_eq: float) -> float:
    """Cycles per 512 B of misses for one SM: the DRAM peak for the row segment the misses read (probe: 64-byte rows
    204 GB/s, 256-byte 233, long streams 237), shared by the blocks in flight."""
    seg = min(h["block_K"], h["block_N"]) * 2 if shape["M"] >= 256 else h["block_N"] * 2
    pts = MC["dram_peak_by_segment_GBs"]; peak = pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if seg <= x1: peak = y0 + (y1 - y0) * (math.log(max(seg, x0)) - math.log(x0)) / (math.log(x1) - math.log(x0)); break
    return 512 / (peak * 1e9 / 2.54e9 / blocks_eq)


def simulate_kernel(sass_path, h: dict, shape: dict, regs: int, smem: int | None = None, warm: bool = True) -> dict:
    ins = kernel_sass(sass_path)
    M, N, K = shape["M"], shape["N"], shape["K"]; KT = math.ceil(K / h["block_K"])
    smem = smem if smem is not None else h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    tiles = math.ceil(M / h["block_M"]) * math.ceil(N / h["block_N"]); Rmax = occupancy(h, regs, smem)
    trips = sass_trips(ins, h, KT - (h["num_stages"] - 1))
    traffic, waves = dram_waves(h, shape, 48 * Rmax)
    fits = M * K * 2 + K * N * 2 + M * N * 2 <= MC["l2_effective_bytes"]
    left, t, w, parts = tiles, 2.6, 0, []
    while left > 0:
        n = min(left, 48 * Rmax); R = math.ceil(n / 48); eq = n / R
        copy_bytes = n * KT * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
        wb = 0.0 if (warm and fits) else (waves[w] if w < len(waves) else 0.0)
        miss = min(1.0, wb / copy_bytes) if copy_bytes else 0.0
        sm = SM(ins, h["threads"], R, trips, active_sms=eq, copy_lines=copy_lines(h), miss_frac=miss, dram_clk_per_512B=dram_clk(h, shape, eq))
        r = sm.run(); us = r["cycles"] / F; t += us
        parts.append({"blocks": n, "per_sm": R, "miss_frac": round(miss, 3), "us": round(us, 2), "busy": {k: round(v / max(1, r["cycles"]), 2) for k, v in r["busy"].items()}})
        left -= n; w += 1
    return {"us": t, "tiles": tiles, "blocks_per_sm": Rmax, "waves": w, "dram_MB": traffic / 1e6, "parts": parts}
