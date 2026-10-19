"""J-hat for the gemm_basic family: predicted time as a function of the six knobs, the shape and the chip terms.
It is a bound, like the paper's cost model: it should not predict slower than the kernel runs... no: it should not
predict FASTER than measured on more than 5 % of configs (the one-sided test of the brief), while ranking the top
configs right (Spearman >= 0.8 on the top-10). Every term is named and comes from the chip contract or from
calibrate.py; nothing is fitted to a kernel inside this file.

What the sweep established about TileLang on sm_121a (and the model takes as structure, not as a fitted term):
- TileLang adds one producer warp group: a kernel asked for T threads launches T + 128 (128 -> 256, 256 -> 384, 512 -> 640).
- __launch_bounds__(T + 128, 1) lets ptxas use up to 65536 / (T + 128) registers per thread; it does (240, 168, ~60),
  so one block fits per SM in every configuration of the family. Occupancy is therefore warps per SM, not blocks per SM.
- Every configuration lowers to TMA loads plus mma; the pipeline has STAGES tiles in flight per block.
"""
from __future__ import annotations

import math

PRODUCER_THREADS = 128


def launched_threads(threads: int) -> int:
    return threads + PRODUCER_THREADS


def regs_per_thread(threads: int, chip: dict) -> int:
    """ptxas fills the launch-bounds budget, rounded down to a multiple of 8, capped at 255."""
    budget = chip["regs_per_sm"] // launched_threads(threads)
    return min(255, budget - budget % 8)


def spills(h: dict, chip: dict) -> bool:
    """The accumulators plus about 24 registers over ptxas's launch-bounds budget: the kernel spills (5x slower at 1024^3)."""
    return h["block_M"] * h["block_N"] // h["threads"] + 24 > regs_per_thread(h["threads"], chip)


def blocks_per_sm(h: dict, chip: dict, dyn_smem: int | None = None) -> int:
    smem = dyn_smem if dyn_smem is not None else h["num_stages"] * (h["block_M"] * h["block_K"] + h["block_K"] * h["block_N"]) * 2
    smem += 1024                                          # the static part (mbarriers)
    lt = launched_threads(h["threads"])
    by_smem = chip["smem_bytes_per_sm"] // max(1, smem)
    by_regs = chip["regs_per_sm"] // (regs_per_thread(h["threads"], chip) * lt)
    by_threads = chip.get("max_threads_per_sm", 1536) // lt
    return max(0, min(by_smem, by_regs, by_threads))


def default_terms(chip: dict) -> dict:
    """First guesses from the measured contract; calibrate.py replaces them with measured values."""
    return {
        "F_tc_per_sm": chip["tensor_flops_per_s_per_sm"]["fp16"],       # tensor-core FLOP/s per SM (B31 mma rate)
        "u_warps": {128: 1.0, 256: 1.0, 512: 1.0},                        # tensor-pipe efficiency by consumer threads per block
        "B_HBM": chip["dram_bytes_per_s"],                                # bytes/s from DRAM
        "B_L2_per_sm": chip["tma_bytes_per_s_per_sm_by_residency"].get("tile=16384,resident=1", 3.1e10),  # TMA bytes/s per SM, one block
        "B_L2_chip": chip["l2_bytes_per_s"],
        "l2_bytes_effective": chip["l2_bytes_effective"],
        "t_tile_latency": chip["tma_bulk_latency_s"].get("16384", 5e-7),  # one TMA tile from L2
        "t_tile_fixed": {128: 0.0, 256: 0.0, 512: 0.0},                   # per tile: prologue, fill, the C store (seconds)
        "t_step": {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},                       # per k-step, by pipeline depth (seconds)
        "launch_s": chip["launch_s"],
        "overlap": "max",                                                 # how copy and tensor time combine per tile: max | sum
        "scale": 1.0,                                                     # one-sided deflation of the per-SM time (calibrate.py)
        "swizzle_penalty": {0: 1.0, 4: 1.0, 8: 1.0, 16: 1.0},             # bank-conflict / traversal factor by SWZ
        "source": "contract defaults",
    }


def traffic_bytes(h: dict, shape: dict, terms: dict) -> dict:
    """Global-memory bytes: each A row-panel is read once per N tile, each B column-panel once per M tile, C written once.
    When A and B fit the effective L2, DRAM sees each byte once; otherwise the swizzle panel decides how much of the
    re-reading L2 absorbs. The model charges the panel's reuse: with panel p, p consecutive tiles share one A panel."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    bm, bn = h["block_M"], h["block_N"]
    tiles_m, tiles_n = math.ceil(M / bm), math.ceil(N / bn)
    a_bytes, b_bytes, c_bytes = M * K * 2, K * N * 2, M * N * 2
    l2_reads = a_bytes * tiles_n + b_bytes * tiles_m                      # what the SMs pull through L2
    if a_bytes + b_bytes <= terms["l2_bytes_effective"]:
        dram = a_bytes + b_bytes + c_bytes
    else:
        # the default order walks N tiles fastest for a fixed M row: the A panel stays, B streams. A panel of p M-rows
        # (use_swizzle) makes p tiles share each B panel. Lower bound on DRAM traffic from that reuse.
        p = max(1, h.get("swizzle", 0) or 1)
        dram = a_bytes + b_bytes * math.ceil(tiles_m / p) + c_bytes
    return {"l2_reads": l2_reads, "dram": dram, "c_bytes": c_bytes}


def _lookup(table: dict, key):
    return table.get(key, table.get(str(key), 0.0))


def predict(h: dict, shape: dict, chip: dict, terms: dict | None = None) -> dict:
    t = terms or default_terms(chip)
    M, N, K = shape["M"], shape["N"], shape["K"]
    bm, bn, bk, st, th = h["block_M"], h["block_N"], h["block_K"], h["num_stages"], h["threads"]
    tiles = math.ceil(M / bm) * math.ceil(N / bn)
    bps = blocks_per_sm(h, chip)
    if bps == 0:
        return {"ok": False, "reason": "does not fit one block per SM", "time_s": math.inf}
    sms = chip["sms"]
    waves = math.ceil(tiles / (sms * bps))
    # per tile: tensor time at the per-SM rate times the warp efficiency; copy time through L2 at the per-SM TMA rate
    k_steps = math.ceil(K / bk)
    flops_tile = 2.0 * bm * bn * (k_steps * bk)
    u = t["u_warps"].get(th, t["u_warps"].get(str(th), 1.0))
    t_compute_tile = flops_tile / (t["F_tc_per_sm"] * u) / bps if bps else math.inf
    bytes_tile = k_steps * (bm + bn) * bk * 2
    t_copy_tile = bytes_tile / t["B_L2_per_sm"]
    swz = t["swizzle_penalty"].get(h.get("swizzle", 0), t["swizzle_penalty"].get(str(h.get("swizzle", 0)), 1.0))
    fixed = _lookup(t["t_tile_fixed"], th); step = _lookup(t["t_step"], st)
    t_pipe = fixed if fixed else min(st, k_steps) * t["t_tile_latency"]   # fill: the first STAGES tiles must land before the pipe is full
    if t.get("overlap", "max") == "sum":        # copies and tensor work charged one after the other
        t_tile = (t_compute_tile + t_copy_tile) * swz + k_steps * step
    else:                                        # perfect overlap: the longer of the two
        t_tile = max(t_compute_tile, t_copy_tile) * swz + k_steps * step
    t_sm = waves * (t_tile + t_pipe) * t.get("scale", 1.0)
    tr = traffic_bytes(h, shape, t)
    t_dram = tr["dram"] / t["B_HBM"]
    t_l2 = tr["l2_reads"] / t["B_L2_chip"]
    time = max(t_sm, t_dram, t_l2) + t["launch_s"]
    bound = max(("tensor", t_sm) if t_compute_tile >= t_copy_tile else ("copy", t_sm), ("dram", t_dram), ("l2", t_l2), key=lambda x: x[1])[0]
    return {"ok": True, "time_s": time, "time_us": time * 1e6, "bound": bound, "waves": waves, "tiles": tiles, "blocks_per_sm": bps,
            "terms": {"t_sm": t_sm, "t_compute_tile": t_compute_tile, "t_copy_tile": t_copy_tile, "t_pipe": t_pipe, "t_dram": t_dram, "t_l2": t_l2,
                      "dram_bytes": tr["dram"], "l2_read_bytes": tr["l2_reads"], "regs_per_thread": regs_per_thread(th, chip),
                      "k_steps": k_steps, "flops_tile": flops_tile, "bytes_tile": bytes_tile,
                      "launched_threads": launched_threads(th)}}
