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


def launched_threads(threads: int, warp_spec: int = 1) -> int:
    return threads + (PRODUCER_THREADS if warp_spec else 0)


def regs_per_thread(threads: int, chip: dict, warp_spec: int = 1, maxrreg: int = 0) -> int:
    """ptxas fills the launch-bounds budget, rounded down to a multiple of 8, capped at 255 (or at maxrregcount)."""
    budget = chip["regs_per_sm"] // launched_threads(threads, warp_spec)
    budget = min(255, budget - budget % 8)
    return min(budget, maxrreg) if maxrreg else budget


def spills(h: dict, chip: dict) -> bool:
    """The accumulators plus about 24 registers over ptxas's launch-bounds budget: the kernel spills (5x slower at 1024^3)."""
    return h["block_M"] * h["block_N"] // h["threads"] + 24 > regs_per_thread(h["threads"], chip, h.get("warp_spec", 1), h.get("maxrreg", 0))


def blocks_per_sm(h: dict, chip: dict, dyn_smem: int | None = None) -> int:
    smem = dyn_smem if dyn_smem is not None else h["num_stages"] * (h["block_M"] * h["block_K"] + h["block_K"] * h["block_N"]) * 2
    smem += 1024                                          # the static part (mbarriers)
    lt = launched_threads(h["threads"], h.get("warp_spec", 1))
    by_smem = chip["smem_bytes_per_sm"] // max(1, smem)
    by_regs = chip["regs_per_sm"] // (regs_per_thread(h["threads"], chip, h.get("warp_spec", 1), h.get("maxrreg", 0)) * lt)
    by_threads = chip.get("max_threads_per_sm", 1536) // lt
    return max(0, min(by_smem, by_regs, by_threads))


def default_terms(chip: dict) -> dict:
    """First guesses from the measured contract; calibrate.py replaces them with measured values."""
    return {
        "F_tc_per_sm": chip["tensor_flops_per_s_per_sm"]["fp16"],       # tensor-core FLOP/s per SM (B31 mma rate)
        "u_warps": {128: 1.0, 256: 1.0, 512: 1.0},                        # tensor-pipe efficiency by consumer threads per block
        "B_HBM": chip["dram_bytes_per_s"],                                # bytes/s from DRAM
        "B_L2_per_sm": chip["tma_bytes_per_s_per_sm_by_residency"].get("tile=16384,resident=1", 3.1e10),  # TMA bytes/s per SM, one block
        # the chip-wide L2 rate TMA loads reach (the TMA/cp.async mix benchmark), above the plain streaming L2 rate
        "B_L2_chip": max(chip["l2_bytes_per_s"], max((chip.get("tma_mix_bytes_per_s") or {}).values(), default=0.0)),
        "l2_bytes_effective": chip["l2_bytes_effective"],
        "t_tile_latency": chip["tma_bulk_latency_s"].get("16384", 5e-7),  # one TMA tile from L2
        "t_tile_fixed": {128: 0.0, 256: 0.0, 512: 0.0},                   # per tile: prologue, fill, the C store (seconds)
        "t_step": {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},                       # per k-step, by pipeline depth (seconds)
        "t_fill": {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},                       # per tile: filling the pipeline, by depth (seconds)
        "launch_s": chip["launch_s"],
        "overlap": "max",                                                 # how copy and tensor time combine per tile: max | sum
        "t_prologue": 0.0,                                                # round 3, mechanism 1: paid once per launch (seconds)
        "R_block": 0.0,                                                   # round 3, mechanism 2: a fixed per-block DRAM streaming rate (bytes/s); 0 = off
        "L_block": 0.0,                                                   # round 3, mechanism 2: per-block rate = bytes in flight / this latency (s); 0 = off
        "scale": 1.0,                                                     # one-sided deflation of the per-SM time (calibrate.py)
        "swizzle_penalty": {0: 1.0, 4: 1.0, 8: 1.0, 16: 1.0},             # bank-conflict / traversal factor by SWZ
        # round 2: one factor per new knob value on the per-SM time, from the knob curves (calibrate.fit_knob_curves);
        # a factor is the smallest ratio measured against the round-1 base, so the product stays a lower bound
        "knob_factors": {},
        "source": "contract defaults",
    }


def traffic_bytes(h: dict, shape: dict, terms: dict, chip: dict | None = None, bps: int = 1) -> dict:
    """Global-memory bytes. The SMs pull A once per N tile and B once per M tile through L2 (l2_reads). What reaches DRAM
    depends on which blocks run together: a wave of W = SMs x blocks-per-SM blocks walks C in panels of p rows
    (p = the swizzle panel, or as many rows as a wave covers in TileLang's default N-fastest order). The blocks of one
    panel share each B column-slice through L2 as they stream it, so B comes from DRAM once per panel of M tiles; an A
    row-panel comes once per group of columns a wave covers. A lower bound on traffic, which keeps J-hat one-sided."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    bm, bn = h["block_M"], h["block_N"]
    tiles_m, tiles_n = math.ceil(M / bm), math.ceil(N / bn)
    a_bytes, b_bytes, c_bytes = M * K * 2, K * N * 2, M * N * 2
    l2_reads = a_bytes * tiles_n + b_bytes * tiles_m
    W = (chip["sms"] if chip else 48) * max(1, bps)
    p = float(h.get("swizzle", 0) or max(1.0, W / tiles_n))      # rows of tiles that share B at the same time
    p = max(1.0, min(p, float(tiles_m)))
    cols_per_wave = max(1.0, W / p)
    if a_bytes + b_bytes <= terms["l2_bytes_effective"]:
        dram = a_bytes + b_bytes + c_bytes
    else:
        # R2: a lower bound on DRAM traffic. A comes once (ncu on the 4096^3 swizzle series shows L2 keeps the panel's
        # A rows). B is streamed once per panel of rows; between two panels the whole of B passes through L2 again and
        # L2 keeps the fraction that fits, so each later panel re-reads only (1 - kept) of B. Measured DRAM bytes for
        # panels 0 / 4 / 8 / 16 at 128x128: 770 / 245 / 170 / 211 MB; this bound gives 422 / 208 / 144 / 112 MB.
        kept = min(1.0, terms["l2_bytes_effective"] / b_bytes)
        b_reads = 1.0 + max(0.0, tiles_m / p - 1.0) * (1.0 - kept)
        dram = a_bytes + b_bytes * b_reads + c_bytes
    return {"l2_reads": l2_reads, "dram": dram, "c_bytes": c_bytes, "panel": p, "wave_blocks": W}


def _lookup(table: dict, key):
    return table.get(key, table.get(str(key), 0.0))


def predict(h: dict, shape: dict, chip: dict, terms: dict | None = None) -> dict:
    t = terms or default_terms(chip)
    M, N, K = shape["M"], shape["N"], shape["K"]
    bm, bn, bk, st, th = h["block_M"], h["block_N"], h["block_K"], h["num_stages"], h["threads"]
    sk = int(h.get("split_k", 1) or 1)
    tiles = math.ceil(M / bm) * math.ceil(N / bn) * sk
    bps = blocks_per_sm(h, chip)
    if bps == 0:
        return {"ok": False, "reason": "does not fit one block per SM", "time_s": math.inf}
    sms = chip["sms"]
    waves = math.ceil(tiles / (sms * bps))
    # per tile: tensor time at the per-SM rate times the warp efficiency; copy time through L2 at the per-SM TMA rate
    k_steps = math.ceil(K / sk / bk)
    flops_tile = 2.0 * bm * bn * (k_steps * bk)
    u = t["u_warps"].get(th, t["u_warps"].get(str(th), 1.0))
    ut = (t.get("u_tile") or {}).get(th) or (t.get("u_tile") or {}).get(str(th))      # R2: the rate by (threads, tile) when known
    if ut and f"{bm}x{bn}@{K}" in ut:
        u = ut[f"{bm}x{bn}@{K}"]
    elif ut and f"{bm}x{bn}" in ut:
        u = ut[f"{bm}x{bn}"]
    t_compute_tile = flops_tile / (t["F_tc_per_sm"] * u) / bps if bps else math.inf
    bytes_tile = k_steps * (bm + bn) * bk * 2
    t_copy_tile = bytes_tile / t["B_L2_per_sm"]
    swz = t["swizzle_penalty"].get(h.get("swizzle", 0), t["swizzle_penalty"].get(str(h.get("swizzle", 0)), 1.0))
    fixed = _lookup(t["t_tile_fixed"], th); step = _lookup(t["t_step"], st); fill = _lookup(t.get("t_fill", {}), st)
    t_pipe = fixed + fill                                                 # per tile: start-up and filling the pipeline (0 under R2's envelopes)
    if t.get("overlap", "max") == "sum":        # copies and tensor work charged one after the other
        t_tile = (t_compute_tile + t_copy_tile) * swz + k_steps * step
    else:                                        # perfect overlap: the longer of the two
        t_tile = max(t_compute_tile, t_copy_tile) * swz + k_steps * step
    startup = (t.get("tile_startup") or {}).get(f"{bm}x{bn}", 0.0)       # round 3: the start-up a tile pays before its pipeline is full (series A)
    t_sm = waves * (t_tile + t_pipe + t.get("t_wave_startup", 0.0) + startup)
    kf = 1.0
    for knob, table in (t.get("knob_factors") or {}).items():
        if knob in h:
            kf *= float(table.get(h[knob], table.get(str(h[knob]), 1.0)))
    t_sm *= kf
    tr = traffic_bytes(h, shape, t, chip, bps)
    if sk > 1:                                                           # the atomics: each part reads and writes its FP32 tile of C
        tr["dram"] += 2.0 * M * N * 4 * sk - M * N * 2
        tr["atomic_bytes"] = 2.0 * M * N * 4 * sk
    t_dram = tr["dram"] / t["B_HBM"]
    R_block = t.get("R_block", 0.0) or 0.0
    L_block = t.get("L_block", 0.0) or 0.0
    if L_block > 0:                                                      # mechanism 2 in the brief's form: bytes in flight / latency
        R_block = st * (bm + bn) * bk * 2 / L_block
    if R_block > 0:
        # mechanism 2: each wave streams its share of the bytes at min(chip rate, blocks in the wave x per-block rate)
        per_wave = sms * bps
        t_dram = 0.0
        left = tiles
        while left > 0:
            blocks_w = min(per_wave, left)
            t_dram += tr["dram"] * (blocks_w / tiles) / min(t["B_HBM"], blocks_w * R_block)
            left -= blocks_w
    t_l2 = tr["l2_reads"] / t["B_L2_chip"]
    time = max(t_sm, t_dram, t_l2) * t.get("scale", 1.0) + t["launch_s"] + t.get("t_prologue", 0.0)   # mechanism 1: the prologue, once per launch
    bound = max(("tensor", t_sm) if t_compute_tile >= t_copy_tile else ("copy", t_sm), ("dram", t_dram), ("l2", t_l2), key=lambda x: x[1])[0]
    return {"ok": True, "time_s": time, "time_us": time * 1e6, "bound": bound, "waves": waves, "tiles": tiles, "blocks_per_sm": bps,
            "terms": {"t_sm": t_sm, "t_compute_tile": t_compute_tile, "t_copy_tile": t_copy_tile, "t_pipe": t_pipe, "t_dram": t_dram, "t_l2": t_l2,
                      "dram_bytes": tr["dram"], "l2_read_bytes": tr["l2_reads"], "regs_per_thread": regs_per_thread(th, chip),
                      "k_steps": k_steps, "flops_tile": flops_tile, "bytes_tile": bytes_tile, "knob_factor": kf,
                      "launched_threads": launched_threads(th)}}
